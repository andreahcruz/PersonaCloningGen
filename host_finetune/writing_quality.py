"""Writing-quality judge schema and calibration gate.

The judge is not a model selector until ``calibration_gate`` passes. Voice is
collected only so that calibration can check the expected ordering. It is not
part of the writing-quality mean.
"""
from __future__ import annotations

WRITING_DIMENSIONS = (
    "coherence",
    "fluency",
    "logical_progression",
    "relevance",
    "specificity",
    "usefulness",
)
CALIBRATION_VOICE = "voice"
CALIBRATION_BUCKETS = ("real", "good_persona", "base", "degraded")
SCORE_MIN = 1
SCORE_MAX = 5
# On a 1–5 scale, a half point is the smallest gap treated as separation.
SEPARATION = 0.5


def writing_quality_unavailable(reason: str = "judge_not_run") -> dict:
    return {
        "available": False,
        "role": "diagnostic_only",
        "calibrated": False,
        "reason": reason,
        "dimensions": {name: None for name in WRITING_DIMENSIONS},
        "mean": None,
        "judge_model": None,
        "judge_family_differs_from_candidate": None,
    }


def writing_prompt(draft: str, topic: str, medium: str) -> str:
    """Blind production prompt. It does not name a model, adapter, or author score."""
    lines = "\n".join(f'  "{name}": <int 1-5>,' for name in WRITING_DIMENSIONS)
    return (
        "Score the DRAFT on writing quality. Return JSON only.\n"
        "coherence: logical flow without internal contradiction.\n"
        "fluency: natural, readable sentences.\n"
        "logical_progression: each part follows from the previous one.\n"
        "relevance: stays on the requested topic and medium.\n"
        "specificity: concrete details rather than empty claims.\n"
        "usefulness: a reader could act on or learn from the draft.\n"
        "{\n"
        f"{lines}\n"
        "}\n\n"
        f"Topic: {topic}\n"
        f"Medium: {medium}\n\n"
        f"DRAFT:\n{draft}\n"
    )


def judge_prompt(draft: str, topic: str, medium: str) -> str:
    """Blind writing-quality prompt. It does not ask the judge to prefer a named adapter."""
    lines = "\n".join(f'  "{name}": <int 1-5>,' for name in (*WRITING_DIMENSIONS, CALIBRATION_VOICE))
    return (
        "Score the DRAFT. Return JSON only.\n"
        "coherence: logical flow without internal contradiction.\n"
        "fluency: natural, readable sentences.\n"
        "logical_progression: each part follows from the previous one.\n"
        "relevance: stays on the requested topic and medium.\n"
        "specificity: concrete details rather than empty claims.\n"
        "usefulness: a reader could act on or learn from the draft.\n"
        "voice: how much the draft sounds like Jason Lemkin. This item is for "
        "calibration ordering only and is not a writing-quality score.\n"
        "{\n"
        f"{lines}\n"
        "}\n\n"
        f"Topic: {topic}\n"
        f"Medium: {medium}\n\n"
        f"DRAFT:\n{draft}\n"
    )


def _in_range(value) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and SCORE_MIN <= float(value) <= SCORE_MAX
    )


def parse_dimension_scores(payload: dict, names: tuple[str, ...]) -> dict | None:
    """Keep a complete in-range score object. A missing dimension is not usable."""
    if not isinstance(payload, dict):
        return None
    parsed = {}
    for name in names:
        value = payload.get(name)
        if not _in_range(value):
            return None
        parsed[name] = float(value)
    return parsed


def parse_writing_scores(payload: dict) -> dict | None:
    """Production writing-quality scores. Voice is not required."""
    return parse_dimension_scores(payload, WRITING_DIMENSIONS)


def parse_judge_scores(payload: dict) -> dict | None:
    """Calibration scores, including the voice item used only for ordering checks."""
    return parse_dimension_scores(payload, (*WRITING_DIMENSIONS, CALIBRATION_VOICE))


def writing_mean(scores: dict) -> float | None:
    values = [scores.get(name) for name in WRITING_DIMENSIONS]
    if any(value is None for value in values):
        return None
    return sum(values) / len(WRITING_DIMENSIONS)


def families_differ(candidate_family: str | None, judge_family: str | None) -> bool:
    if not candidate_family or not judge_family:
        return False
    return candidate_family.strip().lower() != judge_family.strip().lower()


def _mean(rows: list[dict], key: str) -> float | None:
    values = [row[key] for row in rows if row.get(key) is not None]
    if not values:
        return None
    return sum(values) / len(values)


def calibration_gate(samples: list[dict]) -> dict:
    """Pass only when blinded scores match the required ordering.

    Expected, on the 1–5 items ``voice`` and ``coherence``:
    real voice above base and degraded; real and base coherence above degraded;
    good-persona voice above base and good-persona coherence above degraded
    when that bucket is present.
    """
    by_bucket: dict[str, list[dict]] = {name: [] for name in CALIBRATION_BUCKETS}
    for sample in samples:
        bucket = sample.get("bucket")
        voice = sample.get("voice")
        coherence = sample.get("coherence")
        voice_ok = isinstance(voice, (int, float)) and not isinstance(voice, bool) and SCORE_MIN <= float(voice) <= SCORE_MAX
        coherence_ok = (
            isinstance(coherence, (int, float))
            and not isinstance(coherence, bool)
            and SCORE_MIN <= float(coherence) <= SCORE_MAX
        )
        if bucket in by_bucket and voice_ok and coherence_ok:
            by_bucket[bucket].append(sample)
    means = {
        bucket: {
            "voice": _mean(rows, "voice"),
            "coherence": _mean(rows, "coherence"),
            "n": len(rows),
        }
        for bucket, rows in by_bucket.items()
    }
    checks = []

    def require(name: str, ok: bool) -> None:
        checks.append({"name": name, "pass": bool(ok)})

    real = means["real"]
    base = means["base"]
    degraded = means["degraded"]
    present = all(
        means[bucket]["voice"] is not None and means[bucket]["coherence"] is not None
        for bucket in ("real", "base", "degraded")
    )
    require("real_base_degraded_present", present)
    if present:
        require("real_voice_above_base", real["voice"] >= base["voice"] + SEPARATION)
        require("real_voice_above_degraded", real["voice"] >= degraded["voice"] + SEPARATION)
        require("real_coherence_above_degraded", real["coherence"] >= degraded["coherence"] + SEPARATION)
        require("base_coherence_above_degraded", base["coherence"] >= degraded["coherence"] + SEPARATION)
    good = means["good_persona"]
    if good["n"]:
        good_present = good["voice"] is not None and good["coherence"] is not None and present
        require("good_persona_present", good_present)
        if good_present:
            require("good_voice_above_base", good["voice"] >= base["voice"] + SEPARATION)
            require(
                "good_coherence_above_degraded",
                good["coherence"] >= degraded["coherence"] + SEPARATION,
            )
    calibrated = bool(checks) and all(item["pass"] for item in checks)
    return {
        "calibrated": calibrated,
        "role": "selector" if calibrated else "diagnostic_only",
        "reason": "ordering_holds" if calibrated else "judge_failed_sanity_check",
        "separation": SEPARATION,
        "means": means,
        "checks": checks,
    }


def writing_quality_gate(samples: list[dict]) -> dict:
    """Selector gate for the writing judge.

    Real high-quality prose and normal model drafts must both outscore
    deliberately degraded drafts on coherence. Voice is not part of this gate.
    """
    buckets = ("real", "normal", "degraded")
    grouped = {name: [] for name in buckets}
    for sample in samples:
        bucket = sample.get("bucket")
        if bucket in grouped and _in_range(sample.get("coherence")):
            grouped[bucket].append(sample)
    means = {
        bucket: {"coherence": _mean(rows, "coherence"), "n": len(rows)}
        for bucket, rows in grouped.items()
    }
    present = all(means[bucket]["coherence"] is not None for bucket in buckets)
    checks = [{"name": "real_normal_degraded_present", "pass": present}]
    if present:
        checks.append({
            "name": "real_coherence_above_degraded",
            "pass": means["real"]["coherence"] >= means["degraded"]["coherence"] + SEPARATION,
        })
        checks.append({
            "name": "normal_coherence_above_degraded",
            "pass": means["normal"]["coherence"] >= means["degraded"]["coherence"] + SEPARATION,
        })
    calibrated = bool(checks) and all(item["pass"] for item in checks)
    return {
        "calibrated": calibrated,
        "role": "selector" if calibrated else "diagnostic_only",
        "reason": "writing_order_holds" if calibrated else "writing_judge_failed_sanity_check",
        "separation": SEPARATION,
        "means": means,
        "checks": checks,
    }
