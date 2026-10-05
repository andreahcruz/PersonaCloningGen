"""Surface style distance against train-owned posts.

This is a diagnostic. It does not decide which draft sounds like the author.
"""
from __future__ import annotations

import json
import math
import re
import statistics
from pathlib import Path

from host_finetune.train_only_index import (
    DEFAULT_ASSIGNMENTS,
    DEFAULT_DATASET,
    DEFAULT_GOLD_DISPOSITIONS,
    assignment_rows,
    blocked_group_ids,
    load_jsonl,
)

WORD_RE = re.compile(r"[A-Za-z0-9']+")
SENTENCE_RE = re.compile(r"[.!?…]+")
CONTRACTION_RE = re.compile(r"\b\w+'\w+\b")
FIRST_PERSON_RE = re.compile(r"\b(?:I|we|my|our|I'm|we're|I've|we've)\b")
PUNCT_RE = re.compile(r"[,;:—–-]")
SECOND_PERSON_RE = re.compile(r"\b(?:you|your|you're|you've)\b", re.IGNORECASE)
ADVICE_RE = re.compile(
    r"\b(?:should|shouldn't|don't|do not|here's|here is|the answer)\b",
    re.IGNORECASE,
)
SAAS_RE = re.compile(
    r"\b(?:arr|nrr|mrr|churn|acv|quota|pipeline|saas|enterprise|gtm)\b",
    re.IGNORECASE,
)
TRADEOFF_RE = re.compile(
    r"\b(?:but|however|tradeoff|trade-off|instead|rather)\b",
    re.IGNORECASE,
)
FUNCTION_WORDS = (
    "the", "a", "of", "to", "and", "in", "that", "it", "for", "you",
    "we", "i", "is", "are", "was", "be", "this", "but", "not", "on",
)
STYLE_FEATURE_NAMES = (
    "words_per_sentence",
    "questions_per_100w",
    "exclamations_per_100w",
    "contractions_per_100w",
    "numbers_per_100w",
    "first_person_per_100w",
    "punctuation_per_100w",
    "type_token_ratio",
    "paragraphs_per_100w",
    *(f"fw_{word}_per_100w" for word in FUNCTION_WORDS),
)
PERSPECTIVE_FEATURE_NAMES = (
    "second_person_per_100w",
    "advice_cues_per_100w",
    "saas_terms_per_100w",
    "tradeoff_cues_per_100w",
)
# Historical name. Style distance uses the extended surface set, not stance terms.
FEATURE_NAMES = STYLE_FEATURE_NAMES
PLATFORM_TO_MEDIUM = {
    "blog": "blog",
    "linkedin": "linkedin",
    "x": "x",
}
HELD_OUT_PLATFORM_TO_MEDIUM = {
    **PLATFORM_TO_MEDIUM,
    "youtube_jason": "talk",
}
MIN_CALIBRATION_REAL = 4
MIN_CALIBRATION_GENERIC = 2
CALIBRATION_MARGIN = 0.05
NOT_AUTHORSHIP = (
    "Closer surface counts are not authorship accuracy. "
    "This is a style-distance diagnostic, not proof of authorship identity."
)
GENERIC_PROSE = (
    "The committee convened regarding procedural compliance and subsequently approved the quarterly operating plan.",
    "Organizations should leverage synergistic solutions in order to optimize stakeholder outcomes across the enterprise.",
    "A comprehensive framework enables leaders to align strategic initiatives with measurable business objectives.",
    "Furthermore the aforementioned parties commenced an evaluation of cross functional governance during the reporting period.",
)
DEGRADED_PROSE = (
    "word " * 40,
    "!!! ??? the the the the the the the the the the",
)


def _per_100(pattern: re.Pattern[str], raw: str, n_words: int) -> float:
    return 100 * len(pattern.findall(raw)) / n_words


def style_features(text: str) -> dict[str, float]:
    """Surface counts. Stance terms live in ``perspective_features``."""
    raw = text or ""
    words = WORD_RE.findall(raw)
    lowered = [word.lower() for word in words]
    sentences = [part for part in SENTENCE_RE.split(raw) if part.strip()]
    paragraphs = [part for part in re.split(r"\n\s*\n", raw) if part.strip()]
    n_words = max(len(words), 1)
    n_sentences = max(len(sentences), 1)
    counts = {word: 0 for word in FUNCTION_WORDS}
    for word in lowered:
        if word in counts:
            counts[word] += 1
    features = {
        "words_per_sentence": len(words) / n_sentences,
        "questions_per_100w": 100 * raw.count("?") / n_words,
        "exclamations_per_100w": 100 * raw.count("!") / n_words,
        "contractions_per_100w": 100 * len(CONTRACTION_RE.findall(raw)) / n_words,
        "numbers_per_100w": 100 * len(re.findall(r"\d", raw)) / n_words,
        "first_person_per_100w": 100 * len(FIRST_PERSON_RE.findall(raw)) / n_words,
        "punctuation_per_100w": 100 * len(PUNCT_RE.findall(raw)) / n_words,
        "type_token_ratio": len(set(lowered)) / n_words,
        "paragraphs_per_100w": 100 * max(len(paragraphs), 1) / n_words,
        "second_person_per_100w": _per_100(SECOND_PERSON_RE, raw, n_words),
        "advice_cues_per_100w": _per_100(ADVICE_RE, raw, n_words),
        "saas_terms_per_100w": _per_100(SAAS_RE, raw, n_words),
        "tradeoff_cues_per_100w": _per_100(TRADEOFF_RE, raw, n_words),
    }
    for word, count in counts.items():
        features[f"fw_{word}_per_100w"] = 100 * count / n_words
    return features


def perspective_features(text: str) -> dict[str, float]:
    """Advice, SaaS framing, and tradeoff cues. Not punctuation or sentence length."""
    features = style_features(text)
    return {name: features[name] for name in PERSPECTIVE_FEATURE_NAMES}


def reference_profile(
    texts: list[str],
    feature_names: tuple[str, ...] = FEATURE_NAMES,
) -> dict[str, dict[str, float]]:
    """Mean and sample standard deviation for each feature. Empty input is empty."""
    if not texts:
        return {}
    columns = {name: [] for name in feature_names}
    for text in texts:
        features = style_features(text)
        for name in feature_names:
            columns[name].append(features[name])
    profile = {}
    for name, values in columns.items():
        mean = statistics.fmean(values)
        stdev = statistics.stdev(values) if len(values) > 1 else 0.0
        profile[name] = {"mean": mean, "stdev": stdev}
    return profile


def style_distance(text: str, profile: dict[str, dict[str, float]]) -> float | None:
    """Mean absolute z-score. Lower is closer to the reference posts."""
    if not profile:
        return None
    features = style_features(text)
    scores = []
    for name, stats in profile.items():
        stdev = stats["stdev"]
        if stdev <= 0:
            continue
        scores.append(abs(features[name] - stats["mean"]) / stdev)
    if not scores:
        return None
    return statistics.fmean(scores)


def closer_side(distance_a: float | None, distance_b: float | None, margin: float = 0.05) -> str:
    """Return a, b, tie, or unavailable. A small gap stays a tie."""
    if distance_a is None or distance_b is None or math.isnan(distance_a) or math.isnan(distance_b):
        return "unavailable"
    if abs(distance_a - distance_b) <= margin:
        return "tie"
    return "a" if distance_a < distance_b else "b"


def reference_texts(
    dataset_rows: list[dict],
    assignments: list[dict],
    blocked_groups: set[str],
    per_medium: int = 40,
) -> dict[str, list[str]]:
    """Earliest unchanged train posts, grouped by comparison medium."""
    records = assignment_rows(assignments)
    if len(records) != len(dataset_rows):
        raise ValueError(
            f"assignment rows ({len(records)}) != dataset rows ({len(dataset_rows)})"
        )
    chosen = {"blog": [], "linkedin": [], "x": [], "talk": []}
    for row, rec in zip(dataset_rows, records):
        medium = PLATFORM_TO_MEDIUM.get(rec.get("source_platform"))
        if medium is None or len(chosen[medium]) >= per_medium:
            continue
        if rec.get("split") != "train" or rec.get("sft_role") != "unchanged":
            continue
        if rec.get("group_id") in blocked_groups:
            continue
        text = (row.get("output") or "").strip()
        if text:
            chosen[medium].append(text)
    return chosen


def load_reference_texts(per_medium: int = 40) -> dict[str, list[str]]:
    dataset = load_jsonl(DEFAULT_DATASET)
    assignments = load_jsonl(DEFAULT_ASSIGNMENTS)
    dispositions = load_jsonl(DEFAULT_GOLD_DISPOSITIONS)
    return reference_texts(
        dataset, assignments, blocked_group_ids(assignments, dispositions), per_medium
    )


def _medium_for_held_out(record: dict) -> str | None:
    return HELD_OUT_PLATFORM_TO_MEDIUM.get(record.get("source_platform"))


def held_out_reference_texts(
    dataset_rows: list[dict],
    assignments: list[dict],
    blocked_groups: set[str],
    per_medium: int = 80,
) -> dict[str, list[str]]:
    """Unchanged validation posts only. Train rows and gold-overlap families are excluded.

    Blog references stay blog, LinkedIn stays LinkedIn, X stays X, and attributed
    transcripts are the only talk references. Constructed gold briefs are not an input.
    """
    records = assignment_rows(assignments)
    if len(records) != len(dataset_rows):
        raise ValueError(
            f"assignment rows ({len(records)}) != dataset rows ({len(dataset_rows)})"
        )
    chosen = {"blog": [], "linkedin": [], "x": [], "talk": []}
    for row, rec in zip(dataset_rows, records):
        medium = _medium_for_held_out(rec)
        if medium is None or len(chosen[medium]) >= per_medium:
            continue
        if rec.get("split") != "validation" or rec.get("sft_role") != "unchanged":
            continue
        if rec.get("group_id") in blocked_groups:
            continue
        text = (row.get("output") or "").strip()
        if text:
            chosen[medium].append(text)
    return chosen


def iter_train_texts(
    dataset_rows: list[dict],
    assignments: list[dict],
) -> list[str]:
    """Training targets used only for the copying gate. Gold-overlap train text still counts as training text."""
    records = assignment_rows(assignments)
    if len(records) != len(dataset_rows):
        raise ValueError(
            f"assignment rows ({len(records)}) != dataset rows ({len(dataset_rows)})"
        )
    texts = []
    for row, rec in zip(dataset_rows, records):
        if rec.get("split") != "train":
            continue
        text = (row.get("output") or "").strip()
        if text:
            texts.append(text)
    return texts


def partition_references(texts: list[str]) -> tuple[list[str], list[str]]:
    """Split held-out posts into a profile slice and a disjoint real-Lemkin control slice.

    Fewer than eight posts cannot support both a profile and four controls. The
    profile may still be built, and calibration then fails closed.
    """
    if len(texts) < MIN_CALIBRATION_REAL * 2:
        return list(texts), []
    cut = len(texts) // 2
    if len(texts) - cut < MIN_CALIBRATION_REAL:
        cut = len(texts) - MIN_CALIBRATION_REAL
    return list(texts[:cut]), list(texts[cut:])


def _finite_distances(texts: list[str], profile: dict[str, dict[str, float]]) -> list[float]:
    distances = []
    for text in texts:
        distance = style_distance(text, profile)
        if distance is not None and not math.isnan(distance):
            distances.append(distance)
    return distances


def metric_calibrated(
    real_distances: list[float],
    generic_distances: list[float],
    *,
    base_distances: list[float] | None = None,
    degraded_distances: list[float] | None = None,
    margin: float = CALIBRATION_MARGIN,
) -> dict:
    """A style or perspective metric may select only when real held-out Lemkin is closer than generic prose.

    Base-model and degraded controls are required when they are supplied. Missing
    real or generic controls fail closed.
    """
    real = [value for value in real_distances if value is not None and not math.isnan(value)]
    generic = [value for value in generic_distances if value is not None and not math.isnan(value)]
    report = {
        "n_real": len(real),
        "n_generic": len(generic),
        "real_mean": round(statistics.fmean(real), 4) if real else None,
        "generic_mean": round(statistics.fmean(generic), 4) if generic else None,
        "margin": margin,
        "limits": NOT_AUTHORSHIP,
    }
    if len(real) < MIN_CALIBRATION_REAL or len(generic) < MIN_CALIBRATION_GENERIC:
        report["calibrated"] = False
        report["reason"] = "not_enough_controls"
        return report
    closer = report["real_mean"] + margin < report["generic_mean"]
    if not closer:
        report["calibrated"] = False
        report["reason"] = "real_not_closer_than_generic"
        return report
    for label, distances in (("base", base_distances), ("degraded", degraded_distances)):
        if distances is None:
            continue
        kept = [value for value in distances if value is not None and not math.isnan(value)]
        report[f"n_{label}"] = len(kept)
        report[f"{label}_mean"] = round(statistics.fmean(kept), 4) if kept else None
        if len(kept) < MIN_CALIBRATION_GENERIC or report[f"{label}_mean"] is None:
            report["calibrated"] = False
            report["reason"] = f"not_enough_{label}_controls"
            return report
        if not (report["real_mean"] + margin < report[f"{label}_mean"]):
            report["calibrated"] = False
            report["reason"] = f"real_not_closer_than_{label}"
            return report
    report["calibrated"] = True
    report["reason"] = "real_closer_than_controls"
    return report


def score_sheet(sheet: list[dict], references: dict[str, list[str]]) -> dict:
    """Blind distances for each pair. Model names are not an input."""
    profiles = {medium: reference_profile(texts) for medium, texts in references.items()}
    rows = []
    for pair in sheet:
        medium = pair.get("medium")
        profile = profiles.get(medium) or {}
        distance_a = style_distance(pair.get("answer_a") or "", profile)
        distance_b = style_distance(pair.get("answer_b") or "", profile)
        rows.append({
            "pair_id": pair.get("pair_id"),
            "medium": medium,
            "distance_a": None if distance_a is None else round(distance_a, 4),
            "distance_b": None if distance_b is None else round(distance_b, 4),
            "closer": closer_side(distance_a, distance_b),
        })
    return {
        "role": "diagnostic_not_a_voice_decision",
        "method": "mean absolute z-score of surface counts versus unchanged train posts",
        "not_used": "TF-IDF authorship classifier",
        "limits": (
            "Closer surface counts are not authorship accuracy. "
            "Talk has no unchanged train references, so those pairs stay unavailable."
        ),
        "reference_counts": {medium: len(texts) for medium, texts in references.items()},
        "pairs": rows,
    }


def attribute_closer(scored: dict, key_rows: list[dict]) -> dict:
    """Join the blind distances to adapter names. Keep this file out of the reviewer."""
    key = {row["pair_id"]: row for row in key_rows}
    wins = {"repaired": 0, "relabel": 0, "tie": 0, "unavailable": 0}
    by_medium: dict[str, dict[str, int]] = {}
    for pair in scored["pairs"]:
        medium = pair.get("medium") or "unknown"
        bucket = by_medium.setdefault(medium, {"repaired": 0, "relabel": 0, "tie": 0, "unavailable": 0})
        closer = pair["closer"]
        if closer in {"tie", "unavailable"}:
            wins[closer] += 1
            bucket[closer] += 1
            continue
        model = key[pair["pair_id"]][f"answer_{closer}_model"]
        wins[model] = wins.get(model, 0) + 1
        bucket[model] = bucket.get(model, 0) + 1
    return {
        "role": scored["role"],
        "method": scored["method"],
        "limits": scored["limits"],
        "reference_counts": scored["reference_counts"],
        "closer_counts": wins,
        "by_medium": by_medium,
    }


def write_scores(sheet_path: Path, key_path: Path, out_dir: Path) -> dict:
    """Write a blind file and a separate attributed summary."""
    sheet = [json.loads(line) for line in sheet_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    key_rows = [json.loads(line) for line in key_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    scored = score_sheet(sheet, load_reference_texts())
    attributed = attribute_closer(scored, key_rows)
    out_dir.mkdir(parents=True, exist_ok=True)
    blind_path = out_dir / "style_distance_blind.json"
    summary_path = out_dir / "style_distance.json"
    blind_path.write_text(json.dumps(scored, indent=2) + "\n", encoding="utf-8")
    summary_path.write_text(json.dumps(attributed, indent=2) + "\n", encoding="utf-8")
    return attributed
