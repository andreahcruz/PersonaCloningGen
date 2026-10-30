"""Score EXP-010 and re-score EXP-009 with the same strict checks.

EXP-009 raw files are read only.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from host_finetune.compare_adapters import SPECS, _evaluation_resources, score
from host_finetune.eval_gold_rag import load_format_specs
from host_finetune.evidence_capsule import repeats_topic
from host_finetune.factual_reliability import assess_factual, embeds_topic
from host_finetune.judge_client import DEFAULT_JUDGE_MODEL, extract_json_object, ollama_json
from host_finetune.writing_quality import judge_prompt, parse_writing_scores, writing_mean

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parents[1]
PRIOR = ROOT / "experiments" / "EXP-20261029-009-hardened-generation" / "raw"
OUT = HERE / "metrics" / "comparison.json"
ARMS = ("chroma_v2", "pilot_dense", "naive", "mix")


def _load(path: Path) -> tuple[list[dict], dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload["traces"], payload


def _echo(trace: dict) -> bool:
    answer = trace.get("answer") or ""
    topic = trace.get("topic") or ""
    return repeats_topic(answer, topic) or embeds_topic(answer, topic)


def _row(trace: dict, unified: dict) -> dict:
    claims = trace.get("verified_claims") or []
    factual = assess_factual(
        trace.get("answer") or "",
        trace.get("topic") or "",
        claims,
        evidence_status=trace.get("evidence_status") or "verified",
        hits=trace.get("retrieval_hits") or [],
        retrieval_on=True,
    )
    gates = unified["gates"]
    strict_eligible = bool(
        gates["completion"]["pass"]
        and gates["format"]["pass"]
        and gates["copying_memorization"]["pass"]
        and factual["retrieval_source_copying"].get("pass")
        and factual["pass"]
        and not _echo(trace)
    )
    return {
        "id": trace.get("id"),
        "medium": trace.get("medium"),
        "words": len((trace.get("answer") or "").split()),
        "evidence_status": trace.get("evidence_status"),
        "prompt_contract": trace.get("prompt_contract"),
        "topic_echo": _echo(trace),
        "unified_eligible": unified["eligible_for_ranking"],
        "strict_eligible": strict_eligible,
        "completion": gates["completion"]["pass"],
        "format": gates["format"]["pass"],
        "lexical_grounding": gates["grounding"]["pass"],
        "lexical_overlap": gates["grounding"].get("lexical_overlap"),
        "training_copy": gates["copying_memorization"]["pass"],
        "strict_pass": factual["pass"],
        "strict_reasons": factual["reasons"],
        "unsupported_numbers": [item.get("value") for item in (factual["numeric"].get("unsupported") or [])],
        "role_verdicts": [item.get("verdict") for item in factual["role_assertions"]],
        "speaker_status": (factual["speaker_attribution"] or {}).get("status"),
        "retrieval_copy": (factual["retrieval_source_copying"] or {}).get("pass"),
        "style_distance": (unified.get("persona_style") or {}).get("distance"),
        "style_role": (unified.get("persona_style") or {}).get("role"),
    }


def _mean(values: list[bool]) -> float | None:
    if not values:
        return None
    return round(sum(1.0 if value else 0.0 for value in values) / len(values), 4)


def _judge(traces: list[dict]) -> list[dict]:
    rows = []
    for trace in traces:
        answer = (trace.get("answer") or "").strip()
        if not answer:
            rows.append({"id": trace.get("id"), "medium": trace.get("medium"), "writing_mean": None})
            continue
        raw = ollama_json(
            judge_prompt(answer, trace.get("topic") or "", trace.get("medium") or ""),
            DEFAULT_JUDGE_MODEL,
        )
        parsed = parse_writing_scores(extract_json_object(raw) or {})
        rows.append({
            "id": trace.get("id"),
            "medium": trace.get("medium"),
            "writing_mean": None if parsed is None else round(writing_mean(parsed), 4),
            "judge_model": DEFAULT_JUDGE_MODEL,
        })
        print(trace.get("id"), trace.get("medium"), "writing", rows[-1]["writing_mean"], flush=True)
    return rows


def _arm(path: Path, specs, pack, copy_index, do_judge: bool) -> dict:
    traces, payload = _load(path)
    scored = score(traces, specs, pack, copy_index)
    rows = [_row(trace, unified) for trace, unified in zip(traces, scored["unified_rows"])]
    unified = scored["summary"].get("unified") or {}
    style_rows = [row for row in rows if row["unified_eligible"] and row["style_role"] == "selector" and row["style_distance"] is not None]
    arm = {
        "source": str(path.relative_to(ROOT)),
        "wall_seconds": payload.get("wall_seconds"),
        "prompt_contract": payload.get("prompt_contract"),
        "n": len(rows),
        "unified": {
            "n_eligible": unified.get("n_eligible"),
            "eligible_rate": unified.get("eligible_rate"),
            "gates": unified.get("gates"),
            "persona_style": unified.get("persona_style"),
        },
        "strict": {
            "eligible_rate": _mean([row["strict_eligible"] for row in rows]),
            "strict_pass_rate": _mean([row["strict_pass"] for row in rows]),
            "unsupported_number_rate": _mean([bool(row["unsupported_numbers"]) for row in rows]),
            "unsupported_role_rate": _mean(["unsupported" in row["role_verdicts"] for row in rows]),
            "role_not_verifiable_rate": _mean(["not_verifiable" in row["role_verdicts"] for row in rows]),
            "speaker_omitted_rate": _mean([row["speaker_status"] == "omitted" for row in rows]),
            "topic_echo_rate": _mean([row["topic_echo"] for row in rows]),
            "retrieval_copy_pass_rate": _mean([bool(row["retrieval_copy"]) for row in rows]),
            "training_copy_pass_rate": _mean([row["training_copy"] for row in rows]),
            "selector_style_mediums": sorted({row["medium"] for row in style_rows}),
            "selector_style_mean": None if not style_rows else round(sum(row["style_distance"] for row in style_rows) / len(style_rows), 4),
        },
        "rows": rows,
    }
    if do_judge:
        arm["judge_rows"] = _judge(traces)
    return arm


def main() -> None:
    do_judge = "--judge" in sys.argv
    specs = load_format_specs(SPECS)
    pack, copy_index = _evaluation_resources()
    report = {
        "writing_judge": DEFAULT_JUDGE_MODEL if do_judge else "judge_not_run",
        "method": "deterministic_quantity_role_speaker_not_lexical_entailment",
        "arms": {},
        "baseline_exp009": {},
    }
    for name in ARMS:
        current = HERE / "raw" / f"generation_{name}.json"
        if current.is_file():
            report["arms"][name] = _arm(current, specs, pack, copy_index, do_judge)
            print(name, "strict", report["arms"][name]["strict"]["strict_pass_rate"], flush=True)
        prior = PRIOR / f"generation_{name}.json"
        if prior.is_file():
            report["baseline_exp009"][name] = _arm(prior, specs, pack, copy_index, False)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"wrote {OUT}", flush=True)


if __name__ == "__main__":
    main()
