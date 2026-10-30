"""Score saved Relabel traces with the existing unified harness.

Does not load the generator and does not call a retriever.
"""

from __future__ import annotations

import json
from pathlib import Path

from host_finetune.compare_adapters import _evaluation_resources, score
from host_finetune.eval_gold_rag import load_format_specs
from host_finetune.compare_adapters import SPECS

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parents[1] / "metrics" / "unified_scores.json"

ARMS = {
    "matched_dense": ROOT / "experiments/EXP-20261029-006-pilot160-eval/raw/generation_dense.json",
    "naive_chunk_ids": ROOT / "experiments/EXP-20261029-006-pilot160-eval/raw/generation_naive_chunk_ids.json",
    "mix_chunk_ids": ROOT / "experiments/EXP-20261029-006-pilot160-eval/raw/generation_mix_chunk_ids.json",
    "naive_parent_ids": ROOT / "experiments/EXP-20261029-006-pilot160-eval/raw/generation_naive_parent_ids.json",
    "mix_parent_ids": ROOT / "experiments/EXP-20261029-006-pilot160-eval/raw/generation_mix_parent_ids.json",
    "chroma_v2": ROOT / "experiments/EXP-20261029-007-paired-generation/raw/generation_chroma_v2.json",
}


def _load(path: Path) -> list[dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload["traces"] if isinstance(payload, dict) else payload


def _checks(trace: dict) -> dict:
    answer = (trace.get("answer") or "").lower()
    topic_id = trace.get("id")
    checks = {
        "mentions_5_15": "5.15" in answer,
        "mentions_anneka": "anneka" in answer,
        "mentions_benioff": "benioff" in answer,
        "mentions_abernethy": "abernethy" in answer,
        "repeats_topic": (trace.get("topic") or "").strip().lower() == (trace.get("answer") or "").strip().lower(),
    }
    if topic_id == "q_vp_sales_tension":
        checks["mentions_bring_in"] = "bring" in answer or "hire" in answer
        checks["mentions_critic"] = "critic" in answer
    return checks


def _row_brief(trace: dict) -> dict:
    numeric = trace.get("numeric_grounding") or {}
    copying = trace.get("rag_source_copying") or {}
    rejected = trace.get("validation_rejected") or []
    reasons = {}
    for item in rejected:
        reason = item.get("reason") if isinstance(item, dict) else str(item)
        reasons[reason] = reasons.get(reason, 0) + 1
    return {
        "id": trace.get("id"),
        "medium": trace.get("medium"),
        "words": len((trace.get("answer") or "").split()),
        "stop_reason": trace.get("stop_reason"),
        "verified_claims": len(trace.get("verified_claims") or []),
        "rejected_reasons": reasons,
        "numeric_pass": numeric.get("pass"),
        "unsupported_numbers": numeric.get("unsupported") or [],
        "source_copy_pass": copying.get("pass"),
        "source_copy_words": copying.get("max_contiguous_words"),
        "evidence_rejected": bool(trace.get("evidence_rejected")),
        "evidence_stage_reason": trace.get("evidence_stage_reason"),
        "checks": _checks(trace),
        "answer_preview": (trace.get("answer") or "")[:240],
    }


def main() -> None:
    specs = load_format_specs(SPECS)
    pack, copy_index = _evaluation_resources()
    report = {"arms": {}}
    for name, path in ARMS.items():
        if not path.is_file():
            report["arms"][name] = {"missing": str(path)}
            continue
        traces = _load(path)
        scored = score(traces, specs, pack, copy_index)
        summary = scored["summary"]
        summary.pop("by_medium", None)
        unified = summary.get("unified") or {}
        rows = []
        for trace, urow in zip(traces, scored["unified_rows"]):
            brief = _row_brief(trace)
            gates = urow["gates"]
            brief["gates"] = {
                "completion": gates["completion"]["pass"],
                "format": gates["format"]["pass"],
                "grounding": gates["grounding"]["pass"],
                "lexical_overlap": gates["grounding"].get("lexical_overlap"),
                "copying": gates["copying_memorization"]["pass"],
                "eligible": urow["eligible_for_ranking"],
                "style_distance": (urow.get("persona_style") or {}).get("distance"),
            }
            rows.append(brief)
        report["arms"][name] = {
            "source": str(path.relative_to(ROOT)),
            "n": len(traces),
            "load_report": json.loads(path.read_text(encoding="utf-8")).get("load_report"),
            "unified": unified,
            "stopping": summary.get("stopping"),
            "decision": summary.get("decision"),
            "rows": rows,
        }
        print(name, "n", len(traces), "eligible", (unified.get("n_eligible")), flush=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"wrote {OUT}", flush=True)


if __name__ == "__main__":
    main()
