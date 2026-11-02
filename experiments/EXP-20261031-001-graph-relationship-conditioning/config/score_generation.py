"""Score the relationship-conditioning ablation, including failed drafts.

Writing-judge calls are optional and must not run while Relabel is on the GPU.
Fixed-length blog voice needs the separate Kevin reference worktree, which is
not in this checkout. Style distance from the unified pack is recorded instead.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parents[1]
CELLS = OUT / "raw" / "generations"
METRICS = OUT / "metrics"

sys.path.insert(0, str(ROOT))

from host_finetune.compare_adapters import SPECS, _evaluation_resources, score
from host_finetune.evidence_capsule import repeats_topic
from host_finetune.eval_gold_rag import load_format_specs
from host_finetune.factual_reliability import assess_factual, instruction_leak
from host_finetune.judge_client import DEFAULT_JUDGE_MODEL, extract_json_object, ollama_json
from host_finetune.writing_quality import judge_prompt, parse_writing_scores, writing_mean


def _load_traces() -> dict[str, list[dict]]:
    grouped = {"A": [], "B": [], "C": [], "D_blog": []}
    for path in sorted(CELLS.glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        arm = path.stem.split("__", 1)[1]
        for trace in payload.get("traces") or []:
            trace = dict(trace)
            trace["arm"] = arm
            trace["topic_id"] = payload.get("topic_id") or trace.get("id")
            grouped.setdefault(arm, []).append(trace)
    return grouped


def _facts(trace: dict) -> dict:
    relations = trace.get("graph_relationships") or []
    extra = [str(item.get("prompt_fact") or "") for item in relations if item.get("prompt_fact")]
    return assess_factual(
        trace.get("answer") or "",
        trace.get("topic") or "",
        trace.get("verified_claims") or [],
        evidence_status=trace.get("evidence_status") or "verified",
        hits=trace.get("retrieval_hits") or [],
        extra_texts=extra,
    )


def _brief(trace: dict, unified: dict | None) -> dict:
    answer = trace.get("answer") or ""
    facts = _facts(trace)
    numeric = trace.get("numeric_grounding") or {}
    gates = (unified or {}).get("gates") or {}
    relations = trace.get("graph_relationships") or []
    return {
        "id": trace.get("topic_id") or trace.get("id"),
        "medium": trace.get("medium"),
        "words": len(answer.split()),
        "evidence_status": trace.get("evidence_status"),
        "verified_claims": len(trace.get("verified_claims") or []),
        "relationship_count": len(relations),
        "prompt_tokens": trace.get("prompt_tokens"),
        "latency_seconds": trace.get("latency_seconds"),
        "topic_echo": repeats_topic(answer, trace.get("topic") or ""),
        "instruction_leak": instruction_leak(answer),
        "completion": answer.strip().endswith((".", "!", "?")) or not answer.strip(),
        "numeric_pass": numeric.get("pass"),
        "unsupported_numbers": [item.get("surface") or item.get("value") for item in (numeric.get("unsupported") or [])],
        "factual_pass": facts.get("pass"),
        "factual_reasons": facts.get("reasons"),
        "source_copy_pass": (trace.get("rag_source_copying") or {}).get("pass"),
        "eligible_for_ranking": None if unified is None else unified.get("eligible_for_ranking"),
        "style_distance": ((unified or {}).get("persona_style") or {}).get("distance"),
        "perspective_distance": ((unified or {}).get("perspective_fidelity") or {}).get("distance"),
        "gates": {
            "completion": (gates.get("completion") or {}).get("pass"),
            "format": (gates.get("format") or {}).get("pass"),
            "grounding": (gates.get("grounding") or {}).get("pass"),
            "copying": (gates.get("copying_memorization") or {}).get("pass"),
        },
        "answer": answer,
    }


def _judge(traces: list[dict]) -> None:
    for trace in traces:
        answer = (trace.get("answer") or "").strip()
        record = {"writing_mean": None, "reason": "empty_draft"}
        if answer:
            raw = ollama_json(
                judge_prompt(answer, trace.get("topic") or "", trace.get("medium") or ""),
                DEFAULT_JUDGE_MODEL,
            )
            parsed = parse_writing_scores(extract_json_object(raw) or {})
            record = {
                "writing_quality": parsed,
                "writing_mean": None if parsed is None else round(writing_mean(parsed), 4),
                "judge_model": DEFAULT_JUDGE_MODEL,
            }
        trace["writing_judge"] = record
        print(trace.get("topic_id"), trace.get("medium"), record.get("writing_mean"), flush=True)


def main() -> None:
    do_judge = "--judge" in sys.argv
    grouped = _load_traces()
    specs = load_format_specs(SPECS)
    pack, copy_index = _evaluation_resources()
    report = {
        "writing_judge": DEFAULT_JUDGE_MODEL if do_judge else "judge_not_run",
        "fixed_length_blog_voice": "not_supported_in_this_checkout",
        "arms": {},
    }
    for arm, traces in grouped.items():
        if not traces:
            report["arms"][arm] = {"n": 0}
            continue
        if do_judge:
            _judge(traces)
        scored = score(traces, specs, pack, copy_index)
        unified_rows = scored.get("unified_rows") or []
        rows = []
        for index, trace in enumerate(traces):
            unified = unified_rows[index] if index < len(unified_rows) else None
            brief = _brief(trace, unified)
            if do_judge:
                brief["writing_mean"] = (trace.get("writing_judge") or {}).get("writing_mean")
            rows.append(brief)
        by_medium = {}
        for medium in ("blog", "linkedin", "x", "talk"):
            subset = [row for row in rows if row["medium"] == medium]
            if not subset:
                continue
            distances = [row["style_distance"] for row in subset if isinstance(row.get("style_distance"), (int, float))]
            by_medium[medium] = {
                "n": len(subset),
                "mean_words": round(sum(row["words"] for row in subset) / len(subset), 2),
                "factual_pass": sum(1 for row in subset if row["factual_pass"]),
                "numeric_pass": sum(1 for row in subset if row["numeric_pass"] is True),
                "topic_echo": sum(1 for row in subset if row["topic_echo"]),
                "eligible": sum(1 for row in subset if row["eligible_for_ranking"]),
                "mean_style_distance": None if not distances else round(sum(distances) / len(distances), 4),
                "mean_writing": None,
            }
            writing = [row.get("writing_mean") for row in subset if isinstance(row.get("writing_mean"), (int, float))]
            if writing:
                by_medium[medium]["mean_writing"] = round(sum(writing) / len(writing), 4)
        tokens = {}
        report["arms"][arm] = {
            "n": len(traces),
            "unified": scored["summary"].get("unified"),
            "by_medium": by_medium,
            "rows": rows,
        }
        print(arm, "n", len(traces), flush=True)
    paired = []
    by_id = {}
    for arm, payload in report["arms"].items():
        for row in payload.get("rows") or []:
            by_id.setdefault((row["id"], row["medium"]), {})[arm] = row
    for key, arms in sorted(by_id.items()):
        if "A" in arms and "B" in arms:
            paired.append({
                "id": key[0],
                "medium": key[1],
                "token_delta_b_minus_a": (arms["B"].get("prompt_tokens") or 0) - (arms["A"].get("prompt_tokens") or 0),
                "words_a": arms["A"]["words"],
                "words_b": arms["B"]["words"],
                "factual_a": arms["A"]["factual_pass"],
                "factual_b": arms["B"]["factual_pass"],
                "writing_a": arms["A"].get("writing_mean"),
                "writing_b": arms["B"].get("writing_mean"),
                "style_a": arms["A"].get("style_distance"),
                "style_b": arms["B"].get("style_distance"),
                "d_blog_tokens": None if "D_blog" not in arms else arms["D_blog"].get("prompt_tokens"),
            })
    report["paired_ab"] = paired
    METRICS.mkdir(parents=True, exist_ok=True)
    (METRICS / "generation.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print("wrote generation metrics", flush=True)


if __name__ == "__main__":
    main()
