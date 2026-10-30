"""Score the four hardened generation arms. Optional writing judge uses Qwen."""

from __future__ import annotations

import json
import sys
from pathlib import Path

from host_finetune.compare_adapters import SPECS, _evaluation_resources, score
from host_finetune.eval_gold_rag import load_format_specs
from host_finetune.evidence_capsule import repeats_topic
from host_finetune.judge_client import DEFAULT_JUDGE_MODEL, extract_json_object, ollama_json
from host_finetune.writing_quality import judge_prompt, parse_writing_scores, writing_mean

ROOT = Path(__file__).resolve().parents[3]
RAW = Path(__file__).resolve().parents[1] / "raw"
OUT = Path(__file__).resolve().parents[1] / "metrics" / "comparison.json"

ARMS = {
    "chroma_v2": RAW / "generation_chroma_v2.json",
    "pilot_dense": RAW / "generation_pilot_dense.json",
    "naive": RAW / "generation_naive.json",
    "mix": RAW / "generation_mix.json",
}


def _load(path: Path) -> tuple[list[dict], dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload["traces"], payload


def _brief(trace: dict) -> dict:
    numeric = trace.get("numeric_grounding") or {}
    answer = trace.get("answer") or ""
    return {
        "id": trace.get("id"),
        "medium": trace.get("medium"),
        "words": len(answer.split()),
        "evidence_status": trace.get("evidence_status"),
        "evidence_stage_reason": trace.get("evidence_stage_reason"),
        "verified_claims": len(trace.get("verified_claims") or []),
        "numeric_pass": numeric.get("pass"),
        "unsupported_numbers": [item.get("value") for item in (numeric.get("unsupported") or [])],
        "topic_echo": repeats_topic(answer, trace.get("topic") or ""),
        "names_anneka": "anneka" in answer.casefold(),
        "contains_5_15": "5.15" in answer,
        "wall_seconds": trace.get("latency_seconds"),
    }


def _judge(traces: list[dict]) -> list[dict]:
    rows = []
    for trace in traces:
        answer = (trace.get("answer") or "").strip()
        if not answer:
            rows.append({"id": trace.get("id"), "medium": trace.get("medium"), "writing_quality": None, "reason": "empty_draft"})
            continue
        raw = ollama_json(
            judge_prompt(answer, trace.get("topic") or "", trace.get("medium") or ""),
            DEFAULT_JUDGE_MODEL,
        )
        parsed = parse_writing_scores(extract_json_object(raw) or {})
        rows.append({
            "id": trace.get("id"),
            "medium": trace.get("medium"),
            "writing_quality": parsed,
            "writing_mean": None if parsed is None else round(writing_mean(parsed), 4),
            "judge_model": DEFAULT_JUDGE_MODEL,
        })
        print(trace.get("id"), trace.get("medium"), "writing", rows[-1]["writing_mean"], flush=True)
    return rows


def main() -> None:
    do_judge = "--judge" in sys.argv
    specs = load_format_specs(SPECS)
    pack, copy_index = _evaluation_resources()
    report = {"writing_judge": DEFAULT_JUDGE_MODEL if do_judge else "judge_not_run", "arms": {}}
    for name, path in ARMS.items():
        if not path.is_file():
            report["arms"][name] = {"missing": str(path)}
            continue
        traces, payload = _load(path)
        scored = score(traces, specs, pack, copy_index)
        unified = scored["summary"].get("unified") or {}
        rows = []
        for trace, urow in zip(traces, scored["unified_rows"]):
            brief = _brief(trace)
            gates = urow["gates"]
            brief["gates"] = {
                "completion": gates["completion"]["pass"],
                "format": gates["format"]["pass"],
                "grounding": gates["grounding"]["pass"],
                "lexical_overlap": gates["grounding"].get("lexical_overlap"),
                "copying": gates["copying_memorization"]["pass"],
                "eligible": urow["eligible_for_ranking"],
                "style_distance": (urow.get("persona_style") or {}).get("distance"),
                "perspective_distance": (urow.get("perspective_fidelity") or {}).get("distance"),
            }
            rows.append(brief)
        arm = {
            "source": str(path.relative_to(ROOT)),
            "wall_seconds": payload.get("wall_seconds"),
            "n": len(traces),
            "unified": {
                "n_eligible": unified.get("n_eligible"),
                "eligible_rate": unified.get("eligible_rate"),
                "gates": unified.get("gates"),
                "persona_style": unified.get("persona_style"),
                "perspective_fidelity": unified.get("perspective_fidelity"),
                "writing_quality": unified.get("writing_quality"),
            },
            "rows": rows,
        }
        if do_judge:
            arm["judge_rows"] = _judge(traces)
        report["arms"][name] = arm
        print(name, "eligible", unified.get("n_eligible"), "/", len(traces), flush=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"wrote {OUT}", flush=True)


if __name__ == "__main__":
    main()
