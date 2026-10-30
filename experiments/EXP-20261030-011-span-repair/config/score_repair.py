"""Score repaired EXP-009 drafts against the saved 009 and 010 results.

Unchanged drafts reuse the EXP-009 writing scores. Only repaired prose is sent
to the blinded judge.
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
from host_finetune.span_repair import SUBSTANTIVE_ELIGIBILITY, claim_overlap_count, substantive_eligible
from host_finetune.unified_eval import completion_gate, copying_gate, format_gate
from host_finetune.writing_quality import judge_prompt, parse_writing_scores, writing_mean

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parents[1]
PRIOR = ROOT / "experiments" / "EXP-20261029-009-hardened-generation"
CONSTRAINED = ROOT / "experiments" / "EXP-20261030-010-factual-reliability"
ARMS = ("chroma_v2", "pilot_dense", "naive", "mix")


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _echo(trace: dict, answer: str | None = None) -> bool:
    text = trace.get("answer") if answer is None else answer
    topic = trace.get("topic") or ""
    return repeats_topic(text or "", topic) or embeds_topic(text or "", topic)


def _mean(values: list[float]) -> float | None:
    if not values:
        return None
    return round(sum(values) / len(values), 4)


def _rate(flags: list[bool]) -> float | None:
    if not flags:
        return None
    return round(sum(1.0 if flag else 0.0 for flag in flags) / len(flags), 4)


def _writing_lookup(path: Path) -> dict[tuple[str, str], float | None]:
    if not path.is_file():
        return {}
    payload = _load(path)
    found = {}
    for arm, body in (payload.get("arms") or {}).items():
        for row in body.get("judge_rows") or []:
            found[(arm, row.get("id"), row.get("medium"))] = row.get("writing_mean")
    return found


def _judge_one(trace: dict) -> float | None:
    raw = ollama_json(
        judge_prompt(trace.get("answer") or "", trace.get("topic") or "", trace.get("medium") or ""),
        DEFAULT_JUDGE_MODEL,
    )
    parsed = parse_writing_scores(extract_json_object(raw) or {})
    return None if parsed is None else round(writing_mean(parsed), 4)


def _checks(trace: dict, unified: dict, copy_index, answer: str | None = None) -> dict:
    text = trace.get("answer") if answer is None else answer
    claims = trace.get("verified_claims") or []
    factual = assess_factual(
        text or "",
        trace.get("topic") or "",
        claims,
        evidence_status=trace.get("evidence_status") or "verified",
        hits=trace.get("retrieval_hits") or [],
    )
    completion = completion_gate({**trace, "answer": text})
    formatting = format_gate({**trace, "answer": text})
    training = copying_gate({**trace, "answer": text}, copy_index)
    overlap = claim_overlap_count(text or "", claims)
    original_words = len((trace.get("original_answer") or text or "").split())
    words = len((text or "").split())
    retention = 1.0 if original_words == 0 else words / original_words
    row = {
        "completion": completion["pass"],
        "format": formatting["pass"],
        "training_copy": training["pass"],
        "retrieval_copy": bool((factual["retrieval_source_copying"] or {}).get("pass")),
        "topic_echo": _echo(trace, text),
        "strict_pass": factual["pass"],
        "word_retention": retention,
        "accepted_claims": len(claims),
        "claim_overlap": overlap,
        "lexical_grounding": (unified.get("gates") or {}).get("grounding", {}).get("pass"),
        "unified_eligible": unified.get("eligible_for_ranking"),
        "unsupported_numbers": [item.get("value") for item in (factual["numeric"].get("unsupported") or [])],
        "speaker_status": (factual["speaker_attribution"] or {}).get("status"),
        "role_verdicts": [item.get("verdict") for item in factual["role_assertions"]],
        "words": words,
        "strict_reasons": factual["reasons"],
    }
    row["substantive_eligible"] = substantive_eligible(row)
    return row


def _summarize(rows: list[dict]) -> dict:
    return {
        "n": len(rows),
        "unsupported_number_rate": _rate([bool(row["unsupported_numbers"]) for row in rows]),
        "speaker_omitted_rate": _rate([row["speaker_status"] == "omitted" for row in rows]),
        "unsupported_role_rate": _rate(["unsupported" in row["role_verdicts"] for row in rows]),
        "strict_pass_rate": _rate([row["strict_pass"] for row in rows]),
        "mean_words": _mean([row["words"] for row in rows]),
        "mean_claim_overlap": _mean([row["claim_overlap"] for row in rows]),
        "topic_echo_rate": _rate([row["topic_echo"] for row in rows]),
        "completion_rate": _rate([row["completion"] for row in rows]),
        "format_rate": _rate([row["format"] for row in rows]),
        "training_copy_rate": _rate([row["training_copy"] for row in rows]),
        "retrieval_copy_rate": _rate([row["retrieval_copy"] for row in rows]),
        "unified_eligible_rate": _rate([bool(row["unified_eligible"]) for row in rows]),
        "substantive_eligible_rate": _rate([row["substantive_eligible"] for row in rows]),
        "substantive_rule": SUBSTANTIVE_ELIGIBILITY,
    }


def main() -> None:
    do_judge = "--judge" in sys.argv
    specs = load_format_specs(SPECS)
    _pack, copy_index = _evaluation_resources()
    prior_writing = _writing_lookup(PRIOR / "metrics" / "comparison.json")
    constrained_writing = _writing_lookup(CONSTRAINED / "metrics" / "comparison.json")
    report = {
        "writing_judge": DEFAULT_JUDGE_MODEL if do_judge else "judge_not_run_for_new_repairs",
        "substantive_rule": SUBSTANTIVE_ELIGIBILITY,
        "lexical_overlap_is_not_entailment": True,
        "arms": {},
    }
    examples = []
    for arm in ARMS:
        repaired_payload = _load(HERE / "raw" / f"generation_{arm}.json")
        baseline_payload = _load(PRIOR / "raw" / f"generation_{arm}.json")
        constrained_payload = _load(CONSTRAINED / "raw" / f"generation_{arm}.json")
        repaired_scored = score(repaired_payload["traces"], specs, _pack, copy_index)
        baseline_scored = score(baseline_payload["traces"], specs, _pack, copy_index)
        constrained_scored = score(constrained_payload["traces"], specs, _pack, copy_index)
        conditions = {"repair": [], "exp009": [], "exp010": []}
        writing = {"repair": [], "exp009": [], "exp010": []}
        applied = 0
        for trace, base, constrained, u_repair, u_base, u_constrained in zip(
            repaired_payload["traces"],
            baseline_payload["traces"],
            constrained_payload["traces"],
            repaired_scored["unified_rows"],
            baseline_scored["unified_rows"],
            constrained_scored["unified_rows"],
        ):
            key = (arm, trace.get("id"), trace.get("medium"))
            repair_row = _checks(trace, u_repair, copy_index)
            base_row = _checks({**base, "original_answer": base.get("answer")}, u_base, copy_index)
            constrained_row = _checks({**constrained, "original_answer": constrained.get("answer")}, u_constrained, copy_index)
            conditions["repair"].append(repair_row)
            conditions["exp009"].append(base_row)
            conditions["exp010"].append(constrained_row)
            base_score = prior_writing.get(key)
            constrained_score = constrained_writing.get(key)
            writing["exp009"].append(base_score)
            writing["exp010"].append(constrained_score)
            if (trace.get("span_repair") or {}).get("applied"):
                applied += 1
                judged = _judge_one(trace) if do_judge else None
                writing["repair"].append(judged)
                if len(examples) < 6:
                    examples.append({
                        "arm": arm,
                        "id": trace.get("id"),
                        "medium": trace.get("medium"),
                        "applied": True,
                        "before": (trace.get("original_answer") or "")[:700],
                        "after": (trace.get("answer") or "")[:700],
                        "edits": (trace.get("span_repair") or {}).get("edits"),
                        "writing_before": base_score,
                        "writing_after": judged,
                    })
            else:
                writing["repair"].append(base_score)
                if len([item for item in examples if not item["applied"]]) < 3:
                    examples.append({
                        "arm": arm,
                        "id": trace.get("id"),
                        "medium": trace.get("medium"),
                        "applied": False,
                        "warning": (trace.get("span_repair") or {}).get("warning"),
                        "rejection": (trace.get("span_repair") or {}).get("rejection"),
                        "draft": (trace.get("answer") or "")[:700],
                    })
        report["arms"][arm] = {
            "repair_applied": applied,
            "repair": {**_summarize(conditions["repair"]), "writing_mean": _mean([v for v in writing["repair"] if v is not None])},
            "exp009": {**_summarize(conditions["exp009"]), "writing_mean": _mean([v for v in writing["exp009"] if v is not None])},
            "exp010": {**_summarize(conditions["exp010"]), "writing_mean": _mean([v for v in writing["exp010"] if v is not None])},
        }
        print(arm, "applied", applied, "repair strict", report["arms"][arm]["repair"]["strict_pass_rate"], flush=True)
    destination = HERE / "metrics" / "comparison.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2), encoding="utf-8")
    (HERE / "metrics" / "examples.json").write_text(json.dumps(examples, indent=2), encoding="utf-8")
    print(f"wrote {destination}", flush=True)


if __name__ == "__main__":
    main()
