"""Repair saved EXP-009 drafts in place as new files. EXP-009 is not written."""

from __future__ import annotations

import json
from pathlib import Path

from host_finetune.compare_adapters import _evaluation_resources
from host_finetune.evidence_capsule import repeats_topic
from host_finetune.factual_reliability import embeds_topic, instruction_leak
from host_finetune.span_repair import SPAN_REPAIR, repair_draft

ROOT = Path(__file__).resolve().parents[3]
PRIOR = ROOT / "experiments" / "EXP-20261029-009-hardened-generation" / "raw"
LATER = ROOT / "experiments" / "EXP-20261030-010-factual-reliability" / "raw"
OUT = Path(__file__).resolve().parents[1]
ARMS = ("chroma_v2", "pilot_dense", "naive", "mix")


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _words(trace: dict, key: str = "answer") -> int:
    return len((trace.get(key) or "").split())


def _classify(before: dict, after: dict) -> list[str]:
    labels = []
    if before.get("evidence_status") == "extraction_failed" or after.get("evidence_status") == "extraction_failed":
        labels.append("extraction_failed")
    if len(before.get("verified_claims") or []) != len(after.get("verified_claims") or []):
        labels.append("evidence_claim_count_changed")
    answer = after.get("answer") or ""
    if repeats_topic(answer, after.get("topic") or "") or embeds_topic(answer, after.get("topic") or "") or instruction_leak(answer):
        labels.append("prompt_echo_or_instruction_leak")
    first = _words(after, "first_attempt_answer")
    kept = _words(after)
    baseline = _words(before)
    if after.get("accepted_attempt") == 2 and first > 0 and kept < 0.6 * first:
        labels.append("retry_selected_shorter_draft")
    if after.get("accepted_attempt") == 1 and baseline >= 40 and kept < 0.6 * baseline:
        labels.append("restrictive_prompt_short_first_draft")
    if not labels:
        labels.append("no_large_regression_marker")
    return labels


def main() -> None:
    _pack, copy_index = _evaluation_resources()
    audit_rows = []
    raw = OUT / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    for arm in ARMS:
        baseline = _load(PRIOR / f"generation_{arm}.json")
        constrained = _load(LATER / f"generation_{arm}.json")
        repaired_traces = []
        for before, after in zip(baseline["traces"], constrained["traces"]):
            audit_rows.append({
                "arm": arm,
                "id": before.get("id"),
                "medium": before.get("medium"),
                "words_009": _words(before),
                "words_010_first": _words(after, "first_attempt_answer"),
                "words_010": _words(after),
                "attempt_009": before.get("accepted_attempt"),
                "attempt_010": after.get("accepted_attempt"),
                "claims_009": len(before.get("verified_claims") or []),
                "claims_010": len(after.get("verified_claims") or []),
                "evidence_status": before.get("evidence_status"),
                "labels": _classify(before, after),
            })
            result = repair_draft(
                before.get("answer") or "",
                before.get("topic") or "",
                before.get("verified_claims") or [],
                hits=before.get("retrieval_hits") or [],
                prompt=before.get("prompt") or "",
                copy_index=copy_index,
                evidence_status=before.get("evidence_status") or "verified",
                medium=before.get("medium") or "",
            )
            trace = dict(before)
            trace["original_answer"] = before.get("answer") or ""
            trace["answer"] = result["answer"]
            trace["span_repair"] = {key: value for key, value in result.items() if key not in {"answer", "original_answer", "factual"}}
            repaired_traces.append(trace)
            print(arm, before.get("id"), before.get("medium"), "applied" if result["applied"] else result["rejection"], flush=True)
        destination = raw / f"generation_{arm}.json"
        destination.write_text(
            json.dumps(
                {
                    "arm": arm,
                    "repair_contract": SPAN_REPAIR,
                    "source_experiment": "EXP-20261029-009-hardened-generation",
                    "source_file": str((PRIOR / f"generation_{arm}.json").relative_to(ROOT)),
                    "wall_seconds": baseline.get("wall_seconds"),
                    "traces": repaired_traces,
                },
                indent=2,
                default=str,
            ),
            encoding="utf-8",
        )
        print(f"wrote {destination}", flush=True)
    metrics = OUT / "metrics"
    metrics.mkdir(parents=True, exist_ok=True)
    (metrics / "audit.json").write_text(json.dumps(audit_rows, indent=2), encoding="utf-8")
    print(f"wrote {metrics / 'audit.json'}", flush=True)


if __name__ == "__main__":
    main()
