"""Score the four arms. Shared retrieval is not a retrieval improvement."""

from __future__ import annotations

import json
from pathlib import Path

from host_finetune.eval_bench.runner import paired_faithfulness, score_batch
from host_finetune.model_f_experiment.arms import ARMS, EXPERIMENT_ID
from host_finetune.model_f_experiment.runner import MEDIA

NOT_ANNOTATED = {
    "relationship_accuracy": "not_annotated",
    "multi_hop_reasoning": "not_annotated",
    "persona_consistency": "not_scored_length_confounded",
    "retrieval_precision_recall_mrr_ndcg": "not_scored_without_human_grades",
}


def traces_to_records(directory: Path) -> list[dict]:
    records = []
    folder = directory / "raw" / "generations"
    if not folder.is_dir():
        return records
    for path in sorted(folder.glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("status") != "complete":
            continue
        records.append({
            "experiment_id": EXPERIMENT_ID,
            "architecture": payload["arm"],
            "question_id": payload["question_id"],
            "question": payload.get("query") or "",
            "medium": payload.get("medium"),
            "uses_graph": payload["arm"] != "F-SOURCE",
            "retrieval_config": {
                "backend": "lightrag",
                "mode": "hybrid",
                "shared_frozen_payload": True,
                "arm": payload["arm"],
            },
            "generation_config": payload.get("generation_config") or {"arm": payload["arm"]},
            "retrieved_source_ids": list(payload.get("retrieved_source_ids") or []),
            "evidence": list(payload.get("evidence") or []),
            "provenance": list(payload.get("provenance") or []),
            "answer": payload.get("answer") or "",
            "reference_answer": None,
            "claims": list(payload.get("claims") or []),
            "grounding_texts": list(payload.get("grounding_texts") or []),
            "abstention_expected": bool(payload.get("abstention_expected")),
            "latency": payload.get("latency") or {},
            "tokens": payload.get("tokens") or {},
            "context_budget_tokens": (payload.get("tokens") or {}).get("budget_total"),
        })
    return records


def _null_retrieval_comparison(batch: dict) -> None:
    for row in batch["results"]:
        retrieval = row["retrieval"]
        for key in ("precision_at_1", "precision_at_4", "recall_at_1", "recall_at_4", "mrr", "ndcg_at_4"):
            retrieval[key] = None
        retrieval["comparison"] = "not_applicable_shared_frozen_payload"
    for arm in (batch["metrics"].get("arms") or {}).values():
        arm["precision_at_4_mean"] = None
        arm["recall_at_4_mean"] = None
        arm["mrr_mean"] = None
        arm["ndcg_at_4_mean"] = None


def evaluate_directory(directory: Path) -> dict:
    batch = score_batch(traces_to_records(directory))
    _null_retrieval_comparison(batch)
    pairs = {}
    for arm in ARMS:
        if arm == "F-SOURCE":
            continue
        pairs[arm] = paired_faithfulness(batch["results"], "F-SOURCE", arm)
        pairs[arm]["retrieval_difference"] = None
        pairs[arm]["note"] = "Faithfulness only. The frozen passages are the same, so retrieval metrics are not compared."
    report = {
        "experiment_id": EXPERIMENT_ID,
        "automated": ["faithfulness", "numeric_grounding", "citation_marker_leak", "source_copying", "latency", "tokens"],
        "not_annotated": dict(NOT_ANNOTATED),
        "metrics": batch["metrics"],
        "failures": batch["failures"],
        "paired_versus_source": pairs,
        "media": list(MEDIA),
        "composite_score": None,
    }
    (directory / "metrics.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report
