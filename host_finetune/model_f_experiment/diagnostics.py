"""Deterministic Model F metrics and graph diagnostics.

Both the pilot and the 120-prompt run call ``write_model_f_reports``.
Ranked retrieval metrics stay null. Nothing here calls a judge.
"""

from __future__ import annotations

import json
from pathlib import Path

from host_finetune.eval_bench.runner import score_batch
from host_finetune.model_f_experiment.evaluate import _null_retrieval_comparison, traces_to_records
from host_finetune.model_f_experiment.runner import retrieval_path


def _reason_histogram(rows: list) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        reason = str(row.get("rejection_reason") or row.get("reason") or "unspecified")
        counts[reason] = counts.get(reason, 0) + 1
    return counts


def _missing_source_count(histogram: dict[str, int]) -> int:
    needles = ("missing", "provenance", "unknown_graph", "failed")
    return sum(count for reason, count in histogram.items() if any(needle in reason for needle in needles))


def summarize_graph(
    raw: dict | None,
    payload: dict | None,
    catalog: dict | None = None,
    failed_document_ids: set[str] | None = None,
    query: str = "",
) -> dict:
    """Counts from one frozen LightRAG payload. Missing steps stay null."""
    data = (raw or {}).get("data") if isinstance(raw, dict) else {}
    data = data or {}
    entities = [row for row in (data.get("entities") or []) if isinstance(row, dict)]
    relationships = [row for row in (data.get("relationships") or []) if isinstance(row, dict)]
    chunks = [row for row in (data.get("chunks") or []) if isinstance(row, dict)]
    descriptions = [
        row for row in ((payload or {}).get("entity_descriptions") or []) if isinstance(row, dict)
    ]
    nonempty = sum(1 for row in descriptions if str(row.get("description") or "").strip())
    excluded = [row for row in ((payload or {}).get("excluded") or []) if isinstance(row, dict)]
    excluded_reasons = _reason_histogram(excluded)
    summary = {
        "status": "recorded",
        "raw_entity_count": len(entities),
        "raw_relationship_count": len(relationships),
        "raw_chunk_count": len(chunks),
        "entity_description_count": len(descriptions),
        "entity_descriptions_nonempty": nonempty,
        "entity_description_available": nonempty > 0,
        "excluded_reason_counts": excluded_reasons,
        "accepted_relationships": None,
        "rejected_relationships": None,
        "quarantined_relationships": None,
        "relationship_reason_counts": None,
        "missing_source_or_provenance_exclusions": None,
        "relationship_validation_status": "not_recorded",
        "two_hop_path_count": None,
        "two_hop_path_available": None,
        "path_status": "not_recorded",
        "normalization": None,
    }
    if catalog is None:
        summary["relationship_validation_status"] = "not_recorded"
        summary["relationship_validation_reason"] = "catalog_not_supplied"
        return summary
    from host_finetune.graph_relationships import validate_relationships

    checked = validate_relationships(
        (payload or {}).get("relationship_candidates") or [],
        catalog,
        failed_document_ids=failed_document_ids,
    )
    rejected = list(checked.get("rejected") or [])
    quarantined = list(checked.get("quarantined") or [])
    reasons = _reason_histogram(rejected + quarantined)
    summary.update({
        "accepted_relationships": len(checked.get("accepted") or []),
        "rejected_relationships": len(rejected),
        "quarantined_relationships": len(quarantined),
        "relationship_reason_counts": reasons,
        "missing_source_or_provenance_exclusions": _missing_source_count(reasons),
        "relationship_validation_status": "recorded",
    })
    try:
        from host_finetune.model_f_context.pipeline import prepare_context

        prepared = prepare_context(
            payload,
            profile="multihop",
            query=query,
            catalog=catalog,
            failed_document_ids=failed_document_ids,
        )
        paths = (prepared.get("paths") or {}).get("paths") or []
        two_hop = sum(1 for path in paths if int(getattr(path, "hops", 0) or 0) >= 2)
        bundle = prepared.get("bundle") or {}
        normalization = bundle.get("diagnostics") if isinstance(bundle, dict) else None
        summary.update({
            "two_hop_path_count": two_hop,
            "two_hop_path_available": two_hop > 0,
            "path_status": "recorded",
            "direct_edge_count": (prepared.get("paths") or {}).get("direct_edge_count"),
            "normalization": normalization,
        })
    except Exception as exc:
        summary["path_status"] = "not_recorded"
        summary["path_error"] = f"{type(exc).__name__}: {exc}"
    return summary


def _graph_channels(cell: dict) -> dict:
    tokens = ((cell.get("tokens") or {}).get("context_by_channel") or {})
    selected = {
        name: value
        for name, value in tokens.items()
        if name != "primary_source" and isinstance(value, (int, float)) and value > 0
    }
    return {
        "tokens_by_channel": tokens,
        "selected_graph_sections": sorted(selected),
        "graph_context_tokens": sum(selected.values()) if selected else 0,
        "nonempty_graph_context": bool(selected),
    }


def _unmeasured(results: list[dict]) -> dict:
    def present(key: str) -> bool:
        return any(isinstance((row.get("efficiency") or {}).get(key), (int, float)) for row in results)

    gpu = any(isinstance(row.get("gpu_memory_bytes"), (int, float)) for row in results)
    return {
        "gpu_memory_bytes": "recorded" if gpu else "not_measured",
        "retrieval_seconds": "recorded" if present("retrieval_seconds") else "not_recorded",
        "extraction_seconds": "recorded" if any(
            isinstance((row.get("latency") or {}).get("extraction_seconds"), (int, float)) for row in results
        ) else "not_recorded",
        "verification_seconds": "recorded" if any(
            isinstance((row.get("latency") or {}).get("verification_seconds"), (int, float)) for row in results
        ) else "not_recorded",
        "precision_recall_mrr_ndcg": "not_scored_without_human_grades",
        "end_to_end_seconds": "recorded" if present("end_to_end_seconds") else "not_recorded",
    }


def load_graph_summaries(directory: Path) -> dict[str, dict]:
    folder = directory / "raw" / "retrieval"
    found = {}
    if not folder.is_dir():
        return found
    for path in sorted(folder.glob("*.json")):
        try:
            body = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        question_id = str(body.get("question_id") or path.stem)
        summary = body.get("graph_summary")
        if isinstance(summary, dict):
            found[question_id] = summary
        else:
            found[question_id] = {
                "status": "not_recorded",
                "reason": "retrieval freeze has no graph_summary",
            }
    return found


def write_model_f_reports(directory: Path, dest: Path, experiment_id: str) -> dict:
    """Write deterministic metrics and graph diagnostics. Does not call a judge."""
    directory = Path(directory)
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    batch = score_batch(traces_to_records(directory, experiment_id=experiment_id))
    _null_retrieval_comparison(batch)
    summaries = load_graph_summaries(directory)
    cells = []
    graph_cells = []
    for row in batch["results"]:
        config = row.get("generation_config") or {}
        generation = row.get("generation") or {}
        cells.append({
            "question_id": row["question_id"],
            "medium": row.get("medium"),
            "arm": row["architecture"],
            "retrieval_backend": (row.get("retrieval_config") or {}).get("backend"),
            "retrieval_mode": (row.get("retrieval_config") or {}).get("mode"),
            "context_profile": config.get("prepare_profile"),
            "retrieval": row.get("retrieval"),
            "generation": generation,
            "efficiency": row.get("efficiency"),
            "accepted_attempt": config.get("accepted_attempt"),
            "retry_seed": config.get("retry_seed"),
            "retrieved_source_ids": row.get("retrieved_source_ids"),
        })
        graph_cells.append({
            "question_id": row["question_id"],
            "medium": row.get("medium"),
            "arm": row["architecture"],
            "context_profile": config.get("prepare_profile"),
            **_graph_channels(row),
        })
    recorded = [item for item in summaries.values() if item.get("status") == "recorded"]
    usable = [
        question_id
        for question_id, item in summaries.items()
        if item.get("status") == "recorded" and (
            (item.get("accepted_relationships") or 0) > 0
            or item.get("entity_description_available")
            or item.get("two_hop_path_available")
        )
    ]
    fraction = (len(usable) / len(recorded)) if recorded else None
    deterministic = {
        "schema": "model-f-deterministic-v1",
        "experiment_id": experiment_id,
        "judge": None,
        "composite_score": None,
        "ranked_retrieval": "not_scored_without_human_grades",
        "metrics": batch["metrics"],
        "failures": batch["failures"],
        "cells": cells,
        "unmeasured": _unmeasured(batch["results"]),
        "graph_diagnostics": "graph_diagnostics.json",
    }
    graph = {
        "schema": "model-f-graph-diagnostics-v1",
        "experiment_id": experiment_id,
        "role": "diagnostic_not_a_quality_score",
        "questions": summaries,
        "cells": graph_cells,
        "questions_with_summary": len(recorded),
        "questions_with_usable_graph_evidence": len(usable) if recorded else None,
        "fraction_with_usable_graph_evidence": fraction,
        "usable_graph_definition": (
            "A question has usable graph evidence when the frozen summary records "
            "an accepted relationship, a nonempty entity description, or a two-hop path. "
            "This fraction is not added to the arm score."
        ),
        "join": {
            "diagnostic_keys": ["question_id", "medium", "arm"],
            "official_unified_keys": ["id", "medium"],
            "official_rows": "unified_eval/rows.jsonl",
            "note": (
                "Join official id to diagnostic question_id, and join medium. "
                "Graph fields stay in this file. They are not written into unified_eval.v1 rows."
            ),
        },
    }
    (dest / "deterministic_metrics.json").write_text(
        json.dumps(deterministic, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (dest / "graph_diagnostics.json").write_text(
        json.dumps(graph, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return {"cells": len(cells), "failures": len(batch["failures"])}


def retrieval_summary_path(directory: Path, question_id: str) -> Path:
    return retrieval_path(directory, question_id)
