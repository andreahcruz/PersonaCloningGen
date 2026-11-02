"""Operational reports for a LightRAG indexing ledger.

Remaining time is an average of completed document runtimes multiplied by the
selected documents still queued. It is approximate: document length and
failures vary, and the average does not predict a timeout.
"""

from __future__ import annotations

import csv
import json
import statistics
from pathlib import Path


def remaining_estimate(completed_runtimes: list[float], queued: int) -> dict:
    if not completed_runtimes or queued < 0:
        return {
            "label": "approximate",
            "queued_documents": queued,
            "basis": "no completed documents yet",
            "seconds": None,
        }
    mean = statistics.fmean(completed_runtimes)
    return {
        "label": "approximate",
        "queued_documents": queued,
        "completed_sample": len(completed_runtimes),
        "mean_seconds": round(mean, 1),
        "median_seconds": round(statistics.median(completed_runtimes), 1),
        "min_seconds": round(min(completed_runtimes), 1),
        "max_seconds": round(max(completed_runtimes), 1),
        "seconds": round(mean * queued, 1),
        "note": (
            "Approximate. It multiplies the mean completed runtime by the "
            "queued count. Deferred documents and the wall time of attempts "
            "that did not become the completed runtime are excluded. Length "
            "and failure rate vary, so this is not a schedule."
        ),
    }


def write_reports(
    ledger,
    directory: Path,
    *,
    graph_counts: dict | None = None,
    corpus_summary: dict | None = None,
    config_record: dict | None = None,
    pilot_decision: str = "",
) -> dict:
    directory.mkdir(parents=True, exist_ok=True)
    documents = ledger.iter_documents()
    attempts = ledger.iter_attempts()
    counts = ledger.counts()
    queued = sum(
        1
        for row in documents
        if row["selected"] and row["status"] in {"pending", "failed_retryable", "interrupted"}
        and not row.get("defer_reason")
    )
    estimate = remaining_estimate(ledger.runtime_samples(), queued)
    progress = ledger.progress()
    failures = [row for row in documents if row["status"] in {
        "failed_retryable", "failed_permanent", "quarantined", "interrupted"
    }]
    summary = {
        "eligible_documents": counts["eligible"],
        "selected_documents": counts["selected"],
        "completed_graph_indexed": counts["completed_graph"],
        "vector_only_documents": counts["completed_vector_only"],
        "pending_documents": counts["by_status"]["pending"],
        "retryable_failures": counts["by_status"]["failed_retryable"],
        "permanent_failures": counts["by_status"]["failed_permanent"],
        "quarantined_documents": counts["by_status"]["quarantined"],
        "interrupted_documents": counts["by_status"]["interrupted"],
        "skipped_documents": counts["by_status"]["skipped"],
        "deferred_documents": counts["deferred"],
        "successful_chunks": counts["indexed_chunks"],
        "graph_nodes": None if not graph_counts else graph_counts.get("nodes"),
        "graph_edges": None if not graph_counts else graph_counts.get("edges"),
        "graph_count_source": None if not graph_counts else graph_counts.get("source"),
        "elapsed_processing_seconds": round(counts["runtime_seconds"], 3),
        "rolling_throughput_docs_per_hour": _throughput(counts),
        "remaining_compute": estimate,
        "progress": progress,
        "queues": ledger.queue_sizes(),
        "corpus_summary": corpus_summary,
        "configuration": config_record,
        "pilot_incorporation": pilot_decision,
        "by_status": counts["by_status"],
    }
    (directory / "indexing_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    _write_csv(
        directory / "document_status.csv",
        documents,
        [
            "document_id",
            "original_source_id",
            "source_medium",
            "source_attribution_status",
            "source_path",
            "source_hash",
            "selection_rank",
            "selection_priority",
            "selected",
            "status",
            "indexing_mode",
            "attempt_number",
            "runtime_seconds",
            "error_stage",
            "error_category",
            "error_message",
            "retry_eligible",
            "indexed_chunk_count",
            "graph_persistence",
            "vector_persistence",
            "provenance_verified",
            "graph_extracted",
            "vector_only",
            "defer_reason",
            "quality_notes",
        ],
    )
    _write_csv(
        directory / "failures.csv",
        failures,
        [
            "document_id",
            "source_medium",
            "status",
            "attempt_number",
            "error_stage",
            "error_category",
            "error_message",
            "retry_eligible",
            "defer_reason",
            "quality_notes",
        ],
    )
    timing_rows = []
    for attempt in attempts:
        timings = attempt.get("stage_timings") or {}
        extraction = attempt.get("extraction_metrics") or {}
        embedding = attempt.get("embedding_metrics") or {}
        timing_rows.append(
            {
                "document_id": attempt["document_id"],
                "attempt_number": attempt["attempt_number"],
                "status": attempt["status"],
                "wall_seconds_measured": attempt.get("runtime_seconds"),
                "llm_seconds_measured": timings.get("llm_seconds_measured"),
                "embed_seconds_measured": timings.get("embed_seconds_measured"),
                "llm_kind_seconds_estimated": json.dumps(
                    timings.get("llm_seconds_by_kind_estimated") or {},
                    sort_keys=True,
                ),
                "entity_anchor_count": (extraction or {}).get("entity_anchor_count")
                if isinstance(extraction, dict)
                else None,
                "relation_anchor_count": extraction.get("relation_anchor_count")
                if isinstance(extraction, dict)
                else None,
                "embed_texts_measured": embedding.get("texts_measured") if isinstance(embedding, dict) else None,
                "vector_dim": embedding.get("vector_dim") if isinstance(embedding, dict) else None,
                "error_category": attempt.get("error_category"),
            }
        )
    _write_csv(
        directory / "stage_timings.csv",
        timing_rows,
        [
            "document_id",
            "attempt_number",
            "status",
            "wall_seconds_measured",
            "llm_seconds_measured",
            "embed_seconds_measured",
            "llm_kind_seconds_estimated",
            "entity_anchor_count",
            "relation_anchor_count",
            "embed_texts_measured",
            "vector_dim",
            "error_category",
        ],
    )
    manifest = {
        "configuration": config_record,
        "corpus_summary": corpus_summary,
        "documents": [
            {
                "document_id": row["document_id"],
                "original_source_id": row["original_source_id"],
                "source_medium": row["source_medium"],
                "source_hash": row["source_hash"],
                "selection_rank": row["selection_rank"],
                "selected": row["selected"],
                "status": row["status"],
                "defer_reason": row["defer_reason"],
            }
            for row in documents
        ],
    }
    (directory / "index_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    (directory / "quality_audit.md").write_text(
        _quality_markdown(documents, attempts, summary, pilot_decision),
        encoding="utf-8",
    )
    return summary


def _throughput(counts: dict) -> float | None:
    runtime = counts["runtime_seconds"]
    done = counts["completed_graph"] + counts["completed_vector_only"]
    if runtime <= 0 or done <= 0:
        return None
    return round(done / (runtime / 3600), 3)


def _write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fields})


def _quality_markdown(
    documents: list[dict],
    attempts: list[dict],
    summary: dict,
    pilot_decision: str,
) -> str:
    by_category: dict[str, list[str]] = {}
    for row in documents:
        if row["status"] in {"completed", "pending", "skipped"}:
            continue
        category = row.get("error_category") or row["status"]
        by_category.setdefault(str(category), []).append(row["document_id"])
    lines = [
        "# Indexing quality audit",
        "",
        "Operational status is separate from a judgment that the extracted graph is correct.",
        "A completed graph document has a processed LightRAG record, chunk vectors of the configured dimension, source-text provenance, and a persisted entity anchor.",
        "Vector-only documents are not graph-indexed.",
        "",
        "## Coverage",
        "",
        f"- Eligible documents in the ledger: {summary['eligible_documents']}",
        f"- Selected for the default queue: {summary['selected_documents']}",
        f"- Completed graph-indexed: {summary['completed_graph_indexed']}",
        f"- Vector-only completed: {summary['vector_only_documents']}",
        f"- Deferred: {summary['deferred_documents']}",
        f"- Quarantined: {summary['quarantined_documents']}",
        "",
        "## Pilot graph",
        "",
        pilot_decision or "The pilot store was not incorporated.",
        "",
        "## Failures by category",
        "",
    ]
    if not by_category:
        lines.append("No document is currently failed, quarantined, or interrupted.")
    for category in sorted(by_category):
        ids = by_category[category]
        preview = ", ".join(f"`{doc_id}`" for doc_id in ids[:8])
        extra = "" if len(ids) <= 8 else f" and {len(ids) - 8} more"
        lines.append(f"- {category}: {len(ids)}. Examples: {preview}{extra}")
    recovered = []
    for attempt in attempts:
        if attempt.get("status") in {"interrupted", "failed_retryable", "failed_permanent"}:
            recovered.append(
                f"`{attempt['document_id']}` attempt {attempt.get('attempt_number')} "
                f"{attempt.get('status')} ({attempt.get('error_category') or 'unclassified'})"
            )
    if recovered:
        lines.extend(["", "Earlier attempts, including ones a later attempt completed:", ""])
        for item in recovered[:12]:
            lines.append(f"- {item}")
        if len(recovered) > 12:
            lines.append(f"- {len(recovered) - 12} more attempt rows are in `stage_timings.csv`.")
    lines.extend(
        [
            "",
            "## Exclusions",
            "",
            "Graph-ready policy exclusions are quarantined when they appear in the loaded file.",
            "Exact text-hash duplicates are skipped so LightRAG does not create a `dup-` status row.",
            "The eight EXP-20261029-004 failures stay deferred under `pilot160_not_retried` and are not retried by a default session.",
            "Short documents are not dropped. Very long documents stay eligible at a lower priority because extraction cost grows with chunks.",
            "",
            "## Recoverable cases",
            "",
            "Retryable failures have no recorded graph mutation. The next session may delete that document's LightRAG row and enqueue it again.",
            "A document with `kg_write_state=graph_mutation_started` or a node whose `file_path` contains its id is quarantined instead of deleted.",
            "Interrupted documents are retried through LightRAG's own resume when the status is still processing, which purges stored chunk ids before rebuilding.",
            "",
        ]
    )
    return "\n".join(lines)
