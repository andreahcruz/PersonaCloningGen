"""Graph-quality notes that do not change indexing success.

A completed document can have zero committed relations. That is a quality
warning. It is not, by itself, a failed index write.
"""

from __future__ import annotations

import logging
import re

from host_finetune.graph_relationships import is_weak_entity
from host_finetune.lightrag_backend import OMISSION_MARKER

RELATION_DELIMITER = "<|#|>"
_RELATION_LINE = re.compile(r"(?im)^\s*relation\s*<\|#\|>")


def count_malformed_relation_lines(text: str) -> int:
    """Count relation rows that do not have the five LightRAG fields.

    Entity rows are not counted. The delimiter is the one installed in
    LightRAG 1.5.7.
    """
    found = 0
    for line in (text or "").splitlines():
        if not _RELATION_LINE.match(line):
            continue
        fields = [part.strip() for part in line.split(RELATION_DELIMITER)]
        if len(fields) != 5:
            found += 1
    return found


class FormatWarningCounter(logging.Handler):
    """Count LightRAG format warnings without changing the library."""

    def __init__(self, sink: dict[str, int]):
        super().__init__(level=logging.WARNING)
        self.sink = sink

    def emit(self, record: logging.LogRecord) -> None:
        message = record.getMessage()
        if "LLM output format error" not in message:
            return
        if "RELATION" in message:
            self.sink["malformed_relations"] = self.sink.get("malformed_relations", 0) + 1
        else:
            self.sink["other_format"] = self.sink.get("other_format", 0) + 1


def classify_graph_quality(
    *,
    indexing_status: str,
    entity_count: int,
    relation_count: int,
    entity_names: list[str] | None = None,
    malformed_relations: int | None = None,
    extraction_incomplete: bool = False,
    source_text: str = "",
) -> dict:
    """Return a quality judgment that is separate from the ledger status."""
    names = list(entity_names or [])
    weak = [name for name in names if is_weak_entity(name)]
    has_omission = OMISSION_MARKER in (source_text or "")
    if extraction_incomplete or indexing_status in {"failed_permanent", "failed_retryable", "interrupted", "processing"}:
        relation_reason = "extraction_incomplete"
    elif relation_count == 0 and (malformed_relations or 0) > 0:
        relation_reason = "malformed_relations_dropped"
    elif relation_count == 0 and entity_count > 0 and malformed_relations is None:
        relation_reason = "no_relations_committed_cause_not_recorded"
    elif relation_count == 0 and entity_count > 0:
        relation_reason = "no_relations_committed"
    elif relation_count == 0:
        relation_reason = "no_entities_or_relations"
    else:
        relation_reason = "relations_committed"
    return {
        "indexing_status": indexing_status,
        "indexing_success": indexing_status == "completed",
        "entity_count": entity_count,
        "relation_count": relation_count,
        "relation_reason": relation_reason,
        "is_indexing_failure": indexing_status in {"failed_permanent", "failed_retryable", "interrupted"},
        "malformed_relations": malformed_relations,
        "weak_entity_names": weak,
        "omission_marker_in_source": has_omission,
        "relations_from_omission_marked_text": bool(has_omission and relation_count > 0),
    }


def runtime_profile(documents: list[dict], attempts: list[dict] | None = None) -> dict:
    """Summarize measured runtimes. Short documents are not treated as cheap."""
    completed = [row for row in documents if row.get("status") == "completed" and row.get("runtime_seconds") is not None]
    failures = [
        row for row in documents
        if row.get("status") in {"failed_permanent", "failed_retryable", "interrupted", "quarantined"}
    ]
    by_medium: dict[str, list[float]] = {}
    for row in completed:
        by_medium.setdefault(str(row.get("source_medium") or "unknown"), []).append(float(row["runtime_seconds"]))
    expensive = []
    for row in documents:
        runtime = row.get("runtime_seconds")
        words = int(row.get("word_count") or 0)
        deferred = row.get("defer_reason") == "expensive_timeout"
        category = row.get("error_category")
        if deferred or category in {"llm_extraction_timeout", "process_interruption"} or (
            runtime is not None and float(runtime) >= 300 and words < 800
        ):
            expensive.append({
                "document_id": row.get("document_id"),
                "word_count": words,
                "estimated_chunks": row.get("estimated_chunks"),
                "source_medium": row.get("source_medium"),
                "runtime_seconds": runtime,
                "status": row.get("status"),
                "error_category": category,
                "defer_reason": row.get("defer_reason"),
            })
    attempt_failures = [
        {
            "document_id": row.get("document_id"),
            "attempt_number": row.get("attempt_number"),
            "status": row.get("status"),
            "runtime_seconds": row.get("runtime_seconds"),
            "error_category": row.get("error_category"),
        }
        for row in (attempts or [])
        if row.get("status") in {"interrupted", "failed_retryable", "failed_permanent"}
    ]
    return {
        "completed_count": len(completed),
        "completed_mean_seconds": _mean(completed),
        "by_medium_mean_seconds": {key: round(sum(values) / len(values), 3) for key, values in sorted(by_medium.items())},
        "by_estimated_chunks": _bins(completed, "estimated_chunks", ((1, 1), (2, 3), (4, 99))),
        "by_word_count": _bins(completed, "word_count", ((0, 40), (41, 200), (201, 600), (601, 100000))),
        "current_failures": [
            {
                "document_id": row.get("document_id"),
                "status": row.get("status"),
                "error_category": row.get("error_category"),
                "word_count": row.get("word_count"),
                "estimated_chunks": row.get("estimated_chunks"),
                "runtime_seconds": row.get("runtime_seconds"),
                "defer_reason": row.get("defer_reason"),
            }
            for row in failures
        ],
        "expensive_or_deferred": expensive,
        "failed_attempts": attempt_failures,
        "note": (
            "Means use completed document runtimes only. A short document can still "
            "consume the per-document time limit. Those rows stay deferred and do not "
            "block the next selected document."
        ),
    }


def _mean(rows: list[dict]) -> float | None:
    if not rows:
        return None
    return round(sum(float(row["runtime_seconds"]) for row in rows) / len(rows), 3)


def _bins(rows: list[dict], key: str, ranges: tuple[tuple[int, int], ...]) -> list[dict]:
    result = []
    for low, high in ranges:
        chosen = [row for row in rows if low <= int(row.get(key) or 0) <= high]
        result.append({
            "min": low,
            "max": high,
            "documents": len(chosen),
            "mean_seconds": _mean(chosen),
        })
    return result
