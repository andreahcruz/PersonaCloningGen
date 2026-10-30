"""Retrieval comparison records for Model E and LightRAG.

Automatic scores describe coverage, diversity, attribution, and latency.
They are not precision or recall unless a graded relevance set is supplied.
"""

from __future__ import annotations

import statistics
from collections import Counter

from host_finetune.retrieval_contract import EXPERIMENT_CELLS


def retrieval_observation(payload: dict) -> dict:
    hits = payload.get("hits") or []
    sources = [hit.get("graph_document_id") or hit.get("group_id") or hit.get("id") for hit in hits]
    reasons = Counter(
        str((item or {}).get("reason") or "unknown") for item in (payload.get("excluded") or [])
    )
    attributed = bool(hits) and all(
        hit.get("split") == "train" and (hit.get("text") or "").strip() and hit.get("id")
        for hit in hits
    )
    return {
        "backend": payload.get("backend"),
        "mode": payload.get("mode"),
        "query": payload.get("query"),
        "hit_count": len(hits),
        "unique_sources": len(set(sources)),
        "duplicate_source": len(sources) != len(set(sources)),
        "empty": not hits,
        "train_attributed": attributed,
        "excluded_reasons": dict(reasons),
        "seconds": payload.get("seconds"),
        "score_note": payload.get("score_note"),
    }


def summarize_retrieval(rows: list[dict]) -> dict:
    """Measured retrieval behavior. Relevance grades are not inferred."""
    if not rows:
        return {"queries": 0, "precision_recall": "not measured"}
    latencies = [row["seconds"] for row in rows if isinstance(row.get("seconds"), (int, float))]
    return {
        "queries": len(rows),
        "empty_queries": sum(1 for row in rows if row["empty"]),
        "hit_count": sum(row["hit_count"] for row in rows),
        "mean_unique_sources": round(
            statistics.fmean(row["unique_sources"] for row in rows),
            4,
        ),
        "queries_with_duplicate_source": sum(1 for row in rows if row["duplicate_source"]),
        "queries_with_train_attributed_hits": sum(1 for row in rows if row["train_attributed"]),
        "latency_seconds_mean": round(statistics.fmean(latencies), 4) if latencies else None,
        "precision_recall": "not measured; no graded relevance judgments",
        "cells": list(EXPERIMENT_CELLS),
    }
