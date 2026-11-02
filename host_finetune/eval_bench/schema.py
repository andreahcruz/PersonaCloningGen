"""Shared result records for architecture comparisons.

A record can describe a dense retriever. Graph fields are optional and are
not required for validation.
"""

from __future__ import annotations

import json
from typing import Any

SCHEMA_VERSION = "eval-bench-v1"
REQUIRED = (
    "experiment_id",
    "architecture",
    "question_id",
    "retrieval_config",
    "generation_config",
    "retrieved_source_ids",
    "evidence",
    "answer",
    "latency",
)


class RecordError(ValueError):
    """The prediction record is missing a field the scorer needs."""


def _require_dict(value: Any, field: str) -> dict:
    if not isinstance(value, dict):
        raise RecordError(f"{field} must be an object")
    return value


def _require_list(value: Any, field: str) -> list:
    if not isinstance(value, list):
        raise RecordError(f"{field} must be a list")
    return value


def validate_record(record: dict) -> dict:
    """Return a normalized record. Extra keys are kept for later metrics."""
    if not isinstance(record, dict):
        raise RecordError("record must be an object")
    missing = [field for field in REQUIRED if field not in record]
    if missing:
        raise RecordError("missing fields: " + ", ".join(missing))
    experiment_id = str(record["experiment_id"] or "").strip()
    architecture = str(record["architecture"] or "").strip()
    question_id = str(record["question_id"] or "").strip()
    if not experiment_id or not architecture or not question_id:
        raise RecordError("experiment_id, architecture, and question_id must be non-empty")
    evidence = _require_list(record["evidence"], "evidence")
    source_ids = [str(item) for item in _require_list(record["retrieved_source_ids"], "retrieved_source_ids")]
    latency = _require_dict(record["latency"], "latency")
    normalized = dict(record)
    normalized.update({
        "schema_version": SCHEMA_VERSION,
        "experiment_id": experiment_id,
        "architecture": architecture,
        "model_version": None if record.get("model_version") is None else str(record.get("model_version")),
        "question_id": question_id,
        "medium": None if record.get("medium") is None else str(record.get("medium")),
        "retrieval_config": _require_dict(record["retrieval_config"], "retrieval_config"),
        "generation_config": _require_dict(record["generation_config"], "generation_config"),
        "retrieved_source_ids": source_ids,
        "evidence": evidence,
        "provenance": list(record.get("provenance") or []),
        "answer": "" if record.get("answer") is None else str(record.get("answer")),
        "reference_answer": None if not record.get("reference_answer") else str(record.get("reference_answer")),
        "latency": latency,
        "tokens": dict(record.get("tokens") or {}),
        "uses_graph": bool(record.get("uses_graph")),
        "error": None if not record.get("error") else str(record.get("error")),
    })
    return normalized


def prediction_key(record: dict) -> tuple[str, str, str, str]:
    return (
        str(record.get("experiment_id") or ""),
        str(record.get("architecture") or ""),
        str(record.get("question_id") or ""),
        str(record.get("medium") or ""),
    )


def dump_record(record: dict) -> str:
    return json.dumps(record, ensure_ascii=False, sort_keys=True)
