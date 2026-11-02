"""Keep raw entity descriptions available to F-DESC.

``query_documents`` now returns those rows on ``entity_descriptions`` and
still leaves them out of ``hits``. This module remains the reader for a raw
``aquery_data`` payload, which the experiment freezes before normalization.
"""

from __future__ import annotations


def entity_rows(payload: dict | None) -> list[dict]:
    """Entity records from a raw LightRAG payload, including description text."""
    data = (payload or {}).get("data") or {}
    rows = []
    for row in data.get("entities") or []:
        if not isinstance(row, dict):
            continue
        rows.append({
            "entity_name": row.get("entity_name"),
            "entity_type": row.get("entity_type"),
            "description": row.get("description"),
            "source_id": row.get("source_id"),
            "file_path": row.get("file_path"),
        })
    return rows


def attach_raw_entities(normalized: dict, raw_payload: dict | None) -> dict:
    """Add raw entity rows beside hits. Descriptions do not become passages."""
    attached = dict(normalized)
    attached["raw_entities"] = entity_rows(raw_payload)
    hits = list(attached.get("hits") or [])
    for hit in hits:
        if hit.get("provenance") == "entity_description":
            raise ValueError("entity description was stored as a source hit")
    attached["hits"] = hits
    return attached
