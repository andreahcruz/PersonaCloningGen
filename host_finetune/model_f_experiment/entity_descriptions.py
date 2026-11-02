"""Keep raw entity descriptions available to F-DESC.

``lightrag_backend.query_documents`` still drops description text before it
returns. That file is imported by the running indexer, so this module does
not edit it. The experiment freezes the raw query payload instead. After the
indexer exits, ``query_documents`` should return ``raw_entities`` from
``entity_rows`` without adding those rows to ``hits``.
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
