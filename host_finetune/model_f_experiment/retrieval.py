"""One raw LightRAG query per question. Not used while indexing is running.

The returned payload is the ``aquery_data`` body, before description text is
dropped. ``generation_payload`` then adds verbatim hits without removing
those entity rows.
"""

from __future__ import annotations

from pathlib import Path

from host_finetune.model_f_experiment.arms import RETRIEVAL
from host_finetune.model_f_experiment.entity_descriptions import attach_raw_entities
from host_finetune.model_f_experiment.snapshot import SnapshotError, assert_snapshot_path


def generation_payload(raw: dict, catalog: dict) -> dict:
    """Hits for the evidence capsule, plus the raw entity and relationship rows."""
    from host_finetune.lightrag_backend import normalize_lightrag_data

    normalized = normalize_lightrag_data(
        raw,
        catalog,
        mode=RETRIEVAL["mode"],
        evidence_budget=RETRIEVAL["evidence_budget"],
    )
    combined = attach_raw_entities(normalized, raw)
    combined["data"] = (raw or {}).get("data") or {}
    combined["metadata"] = (raw or {}).get("metadata") or {}
    combined["status"] = (raw or {}).get("status")
    combined["query"] = (raw or {}).get("query")
    return combined


async def retrieve_raw(query: str, snapshot: Path) -> dict:
    """Query a snapshot copy. Raises before any call if the path is the live store."""
    assert_snapshot_path(snapshot)
    from host_finetune.lightrag_backend import (
        DEFAULT_EMBED_MODEL,
        DEFAULT_EXTRACTION_MODEL,
        PROJECT_ENTITY_GUIDANCE,
        build_lightrag,
        read_catalog,
        require_ollama_models,
    )

    base_url = "http://localhost:11434"
    require_ollama_models(base_url, DEFAULT_EXTRACTION_MODEL, DEFAULT_EMBED_MODEL)
    catalog = read_catalog(snapshot / "catalog.jsonl")
    if not catalog:
        raise SnapshotError(f"snapshot catalog is empty: {snapshot}")
    from lightrag import QueryParam

    rag = build_lightrag(
        snapshot,
        base_url,
        DEFAULT_EXTRACTION_MODEL,
        DEFAULT_EMBED_MODEL,
        entity_types_guidance=PROJECT_ENTITY_GUIDANCE,
        num_ctx=16384,
    )
    await rag.initialize_storages()
    try:
        chunk_pool = max(RETRIEVAL["evidence_budget"] * 3, RETRIEVAL["evidence_budget"])
        payload = await rag.aquery_data(
            query,
            param=QueryParam(
                mode=RETRIEVAL["mode"],
                only_need_context=True,
                top_k=RETRIEVAL["top_k"],
                chunk_top_k=chunk_pool,
                enable_rerank=False,
            ),
        )
    finally:
        await rag.finalize_storages()
    if not isinstance(payload, dict):
        raise SnapshotError("LightRAG query did not return an object")
    payload = dict(payload)
    payload["query"] = query
    return payload
