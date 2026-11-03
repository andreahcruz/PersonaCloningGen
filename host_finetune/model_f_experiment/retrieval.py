"""One raw LightRAG query per question. Not used while indexing is running.

The returned payload is the ``aquery_data`` body, including entity rows.
``generation_payload`` adds verbatim hits and keeps those rows for F-DESC.
``query_documents`` exposes the same text on ``entity_descriptions``.
"""

from __future__ import annotations

import shutil
from pathlib import Path

from host_finetune.model_f_experiment.arms import RETRIEVAL
from host_finetune.model_f_experiment.entity_descriptions import attach_raw_entities
from host_finetune.model_f_experiment.snapshot import SnapshotError, assert_snapshot_path


def generation_payload(
    raw: dict,
    catalog: dict,
    *,
    failed_document_ids: set[str] | None = None,
) -> dict:
    """Hits for the evidence capsule, plus the raw entity and relationship rows.

    Failed graph documents stay in the raw payload and are left out of hits,
    matching ``query_documents``.
    """
    from host_finetune.lightrag_backend import normalize_lightrag_data

    normalized = normalize_lightrag_data(
        raw,
        catalog,
        mode=RETRIEVAL["mode"],
        evidence_budget=RETRIEVAL["evidence_budget"],
        failed_document_ids=failed_document_ids,
    )
    combined = attach_raw_entities(normalized, raw)
    combined["data"] = (raw or {}).get("data") or {}
    combined["metadata"] = (raw or {}).get("metadata") or {}
    combined["status"] = (raw or {}).get("status")
    combined["query"] = (raw or {}).get("query")
    observed = ((raw or {}).get("metadata") or {}).get("query_mode") or (raw or {}).get("mode")
    combined["backend"] = RETRIEVAL["backend"]
    if observed:
        combined["mode"] = observed
        combined["mode_source"] = "payload"
    else:
        combined["mode"] = RETRIEVAL["mode"]
        combined["mode_source"] = "experiment_query"
    return combined


def prepare_query_workdir(snapshot: Path, workdir: Path) -> Path:
    """Copy the verified snapshot once. Queries write LightRAG cache into the copy."""
    snapshot = Path(snapshot).resolve()
    workdir = Path(workdir).resolve()
    assert_snapshot_path(snapshot)
    if workdir == snapshot or snapshot in workdir.parents or workdir in snapshot.parents:
        raise SnapshotError("the LightRAG query copy must stay outside the verified snapshot")
    if not snapshot.is_dir():
        raise SnapshotError(f"snapshot is not a directory: {snapshot}")
    if not workdir.exists():
        workdir.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(snapshot, workdir)
    return workdir


async def retrieve_raw(query: str, snapshot: Path) -> dict:
    """Query a working copy. Raises before any call if the path is the live store."""
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
