"""One context-only query against the expansion graph. Does not change rag_active.json."""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from host_finetune.graph_relationships import validate_relationships
from host_finetune.lightrag_backend import (
    build_lightrag,
    normalize_lightrag_data,
    read_catalog,
    read_failed_document_ids,
    require_lightrag_mode,
    require_ollama_models,
)

WORKDIR = ROOT / "artifacts" / "lightrag" / "model_f" / "storage_expand_v1"
OUT = Path(__file__).resolve().parent / "retrieval_smoke.json"
QUERY = "What changed at SaaStr Annual between last year and this year?"


async def _shutdown(rag) -> None:
    targets = []
    embedding = getattr(rag, "embedding_func", None)
    if embedding is not None:
        targets.append(getattr(embedding, "func", None))
    for state in getattr(rag, "_role_llm_states", {}).values():
        targets.append(getattr(state, "wrapped", None))
    for func in targets:
        shutdown = getattr(func, "shutdown", None)
        if callable(shutdown):
            await shutdown(graceful=False, timeout=1)


async def main() -> None:
    mode = require_lightrag_mode("mix")
    require_ollama_models("http://localhost:11434", "llama3.1:latest", "nomic-embed-text")
    catalog = read_catalog(WORKDIR / "catalog.jsonl")
    from lightrag import QueryParam

    rag = build_lightrag(
        WORKDIR,
        "http://localhost:11434",
        "llama3.1:latest",
        "nomic-embed-text",
        num_ctx=16384,
    )
    await rag.initialize_storages()
    try:
        payload = await rag.aquery_data(
            QUERY,
            param=QueryParam(
                mode=mode,
                only_need_context=True,
                top_k=10,
                chunk_top_k=9,
                enable_rerank=False,
            ),
        )
    finally:
        await rag.finalize_storages()
        await _shutdown(rag)
    failed_ids = read_failed_document_ids(WORKDIR)
    normalized = normalize_lightrag_data(
        payload,
        catalog,
        mode=mode,
        evidence_budget=3,
        max_per_source=1,
        failed_document_ids=failed_ids,
    )
    relationships = validate_relationships(
        normalized.get("relationship_candidates") or [],
        catalog,
        failed_document_ids=failed_ids,
    )
    hits = []
    for hit in normalized["hits"]:
        hits.append(
            {
                "graph_document_id": hit.get("graph_document_id"),
                "medium": hit.get("medium"),
                "source_path": hit.get("source_file"),
            }
        )
    result = {
        "query": QUERY,
        "mode": mode,
        "num_ctx": 16384,
        "status": normalized["status"],
        "message": normalized["message"],
        "hit_count": len(normalized["hits"]),
        "hits": hits,
        "relationships": {
            "accepted": len(relationships.get("accepted") or []),
            "quarantined": len(relationships.get("quarantined") or []),
            "rejected": len(relationships.get("rejected") or []),
        },
        "excluded_count": len(normalized["excluded"]),
    }
    OUT.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
