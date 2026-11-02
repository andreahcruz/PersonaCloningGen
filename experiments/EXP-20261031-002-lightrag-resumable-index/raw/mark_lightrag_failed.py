"""Mark the sealed expansion document failed inside LightRAG.

The row stays. Chunk text and vectors are not deleted. Status ``failed`` keeps
the next ingest from auto-resuming a document whose extraction never finished.
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from host_finetune.lightrag_backend import build_lightrag

WORKDIR = ROOT / "artifacts" / "lightrag" / "model_f" / "storage_expand_v1"
DOC_ID = "gd_002be28d89a6ca9924e73be2"


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
    rag = build_lightrag(
        WORKDIR,
        "http://localhost:11434",
        "llama3.1:latest",
        "nomic-embed-text",
        num_ctx=16384,
    )
    await rag.initialize_storages()
    try:
        row = await rag.doc_status.get_by_id(DOC_ID)
        if row is None:
            raise SystemExit(f"missing LightRAG status for {DOC_ID}")
        data = dict(row)
        data["status"] = "failed"
        data["error_msg"] = (
            "document time limit cancelled extraction twice; "
            "left failed so a later ingest does not auto-resume it"
        )
        await rag.doc_status.upsert({DOC_ID: data})
        await rag.finalize_storages()
    finally:
        await _shutdown(rag)
    status = json.loads((WORKDIR / "kv_store_doc_status.json").read_text(encoding="utf-8"))
    counts = {}
    for item in status.values():
        name = str(item.get("status"))
        counts[name] = counts.get(name, 0) + 1
    failed = status[DOC_ID]
    print(json.dumps({
        "doc_status": failed.get("status"),
        "kg_write_state": (failed.get("metadata") or {}).get("kg_write_state"),
        "file_path": failed.get("file_path"),
        "status_counts": counts,
    }, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
