"""Seal the twice-cancelled expansion document out of the default queue.

This does not call a model and does not delete LightRAG rows.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from host_finetune.indexing_ledger import IndexingLedger
WORKDIR = ROOT / "artifacts" / "lightrag" / "model_f" / "storage_expand_v1"
DOC_ID = "gd_002be28d89a6ca9924e73be2"


def main() -> None:
    ledger = IndexingLedger(WORKDIR / "indexing_ledger.sqlite")
    try:
        current = ledger.document(DOC_ID)
        if current is None:
            raise SystemExit(f"missing {DOC_ID}")
        if current["status"] != "failed_permanent":
            ledger.reconcile_status(
                DOC_ID,
                {
                    "status": "failed_permanent",
                    "error_stage": "interrupt",
                    "error_category": "process_interruption",
                    "error_message": (
                        "document time limit cancelled extraction twice before a chunk "
                        "finished; sealed out of the default queue"
                    ),
                    "retry_eligible": False,
                    "defer_reason": "expensive_timeout",
                    "persisted_lightrag_id": DOC_ID,
                    "indexed_chunk_count": current.get("indexed_chunk_count"),
                    "graph_persistence": current.get("graph_persistence") or "not_verified",
                    "vector_persistence": current.get("vector_persistence") or "present_unverified",
                    "provenance_verified": None,
                    "vector_only": False,
                    "graph_extracted": False,
                    "quality_notes": (
                        "Two 420-second document-timer cancels during extraction. "
                        "The retry purged staged chunks and found no entity anchors. "
                        "A default resume does not select this document."
                    ),
                },
            )
        sealed = ledger.document(DOC_ID)
        nxt = ledger.next_document(include_deferred=False, retry_timeouts=False)
        status = json.loads((WORKDIR / "kv_store_doc_status.json").read_text(encoding="utf-8"))
        row = status.get(DOC_ID) or {}
        meta = row.get("metadata") or {}
        chunks = json.loads((WORKDIR / "vdb_chunks.json").read_text(encoding="utf-8"))
        print(json.dumps({
            "sealed_status": sealed["status"],
            "defer_reason": sealed["defer_reason"],
            "attempt_number": sealed["attempt_number"],
            "word_count": sealed["word_count"],
            "estimated_chunks": sealed["estimated_chunks"],
            "next_default_document": None if nxt is None else nxt["document_id"],
            "lightrag_status": row.get("status"),
            "kg_write_state": meta.get("kg_write_state") if isinstance(meta, dict) else None,
            "lightrag_chunks_count": row.get("chunks_count"),
            "chunk_vectors": len(chunks.get("data") or []),
            "embedding_dim": chunks.get("embedding_dim"),
        }, indent=2))
    finally:
        ledger.close()


if __name__ == "__main__":
    main()
