"""Small, deployment-safe wrapper around Kevin's LightRAG retrieval contract.

The service opens a *runtime copy* of the verified Model F snapshot.  It never
indexes documents, mutates the source corpus, or returns generated entity
descriptions as citations.  The latter are intentionally omitted from the
response because they are graph-derived text rather than source passages.
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

import requests

REQUIRED_LIGHTRAG_VERSION = "1.5.7"
EMBED_DIMENSION = 768
ALLOWED_MODES = {"naive", "local", "global", "hybrid", "mix"}
REQUIRED_SNAPSHOT_FILES = {
    "catalog.jsonl",
    "graph_chunk_entity_relation.graphml",
    "kv_store_full_docs.json",
    "kv_store_text_chunks.json",
    "vdb_chunks.json",
    "vdb_entities.json",
    "vdb_relationships.json",
    "snapshot_hashes.json",
}


class LightRAGUnavailable(RuntimeError):
    """Raised when the runtime image, snapshot, or Ollama dependency is missing."""


def _normalise(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def _verbatim_span(chunk: str, source: str) -> str | None:
    """Only return text that really occurs in its catalogued source document."""
    if chunk and chunk in source:
        return chunk
    # A whitespace-folded match is useful diagnostically, but is not safe to
    # turn into a citable slice because we would be inventing its boundaries.
    if _normalise(chunk) and _normalise(chunk) in _normalise(source):
        return None
    return None


def snapshot_status(workdir: Path) -> dict[str, Any]:
    """Fast structural readiness check; full hashing is a separate setup step."""
    workdir = Path(workdir)
    missing = sorted(name for name in REQUIRED_SNAPSHOT_FILES if not (workdir / name).is_file())
    if missing:
        return {
            "ready": False,
            "detail": f"LightRAG runtime snapshot is incomplete: missing {', '.join(missing)}",
            "documents": 0,
            "embedding_dimension": None,
        }
    try:
        with (workdir / "catalog.jsonl").open(encoding="utf-8") as handle:
            documents = sum(1 for line in handle if line.strip())
        with (workdir / "vdb_chunks.json").open(encoding="utf-8") as handle:
            payload = json.load(handle)
        dimension = int(payload.get("embedding_dim", 0))
    except (OSError, ValueError, json.JSONDecodeError) as error:
        return {
            "ready": False,
            "detail": f"Could not read LightRAG runtime snapshot: {error}",
            "documents": 0,
            "embedding_dimension": None,
        }
    if dimension != EMBED_DIMENSION:
        return {
            "ready": False,
            "detail": f"LightRAG snapshot uses {dimension} dimensions; expected {EMBED_DIMENSION}.",
            "documents": documents,
            "embedding_dimension": dimension,
        }
    return {
        "ready": documents > 0,
        "detail": f"{documents} catalogued snapshot documents, {dimension}-dimension embeddings",
        "documents": documents,
        "embedding_dimension": dimension,
    }


def verify_snapshot(workdir: Path) -> dict[str, Any]:
    """Recompute the archive's recorded file hashes before first deployment use."""
    status = snapshot_status(workdir)
    if not status["ready"]:
        raise LightRAGUnavailable(status["detail"])
    try:
        manifest = json.loads((Path(workdir) / "snapshot_hashes.json").read_text(encoding="utf-8"))
        expected: dict[str, str] = manifest["files"]
    except (OSError, KeyError, TypeError, json.JSONDecodeError) as error:
        raise LightRAGUnavailable(f"Snapshot hash manifest is invalid: {error}") from error
    mismatches: list[str] = []
    for relative, digest in expected.items():
        target = Path(workdir) / relative
        hasher = hashlib.sha256()
        try:
            with target.open("rb") as handle:
                for block in iter(lambda: handle.read(1024 * 1024), b""):
                    hasher.update(block)
        except OSError:
            mismatches.append(relative)
            continue
        if hasher.hexdigest() != digest:
            mismatches.append(relative)
    if mismatches:
        raise LightRAGUnavailable(f"Snapshot hash verification failed: {', '.join(mismatches)}")
    return {**status, "verified_files": len(expected)}


def _require_lightrag() -> None:
    try:
        from lightrag._version import __version__
    except ImportError as error:
        raise LightRAGUnavailable("lightrag-hku is not installed in the retrieval service") from error
    if __version__ != REQUIRED_LIGHTRAG_VERSION:
        raise LightRAGUnavailable(
            f"LightRAG {__version__} is installed; Model F requires {REQUIRED_LIGHTRAG_VERSION}."
        )


def _require_ollama_models(base_url: str, extraction_model: str, embed_model: str) -> None:
    try:
        response = requests.get(f"{base_url.rstrip('/')}/api/tags", timeout=10)
        response.raise_for_status()
        names = {str(item.get("name") or "") for item in response.json().get("models") or []}
    except (requests.RequestException, ValueError) as error:
        raise LightRAGUnavailable(f"Cannot reach Ollama at {base_url}: {error}") from error
    missing = [
        model
        for model in (extraction_model, embed_model)
        if model not in names and f"{model}:latest" not in names
    ]
    if missing:
        raise LightRAGUnavailable(f"Missing LightRAG Ollama model(s): {', '.join(missing)}")


def _load_catalog(path: Path) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    try:
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                if row.get("included", True) and not row.get("exclusion_reasons"):
                    rows[str(row["graph_document_id"])] = row
    except (OSError, KeyError, json.JSONDecodeError) as error:
        raise LightRAGUnavailable(f"Could not load LightRAG catalog: {error}") from error
    if not rows:
        raise LightRAGUnavailable("LightRAG catalog is empty")
    return rows


def _failed_document_ids(workdir: Path) -> set[str]:
    """Keep chunks from failed graph extraction attempts out of results."""
    path = workdir / "kv_store_doc_status.json"
    if not path.is_file():
        return set()
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return set()
    rows = payload.values() if isinstance(payload, dict) else payload
    failed: set[str] = set()
    for row in rows or []:
        if not isinstance(row, dict) or str(row.get("status") or "").lower() != "failed":
            continue
        for key in ("file_path", "id", "doc_id", "_id"):
            if row.get(key):
                failed.add(str(row[key]))
    return failed


def _embedding_func(base_url: str, embed_model: str):
    _require_lightrag()
    import numpy as np
    from lightrag.llm.ollama import ollama_embed
    from lightrag.utils import wrap_embedding_func_with_attrs

    @wrap_embedding_func_with_attrs(
        embedding_dim=EMBED_DIMENSION,
        max_token_size=8192,
        model_name=embed_model,
        supports_asymmetric=True,
    )
    async def embed(texts: list[str], **kwargs) -> np.ndarray:
        for key in ("embed_model", "host", "embedding_dim"):
            kwargs.pop(key, None)
        return await ollama_embed.func(
            texts,
            embed_model=embed_model,
            embedding_dim=EMBED_DIMENSION,
            host=base_url,
            query_prefix="search_query: ",
            document_prefix="search_document: ",
            **kwargs,
        )

    return embed


def _build_lightrag(workdir: Path, base_url: str, extraction_model: str, embed_model: str, num_ctx: int):
    _require_lightrag()
    from lightrag import LightRAG
    from lightrag.llm.ollama import ollama_model_complete

    return LightRAG(
        working_dir=str(workdir),
        llm_model_func=ollama_model_complete,
        llm_model_name=extraction_model,
        llm_model_kwargs={"host": base_url, "options": {"num_ctx": num_ctx, "temperature": 0}},
        embedding_func=_embedding_func(base_url, embed_model),
        llm_model_max_async=1,
        embedding_func_max_async=1,
        chunk_token_size=400,
        entity_extract_max_gleaning=1,
        default_llm_timeout=600,
        addon_params={"language": "English"},
    )


def _normalise_hits(
    raw: dict[str, Any] | None,
    catalog: dict[str, dict[str, Any]],
    *,
    mode: str,
    evidence_budget: int,
    failed_ids: set[str],
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    data = (raw or {}).get("data") or {}
    hits: list[dict[str, Any]] = []
    rejected = {"unknown_document": 0, "failed_document": 0, "non_verbatim": 0, "duplicate": 0}
    seen: set[str] = set()
    for row in data.get("chunks") or []:
        if not isinstance(row, dict):
            continue
        doc_id = str(row.get("file_path") or "")
        document = catalog.get(doc_id)
        if document is None or document.get("split") != "train":
            rejected["unknown_document"] += 1
            continue
        if doc_id in failed_ids or str(document.get("parent_graph_document_id") or doc_id) in failed_ids:
            rejected["failed_document"] += 1
            continue
        text = _verbatim_span(str(row.get("content") or row.get("text") or ""), str(document.get("text") or ""))
        if not text:
            rejected["non_verbatim"] += 1
            continue
        if text in seen:
            rejected["duplicate"] += 1
            continue
        seen.add(text)
        hits.append(
            {
                "id": str(document.get("parent_graph_document_id") or doc_id),
                "chunk_text": text,
                "source": str(document.get("medium") or "unknown"),
                "title": document.get("title"),
                "url": document.get("url"),
                "date": document.get("date"),
                "graph_document_id": doc_id,
                "backend": "lightrag",
                "retrieval_mode": mode,
                "score_type": "unscored_rank",
                "rank": len(hits) + 1,
            }
        )
        if len(hits) >= evidence_budget:
            break
    return hits, rejected


async def retrieve(
    query: str,
    workdir: Path,
    *,
    base_url: str,
    extraction_model: str,
    embed_model: str,
    mode: str = "hybrid",
    evidence_budget: int = 4,
    top_k: int = 20,
    num_ctx: int = 16384,
) -> dict[str, Any]:
    if mode not in ALLOWED_MODES:
        raise LightRAGUnavailable(f"Unsupported LightRAG mode {mode!r}")
    if not query.strip():
        raise LightRAGUnavailable("A query is required")
    if not 1 <= evidence_budget <= 8:
        raise LightRAGUnavailable("Evidence budget must be between 1 and 8")
    status = snapshot_status(workdir)
    if not status["ready"]:
        raise LightRAGUnavailable(status["detail"])
    _require_ollama_models(base_url, extraction_model, embed_model)
    catalog = _load_catalog(Path(workdir) / "catalog.jsonl")
    from lightrag import QueryParam

    rag = _build_lightrag(Path(workdir), base_url, extraction_model, embed_model, num_ctx)
    await rag.initialize_storages()
    try:
        raw = await rag.aquery_data(
            query,
            param=QueryParam(
                mode=mode,
                only_need_context=True,
                top_k=top_k,
                chunk_top_k=max(evidence_budget * 3, evidence_budget),
                enable_rerank=False,
            ),
        )
    finally:
        await rag.finalize_storages()

    hits, rejected = _normalise_hits(
        raw,
        catalog,
        mode=mode,
        evidence_budget=evidence_budget,
        failed_ids=_failed_document_ids(Path(workdir)),
    )
    return {
        "query": query,
        "backend": "lightrag",
        "mode": mode,
        "hits": hits,
        "source_count": len(hits),
        "snapshot_documents": status["documents"],
        "embedding_dimension": status["embedding_dimension"],
        "rejected": rejected,
        # Generated graph summaries are deliberately not returned as sources.
        "non_citable_graph_context": {
            "entity_descriptions_available": len(((raw or {}).get("data") or {}).get("entities") or []),
            "relationship_candidates_available": len(((raw or {}).get("data") or {}).get("relationships") or []),
        },
    }
