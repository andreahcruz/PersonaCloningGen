"""LightRAG 1.5.7 retrieval behind the Model E evidence contract.

Indexing and querying stay out of the frozen Chroma directories. Retrieval
uses ``query_data`` so LightRAG does not write the final article. Generated
entity and relationship summaries are not citable passages. Relationship rows
are returned separately and stay out of claim validation until
``graph_relationships`` accepts a source span.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import re
from pathlib import Path

import requests

from host_finetune.retrieval_contract import require_lightrag_mode

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[1]
REQUIRED_LIGHTRAG_VERSION = "1.5.7"
DEFAULT_WORKDIR = ROOT / "artifacts" / "lightrag" / "model_f" / "storage"
DEFAULT_GRAPH_READY = ROOT / "artifacts" / "lightrag" / "model_f" / "graph_ready"
DEFAULT_EXTRACTION_MODEL = "llama3.1:latest"
DEFAULT_EMBED_MODEL = "nomic-embed-text"
EMBED_DIM = 768
NUM_CTX = 32768
LIGHTRAG_MODES = ("naive", "local", "global", "hybrid", "mix")


class LightRAGUnavailable(RuntimeError):
    """The local model, package, or index required for a query is not available."""


def require_lightrag_package() -> str:
    try:
        import lightrag
        from lightrag._version import __version__
    except ImportError as exc:
        raise LightRAGUnavailable(
            "lightrag-hku is not installed. Install host_finetune/requirements-lightrag.txt "
            "into the host fine-tune environment, then restore opentelemetry-api==1.41.1 "
            "so Chroma 1.5.5 still imports."
        ) from exc
    if __version__ != REQUIRED_LIGHTRAG_VERSION:
        raise LightRAGUnavailable(
            f"LightRAG {__version__} is installed; this project was written for "
            f"{REQUIRED_LIGHTRAG_VERSION}."
        )
    return __version__


def ollama_model_names(base_url: str, timeout: float = 10) -> set[str]:
    try:
        response = requests.get(f"{base_url.rstrip('/')}/api/tags", timeout=timeout)
        response.raise_for_status()
        payload = response.json()
    except (requests.RequestException, ValueError) as exc:
        raise LightRAGUnavailable(f"Ollama is not reachable at {base_url}: {exc}") from exc
    names = set()
    for row in payload.get("models") or []:
        name = str(row.get("name") or "").strip()
        if name:
            names.add(name)
    return names


def require_ollama_models(base_url: str, extraction_model: str, embed_model: str) -> None:
    names = ollama_model_names(base_url)
    missing = []
    for model in (extraction_model, embed_model):
        if model not in names and f"{model}:latest" not in names:
            missing.append(model)
    if missing:
        raise LightRAGUnavailable(
            "missing Ollama models: " + ", ".join(missing) + f". Installed: {sorted(names)}"
        )


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def catalog_from_documents(documents: list[dict]) -> dict[str, dict]:
    catalog = {}
    for document in documents:
        if not document.get("included", True):
            continue
        if document.get("exclusion_reasons"):
            continue
        doc_id = document["graph_document_id"]
        catalog[doc_id] = {
            "graph_document_id": doc_id,
            "text": document["text"],
            "text_sha256": document.get("text_sha256")
            or hashlib.sha256(document["text"].encode("utf-8")).hexdigest(),
            "medium": document.get("medium"),
            "group_id": document.get("group_id"),
            "group_ids": document.get("group_ids") or [],
            "split": document.get("split"),
            "source_file": document.get("source_file"),
            "source_line": document.get("source_line"),
            "experiment_row_ids": document.get("experiment_row_ids") or [],
            "primary_row_index": document.get("primary_row_index"),
            "attribution_status": document.get("attribution_status"),
            "author_position": document.get("author_position"),
            "quoted_speakers": list(document.get("quoted_speakers") or []),
            "parent_graph_document_id": document.get("parent_graph_document_id") or doc_id,
            "reconstruction_method": document.get("reconstruction_method"),
        }
    return catalog


def write_catalog(catalog: dict[str, dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [catalog[key] for key in sorted(catalog)]
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def read_catalog(path: Path) -> dict[str, dict]:
    if not path.is_file():
        raise LightRAGUnavailable(f"LightRAG catalog is missing: {path}")
    rows = load_jsonl(path)
    return {row["graph_document_id"]: row for row in rows}


def _verbatim_span(chunk: str, source: str) -> str | None:
    if chunk and chunk in source:
        return chunk
    folded_chunk = _norm(chunk)
    folded_source = _norm(source)
    if not folded_chunk or folded_chunk not in folded_source:
        return None
    # The chunk survives whitespace folding but is not an exact slice.
    # Do not invent a span that the validator could fail to find.
    return None


OMISSION_MARKER = "[Omitted same-source text"


def read_failed_document_ids(workdir: Path) -> set[str]:
    """Failed and skipped graph ids. Missing files mean there is nothing to exclude."""
    found: set[str] = set()
    status_path = workdir / "kv_store_doc_status.json"
    if status_path.is_file():
        try:
            payload = json.loads(status_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            payload = {}
        rows = payload.values() if isinstance(payload, dict) else payload
        for row in rows or []:
            if not isinstance(row, dict):
                continue
            if str(row.get("status") or "").lower() != "failed":
                continue
            for key in ("file_path", "id", "doc_id", "_id"):
                if row.get(key):
                    found.add(str(row[key]))
    skip_path = workdir / "extraction_skip.json"
    if skip_path.is_file():
        try:
            skipped = json.loads(skip_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            skipped = []
        if isinstance(skipped, dict):
            skipped = skipped.get("ids") or skipped.get("documents") or []
        for item in skipped or []:
            if isinstance(item, str):
                found.add(item)
            elif isinstance(item, dict) and item.get("id"):
                found.add(str(item["id"]))
    return found


def normalize_lightrag_data(
    payload: dict | None,
    catalog: dict[str, dict],
    *,
    mode: str,
    evidence_budget: int,
    max_per_source: int = 1,
    failed_document_ids: set[str] | None = None,
) -> dict:
    """Turn a ``query_data`` payload into Model E hits plus excluded graph rows.

    Hits contain verbatim source chunks only. Relationship rows are copied
    to ``relationship_candidates`` for a later source-span check. Their text
    is still not a hit. Entity rows stay out of hits and out of the excluded
    record's text. ``entity_descriptions`` keeps ``entity_name``,
    ``entity_type``, ``description``, ``source_id``, and ``file_path`` so a
    generated summary can be shown as derived context without becoming a
    citable passage or an allowed quantity.
    """
    mode = require_lightrag_mode(mode)
    if evidence_budget < 1:
        raise ValueError("evidence_budget must be positive")
    if max_per_source < 1:
        raise ValueError("max_per_source must be positive")
    data = (payload or {}).get("data") or {}
    chunks = data.get("chunks") or []
    hits = []
    excluded = []
    failed_ids = set(failed_document_ids or ())
    per_source: dict[str, int] = {}
    seen_text = set()
    for chunk in chunks:
        if not isinstance(chunk, dict):
            excluded.append({"reason": "malformed_chunk"})
            continue
        file_path = str(chunk.get("file_path") or "")
        content = chunk.get("content") or chunk.get("text") or ""
        document = catalog.get(file_path)
        if document is None:
            excluded.append({"reason": "unknown_graph_document", "file_path": file_path})
            continue
        if document.get("split") != "train":
            excluded.append({"reason": "non_train_document", "file_path": file_path})
            continue
        parent_id = document.get("parent_graph_document_id") or file_path
        if file_path in failed_ids or parent_id in failed_ids:
            excluded.append({"reason": "failed_extraction_chunk", "file_path": file_path})
            continue
        verbatim = _verbatim_span(str(content), document["text"])
        if verbatim and OMISSION_MARKER in verbatim:
            excluded.append({"reason": "omission_marker_span", "file_path": file_path})
            continue
        if verbatim is None:
            excluded.append({"reason": "chunk_not_in_source", "file_path": file_path})
            continue
        if verbatim in seen_text:
            excluded.append({"reason": "duplicate_chunk", "file_path": file_path})
            continue
        if per_source.get(file_path, 0) >= max_per_source:
            excluded.append({"reason": "source_diversity_cap", "file_path": file_path})
            continue
        seen_text.add(verbatim)
        per_source[file_path] = per_source.get(file_path, 0) + 1
        chunk_id = str(chunk.get("chunk_id") or len(hits) + 1)
        row_ids = document.get("experiment_row_ids") or []
        # The claim extractor copies this id. A file_path:chunk_id string is
        # rejected as an unknown source, which drops every claim.
        evidence_id = document.get("parent_graph_document_id") or file_path
        hits.append(
            {
                "id": evidence_id,
                "text": verbatim,
                "medium": document.get("medium"),
                "group_id": document.get("group_id"),
                "row_id": document.get("primary_row_index"),
                "split": "train",
                "distance": None,
                "rank": len(hits) + 1,
                "score": None,
                "score_type": "unscored_rank",
                "backend": "lightrag",
                "retrieval_mode": mode,
                "attribution_status": document.get("attribution_status"),
                "author_position": document.get("author_position"),
                "quoted_speakers": list(document.get("quoted_speakers") or []),
                "graph_document_id": file_path,
                "parent_graph_document_id": document.get("parent_graph_document_id") or file_path,
                "chunk_id": chunk_id,
                "provenance": "source_chunk",
                "source_row_ids": list(row_ids),
                "source_file": document.get("source_file"),
                "source_line": document.get("source_line"),
            }
        )
        if len(hits) >= evidence_budget:
            break
    relationship_candidates = []
    entity_descriptions = []
    for kind in ("entities", "relationships"):
        for row in data.get(kind) or []:
            if not isinstance(row, dict):
                continue
            if kind == "relationships":
                relationship_candidates.append(dict(row))
            else:
                # Name is required for the description channel. The other four
                # fields are the ones this function used to drop.
                entity_descriptions.append({
                    "entity_name": row.get("entity_name"),
                    "entity_type": row.get("entity_type"),
                    "description": row.get("description"),
                    "source_id": row.get("source_id"),
                    "file_path": row.get("file_path"),
                })
            excluded.append(
                {
                    "reason": "generated_summary_not_a_source_span",
                    "kind": kind,
                    "name": row.get("entity_name") or row.get("src_id"),
                    "file_path": row.get("file_path"),
                    "source_id": row.get("source_id"),
                }
            )
    status = (payload or {}).get("status")
    return {
        "hits": hits,
        "excluded": excluded,
        "relationship_candidates": relationship_candidates,
        "entity_descriptions": entity_descriptions,
        "status": status,
        "metadata": (payload or {}).get("metadata") or {},
        "message": (payload or {}).get("message"),
    }


def _status_text(row: dict | None) -> str:
    if not row:
        return ""
    raw = row.get("status")
    value = getattr(raw, "value", raw)
    return str(value or "").strip().lower()


def make_embedding_func(base_url: str, embed_model: str):
    require_lightrag_package()
    import numpy as np
    from lightrag.llm.ollama import ollama_embed
    from lightrag.utils import wrap_embedding_func_with_attrs

    @wrap_embedding_func_with_attrs(
        embedding_dim=EMBED_DIM,
        max_token_size=8192,
        model_name=embed_model,
        supports_asymmetric=True,
    )
    async def embedding_func(texts: list[str], **kwargs) -> np.ndarray:
        kwargs.pop("embed_model", None)
        kwargs.pop("host", None)
        kwargs.pop("embedding_dim", None)
        return await ollama_embed.func(
            texts,
            embed_model=embed_model,
            embedding_dim=EMBED_DIM,
            host=base_url,
            query_prefix="search_query: ",
            document_prefix="search_document: ",
            **kwargs,
        )

    return embedding_func


PROJECT_ENTITY_GUIDANCE = """Classify an entity only when it is a specific named thing in the input text. If it does not fit, omit it.

- Person: a named human. A role word alone is not a person.
- Organization: a named company, fund, or institution.
- Product: a named product or service.
- Event: a named event.
- Concept: a specific business idea of more than one word. A single generic noun is not a concept.

Do not create an entity for a weekday, a month, a year, a clock time, a percentage, a currency amount, or an accuracy range such as "93%-95% Correct". Put a number in a description only when the input states that number. Do not invent a metric.
Do not create an entity for a generic noun such as Company, Partner, Investment, Investor, Round, Startup, Basic, or Update.
Do not extract entities from a line that begins with "[Attribution" or "[Quoted speaker" or "[End attribution".
When the input attributes a quotation to a named person, that person is the speaker of the quotation. Do not assign that speaker every organization mentioned later in a title or a link.
"""


def build_lightrag(
    workdir: Path,
    base_url: str,
    extraction_model: str,
    embed_model: str,
    *,
    entity_types_guidance: str | None = PROJECT_ENTITY_GUIDANCE,
    num_ctx: int = NUM_CTX,
):
    """Construct a LightRAG instance. The caller initializes storage.

    ``entity_types_guidance=None`` keeps the LightRAG 1.5.7 default guidance.
    The project guidance is the pilot default. The three-document smoke index
    was built before this argument existed and is not rebuilt by this function.
    """
    require_lightrag_package()
    from lightrag import LightRAG
    from lightrag.llm.ollama import ollama_model_complete

    workdir.mkdir(parents=True, exist_ok=True)
    addon_params: dict = {"language": "English"}
    if entity_types_guidance:
        addon_params["entity_types_guidance"] = entity_types_guidance
    return LightRAG(
        working_dir=str(workdir),
        llm_model_func=ollama_model_complete,
        llm_model_name=extraction_model,
        llm_model_kwargs={
            "host": base_url,
            "options": {"num_ctx": num_ctx, "temperature": 0},
        },
        embedding_func=make_embedding_func(base_url, embed_model),
        llm_model_max_async=1,
        embedding_func_max_async=1,
        chunk_token_size=400,
        entity_extract_max_gleaning=1,
        default_llm_timeout=600,
        addon_params=addon_params,
    )


async def _processed_ids(rag, doc_ids: list[str]) -> set[str]:
    processed = set()
    for doc_id in doc_ids:
        row = await rag.doc_status.get_by_id(doc_id)
        if _status_text(row) == "processed":
            processed.add(doc_id)
    return processed


def _manifest_path(workdir: Path) -> Path:
    return workdir / "index_manifest.json"


def read_manifest(workdir: Path) -> dict:
    path = _manifest_path(workdir)
    if not path.is_file():
        return {"documents": {}, "version": REQUIRED_LIGHTRAG_VERSION}
    return json.loads(path.read_text(encoding="utf-8"))


def write_manifest(workdir: Path, manifest: dict) -> None:
    _manifest_path(workdir).write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


async def index_documents(
    documents: list[dict],
    workdir: Path,
    *,
    base_url: str,
    extraction_model: str = DEFAULT_EXTRACTION_MODEL,
    embed_model: str = DEFAULT_EMBED_MODEL,
    batch_size: int = 2,
    entity_types_guidance: str | None = PROJECT_ENTITY_GUIDANCE,
    num_ctx: int = NUM_CTX,
) -> dict:
    """Insert documents that are not already PROCESSED. Does not delete storage."""
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    require_ollama_models(base_url, extraction_model, embed_model)
    catalog_path = workdir / "catalog.jsonl"
    catalog = read_catalog(catalog_path) if catalog_path.is_file() else {}
    incoming = catalog_from_documents(documents)
    for doc_id, document in incoming.items():
        previous = catalog.get(doc_id)
        if previous and previous.get("text_sha256") not in {None, document["text_sha256"]}:
            raise LightRAGUnavailable(
                f"refusing to replace catalog text for {doc_id}"
            )
        catalog[doc_id] = document
    write_catalog(catalog, catalog_path)
    manifest = read_manifest(workdir)
    known = manifest.setdefault("documents", {})
    rag = build_lightrag(
        workdir,
        base_url,
        extraction_model,
        embed_model,
        entity_types_guidance=entity_types_guidance,
        num_ctx=num_ctx,
    )
    await rag.initialize_storages()
    inserted = []
    skipped = []
    refused = []
    extraction_failed = []
    try:
        ordered = [catalog[key] for key in sorted(catalog)]
        processed = await _processed_ids(rag, [doc["graph_document_id"] for doc in ordered])
        skip_path = workdir / "extraction_skip.json"
        skip_ids = set(json.loads(skip_path.read_text(encoding="utf-8"))) if skip_path.is_file() else set()
        pending = []
        for document in ordered:
            doc_id = document["graph_document_id"]
            recorded = known.get(doc_id)
            if recorded and recorded.get("text_sha256") not in {None, document["text_sha256"]}:
                refused.append(doc_id)
                continue
            if doc_id in processed:
                skipped.append(doc_id)
                known[doc_id] = {
                    "text_sha256": document["text_sha256"],
                    "status": "processed",
                }
                continue
            if doc_id in skip_ids:
                row = await rag.doc_status.get_by_id(doc_id)
                if _status_text(row) in {"failed", "processing", "pending"}:
                    await rag.adelete_by_doc_id(doc_id)
                extraction_failed.append(
                    {"id": doc_id, "error": "listed in extraction_skip.json"}
                )
                continue
            pending.append(document)
        if refused:
            raise LightRAGUnavailable(
                "refusing to overwrite processed documents whose text changed: "
                + ", ".join(refused[:10])
            )
        for start in range(0, len(pending), batch_size):
            batch = pending[start : start + batch_size]
            error = ""
            for item in batch:
                doc_id = item["graph_document_id"]
                row = await rag.doc_status.get_by_id(doc_id)
                if _status_text(row) in {"failed", "processing"}:
                    await rag.adelete_by_doc_id(doc_id)
            try:
                await rag.ainsert(
                    [item["text"] for item in batch],
                    ids=[item["graph_document_id"] for item in batch],
                    file_paths=[item["graph_document_id"] for item in batch],
                )
            except Exception as exc:
                error = str(exc)
                logger.exception("LightRAG batch failed")
            done = await _processed_ids(rag, [item["graph_document_id"] for item in batch])
            for item in batch:
                doc_id = item["graph_document_id"]
                if doc_id in done:
                    inserted.append(doc_id)
                    known[doc_id] = {
                        "text_sha256": item["text_sha256"],
                        "status": "processed",
                    }
                else:
                    extraction_failed.append({"id": doc_id, "error": error[:500]})
            write_manifest(workdir, manifest)
            logger.info(
                "indexed %d / %d pending, failed %d",
                len(inserted),
                len(pending),
                len(extraction_failed),
            )
        write_manifest(workdir, manifest)
    finally:
        await rag.finalize_storages()
    return {
        "inserted": inserted,
        "skipped_processed": skipped,
        "extraction_failed": extraction_failed,
        "catalog_documents": len(catalog),
        "workdir": str(workdir),
        "extraction_model": extraction_model,
        "embed_model": embed_model,
        "embedding_dim": EMBED_DIM,
        "num_ctx": num_ctx,
        "entity_guidance": (
            "library_default"
            if entity_types_guidance is None
            else "project"
            if entity_types_guidance == PROJECT_ENTITY_GUIDANCE
            else "custom"
        ),
        "lightrag_version": REQUIRED_LIGHTRAG_VERSION,
    }


async def query_documents(
    query: str,
    workdir: Path,
    mode: str,
    *,
    base_url: str,
    extraction_model: str = DEFAULT_EXTRACTION_MODEL,
    embed_model: str = DEFAULT_EMBED_MODEL,
    evidence_budget: int = 4,
    max_per_source: int = 1,
    top_k: int = 20,
    entity_types_guidance: str | None = PROJECT_ENTITY_GUIDANCE,
    num_ctx: int = NUM_CTX,
) -> dict:
    mode = require_lightrag_mode(mode)
    require_ollama_models(base_url, extraction_model, embed_model)
    catalog = read_catalog(workdir / "catalog.jsonl")
    if not catalog:
        raise LightRAGUnavailable(f"LightRAG catalog is empty: {workdir}")
    from lightrag import QueryParam

    rag = build_lightrag(
        workdir,
        base_url,
        extraction_model,
        embed_model,
        entity_types_guidance=entity_types_guidance,
        num_ctx=num_ctx,
    )
    await rag.initialize_storages()
    try:
        # chunk_top_k is a chunk pool. The evidence budget is applied afterwards
        # so one source cannot fill every slot. top_k counts entities or
        # relationships, depending on the mode, and is not the document budget.
        chunk_pool = max(evidence_budget * max_per_source * 3, evidence_budget)
        payload = await rag.aquery_data(
            query,
            param=QueryParam(
                mode=mode,
                only_need_context=True,
                top_k=top_k,
                chunk_top_k=chunk_pool,
                enable_rerank=False,
            ),
        )
    finally:
        await rag.finalize_storages()
    failed_ids = read_failed_document_ids(workdir)
    normalized = normalize_lightrag_data(
        payload,
        catalog,
        mode=mode,
        evidence_budget=evidence_budget,
        max_per_source=max_per_source,
        failed_document_ids=failed_ids,
    )
    from host_finetune.graph_relationships import validate_relationships

    relationships = validate_relationships(
        normalized.get("relationship_candidates") or [],
        catalog,
        failed_document_ids=failed_ids,
    )
    return {
        "query": query,
        "filter": None,
        "k": evidence_budget,
        "backend": "lightrag",
        "mode": mode,
        "hits": normalized["hits"],
        "excluded": normalized["excluded"],
        "relationship_candidates": normalized.get("relationship_candidates") or [],
        "entity_descriptions": normalized.get("entity_descriptions") or [],
        "relationships": relationships,
        "status": normalized["status"],
        "metadata": normalized["metadata"],
        "message": normalized["message"],
        "score_note": "LightRAG ranks are not cosine distances and are not compared with Chroma distances.",
    }


def index_documents_sync(documents: list[dict], workdir: Path, **kwargs) -> dict:
    return asyncio.run(index_documents(documents, workdir, **kwargs))


def query_documents_sync(query: str, workdir: Path, mode: str, **kwargs) -> dict:
    return asyncio.run(query_documents(query, workdir, mode, **kwargs))


def build_lightrag_retriever(args):
    """Retriever callable with the same topic/medium signature as dense retrieval."""
    import time

    workdir = Path(args.lightrag_workdir)
    mode = require_lightrag_mode(args.lightrag_mode)
    budget = args.evidence_budget or args.k
    if budget < 1:
        raise LightRAGUnavailable("evidence budget must be positive")

    def retriever(topic: str, comparison_medium: str) -> dict:
        del comparison_medium
        started = time.perf_counter()
        payload = query_documents_sync(
            topic,
            workdir,
            mode,
            base_url=args.ollama_base,
            extraction_model=args.lightrag_model,
            embed_model=args.embed_model,
            evidence_budget=budget,
            max_per_source=args.lightrag_max_per_source,
        )
        payload["seconds"] = round(time.perf_counter() - started, 3)
        return payload

    return retriever
