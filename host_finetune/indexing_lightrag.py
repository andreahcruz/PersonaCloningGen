"""LightRAG 1.5.7 adapter for one-document indexing.

This module does not patch LightRAG. It calls the installed enqueue and
process methods, wraps the LLM and embedding callables this project already
supplies, and reads document status back after each call.

``ainsert`` cannot skip graph extraction. Vector-only mode passes
``process_options='!'`` to ``apipeline_enqueue_documents``, which LightRAG
1.5.7 honors as ``skip_kg``. Graph mode passes no process options, matching
``ainsert`` and the 160-document pilot chunker.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import time
from dataclasses import dataclass
from pathlib import Path

from host_finetune.indexing_classify import classify_failure, retry_eligible, terminal_status
from host_finetune.indexing_quality import FormatWarningCounter
from host_finetune.indexing_timeout import abortable
from host_finetune.lightrag_backend import (
    EMBED_DIM,
    PROJECT_ENTITY_GUIDANCE,
    catalog_from_documents,
    read_catalog,
    require_lightrag_package,
    write_catalog,
)

GRAPH_PATH_SEPARATOR = "<SEP>"


@dataclass(frozen=True)
class IndexConfig:
    extraction_model: str
    embed_model: str
    base_url: str
    num_ctx: int = 16384
    chunk_token_size: int = 400
    gleaning: int = 1
    llm_timeout: int = 600
    temperature: float = 0.0
    embedding_dim: int = EMBED_DIM
    entity_guidance: str | None = PROJECT_ENTITY_GUIDANCE
    store_role: str = "graph"
    workspace: str = ""
    omission_segments: bool = False

    @property
    def worker_timeout_seconds(self) -> int:
        """LightRAG sets the worker cap to ``llm_timeout * 2`` when no explicit cap is passed."""
        return self.llm_timeout * 2


def configuration_record(config: IndexConfig) -> dict:
    guidance = config.entity_guidance or ""
    record = {
        "lightrag_version": "1.5.7",
        "extraction_model": config.extraction_model,
        "embed_model": config.embed_model,
        "embedding_dim": config.embedding_dim,
        "embedding_prefixes": ["search_query: ", "search_document: "],
        "chunk_token_size": config.chunk_token_size,
        "chunk_overlap_token_size": "library_default_not_overridden",
        "entity_guidance_sha256": hashlib.sha256(guidance.encode("utf-8")).hexdigest(),
        "entity_guidance": "project" if config.entity_guidance == PROJECT_ENTITY_GUIDANCE else (
            "library_default" if config.entity_guidance is None else "custom"
        ),
        "num_ctx": config.num_ctx,
        "temperature": config.temperature,
        "entity_extract_max_gleaning": config.gleaning,
        "llm_timeout_seconds": config.llm_timeout,
        "worker_timeout_seconds": config.worker_timeout_seconds,
        "llm_model_max_async": 1,
        "embedding_func_max_async": 1,
        "store_role": config.store_role,
        "vector_only_option": "process_options '!' (skip_kg)" if config.store_role == "vector_only" else "not used",
    }
    if config.omission_segments:
        record["omission_segments"] = "split_on_marker"
    return record


def configuration_fingerprint(config: IndexConfig) -> str:
    import json

    payload = configuration_record(config)
    encoded = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


class StageMeter:
    """Measured call timings for the callables this project passes into LightRAG.

    The split of LLM time into extraction versus merge-summary is estimated
    from prompt text. The totals are wall-clock measurements around the calls.
    """

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.llm_calls = 0
        self.llm_seconds = 0.0
        self.embed_calls = 0
        self.embed_texts = 0
        self.embed_seconds = 0.0
        self.llm_seconds_by_kind: dict[str, float] = {}
        self.llm_calls_by_kind: dict[str, int] = {}

    def record_llm(self, kind: str, seconds: float) -> None:
        self.llm_calls += 1
        self.llm_seconds += seconds
        self.llm_seconds_by_kind[kind] = self.llm_seconds_by_kind.get(kind, 0.0) + seconds
        self.llm_calls_by_kind[kind] = self.llm_calls_by_kind.get(kind, 0) + 1

    def record_embed(self, seconds: float, texts: int) -> None:
        self.embed_calls += 1
        self.embed_texts += texts
        self.embed_seconds += seconds

    def as_dict(self) -> dict:
        return {
            "llm_calls_measured": self.llm_calls,
            "llm_seconds_measured": round(self.llm_seconds, 3),
            "llm_seconds_by_kind_estimated": {
                key: round(value, 3) for key, value in sorted(self.llm_seconds_by_kind.items())
            },
            "llm_calls_by_kind_estimated": dict(sorted(self.llm_calls_by_kind.items())),
            "embed_calls_measured": self.embed_calls,
            "embed_texts_measured": self.embed_texts,
            "embed_seconds_measured": round(self.embed_seconds, 3),
            "timing_note": (
                "LLM and embedding totals are measured around the project callbacks. "
                "The extraction versus merge split is estimated from prompt text."
            ),
        }


def llm_prompt_kind(prompt: str) -> str:
    text = prompt or ""
    if "missed or incorrectly formatted" in text:
        return "extraction_gleaning_estimated"
    lowered = text.lower()
    if "summar" in lowered:
        return "merge_summary_estimated"
    if "---Input Text---" in text or "Knowledge Graph Specialist" in text or "<|#|>" in text:
        return "extraction_estimated"
    return "llm_other_estimated"


def _status_text(row: dict | None) -> str:
    if not row:
        return "absent"
    raw = row.get("status")
    value = getattr(raw, "value", raw)
    return str(value or "").strip().lower()


def _metadata(row: dict | None) -> dict:
    if not row:
        return {}
    meta = row.get("metadata") or {}
    return meta if isinstance(meta, dict) else {}


def _path_has_doc(file_path: object, doc_id: str) -> bool:
    text = str(file_path or "")
    parts = [part.strip() for part in text.split(GRAPH_PATH_SEPARATOR)]
    return doc_id in parts or doc_id == text.strip()


class LightRAGIndexAdapter:
    """One LightRAG instance, one writer, documents indexed one at a time."""

    def __init__(
        self,
        workdir: Path,
        config: IndexConfig,
        *,
        llm_model_func=None,
        embedding_func=None,
        require_models: bool = True,
    ):
        self.workdir = Path(workdir)
        self.config = config
        self.meter = StageMeter()
        self._llm_override = llm_model_func
        self._embed_override = embedding_func
        self._require_models = require_models and llm_model_func is None
        self.rag = None
        self._ready = False
        self._cancel_generation = asyncio.Event()
        self._inflight: set[asyncio.Task] = set()
        self.normalization_counts: dict[str, int] = {}

    async def open(self) -> None:
        if self._ready:
            return
        require_lightrag_package()
        if self._require_models:
            from host_finetune.lightrag_backend import require_ollama_models

            require_ollama_models(
                self.config.base_url,
                self.config.extraction_model,
                self.config.embed_model,
            )
        self.rag = self._build()
        await self.rag.initialize_storages()
        from host_finetune.indexing_extraction_audit import POLICY_VERSION, install_parser_normalizer

        install_parser_normalizer(self.normalization_counts)
        self.normalization_policy = POLICY_VERSION
        self._ready = True

    async def close(self) -> None:
        try:
            await self._drain_generation()
            self._cancel_generation = asyncio.Event()
            if self.rag is not None and self._ready:
                # Flush while the embedding workers are still alive. A cancelled
                # document can leave chunk upserts that finalize still has to embed.
                try:
                    await self.rag.finalize_storages()
                except asyncio.CancelledError:
                    logging.getLogger(__name__).warning("storage finalize was cancelled")
            if self.rag is not None:
                await self._shutdown_background_workers()
        finally:
            from host_finetune.indexing_extraction_audit import uninstall_parser_normalizer

            uninstall_parser_normalizer()
            self._ready = False

    async def _shutdown_background_workers(self) -> None:
        """Stop LightRAG's priority-queue workers before the event loop closes.

        ``finalize_storages`` flushes files and does not join those workers.
        After a cancelled extraction they stay pending, and ``asyncio.run``
        then blocks while cancelling them.
        """
        targets = []
        embedding = getattr(self.rag, "embedding_func", None)
        if embedding is not None:
            targets.append(getattr(embedding, "func", None))
        for state in getattr(self.rag, "_role_llm_states", {}).values():
            targets.append(getattr(state, "wrapped", None))
        for func in targets:
            shutdown = getattr(func, "shutdown", None)
            if not callable(shutdown):
                continue
            try:
                await shutdown(graceful=False, timeout=1)
            except Exception:
                logger = __import__("logging").getLogger(__name__)
                logger.warning("LightRAG worker shutdown failed", exc_info=True)

    def _build(self):
        from lightrag import LightRAG
        from lightrag.llm.ollama import ollama_embed, ollama_model_complete
        from lightrag.utils import wrap_embedding_func_with_attrs

        meter = self.meter
        config = self.config

        if self._llm_override is not None:
            async def llm_model_func(prompt, **kwargs):
                started = time.perf_counter()
                kind = llm_prompt_kind(str(prompt))
                try:
                    return await abortable(
                        self._llm_override(prompt, **kwargs),
                        self._cancel_generation,
                        self._inflight,
                    )
                finally:
                    meter.record_llm(kind, time.perf_counter() - started)
        else:
            async def llm_model_func(prompt, **kwargs):
                started = time.perf_counter()
                kind = llm_prompt_kind(str(prompt))
                try:
                    return await abortable(
                        ollama_model_complete(prompt, **kwargs),
                        self._cancel_generation,
                        self._inflight,
                    )
                finally:
                    meter.record_llm(kind, time.perf_counter() - started)

        if self._embed_override is not None:
            embedding_func = self._embed_override
        else:
            @wrap_embedding_func_with_attrs(
                embedding_dim=config.embedding_dim,
                max_token_size=8192,
                model_name=config.embed_model,
                supports_asymmetric=True,
            )
            async def embedding_func(texts: list[str], **kwargs):
                started = time.perf_counter()
                kwargs.pop("embed_model", None)
                kwargs.pop("host", None)
                kwargs.pop("embedding_dim", None)
                try:
                    return await ollama_embed.func(
                        texts,
                        embed_model=config.embed_model,
                        embedding_dim=config.embedding_dim,
                        host=config.base_url,
                        query_prefix="search_query: ",
                        document_prefix="search_document: ",
                        **kwargs,
                    )
                finally:
                    meter.record_embed(time.perf_counter() - started, len(texts))

        self.workdir.mkdir(parents=True, exist_ok=True)
        addon_params: dict = {"language": "English"}
        if config.entity_guidance:
            addon_params["entity_types_guidance"] = config.entity_guidance
        return LightRAG(
            working_dir=str(self.workdir),
            workspace=config.workspace,
            llm_model_func=llm_model_func,
            llm_model_name=config.extraction_model,
            llm_model_kwargs={
                "host": config.base_url,
                "options": {"num_ctx": config.num_ctx, "temperature": config.temperature},
            },
            embedding_func=embedding_func,
            llm_model_max_async=1,
            embedding_func_max_async=1,
            chunk_token_size=config.chunk_token_size,
            entity_extract_max_gleaning=config.gleaning,
            default_llm_timeout=config.llm_timeout,
            addon_params=addon_params,
        )

    async def graph_counts(self) -> dict:
        nodes = await self.rag.chunk_entity_relation_graph.get_all_nodes()
        edges = await self.rag.chunk_entity_relation_graph.get_all_edges()
        return {"nodes": len(nodes), "edges": len(edges), "source": "measured_after_shutdown_flush"}

    async def inspect(self, doc_id: str, source_text: str) -> dict:
        row = await self.rag.doc_status.get_by_id(doc_id)
        if not row:
            return {
                "present": False,
                "lightrag_status": "absent",
                "skip_kg": False,
                "graph_mutated": False,
                "chunks_count": 0,
                "chunk_ids": [],
                "entity_anchor_count": 0,
                "relation_anchor_count": 0,
                "nodes_with_doc": 0,
                "vector_dim": None,
                "provenance_ok": None,
                "error_msg": "",
                "verified": False,
            }
        meta = _metadata(row)
        status = _status_text(row)
        chunk_ids = [str(item) for item in (row.get("chunks_list") or []) if item]
        entities = await self.rag.full_entities.get_by_id(doc_id)
        relations = await self.rag.full_relations.get_by_id(doc_id)
        entity_names = list((entities or {}).get("entity_names") or [])
        relation_count = int((relations or {}).get("count") or 0)
        nodes_with_doc = 0
        for name in entity_names:
            node = await self.rag.chunk_entity_relation_graph.get_node(name)
            if node and _path_has_doc(node.get("file_path"), doc_id):
                nodes_with_doc += 1
        vector_dim = None
        if chunk_ids:
            vectors = await self.rag.chunks_vdb.get_vectors_by_ids(chunk_ids[:1])
            vector = (vectors or {}).get(chunk_ids[0])
            if vector is not None:
                vector_dim = len(vector)
        provenance = await self._provenance(chunk_ids, source_text) if chunk_ids else None
        mutated = meta.get("kg_write_state") == "graph_mutation_started" or nodes_with_doc > 0
        skip_kg = bool(meta.get("skip_kg")) or "!" in str(meta.get("process_options") or "")
        verified = self._verified(
            status=status,
            skip_kg=skip_kg,
            chunks=len(chunk_ids),
            entity_row_present=entities is not None,
            entity_count=len(entity_names),
            nodes_with_doc=nodes_with_doc,
            vector_dim=vector_dim,
            provenance=provenance,
        )
        return {
            "present": True,
            "lightrag_status": status,
            "skip_kg": skip_kg,
            "graph_mutated": mutated,
            "chunks_count": int(row.get("chunks_count") or len(chunk_ids)),
            "chunk_ids": chunk_ids,
            "entity_anchor_count": len(entity_names),
            "relation_anchor_count": relation_count,
            "nodes_with_doc": nodes_with_doc,
            "vector_dim": vector_dim,
            "provenance_ok": provenance,
            "error_msg": str(row.get("error_msg") or ""),
            "verified": verified,
            "kg_write_state": meta.get("kg_write_state"),
            "process_options": meta.get("process_options"),
        }

    def _verified(
        self,
        *,
        status: str,
        skip_kg: bool,
        chunks: int,
        entity_row_present: bool,
        entity_count: int,
        nodes_with_doc: int,
        vector_dim: int | None,
        provenance: bool | None,
    ) -> bool:
        if status != "processed" or chunks < 1:
            return False
        if vector_dim != self.config.embedding_dim:
            return False
        if provenance is not True:
            return False
        if self.config.store_role == "vector_only":
            return skip_kg and entity_count == 0 and nodes_with_doc == 0
        if skip_kg:
            return False
        if not entity_row_present:
            return False
        if entity_count > 0 and nodes_with_doc < 1:
            return False
        return True

    async def _provenance(self, chunk_ids: list[str], source_text: str) -> bool:
        checked = 0
        for chunk_id in chunk_ids:
            row = await self.rag.text_chunks.get_by_id(chunk_id)
            content = ""
            if isinstance(row, dict):
                content = str(row.get("content") or row.get("text") or "")
            if not content:
                return False
            checked += 1
            if content not in source_text:
                return False
        return checked > 0

    def remember_catalog(self, document: dict) -> None:
        path = self.workdir / "catalog.jsonl"
        catalog = read_catalog(path) if path.is_file() else {}
        incoming = catalog_from_documents([document])
        for doc_id, row in incoming.items():
            previous = catalog.get(doc_id)
            if previous and previous.get("text_sha256") not in {None, row["text_sha256"]}:
                raise RuntimeError(f"refusing to replace catalog text for {doc_id}")
            catalog[doc_id] = row
        if incoming:
            write_catalog(catalog, path)

    async def index_document(
        self,
        document: dict,
        *,
        attempt: int,
        max_attempts: int,
        checkpoint_id: str,
        max_document_seconds: float | None,
        grace_seconds: float,
        cancel_wait_seconds: float,
        merge_wait_seconds: float,
        stop,
    ) -> dict:
        started = time.perf_counter()
        self.meter.reset()
        self._document_timer_cancel = False
        self._cancel_generation = asyncio.Event()
        self._format_warnings = {"malformed_relations": 0, "other_format": 0}
        doc_id = document["graph_document_id"]
        source_text = document.get("text") or ""
        source_hash = hashlib.sha256(source_text.encode("utf-8")).hexdigest()
        recorded_hash = document.get("text_sha256") or source_hash
        if recorded_hash != source_hash:
            return self._finish(
                started,
                checkpoint_id,
                status="quarantined",
                error_stage="source",
                error_category="source_validation",
                error_message="source text hash does not match the document",
                retry_eligible=False,
                persisted_id=doc_id,
            )
        vram_before = sample_vram()
        inspection = await self.inspect(doc_id, source_text)
        plan = self._plan(inspection)
        if plan == "skip_completed":
            return self._success(started, checkpoint_id, doc_id, inspection, note="already persisted; model was not called")
        if plan == "quarantine_unverified":
            return self._finish(
                started,
                checkpoint_id,
                status="quarantined",
                error_stage="persist",
                error_category="persistence_failure",
                error_message="processed record failed the graph or vector integrity check",
                retry_eligible=False,
                persisted_id=doc_id,
                inspection=inspection,
            )
        if plan == "quarantine_residue":
            return self._finish(
                started,
                checkpoint_id,
                status="quarantined",
                error_stage="merge",
                error_category="graph_store_write",
                error_message=(
                    "document has graph residue and was not deleted or reinserted; "
                    + (inspection.get("error_msg") or "manual review required")
                ),
                retry_eligible=False,
                persisted_id=doc_id,
                inspection=inspection,
            )
        handler = FormatWarningCounter(self._format_warnings)
        logging.getLogger("lightrag").addHandler(handler)
        try:
            self.remember_catalog(document)
            if plan == "delete_then_enqueue":
                deleted = await self.rag.adelete_by_doc_id(doc_id)
                if deleted.status not in {"success", "not_found"}:
                    return self._finish(
                        started,
                        checkpoint_id,
                        status="interrupted",
                        error_stage="persist",
                        error_category="process_interruption",
                        error_message=f"delete before retry was not allowed: {deleted.status} {deleted.message}",
                        retry_eligible=True,
                        persisted_id=doc_id,
                        inspection=inspection,
                    )
                await self._enqueue(document, checkpoint_id)
            elif plan == "enqueue":
                await self._enqueue(document, checkpoint_id)
            await self._supervise(
                doc_id,
                max_document_seconds=max_document_seconds,
                grace_seconds=grace_seconds,
                cancel_wait_seconds=cancel_wait_seconds,
                merge_wait_seconds=merge_wait_seconds,
                stop=stop,
            )
        except asyncio.CancelledError:
            inspection = await self.inspect(doc_id, source_text)
            mutated = bool(inspection.get("graph_mutated"))
            defer = "expensive_timeout" if self._document_timer_cancel and not mutated else None
            status = "quarantined" if mutated else "interrupted"
            eligible = not mutated
            if defer and attempt >= max_attempts:
                status = "failed_permanent"
                eligible = False
            return self._finish(
                started,
                checkpoint_id,
                status=status,
                error_stage="interrupt",
                error_category="process_interruption",
                error_message="generation was cancelled before the document task returned",
                retry_eligible=eligible and status == "failed_retryable",
                persisted_id=doc_id,
                inspection=inspection,
                vram_before=vram_before,
                defer_reason=defer,
            )
        except Exception as exc:
            inspection = await self.inspect(doc_id, source_text)
            stage, category = classify_failure(str(exc), exception_name=type(exc).__name__)
            eligible = retry_eligible(category, graph_mutated=bool(inspection.get("graph_mutated")))
            status = terminal_status(category, eligible=eligible, attempt=attempt, max_attempts=max_attempts)
            return self._finish(
                started,
                checkpoint_id,
                status=status,
                error_stage=stage,
                error_category=category,
                error_message=str(exc)[:500],
                retry_eligible=eligible and status == "failed_retryable",
                persisted_id=doc_id,
                inspection=inspection,
                vram_before=vram_before,
            )
        finally:
            logging.getLogger("lightrag").removeHandler(handler)
        inspection = await self.inspect(doc_id, source_text)
        if inspection["verified"]:
            return self._success(started, checkpoint_id, doc_id, inspection, vram_before=vram_before)
        message = inspection.get("error_msg") or f"lightRAG status {inspection['lightrag_status']}"
        if stop is not None and (stop.stop_requested() or stop.cancel_requested()):
            stage, category = "interrupt", "process_interruption"
        else:
            stage, category = classify_failure(message, stage_hint=_stage_hint(inspection))
        if inspection["lightrag_status"] in {"processing", "parsing", "analyzing", "pending"} and category != "llm_extraction_timeout":
            stage, category = "interrupt", "process_interruption"
        mutated = bool(inspection.get("graph_mutated"))
        eligible = retry_eligible(category, graph_mutated=mutated)
        status = terminal_status(category, eligible=eligible, attempt=attempt, max_attempts=max_attempts)
        if mutated and status in {"failed_retryable", "interrupted"}:
            status = "quarantined"
            eligible = False
        defer = "expensive_timeout" if category == "llm_extraction_timeout" else None
        if self._document_timer_cancel and not mutated:
            defer = "expensive_timeout"
            if attempt >= max_attempts:
                status = "failed_permanent"
                eligible = False
        return self._finish(
            started,
            checkpoint_id,
            status=status,
            error_stage=stage,
            error_category=category,
            error_message=message[:500],
            retry_eligible=eligible and status == "failed_retryable",
            persisted_id=doc_id,
            inspection=inspection,
            vram_before=vram_before,
            defer_reason=defer,
        )

    def _plan(self, inspection: dict) -> str:
        if not inspection["present"]:
            return "enqueue"
        status = inspection["lightrag_status"]
        if status == "processed" and inspection["verified"]:
            return "skip_completed"
        if status == "processed":
            return "quarantine_unverified"
        if status in {"pending", "processing", "parsing", "analyzing"}:
            return "process_existing"
        if status == "failed":
            residue = bool(inspection["graph_mutated"]) or int(inspection["entity_anchor_count"]) > 0
            if residue:
                return "quarantine_residue"
            return "delete_then_enqueue"
        return "quarantine_unverified"

    async def _enqueue(self, document: dict, checkpoint_id: str) -> None:
        doc_id = document["graph_document_id"]
        options = "!" if self.config.store_role == "vector_only" else None
        await self.rag.apipeline_enqueue_documents(
            document["text"],
            ids=[doc_id],
            file_paths=[doc_id],
            track_id=checkpoint_id,
            process_options=options,
        )

    async def _supervise(
        self,
        doc_id: str,
        *,
        max_document_seconds: float | None,
        grace_seconds: float,
        cancel_wait_seconds: float,
        merge_wait_seconds: float,
        stop,
    ) -> None:
        parked = await self._park_other_unfinished(doc_id)
        if parked:
            logging.getLogger(__name__).info(
                "held unfinished documents out of this ingest: %s", ", ".join(parked)
            )
        task = asyncio.create_task(self.rag.apipeline_process_enqueue_documents())
        started = time.monotonic()
        grace_deadline = None
        cancel_deadline = None
        extended_for_merge = False
        try:
            while not task.done():
                now = time.monotonic()
                timed_out = max_document_seconds is not None and (now - started) >= max_document_seconds
                if stop is not None and stop.cancel_requested() and cancel_deadline is None:
                    await self._begin_document_cancel(doc_id)
                    cancel_deadline = now + cancel_wait_seconds
                elif stop is not None and stop.stop_requested() and grace_deadline is None and cancel_deadline is None:
                    grace_deadline = now + grace_seconds
                if timed_out and cancel_deadline is None:
                    if not extended_for_merge and await self._mutation_started(doc_id):
                        extended_for_merge = True
                        cancel_deadline = now + merge_wait_seconds
                    else:
                        self._document_timer_cancel = True
                        await self._begin_document_cancel(doc_id)
                        cancel_deadline = now + cancel_wait_seconds
                if grace_deadline is not None and now >= grace_deadline and cancel_deadline is None:
                    if not extended_for_merge and await self._mutation_started(doc_id):
                        extended_for_merge = True
                        grace_deadline = now + merge_wait_seconds
                    else:
                        await self._begin_document_cancel(doc_id)
                        cancel_deadline = now + cancel_wait_seconds
                if cancel_deadline is not None and now >= cancel_deadline and not task.done():
                    if not extended_for_merge and await self._mutation_started(doc_id):
                        extended_for_merge = True
                        cancel_deadline = now + merge_wait_seconds
                    else:
                        if not await self._mutation_started(doc_id):
                            self._cancel_generation.set()
                        task.cancel()
                        break
                await asyncio.sleep(0.05)
            try:
                await task
            except asyncio.CancelledError:
                return
        finally:
            if not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
            await self._drain_generation()

    async def _begin_document_cancel(self, doc_id: str) -> None:
        """Stop an in-flight generation unless this document's graph merge has started."""
        if not await self._mutation_started(doc_id):
            self._cancel_generation.set()
        await self._request_pipeline_cancel()

    async def _drain_generation(self) -> None:
        pending = [task for task in list(self._inflight) if not task.done()]
        if pending:
            await asyncio.wait(pending, timeout=5)

    async def _park_other_unfinished(self, current_id: str) -> list[str]:
        """Stop a later ingest from resuming a different unfinished document.

        LightRAG processes every pending or processing row in one call. A
        document cancelled by the per-document timer is marked failed here
        when its own graph merge has not started. Vectors are not deleted.
        """
        from lightrag.base import DocStatus

        found = await self.rag.doc_status.get_docs_by_statuses([
            DocStatus.PENDING,
            DocStatus.PROCESSING,
            DocStatus.PARSING,
            DocStatus.ANALYZING,
        ])
        updates = {}
        for other_id in found:
            if other_id == current_id:
                continue
            row = await self.rag.doc_status.get_by_id(other_id)
            if not row:
                continue
            data = dict(row)
            if (data.get("metadata") or {}).get("kg_write_state") == "graph_mutation_started":
                continue
            data["status"] = "failed"
            data["error_msg"] = (
                "held out of this ingest so a different document can be indexed; "
                "vectors and graph rows were not deleted"
            )
            if not data.get("file_path"):
                data["file_path"] = other_id
            updates[other_id] = data
        if updates:
            await self.rag.doc_status.upsert(updates)
        return sorted(updates)

    async def _mutation_started(self, doc_id: str) -> bool:
        row = await self.rag.doc_status.get_by_id(doc_id)
        return _metadata(row).get("kg_write_state") == "graph_mutation_started"

    async def _request_pipeline_cancel(self) -> None:
        from lightrag.kg.shared_storage import get_namespace_data, get_namespace_lock

        status = await get_namespace_data("pipeline_status", workspace=self.rag.workspace)
        lock = get_namespace_lock("pipeline_status", workspace=self.rag.workspace)
        async with lock:
            status["cancellation_requested"] = True

    def _success(self, started: float, checkpoint_id: str, doc_id: str, inspection: dict, note: str | None = None, vram_before=None) -> dict:
        vector_only = self.config.store_role == "vector_only"
        quality = note
        if inspection["entity_anchor_count"] == 0 and not vector_only:
            quality = "; ".join(part for part in (quality, "extraction persisted with zero entities") if part)
        return self._finish(
            started,
            checkpoint_id,
            status="completed",
            error_stage=None,
            error_category=None,
            error_message=None,
            retry_eligible=False,
            persisted_id=doc_id,
            inspection=inspection,
            quality_notes=quality,
            vram_before=vram_before,
        )

    def _finish(
        self,
        started: float,
        checkpoint_id: str,
        *,
        status: str,
        error_stage: str | None,
        error_category: str | None,
        error_message: str | None,
        retry_eligible: bool,
        persisted_id: str | None,
        inspection: dict | None = None,
        quality_notes: str | None = None,
        vram_before=None,
        defer_reason: str | None = None,
    ) -> dict:
        inspection = inspection or {}
        vector_only = bool(inspection.get("skip_kg")) or (
            self.config.store_role == "vector_only" and status == "completed"
        )
        graph_extracted = status == "completed" and not vector_only
        if status == "completed" and self.config.store_role == "graph":
            if inspection.get("entity_anchor_count", 0) == 0:
                graph_persistence = "verified_empty"
            elif inspection.get("nodes_with_doc", 0) > 0:
                graph_persistence = "verified"
            else:
                graph_persistence = "verified_empty"
        elif inspection.get("graph_mutated"):
            graph_persistence = "residue"
        else:
            graph_persistence = "not_applied" if vector_only else "not_verified"
        if status == "completed" and inspection.get("vector_dim") == self.config.embedding_dim:
            vector_persistence = "verified"
        elif inspection.get("chunks_count"):
            vector_persistence = "present_unverified"
        else:
            vector_persistence = "absent"
        timings = self.meter.as_dict()
        timings["document_wall_seconds_measured"] = round(time.perf_counter() - started, 3)
        warnings = getattr(self, "_format_warnings", {}).get("malformed_relations", 0)
        if status == "completed" and warnings and not (inspection or {}).get("relation_anchor_count"):
            extra = (
                "zero committed relations after malformed relation warnings; "
                "indexing success is unchanged"
            )
            quality_notes = "; ".join(part for part in (quality_notes, extra) if part)
        vram_after = sample_vram()
        if vram_before or vram_after:
            timings["vram"] = {"before": vram_before, "after": vram_after, "kind": "sampled_not_peak"}
        return {
            "status": status,
            "error_stage": error_stage,
            "error_category": error_category,
            "error_message": error_message,
            "retry_eligible": retry_eligible,
            "checkpoint_id": checkpoint_id,
            "persisted_lightrag_id": persisted_id,
            "indexed_chunk_count": inspection.get("chunks_count"),
            "extraction_metrics": {
                "entity_anchor_count": inspection.get("entity_anchor_count"),
                "relation_anchor_count": inspection.get("relation_anchor_count"),
                "nodes_with_doc": inspection.get("nodes_with_doc"),
                "skip_kg": inspection.get("skip_kg"),
                "llm": {
                    "calls_measured": self.meter.llm_calls,
                    "seconds_measured": round(self.meter.llm_seconds, 3),
                    "by_kind_estimated": timings["llm_seconds_by_kind_estimated"],
                },
                "malformed_relation_warnings_measured": self._format_warnings.get("malformed_relations", 0),
                "other_format_warnings_measured": self._format_warnings.get("other_format", 0),
            },
            "embedding_metrics": {
                "calls_measured": self.meter.embed_calls,
                "texts_measured": self.meter.embed_texts,
                "seconds_measured": round(self.meter.embed_seconds, 3),
                "vector_dim": inspection.get("vector_dim"),
                "expected_dim": self.config.embedding_dim,
            },
            "graph_persistence": graph_persistence,
            "vector_persistence": vector_persistence,
            "provenance_verified": inspection.get("provenance_ok"),
            "vector_only": vector_only and status == "completed",
            "graph_extracted": graph_extracted,
            "stage_timings": timings,
            "runtime_seconds": timings["document_wall_seconds_measured"],
            "quality_notes": quality_notes,
            "defer_reason": defer_reason,
        }


def _stage_hint(inspection: dict) -> str | None:
    message = (inspection.get("error_msg") or "").lower()
    if "merge" in message:
        return "merge"
    if "extract" in message:
        return "extract"
    return None


def sample_vram() -> dict | None:
    """One nvidia-smi sample. This is not a peak-memory measurement."""
    import subprocess

    try:
        output = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.used,memory.total",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            timeout=5,
            stderr=subprocess.DEVNULL,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    line = output.strip().splitlines()[0] if output.strip() else ""
    parts = [part.strip() for part in line.split(",")]
    if len(parts) < 3:
        return None
    return {
        "kind": "sampled_not_peak",
        "name": parts[0],
        "memory_used_mib": _float(parts[1]),
        "memory_total_mib": _float(parts[2]),
    }


def _float(value: str) -> float | None:
    try:
        return float(value)
    except ValueError:
        return None
