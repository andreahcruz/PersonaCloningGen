"""Checkpoint and recovery tests for the LightRAG indexing ledger.

The session tests use a fake model. One test drives installed LightRAG 1.5.7
with a stub LLM and stub embeddings in a temporary directory. It does not
open the pilot store or call Ollama.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from host_finetune.indexing_classify import classify_failure, terminal_status
from host_finetune.indexing_ledger import IndexingLedger
from host_finetune.indexing_lightrag import IndexConfig, LightRAGIndexAdapter
from host_finetune.indexing_schedule import HISTORICAL_PILOT_IDS, build_schedule, validation_batch
from host_finetune.indexing_session import (
    SessionLimits,
    StopController,
    assert_workdir_allowed,
    run_indexing_session,
)
from host_finetune.lightrag_backend import ROOT


def _text_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _doc(doc_id: str, text: str, medium: str, **extra) -> dict:
    row = {
        "graph_document_id": doc_id,
        "text": text,
        "text_sha256": _text_hash(text),
        "medium": medium,
        "word_count": len(text.split()),
        "included": True,
        "split": "train",
        "attribution_status": extra.pop("attribution_status", "author_account"),
        "exclusion_reasons": extra.pop("exclusion_reasons", []),
        "source_file": f"{medium}.jsonl",
        "source_line": extra.pop("source_line", 1),
        "reconstruction_method": extra.pop("reconstruction_method", "single_chunk"),
        "numeric_surface_count": extra.pop("numeric_surface_count", 0),
        "flags": extra.pop("flags", []),
        "parent_graph_document_id": doc_id,
    }
    row.update(extra)
    return row


def _limits(**kwargs) -> SessionLimits:
    values = dict(
        max_documents=10,
        max_runtime_seconds=None,
        max_attempts=2,
        max_document_seconds=None,
        stop_after_current=False,
        grace_seconds=0.2,
        cancel_wait_seconds=0.2,
        merge_wait_seconds=0.2,
        include_deferred=False,
        retry_timeouts=False,
        allow_ids=None,
    )
    values.update(kwargs)
    return SessionLimits(**values)


class FakeAdapter:
    def __init__(self):
        self.calls: list[str] = []
        self.inspections: dict[str, dict] = {}
        self.fail_ids: dict[str, dict] = {}
        self.closed = False

    async def open(self) -> None:
        return None

    async def close(self) -> None:
        self.closed = True

    async def graph_counts(self) -> dict:
        return {"nodes": len(self.calls), "edges": 0, "source": "fake"}

    async def inspect(self, doc_id: str, source_text: str) -> dict:
        del source_text
        return self.inspections.get(
            doc_id,
            {
                "present": False,
                "verified": False,
                "lightrag_status": "absent",
                "graph_mutated": False,
                "chunks_count": 0,
                "error_msg": "",
                "provenance_ok": None,
                "entity_anchor_count": 0,
                "skip_kg": False,
            },
        )

    def _success(self, started, checkpoint_id, doc_id, inspection, note=None, vram_before=None):
        del started, vram_before
        return {
            "status": "completed",
            "error_stage": None,
            "error_category": None,
            "error_message": None,
            "retry_eligible": False,
            "checkpoint_id": checkpoint_id,
            "persisted_lightrag_id": doc_id,
            "indexed_chunk_count": inspection.get("chunks_count") or 1,
            "extraction_metrics": {},
            "embedding_metrics": {},
            "graph_persistence": "verified",
            "vector_persistence": "verified",
            "provenance_verified": True,
            "vector_only": False,
            "graph_extracted": True,
            "stage_timings": {},
            "runtime_seconds": 0.01,
            "quality_notes": note,
            "defer_reason": None,
        }

    async def index_document(self, document, **kwargs):
        doc_id = document["graph_document_id"]
        self.calls.append(doc_id)
        stop = kwargs["stop"]
        if doc_id in self.fail_ids:
            failure = dict(self.fail_ids[doc_id])
            failure.setdefault("checkpoint_id", kwargs["checkpoint_id"])
            failure.setdefault("runtime_seconds", 0.01)
            failure.setdefault("stage_timings", {})
            failure.setdefault("extraction_metrics", {})
            failure.setdefault("embedding_metrics", {})
            return failure
        if stop is not None and stop.cancel_requested():
            return self._interrupted(doc_id, kwargs["checkpoint_id"])
        return {
            "status": "completed",
            "error_stage": None,
            "error_category": None,
            "error_message": None,
            "retry_eligible": False,
            "checkpoint_id": kwargs["checkpoint_id"],
            "persisted_lightrag_id": doc_id,
            "indexed_chunk_count": 1,
            "extraction_metrics": {"entity_anchor_count": 1},
            "embedding_metrics": {"vector_dim": 768, "seconds_measured": 0.01},
            "graph_persistence": "verified",
            "vector_persistence": "verified",
            "provenance_verified": True,
            "vector_only": False,
            "graph_extracted": True,
            "stage_timings": {"document_wall_seconds_measured": 0.01},
            "runtime_seconds": 0.01,
            "quality_notes": None,
            "defer_reason": None,
        }

    def _interrupted(self, doc_id: str, checkpoint_id: str) -> dict:
        return {
            "status": "interrupted",
            "error_stage": "interrupt",
            "error_category": "process_interruption",
            "error_message": "stop requested",
            "retry_eligible": True,
            "checkpoint_id": checkpoint_id,
            "persisted_lightrag_id": doc_id,
            "indexed_chunk_count": 0,
            "extraction_metrics": {},
            "embedding_metrics": {},
            "graph_persistence": "not_verified",
            "vector_persistence": "absent",
            "provenance_verified": None,
            "vector_only": False,
            "graph_extracted": False,
            "stage_timings": {},
            "runtime_seconds": 0.01,
            "quality_notes": None,
            "defer_reason": None,
        }


def _prepare(tmp_path: Path, documents: list[dict]) -> tuple[IndexingLedger, dict]:
    rows, summary = build_schedule(
        documents,
        corpus_hash="corpus-hash",
        fingerprint="finger-a",
        indexing_mode="graph",
    )
    ledger = IndexingLedger(tmp_path / "indexing_ledger.sqlite")
    ledger.sync_corpus(rows, corpus_hash="corpus-hash", fingerprint="finger-a")
    return ledger, {row["graph_document_id"]: row for row in documents}


def test_classifier_separates_timeout_from_embedding_and_interruption():
    assert classify_failure("Worker timeout for task after 1200s", exception_name="WorkerTimeoutError") == (
        "extract",
        "llm_extraction_timeout",
    )
    assert classify_failure("embedding request timed out")[1] == "embedding_model_failure"
    assert classify_failure("User cancelled during extract")[1] == "process_interruption"
    assert classify_failure("context length exceeded")[1] == "context_window_overflow"
    assert terminal_status("llm_extraction_timeout", eligible=True, attempt=1, max_attempts=2) == "failed_retryable"
    assert terminal_status("source_validation", eligible=False, attempt=1, max_attempts=2) == "quarantined"


def test_schedule_keeps_short_posts_and_defers_pilot_failures():
    historical = next(iter(HISTORICAL_PILOT_IDS))
    documents = [
        _doc("gd_short", "Hire well.", "x", word_count=2),
        _doc(historical, "A blog that timed out in the pilot and should stay deferred.", "blog", word_count=12, attribution_status="publication_page"),
        _doc("gd_li", "The VP of Sales missed quota after the first hire failed. " * 20, "linkedin", attribution_status="named_author"),
        _doc("gd_omit", "Section one.\n\n[Omitted same-source text between these sections: 4 words.]\n\nSection two is still useful.", "blog", attribution_status="publication_page", reconstruction_method="ordered_eligible_chunks"),
        _doc("gd_bad", "not indexed", "blog", included=False, exclusion_reasons=["other_author"]),
        _doc("gd_copy", "Hire well.", "x", word_count=2, source_line=2),
    ]
    rows, summary = build_schedule(documents, corpus_hash="c", fingerprint="f", indexing_mode="graph")
    by_id = {row["document_id"]: row for row in rows}
    short_rows = [row for row in rows if row["source_hash"] == _text_hash("Hire well.")]
    assert any(row["status"] == "pending" and row["selected"] for row in short_rows)
    assert any(row["status"] == "skipped" for row in short_rows)
    assert by_id[historical]["defer_reason"] == "pilot160_not_retried"
    assert not by_id[historical]["selected"]
    assert by_id["gd_bad"]["status"] == "quarantined"
    assert by_id["gd_omit"]["has_omission"] in {1, True}
    skipped = [row for row in rows if row["status"] == "skipped"]
    assert len(skipped) == 1
    assert summary["deferred_pilot"] == 1
    batch = validation_batch(rows)
    assert "gd_li" in batch
    assert historical not in batch


def test_ledger_preserves_attempt_history_and_refuses_config_change(tmp_path: Path):
    documents = [_doc("gd_a", "A substantive hiring note about the first VP of Sales.", "linkedin")]
    ledger, _ = _prepare(tmp_path, documents)
    ledger.mark_processing("gd_a", "ckpt-1")
    ledger.record_outcome(
        "gd_a",
        {
            "status": "failed_retryable",
            "error_stage": "extract",
            "error_category": "llm_extraction_timeout",
            "error_message": "worker timeout 1200",
            "retry_eligible": True,
            "checkpoint_id": "ckpt-1",
            "runtime_seconds": 12,
            "defer_reason": "expensive_timeout",
            "graph_persistence": "not_verified",
            "vector_persistence": "present_unverified",
            "provenance_verified": None,
            "vector_only": False,
            "graph_extracted": False,
            "stage_timings": {"llm_seconds_measured": 12},
            "extraction_metrics": {},
            "embedding_metrics": {},
        },
    )
    assert ledger.next_document(include_deferred=False, retry_timeouts=False) is None
    assert ledger.next_document(include_deferred=True, retry_timeouts=True)["document_id"] == "gd_a"
    ledger.mark_processing("gd_a", "ckpt-2")
    ledger.record_outcome(
        "gd_a",
        {
            "status": "completed",
            "error_stage": None,
            "error_category": None,
            "error_message": None,
            "retry_eligible": False,
            "checkpoint_id": "ckpt-2",
            "runtime_seconds": 3,
            "graph_persistence": "verified",
            "vector_persistence": "verified",
            "provenance_verified": True,
            "vector_only": False,
            "graph_extracted": True,
            "indexed_chunk_count": 1,
            "stage_timings": {},
            "extraction_metrics": {"entity_anchor_count": 1},
            "embedding_metrics": {"vector_dim": 768},
        },
    )
    history = ledger.attempts("gd_a")
    assert [item["error_message"] for item in history][0] == "worker timeout 1200"
    assert history[-1]["status"] == "completed"
    with pytest.raises(SystemExit):
        ledger.sync_corpus([], corpus_hash="corpus-hash", fingerprint="finger-b")
    ledger.close()


def test_session_stop_resume_and_export(tmp_path: Path):
    documents = [
        _doc("gd_one", "Hire the second seller only after the first seller produces.", "x"),
        _doc("gd_two", "A LinkedIn note about a VP of Sales who missed the number. " * 5, "linkedin", attribution_status="named_author"),
    ]
    ledger, by_id = _prepare(tmp_path, documents)
    ledger.close()
    adapter = FakeAdapter()
    stop = StopController(tmp_path / "STOP")

    async def first():
        return await run_indexing_session(
            workdir=tmp_path,
            graph_ready=tmp_path,
            config=_config(),
            limits=_limits(max_documents=5, stop_after_current=True),
            stop=stop,
            adapter=adapter,
            documents_by_id=by_id,
            session_id="sess-test-1",
        )

    summary = asyncio.run(first())
    assert summary["documents_handled"] == 1
    assert adapter.calls == [adapter.calls[0]]
    done = adapter.calls[0]

    async def second():
        return await run_indexing_session(
            workdir=tmp_path,
            graph_ready=tmp_path,
            config=_config(),
            limits=_limits(max_documents=5),
            stop=StopController(),
            adapter=adapter,
            documents_by_id=by_id,
            session_id="sess-test-2",
        )

    asyncio.run(second())
    assert adapter.calls == [done, "gd_one" if done != "gd_one" else "gd_two"]
    assert len(set(adapter.calls)) == 2
    report = tmp_path / "reports" / "indexing_summary.json"
    assert report.is_file()
    assert (tmp_path / "reports" / "document_status.csv").is_file()
    assert (tmp_path / "reports" / "failures.csv").is_file()
    assert (tmp_path / "reports" / "stage_timings.csv").is_file()
    assert (tmp_path / "reports" / "index_manifest.json").is_file()
    assert (tmp_path / "reports" / "quality_audit.md").is_file()


def test_stale_processing_is_reconciled_then_retried(tmp_path: Path):
    documents = [_doc("gd_mid", "Pricing moved from ten dollars to twelve after the renewal.", "blog", attribution_status="publication_page")]
    ledger, by_id = _prepare(tmp_path, documents)
    ledger.mark_processing("gd_mid", "ckpt-crash")
    ledger.close()
    adapter = FakeAdapter()
    summary = asyncio.run(
        run_indexing_session(
            workdir=tmp_path,
            graph_ready=tmp_path,
            config=_config(),
            limits=_limits(max_documents=2),
            stop=StopController(),
            adapter=adapter,
            documents_by_id=by_id,
            session_id="sess-recover",
        )
    )
    assert adapter.calls == ["gd_mid"]
    assert summary["documents_handled"] == 1
    check = IndexingLedger(tmp_path / "indexing_ledger.sqlite")
    try:
        history = check.attempts("gd_mid")
        assert any(item["status"] == "reconciled" for item in history)
        assert check.document("gd_mid")["status"] == "completed"
    finally:
        check.close()


def test_timeout_is_not_retried_in_the_same_default_queue(tmp_path: Path):
    documents = [
        _doc("gd_slow", "A moderate blog about retention and pricing for the team.", "blog", attribution_status="publication_page"),
        _doc("gd_next", "An X post about hiring a second seller this quarter.", "x"),
    ]
    ledger, by_id = _prepare(tmp_path, documents)
    ledger.close()
    adapter = FakeAdapter()
    adapter.fail_ids["gd_slow"] = {
        "status": "failed_retryable",
        "error_stage": "extract",
        "error_category": "llm_extraction_timeout",
        "error_message": "Worker timeout after 1200s",
        "retry_eligible": True,
        "defer_reason": "expensive_timeout",
        "graph_persistence": "not_verified",
        "vector_persistence": "present_unverified",
        "provenance_verified": None,
        "vector_only": False,
        "graph_extracted": False,
        "indexed_chunk_count": 1,
        "persisted_lightrag_id": "gd_slow",
    }
    # The scheduler may pick either id first. Force the slow one by allow list order
    # is not rank order, so run the whole queue and assert the slow id is called once.
    asyncio.run(
        run_indexing_session(
            workdir=tmp_path,
            graph_ready=tmp_path,
            config=_config(),
            limits=_limits(max_documents=5),
            stop=StopController(),
            adapter=adapter,
            documents_by_id=by_id,
            session_id="sess-timeout",
        )
    )
    assert adapter.calls.count("gd_slow") == 1
    assert "gd_next" in adapter.calls


def test_document_timer_interrupt_is_not_the_next_document(tmp_path: Path):
    documents = [
        _doc("gd_hung", "A blog about fundraising and retention that stalls.", "blog", attribution_status="publication_page"),
        _doc("gd_next", "An X post about hiring a second seller this quarter.", "x"),
    ]
    ledger, by_id = _prepare(tmp_path, documents)
    ledger.close()
    adapter = FakeAdapter()
    adapter.fail_ids["gd_hung"] = {
        "status": "interrupted",
        "error_stage": "interrupt",
        "error_category": "process_interruption",
        "error_message": "document time limit",
        "retry_eligible": True,
        "defer_reason": "expensive_timeout",
        "graph_persistence": "not_verified",
        "vector_persistence": "absent",
        "provenance_verified": None,
        "vector_only": False,
        "graph_extracted": False,
        "indexed_chunk_count": 0,
        "persisted_lightrag_id": "gd_hung",
    }
    asyncio.run(
        run_indexing_session(
            workdir=tmp_path,
            graph_ready=tmp_path,
            config=_config(),
            limits=_limits(max_documents=5),
            stop=StopController(),
            adapter=adapter,
            documents_by_id=by_id,
            session_id="sess-timer",
        )
    )
    assert adapter.calls.count("gd_hung") == 1
    assert "gd_next" in adapter.calls
    stored = IndexingLedger(tmp_path / "indexing_ledger.sqlite")
    try:
        hung = stored.document("gd_hung")
        assert hung["status"] == "interrupted"
        assert hung["defer_reason"] == "expensive_timeout"
        skipped = [
            row["document_id"]
            for row in stored.peek_documents(10, include_deferred=False, retry_timeouts=False)
        ]
        retried = [
            row["document_id"]
            for row in stored.peek_documents(10, include_deferred=False, retry_timeouts=True)
        ]
    finally:
        stored.close()
    assert "gd_hung" not in skipped
    assert "gd_hung" in retried


def test_protected_pilot_workdir_is_refused():
    with pytest.raises(SystemExit):
        assert_workdir_allowed(ROOT / "artifacts" / "lightrag" / "model_f" / "storage_pilot160")


def _config(**kwargs) -> IndexConfig:
    values = dict(
        extraction_model="stub-llm",
        embed_model="stub-nomic",
        base_url="http://127.0.0.1:9",
        workspace="unit",
    )
    values.update(kwargs)
    return IndexConfig(**values)


def resume_graph_store(workdir: str) -> None:
    """Resume a store in a fresh process.

    A cancelled LightRAG pipeline can leave this process's shared locks
    unusable. A new process is how a later ``--resume`` command actually
    starts, so the recovery check uses one.
    """
    import json

    root = Path(workdir)
    payload = json.loads((root / "resume_inputs.json").read_text(encoding="utf-8"))
    documents = payload["documents"]
    by_id = {row["graph_document_id"]: row for row in documents}

    async def llm(prompt, **kwargs):
        del kwargs
        text = str(prompt)
        if "missed or incorrectly formatted" in text:
            return "<|COMPLETE|>"
        if "summar" in text.lower():
            return "A short summary."
        if "Acme" in text:
            return "entity<|#|>Acme<|#|>Organization<|#|>A company in the text\n<|COMPLETE|>"
        return "entity<|#|>Northwind<|#|>Organization<|#|>A fund named in the text\n<|COMPLETE|>"

    from lightrag.utils import wrap_embedding_func_with_attrs

    @wrap_embedding_func_with_attrs(
        embedding_dim=768,
        max_token_size=8192,
        model_name="stub-nomic",
        supports_asymmetric=True,
    )
    async def embed(texts, **kwargs):
        del kwargs
        rows = []
        for text in texts:
            vec = np.zeros(768, dtype=np.float32)
            vec[0] = 1.0
            vec[1] = float(len(text) % 50)
            rows.append(vec)
        return np.vstack(rows)

    config = IndexConfig(
        extraction_model="stub-llm",
        embed_model="stub-nomic",
        base_url="http://127.0.0.1:9",
        workspace="graphcase",
        gleaning=1,
        llm_timeout=30,
    )
    adapter = LightRAGIndexAdapter(
        root,
        config,
        llm_model_func=llm,
        embedding_func=embed,
        require_models=False,
    )
    summary = asyncio.run(
        run_indexing_session(
            workdir=root,
            graph_ready=root,
            config=config,
            limits=_limits(max_documents=2, retry_timeouts=True),
            stop=StopController(),
            adapter=adapter,
            documents_by_id=by_id,
            session_id="sess-live-3",
        )
    )
    if summary["completed_graph_indexed"] != 2:
        raise SystemExit(f"resume completed {summary['completed_graph_indexed']} graph documents")


def test_lightrag_stub_completion_interrupt_and_vector_only(tmp_path: Path):
    pytest.importorskip("lightrag")
    from lightrag.utils import wrap_embedding_func_with_attrs

    graph_dir = tmp_path / "graph"
    vector_dir = tmp_path / "vector"
    graph_dir.mkdir()
    vector_dir.mkdir()
    first = _doc("gd_alpha", "Acme reported revenue growth after hiring a seller.", "blog", attribution_status="publication_page")
    second = _doc("gd_beta", "Northwind raised a fund for SaaS founders.", "linkedin", attribution_status="named_author")
    hang = {"on": False}

    async def llm(prompt, **kwargs):
        del kwargs
        text = str(prompt)
        if hang["on"] and "missed or incorrectly formatted" not in text:
            await asyncio.sleep(5)
        if "missed or incorrectly formatted" in text:
            return "<|COMPLETE|>"
        if "summar" in text.lower():
            return "A short summary."
        if "Acme" in text:
            return "entity<|#|>Acme<|#|>Organization<|#|>A company in the text\n<|COMPLETE|>"
        return "entity<|#|>Northwind<|#|>Organization<|#|>A fund named in the text\n<|COMPLETE|>"

    @wrap_embedding_func_with_attrs(
        embedding_dim=768,
        max_token_size=8192,
        model_name="stub-nomic",
        supports_asymmetric=True,
    )
    async def embed(texts, **kwargs):
        del kwargs
        rows = []
        for text in texts:
            vec = np.zeros(768, dtype=np.float32)
            vec[0] = 1.0
            vec[1] = float(len(text) % 50)
            rows.append(vec)
        return np.vstack(rows)

    by_id = {row["graph_document_id"]: row for row in (first, second)}
    rows, _ = build_schedule([first, second], corpus_hash="c", fingerprint="f", indexing_mode="graph")
    ledger = IndexingLedger(graph_dir / "indexing_ledger.sqlite")
    ledger.sync_corpus(rows, corpus_hash="c", fingerprint="f")
    ledger.close()
    config = IndexConfig(
        extraction_model="stub-llm",
        embed_model="stub-nomic",
        base_url="http://127.0.0.1:9",
        workspace="graphcase",
        gleaning=1,
        llm_timeout=30,
    )
    adapter = LightRAGIndexAdapter(
        graph_dir,
        config,
        llm_model_func=llm,
        embedding_func=embed,
        require_models=False,
    )

    async def once(limits, stop, session_id):
        return await run_indexing_session(
            workdir=graph_dir,
            graph_ready=graph_dir,
            config=config,
            limits=limits,
            stop=stop,
            adapter=adapter,
            documents_by_id=by_id,
            session_id=session_id,
        )

    async def scenario():
        vector_doc = _doc("gd_vec", "Only the chunk vector should be stored for this note.", "x")
        vector_rows, _ = build_schedule(
            [vector_doc], corpus_hash="cv", fingerprint="fv", indexing_mode="vector_only"
        )
        vledger = IndexingLedger(vector_dir / "indexing_ledger.sqlite")
        vledger.sync_corpus(vector_rows, corpus_hash="cv", fingerprint="fv")
        vledger.close()
        vector_config = IndexConfig(
            extraction_model="stub-llm",
            embed_model="stub-nomic",
            base_url="http://127.0.0.1:9",
            workspace="vectorcase",
            store_role="vector_only",
            gleaning=0,
            llm_timeout=30,
        )
        vector_adapter = LightRAGIndexAdapter(
            vector_dir,
            vector_config,
            llm_model_func=llm,
            embedding_func=embed,
            require_models=False,
        )
        vector_summary = await run_indexing_session(
            workdir=vector_dir,
            graph_ready=vector_dir,
            config=vector_config,
            limits=_limits(max_documents=1),
            stop=StopController(),
            adapter=vector_adapter,
            documents_by_id={"gd_vec": vector_doc},
            session_id="sess-vector",
        )
        assert vector_summary["vector_only_documents"] == 1
        assert vector_summary["completed_graph_indexed"] == 0
        assert vector_summary["graph_nodes"] == 0

        first_summary = await once(_limits(max_documents=1), StopController(), "sess-live-1")
        assert first_summary["documents_handled"] == 1
        assert first_summary["completed_graph_indexed"] == 1
        assert first_summary["graph_nodes"] >= 1

        hang["on"] = True
        second_summary = await once(
            _limits(max_document_seconds=0.4, grace_seconds=0.1, cancel_wait_seconds=0.2, max_documents=1),
            StopController(),
            "sess-live-2",
        )
        assert second_summary["documents_handled"] == 1
        check = IndexingLedger(graph_dir / "indexing_ledger.sqlite")
        try:
            statuses = {row["document_id"]: row["status"] for row in check.iter_documents()}
        finally:
            check.close()
        assert "completed" in statuses.values()
        assert any(status != "completed" for status in statuses.values())
        (graph_dir / "resume_inputs.json").write_text(
            json.dumps({"documents": [first, second]}),
            encoding="utf-8",
        )

    asyncio.run(scenario())
    import subprocess
    import sys

    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            "from tests.test_indexing_checkpoint import resume_graph_store; "
            "resume_graph_store(r'''" + str(graph_dir) + "''')",
        ],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        timeout=90,
    )
    if completed.returncode != 0:
        raise AssertionError(completed.stderr[-4000:] or completed.stdout[-4000:])
    final = IndexingLedger(graph_dir / "indexing_ledger.sqlite")
    try:
        rows_now = {row["document_id"]: row for row in final.iter_documents()}
        assert rows_now["gd_alpha"]["status"] == "completed"
        assert rows_now["gd_beta"]["status"] == "completed"
        assert rows_now["gd_alpha"]["persisted_lightrag_id"] == "gd_alpha"
        assert rows_now["gd_beta"]["persisted_lightrag_id"] == "gd_beta"
        assert rows_now["gd_alpha"]["provenance_verified"] in {1, True}
        embed_metrics = rows_now["gd_alpha"]["embedding_metrics"] or {}
        assert embed_metrics.get("vector_dim") == 768
        assert rows_now["gd_alpha"]["graph_extracted"] in {1, True}
        assert rows_now["gd_alpha"]["vector_only"] in {0, False}
    finally:
        final.close()
