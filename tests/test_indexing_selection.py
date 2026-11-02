"""Blog-only scheduling. These tests do not open the expansion store."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from host_finetune.indexing_ledger import IndexingLedger
from host_finetune.indexing_lightrag import IndexConfig, configuration_fingerprint, configuration_record
from host_finetune.indexing_schedule import build_schedule
from host_finetune.indexing_selection import (
    ORIGINAL_MANIFEST_NAME,
    POLICY_VERSION,
    build_blog_manifest,
    export_blog_manifest_readonly,
    identity_sha256,
    write_blog_manifest,
)
from host_finetune.indexing_timeout import ADAPTIVE_V1, select_timeout


def _documents() -> list[dict]:
    rows = []
    for doc_id, medium, words in (
        ("gd_blog", "blog", 220),
        ("gd_blog_done", "blog", 80),
        ("gd_li", "linkedin", 40),
        ("gd_x", "x", 12),
    ):
        text = f"{medium} source text " + ("word " * words)
        rows.append({
            "graph_document_id": doc_id,
            "parent_graph_document_id": f"parent-{doc_id}",
            "text": text.strip(),
            "text_sha256": hashlib.sha256(text.strip().encode("utf-8")).hexdigest(),
            "medium": medium,
            "split": "train",
            "attribution_status": "publication_page",
            "word_count": words,
            "included": True,
            "source_file": f"{medium}.jsonl",
            "source_line": 1,
        })
    return rows


def _ledger(tmp_path: Path) -> IndexingLedger:
    rows, _summary = build_schedule(_documents(), corpus_hash="corpus", fingerprint="finger", indexing_mode="graph")
    for row in rows:
        if row["document_id"] == "gd_blog_done":
            row["status"] = "completed"
    ledger = IndexingLedger(tmp_path / "indexing_ledger.sqlite")
    ledger.sync_corpus(rows, corpus_hash="corpus", fingerprint="finger")
    return ledger


def test_non_train_blog_is_not_eligible(tmp_path: Path):
    documents = _documents()
    documents.append({
        **documents[0],
        "graph_document_id": "gd_blog_val",
        "parent_graph_document_id": "parent-gd_blog_val",
        "split": "validation",
        "text": "validation blog text " + ("word " * 40),
    })
    documents[-1]["text"] = documents[-1]["text"].strip()
    documents[-1]["text_sha256"] = hashlib.sha256(documents[-1]["text"].encode("utf-8")).hexdigest()
    rows, _summary = build_schedule(documents, corpus_hash="corpus", fingerprint="finger", indexing_mode="graph")
    held = next(row for row in rows if row["document_id"] == "gd_blog_val")
    assert held["status"] == "quarantined"
    manifest = build_blog_manifest(rows)
    assert "gd_blog_val" not in {row["document_id"] for row in manifest}
    ledger = _ledger(tmp_path)
    try:
        rows = ledger.iter_documents()
    finally:
        ledger.close()
    first = build_blog_manifest(rows)
    second = build_blog_manifest(rows)
    assert {row["document_id"] for row in first} == {"gd_blog", "gd_blog_done"}
    assert identity_sha256(first) == identity_sha256(second)
    assert all(row["source_medium"] == "blog" for row in first)
    assert all(row["selection_policy_version"] == POLICY_VERSION for row in first)
    assert len({row["document_id"] for row in first}) == len(first)
    pending = [row for row in first if row["scheduled_in_normal_queue"]]
    assert [row["document_id"] for row in pending] == ["gd_blog"]
    done = next(row for row in first if row["document_id"] == "gd_blog_done")
    assert done["indexing_status"] == "completed"
    assert done["scheduled_in_normal_queue"] is False
    assert done["source_hash"]


def test_blog_queue_skips_other_media_and_completed_blogs(tmp_path: Path):
    ledger = _ledger(tmp_path)
    closed = False
    try:
        nxt = ledger.next_document(include_deferred=False, retry_timeouts=False, source_media=("blog",))
        assert nxt["document_id"] == "gd_blog"
        assert nxt["source_medium"] == "blog"
        assert select_timeout(ADAPTIVE_V1, nxt["estimated_chunks"], retry=False, fixed_seconds=420) >= 60
        all_media = ledger.next_document(include_deferred=False, retry_timeouts=False)
        assert all_media["document_id"] in {"gd_blog", "gd_li", "gd_x"}
        ledger.close()
        closed = True
        reopened = IndexingLedger(tmp_path / "indexing_ledger.sqlite")
        try:
            again = reopened.next_document(include_deferred=False, retry_timeouts=False, source_media=("blog",))
            assert again["document_id"] == "gd_blog"
            assert reopened.document("gd_li")["status"] == "pending"
            assert reopened.document("gd_blog_done")["status"] == "completed"
            summary = reopened.selection_summary()
            assert summary["completed_blogs"] == 1
            assert summary["pending_blogs_normal_queue"] == 1
            assert summary["completed_nonblog_seed"] == 0
            assert summary["description"] == "mixed_media_seed_with_blog_only_expansion"
        finally:
            reopened.close()
    finally:
        if not closed:
            ledger.close()


def test_deferred_blog_is_not_in_the_normal_blog_queue(tmp_path: Path):
    ledger = _ledger(tmp_path)
    try:
        rows = ledger.iter_documents()
        with ledger._tx() as connection:
            connection.execute(
                """
                UPDATE documents
                SET status = 'interrupted', defer_reason = 'expensive_timeout'
                WHERE document_id = 'gd_blog'
                """
            )
        del rows
        assert ledger.next_document(include_deferred=False, retry_timeouts=False, source_media=("blog",)) is None
        retried = ledger.next_document(include_deferred=False, retry_timeouts=True, source_media=("blog",))
        assert retried["document_id"] == "gd_blog"
        assert select_timeout(ADAPTIVE_V1, retried["estimated_chunks"], retry=True, fixed_seconds=420) == 420
    finally:
        ledger.close()


def test_empty_blog_queue_and_original_manifest_are_protected(tmp_path: Path):
    only_short = [row for row in _documents() if row["medium"] != "blog"]
    rows, _summary = build_schedule(only_short, corpus_hash="corpus", fingerprint="finger", indexing_mode="graph")
    ledger = IndexingLedger(tmp_path / "indexing_ledger.sqlite")
    try:
        ledger.sync_corpus(rows, corpus_hash="corpus", fingerprint="finger")
        assert ledger.next_document(include_deferred=False, retry_timeouts=False, source_media=("blog",)) is None
        original = tmp_path / ORIGINAL_MANIFEST_NAME
        original.write_text("original\n", encoding="utf-8")
        before = original.read_bytes()
        try:
            write_blog_manifest([], original)
            raised = False
        except SystemExit:
            raised = True
        assert raised
        assert original.read_bytes() == before
    finally:
        ledger.close()


def test_manifest_export_preserves_hashes_and_fingerprint(tmp_path: Path):
    ledger = _ledger(tmp_path)
    ledger.close()
    original = tmp_path / ORIGINAL_MANIFEST_NAME
    original.write_text('{"all":"media"}\n', encoding="utf-8")
    written = export_blog_manifest_readonly(
        tmp_path / "indexing_ledger.sqlite",
        tmp_path / "selection_blog_only_v1.jsonl",
    )
    body = (tmp_path / "selection_blog_only_v1.jsonl").read_text(encoding="utf-8")
    assert hashlib.sha256(body.encode("utf-8")).hexdigest() == written["sha256"]
    assert original.read_text(encoding="utf-8") == '{"all":"media"}\n'
    loaded = [json.loads(line) for line in body.splitlines()]
    assert {row["document_id"] for row in loaded} == {"gd_blog", "gd_blog_done"}
    config = IndexConfig(
        extraction_model="llama3.1:latest",
        embed_model="nomic-embed-text",
        base_url="http://localhost:11434",
    )
    assert "selection_policy" not in configuration_record(config)
    assert configuration_fingerprint(config) == configuration_fingerprint(config)
