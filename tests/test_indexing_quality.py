"""Quality, orphan, and omission checks. These tests do not open the expansion store."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pytest

from host_finetune.indexing_lightrag import IndexConfig, configuration_fingerprint, configuration_record
from host_finetune.indexing_ledger import IndexingLedger
from host_finetune.indexing_quality import (
    FormatWarningCounter,
    classify_graph_quality,
    count_malformed_relation_lines,
    runtime_profile,
)
from host_finetune.indexing_reconcile import reconcile_records
from host_finetune.indexing_schedule import build_schedule
from host_finetune.indexing_segments import split_omission_segments, write_segment_corpus
from host_finetune.lightrag_backend import normalize_lightrag_data


def _ledger_row(doc_id: str, status: str, **extra) -> dict:
    row = {
        "document_id": doc_id,
        "status": status,
        "word_count": extra.pop("word_count", 100),
        "estimated_chunks": extra.pop("estimated_chunks", 1),
        "source_medium": extra.pop("source_medium", "blog"),
        "runtime_seconds": extra.pop("runtime_seconds", None),
        "error_category": extra.pop("error_category", None),
        "defer_reason": extra.pop("defer_reason", None),
        "has_omission": extra.pop("has_omission", 0),
    }
    row.update(extra)
    return row


def test_orphan_failed_vectors_are_covered_by_the_query_filter():
    report = reconcile_records(
        ledger_rows=[_ledger_row("gd_ok", "completed"), _ledger_row("gd_bad", "failed_permanent")],
        doc_status={
            "gd_ok": {"status": "processed", "metadata": {"kg_write_state": "graph_mutation_started"}},
            "gd_bad": {"status": "failed", "metadata": {"kg_write_state": "pre_graph"}, "file_path": "gd_bad"},
        },
        chunk_vectors=[
            {"__id__": "gd_ok-chunk-000", "full_doc_id": "gd_ok", "file_path": "gd_ok"},
            {"__id__": "gd_bad-chunk-000", "full_doc_id": "gd_bad", "file_path": "gd_bad"},
        ],
        text_chunks={"gd_ok-chunk-000": {"full_doc_id": "gd_ok"}},
        full_entities={"gd_ok": {"count": 2, "entity_names": ["SaaStr"]}},
        full_relations={"gd_ok": {"count": 1}},
        graph_file_paths=["gd_ok"],
        graph_node_ids=["SaaStr"],
    )
    assert report["query_filter"]["failed_vectors_not_covered"] == []
    assert report["query_filter"]["failed_vectors_covered"][0]["chunk_id"] == "gd_bad-chunk-000"
    assert report["unknown_or_unowned_vectors"][0]["reason"] == "vector_without_text_chunk"
    assert report["anchor_conflicts"] == []
    assert report["completed_with_relations"][0]["document_id"] == "gd_ok"


def test_failed_chunk_cannot_become_a_hit():
    catalog = {
        "gd_bad": {
            "graph_document_id": "gd_bad",
            "text": "Capital One acquired Brex.",
            "split": "train",
            "medium": "blog",
        }
    }
    result = normalize_lightrag_data(
        {
            "data": {
                "chunks": [{
                    "content": "Capital One acquired Brex.",
                    "file_path": "gd_bad",
                    "chunk_id": "gd_bad-chunk-000",
                }],
                "entities": [{"entity_name": "Brex", "file_path": "gd_bad", "description": "A company."}],
                "relationships": [],
            }
        },
        catalog,
        mode="naive",
        evidence_budget=4,
        failed_document_ids={"gd_bad"},
    )
    assert result["hits"] == []
    assert any(item["reason"] == "failed_extraction_chunk" for item in result["excluded"])
    assert any(item["reason"] == "generated_summary_not_a_source_span" for item in result["excluded"])


def test_entity_only_completion_is_not_an_indexing_failure():
    quality = classify_graph_quality(
        indexing_status="completed",
        entity_count=11,
        relation_count=0,
        entity_names=["SaaStr", "I", "AI"],
        malformed_relations=8,
        source_text="A short blog.",
    )
    assert quality["indexing_success"] is True
    assert quality["is_indexing_failure"] is False
    assert quality["relation_reason"] == "malformed_relations_dropped"
    assert "I" in quality["weak_entity_names"]
    assert "SaaStr" not in quality["weak_entity_names"]

    legitimate = classify_graph_quality(
        indexing_status="completed",
        entity_count=2,
        relation_count=0,
        entity_names=["SaaStr"],
        malformed_relations=0,
    )
    assert legitimate["relation_reason"] == "no_relations_committed"
    assert legitimate["is_indexing_failure"] is False
    unknown = classify_graph_quality(
        indexing_status="completed",
        entity_count=11,
        relation_count=0,
        malformed_relations=None,
    )
    assert unknown["relation_reason"] == "no_relations_committed_cause_not_recorded"
    assert unknown["is_indexing_failure"] is False


def test_malformed_relation_lines_and_warning_counter():
    text = "\n".join([
        "entity<|#|>SaaStr<|#|>Organization<|#|>A conference",
        "relation<|#|>SaaStr<|#|>AI<|#|>only-four",
        "relation<|#|>SaaStr<|#|>AI<|#|>topic<|#|>A complete relation",
    ])
    assert count_malformed_relation_lines(text) == 1
    sink = {"malformed_relations": 0, "other_format": 0}
    handler = FormatWarningCounter(sink)
    logging.getLogger("lightrag").addHandler(handler)
    try:
        logging.getLogger("lightrag").warning(
            "gd_demo-chunk-000: LLM output format error; found 4/5 fields on RELATION `A`~`B`"
        )
    finally:
        logging.getLogger("lightrag").removeHandler(handler)
    assert sink["malformed_relations"] == 1


def test_omission_segments_keep_provenance_and_stable_ids():
    marked = (
        "Keep the price.\n\n"
        "[Omitted same-source text between these sections: 40 words.]\n\n"
        "Then hire a seller."
    )
    document = {
        "graph_document_id": "gd_parent",
        "parent_graph_document_id": "gd_v1",
        "text": marked,
        "split": "train",
        "medium": "blog",
        "attribution_status": "publication_page",
        "source_file": "blog.jsonl",
        "source_line": 4,
    }
    first = split_omission_segments(document)
    second = split_omission_segments(document)
    assert [row["graph_document_id"] for row in first] == [row["graph_document_id"] for row in second]
    assert all(row["graph_document_id"].startswith("gs_") for row in first)
    assert all("[Omitted same-source text" not in row["text"] for row in first)
    assert first[0]["text"] == "Keep the price."
    assert first[1]["text"] == "Then hire a seller."
    assert first[0]["source_graph_document_id"] == "gd_parent"
    assert first[0]["parent_graph_document_id"] == "gd_v1"
    assert first[0]["text_sha256"]
    assert first[1]["omitted_words_before"] == 40
    assert first[0]["split"] == "train"
    whole = split_omission_segments({**document, "text": "No gap here."})
    assert whole[0]["graph_document_id"] == "gd_parent"
    assert whole[0]["representation_changed"] is False
    with pytest.raises(ValueError):
        split_omission_segments({**document, "split": "validation"})


def test_segment_writer_refuses_the_source_directory(tmp_path: Path):
    source = tmp_path / "graph_ready_v2"
    source.mkdir()
    (source / "documents.jsonl").write_text(
        json.dumps({
            "graph_document_id": "gd_a",
            "text": "A train post.",
            "split": "train",
        }) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(SystemExit):
        write_segment_corpus(source, source)


def test_short_document_can_be_expensive_and_deferred():
    profile = runtime_profile([
        _ledger_row("gd_fast", "completed", runtime_seconds=8.0, word_count=20, source_medium="x"),
        _ledger_row(
            "gd_hung",
            "failed_permanent",
            runtime_seconds=436.0,
            word_count=318,
            estimated_chunks=2,
            error_category="process_interruption",
            defer_reason="expensive_timeout",
        ),
    ])
    assert profile["by_word_count"][0]["documents"] == 1
    assert profile["expensive_or_deferred"][0]["document_id"] == "gd_hung"
    assert profile["expensive_or_deferred"][0]["word_count"] == 318


def test_completed_document_is_not_reselected(tmp_path: Path):
    document = {
        "graph_document_id": "gd_done",
        "text": "SaaStr annual showed sales and marketing together.",
        "text_sha256": "abc",
        "medium": "blog",
        "split": "train",
        "attribution_status": "publication_page",
        "word_count": 8,
        "included": True,
    }
    rows, _ = build_schedule([document], corpus_hash="c", fingerprint="f", indexing_mode="graph")
    rows[0]["status"] = "completed"
    ledger = IndexingLedger(tmp_path / "indexing_ledger.sqlite")
    try:
        ledger.sync_corpus(rows, corpus_hash="c", fingerprint="f")
        assert ledger.next_document(include_deferred=False, retry_timeouts=False) is None
    finally:
        ledger.close()


def test_interrupted_partial_write_is_visible_without_a_graph_anchor():
    report = reconcile_records(
        ledger_rows=[_ledger_row("gd_stop", "interrupted")],
        doc_status={"gd_stop": {"status": "processing", "metadata": {"kg_write_state": "pre_graph"}}},
        chunk_vectors=[{"__id__": "gd_stop-chunk-000", "full_doc_id": "gd_stop", "file_path": "gd_stop"}],
        text_chunks={},
        full_entities={},
        full_relations={},
        graph_file_paths=[],
    )
    assert report["interrupted_or_partial"][0]["document_id"] == "gd_stop"
    assert report["anchor_conflicts"] == []
    assert report["completed_with_relations"] == []


def test_other_unfinished_documents_are_parked_before_ingest():
    import asyncio

    from lightrag.base import DocStatus

    from host_finetune.indexing_lightrag import LightRAGIndexAdapter

    class StatusStore:
        def __init__(self):
            self.rows = {
                "gd_current": {"status": "pending", "file_path": "gd_current", "metadata": {}},
                "gd_other": {
                    "status": "processing",
                    "file_path": "gd_other",
                    "metadata": {"kg_write_state": "pre_graph"},
                },
                "gd_merging": {
                    "status": "processing",
                    "file_path": "gd_merging",
                    "metadata": {"kg_write_state": "graph_mutation_started"},
                },
            }

        async def get_docs_by_statuses(self, statuses):
            wanted = {item.value if isinstance(item, DocStatus) else item for item in statuses}
            return {key: value for key, value in self.rows.items() if value["status"] in wanted}

        async def get_by_id(self, doc_id):
            return dict(self.rows[doc_id])

        async def upsert(self, data):
            self.rows.update(data)

    class Rag:
        def __init__(self):
            self.doc_status = StatusStore()

    adapter = LightRAGIndexAdapter.__new__(LightRAGIndexAdapter)
    adapter.rag = Rag()
    parked = asyncio.run(adapter._park_other_unfinished("gd_current"))
    assert parked == ["gd_other"]
    assert adapter.rag.doc_status.rows["gd_other"]["status"] == "failed"
    assert adapter.rag.doc_status.rows["gd_other"]["file_path"] == "gd_other"
    assert adapter.rag.doc_status.rows["gd_current"]["status"] == "pending"
    assert adapter.rag.doc_status.rows["gd_merging"]["status"] == "processing"


def test_default_configuration_fingerprint_does_not_include_segmentation():
    base = IndexConfig(extraction_model="llama3.1:latest", embed_model="nomic-embed-text", base_url="http://localhost:11434")
    same = IndexConfig(
        extraction_model="llama3.1:latest",
        embed_model="nomic-embed-text",
        base_url="http://localhost:11434",
        omission_segments=False,
    )
    split = IndexConfig(
        extraction_model="llama3.1:latest",
        embed_model="nomic-embed-text",
        base_url="http://localhost:11434",
        omission_segments=True,
    )
    assert configuration_fingerprint(base) == configuration_fingerprint(same)
    assert "omission_segments" not in configuration_record(base)
    assert configuration_fingerprint(split) != configuration_fingerprint(base)
    assert configuration_record(split)["omission_segments"] == "split_on_marker"
