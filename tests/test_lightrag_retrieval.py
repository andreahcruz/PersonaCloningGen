"""LightRAG evidence normalization and Model F selection."""

import json

import pytest

from host_finetune.compare_adapters import build_selected_retriever, parse_args
from host_finetune.evidence_capsule import extraction_user_prompt, freeze_hits
from host_finetune.lightrag_backend import (
    LightRAGUnavailable,
    normalize_lightrag_data,
    read_failed_document_ids,
    require_ollama_models,
)
from host_finetune.lightrag_eval import retrieval_observation, summarize_retrieval
from host_finetune.retrieval_contract import LIGHTRAG_MODES, RetrievalConfigError, require_lightrag_mode


CATALOG = {
    "gd_a": {
        "graph_document_id": "gd_a",
        "text": "Net retention stayed above 100 percent after the price change.",
        "medium": "blog",
        "group_id": "g",
        "split": "train",
        "source_file": "blog.jsonl",
        "source_line": 1,
        "experiment_row_ids": ["train_4"],
        "primary_row_index": 4,
        "attribution_status": "publication_page",
        "author_position": "not_verified",
    },
    "gd_b": {
        "graph_document_id": "gd_b",
        "text": "Hire the second seller only after the first one is productive.",
        "medium": "x",
        "group_id": "h",
        "split": "train",
        "experiment_row_ids": ["train_8"],
        "primary_row_index": 8,
        "attribution_status": "author_account",
        "author_position": "account_or_name_attributed",
    },
}


def _payload():
    return {
        "status": "success",
        "data": {
            "chunks": [
                {"content": "Net retention stayed above 100 percent", "file_path": "gd_a", "chunk_id": "c1"},
                {"content": "after the price change.", "file_path": "gd_a", "chunk_id": "c2"},
                {"content": "Hire the second seller only after the first one is productive.", "file_path": "gd_b", "chunk_id": "c3"},
                {"content": "not in the source", "file_path": "gd_a", "chunk_id": "c4"},
                {"content": "orphan", "file_path": "missing", "chunk_id": "c5"},
            ],
            "entities": [
                {"entity_name": "Net retention", "description": "A generated summary.", "file_path": "gd_a"}
            ],
            "relationships": [],
        },
        "metadata": {"query_mode": "hybrid"},
    }


def test_normalization_keeps_verbatim_chunks_and_drops_graph_summaries():
    result = normalize_lightrag_data(
        _payload(),
        CATALOG,
        mode="hybrid",
        evidence_budget=4,
        max_per_source=1,
    )
    assert [hit["graph_document_id"] for hit in result["hits"]] == ["gd_a", "gd_b"]
    assert [hit["id"] for hit in result["hits"]] == ["gd_a", "gd_b"]
    assert result["hits"][0]["chunk_id"] == "c1"
    assert result["hits"][0]["distance"] is None
    assert result["hits"][0]["score_type"] == "unscored_rank"
    assert result["hits"][0]["split"] == "train"
    assert "generated summary" not in result["hits"][0]["text"].lower()
    reasons = {item["reason"] for item in result["excluded"]}
    assert "source_diversity_cap" in reasons
    assert "chunk_not_in_source" in reasons
    assert "unknown_graph_document" in reasons
    assert "generated_summary_not_a_source_span" in reasons


def test_entity_descriptions_are_retained_outside_hits():
    result = normalize_lightrag_data(
        _payload(),
        CATALOG,
        mode="hybrid",
        evidence_budget=4,
        max_per_source=1,
    )
    assert result["entity_descriptions"] == [{
        "entity_name": "Net retention",
        "entity_type": None,
        "description": "A generated summary.",
        "source_id": None,
        "file_path": "gd_a",
    }]
    assert all("description" not in item for item in result["excluded"])
    assert all(hit.get("provenance") == "source_chunk" for hit in result["hits"])
    assert "generated summary" not in json.dumps(result["hits"]).lower()
    assert "generated summary" not in json.dumps(result["excluded"]).lower()


def test_evidence_id_uses_the_parent_document_when_present():
    catalog = {
        "gd_child": {
            **CATALOG["gd_a"],
            "graph_document_id": "gd_child",
            "parent_graph_document_id": "gd_parent",
            "text": "Anneka Gupta said the product has to be personal.",
        }
    }
    payload = {
        "status": "success",
        "data": {
            "chunks": [{
                "content": "Anneka Gupta said the product has to be personal.",
                "file_path": "gd_child",
                "chunk_id": "chunk-000",
            }],
            "entities": [],
            "relationships": [],
        },
    }
    result = normalize_lightrag_data(
        payload, catalog, mode="naive", evidence_budget=4, max_per_source=1,
    )
    assert result["hits"][0]["id"] == "gd_parent"
    assert result["hits"][0]["chunk_id"] == "chunk-000"
    assert result["hits"][0]["graph_document_id"] == "gd_child"


def test_failed_and_omitted_chunks_are_not_evidence():
    catalog = {
        "gd_failed": {
            **CATALOG["gd_a"],
            "graph_document_id": "gd_failed",
            "text": "Capital One acquired Brex for $5.15 billion.",
        },
        "gd_gap": {
            **CATALOG["gd_b"],
            "graph_document_id": "gd_gap",
            "text": "Keep the price. [Omitted same-source text between these sections: 40 words.] Then hire.",
        },
    }
    payload = {
        "status": "success",
        "data": {
            "chunks": [
                {
                    "content": "Capital One acquired Brex for $5.15 billion.",
                    "file_path": "gd_failed",
                    "chunk_id": "c1",
                },
                {
                    "content": "Keep the price. [Omitted same-source text between these sections: 40 words.] Then hire.",
                    "file_path": "gd_gap",
                    "chunk_id": "c2",
                },
            ],
            "entities": [],
            "relationships": [],
        },
    }
    result = normalize_lightrag_data(
        payload,
        catalog,
        mode="naive",
        evidence_budget=4,
        max_per_source=1,
        failed_document_ids={"gd_failed"},
    )
    assert result["hits"] == []
    reasons = {item["reason"] for item in result["excluded"]}
    assert reasons == {"failed_extraction_chunk", "omission_marker_span"}


def test_failed_ids_come_from_file_path(tmp_path):
    (tmp_path / "kv_store_doc_status.json").write_text(
        json.dumps({"row": {"status": "failed", "file_path": "gd_failed"}}),
        encoding="utf-8",
    )
    (tmp_path / "extraction_skip.json").write_text(
        json.dumps(["gd_skipped"]),
        encoding="utf-8",
    )
    assert read_failed_document_ids(tmp_path) == {"gd_failed", "gd_skipped"}


def test_bypass_mode_is_rejected():
    with pytest.raises(RetrievalConfigError):
        require_lightrag_mode("bypass")
    for mode in LIGHTRAG_MODES:
        require_lightrag_mode(mode)


def test_missing_ollama_model_fails_closed(monkeypatch):
    monkeypatch.setattr(
        "host_finetune.lightrag_backend.ollama_model_names",
        lambda base_url: {"nomic-embed-text:latest"},
    )
    with pytest.raises(LightRAGUnavailable):
        require_ollama_models("http://localhost:11434", "llama3.1:latest", "nomic-embed-text")


def test_dense_prompt_is_unchanged_when_attribution_is_absent():
    hits = [{"id": "train_1", "text": "Keep the price.", "distance": 0.2}]
    frozen = freeze_hits(hits)
    assert "attribution_status" not in frozen[0]
    prompt = extraction_user_prompt("pricing", frozen)
    assert "Attribution status" not in prompt
    assert "[train_1]\nKeep the price." in prompt


def test_attributed_hit_adds_a_label_without_replacing_the_span():
    hits = [{
        "id": "gd_a:c1",
        "text": "Net retention stayed above 100 percent",
        "attribution_status": "publication_page",
        "source_row_ids": ["train_4"],
    }]
    frozen = freeze_hits(hits)
    assert frozen[0]["source_row_ids"] == ["train_4"]
    prompt = extraction_user_prompt("pricing", frozen)
    assert "Attribution status: publication_page." in prompt
    assert "Net retention stayed above 100 percent" in prompt


def test_retrieval_summary_does_not_invent_precision():
    row = retrieval_observation(
        {
            "backend": "lightrag",
            "mode": "naive",
            "query": "retention",
            "hits": [{
                "id": "gd_a:c1",
                "text": "Net retention stayed above 100 percent",
                "split": "train",
                "graph_document_id": "gd_a",
            }],
            "excluded": [],
            "seconds": 1.5,
        }
    )
    summary = summarize_retrieval([row, {**row, "empty": True, "hit_count": 0, "unique_sources": 0, "train_attributed": False, "duplicate_source": False}])
    assert summary["precision_recall"].startswith("not measured")
    assert summary["queries"] == 2
    assert summary["queries_with_train_attributed_hits"] == 1


def test_parse_args_keep_dense_retrieval_off(monkeypatch):
    monkeypatch.setattr(
        "sys.argv",
        ["compare_adapters", "--max-topics", "1"],
    )
    args = parse_args()
    assert args.retrieval is False
    assert args.retrieval_backend == "dense"
    assert args.lightrag_fallback == "error"
    assert args.lightrag_mode == "hybrid"


def test_missing_lightrag_catalog_falls_back_only_when_asked(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "sys.argv",
        ["compare_adapters"],
    )
    args = parse_args()
    args.retrieval_backend = "lightrag"
    args.lightrag_workdir = tmp_path
    args.lightrag_fallback = "error"
    with pytest.raises(SystemExit):
        build_selected_retriever(args)
    seen = {}

    def fake_dense(runtime_args):
        seen["args"] = runtime_args
        return lambda topic, medium: {"backend": "dense", "hits": []}

    monkeypatch.setattr("host_finetune.compare_adapters.build_runtime_retriever", fake_dense)
    args.lightrag_fallback = "dense"
    retriever = build_selected_retriever(args)
    assert retriever("topic", "blog")["backend"] == "dense"
    assert seen["args"] is args


def test_installed_lightrag_modes_match_the_pinned_release():
    pytest.importorskip("lightrag")
    from lightrag import QueryParam
    from lightrag._version import __version__

    assert __version__ == "1.5.7"
    annotation = str(QueryParam.__annotations__["mode"])
    for mode in LIGHTRAG_MODES:
        assert mode in annotation
    assert "bypass" in annotation


def test_frozen_retrieval_round_trip_does_not_call_the_retriever_again(tmp_path):
    from host_finetune.compare_adapters import load_frozen_retriever, write_frozen_retrieval

    calls = []

    def retriever(topic, medium):
        calls.append((topic, medium))
        return {"query": topic, "backend": "lightrag", "mode": "hybrid", "hits": [{"id": "gd_a", "text": "Keep net retention above 100 percent."}]}

    path = tmp_path / "frozen.jsonl"
    write_frozen_retrieval(path, ["pricing"], retriever)
    assert calls == [("pricing", "blog")]
    frozen = load_frozen_retriever(path)
    payload = frozen("pricing", "x")
    assert payload["hits"][0]["id"] == "gd_a"
    assert payload["backend"] == "lightrag"
    with pytest.raises(SystemExit):
        frozen("missing topic", "blog")


def test_partial_catalog_round_trip(tmp_path):
    from host_finetune.lightrag_backend import read_catalog, write_catalog

    write_catalog(CATALOG, tmp_path / "catalog.jsonl")
    loaded = read_catalog(tmp_path / "catalog.jsonl")
    assert loaded["gd_a"]["split"] == "train"
    payload = json.loads((tmp_path / "catalog.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert payload["graph_document_id"]
