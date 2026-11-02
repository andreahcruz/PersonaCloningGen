"""CPU tests for the four-arm plan. No retrieval and no generation."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from host_finetune.model_f_context.budget import BUDGETS
from host_finetune.model_f_experiment.arms import ARMS, experiment_budget, fingerprint
from host_finetune.model_f_experiment.derive import derive_arm
from host_finetune.model_f_experiment.entity_descriptions import attach_raw_entities, entity_rows
from host_finetune.model_f_experiment.evaluate import evaluate_directory
from host_finetune.model_f_experiment.graph_audit import audit_directory, audit_records
from host_finetune.model_f_experiment.runner import (
    cell_record,
    freeze_evidence,
    freeze_retrieval,
    is_complete,
    load_evidence,
    load_questions,
    pending,
    run_pending,
    write_generation,
    write_manifest,
)
from host_finetune.model_f_experiment.snapshot import SnapshotError, copy_snapshot

SOURCE_A = "Company X hired Executive Y after the Series B. The board wanted a sales leader."
SOURCE_B = "Executive Y implemented Strategy Z across the sales team in 2024."
MEDIUM = {
    "medium": "blog",
    "format": "blog_draft",
    "instruction": "Write a blog post in the style of Jason Lemkin about: {topic}",
    "max_new_tokens": 768,
}


def _catalog():
    return {
        "gd_a": {"graph_document_id": "gd_a", "parent_graph_document_id": "gd_a", "text": SOURCE_A, "split": "train", "medium": "blog", "group_id": "g", "attribution_status": "publication_page", "author_position": "not_verified"},
        "gd_b": {"graph_document_id": "gd_b", "parent_graph_document_id": "gd_b", "text": SOURCE_B, "split": "train", "medium": "blog", "group_id": "h", "attribution_status": "publication_page", "author_position": "not_verified"},
    }


def _payload():
    return {
        "status": "success",
        "query": "What strategy followed after Company X hired Executive Y?",
        "metadata": {"query_mode": "hybrid"},
        "data": {
            "chunks": [
                {"content": SOURCE_A, "file_path": "gd_a", "chunk_id": "gd_a-chunk-0"},
                {"content": SOURCE_B, "file_path": "gd_b", "chunk_id": "gd_b-chunk-0"},
            ],
            "entities": [{
                "entity_name": "Company X",
                "entity_type": "organization",
                "description": "Company X is a SaaS company<SEP>A generated gloss that is not a source sentence.",
                "source_id": "gd_a-chunk-0",
                "file_path": "gd_a",
            }],
            "relationships": [
                {"src_id": "Company X", "tgt_id": "Executive Y", "description": "Company X hired Executive Y after the Series B.", "keywords": "hired after", "source_id": "gd_a-chunk-0", "file_path": "gd_a", "weight": 2.0},
                {"src_id": "Executive Y", "tgt_id": "Strategy Z", "description": "Executive Y implemented Strategy Z across the sales team in 2024.", "keywords": "implemented strategy", "source_id": "gd_b-chunk-0", "file_path": "gd_b", "weight": 1.0},
                {"src_id": "Company X", "tgt_id": "Strategy Z", "description": "Company X endorsed Strategy Z as company policy.", "keywords": "endorsed", "source_id": "gd_a-chunk-0<SEP>gd_b-chunk-0", "file_path": "gd_a<SEP>gd_b", "weight": 9.0},
            ],
        },
    }


def _claims():
    return [
        {"claim": "The board wanted a sales leader.", "source_id": "gd_a"},
    ]


def test_four_arms_share_passages_and_isolate_sections():
    payload = _payload()
    plans = {
        arm: derive_arm(payload, payload["query"], arm, _claims(), MEDIUM, catalog=_catalog(), count_tokens=lambda text: len(text.split()) or 1)
        for arm in ARMS
    }
    passages = [plan["passages_retained"] for plan in plans.values()]
    assert passages[0] == passages[1] == passages[2] == passages[3]
    assert any("gd_a" in item for item in passages[0])
    assert any("gd_b" in item for item in passages[0])
    assert "VERIFIED RELATIONSHIPS" not in plans["F-SOURCE"]["instruction"]
    assert "SUPPLEMENTARY GRAPH-DERIVED DESCRIPTIONS" not in plans["F-SOURCE"]["instruction"]
    assert "MULTI-HOP EVIDENCE PATHS" not in plans["F-SOURCE"]["instruction"]
    assert "VERIFIED RELATIONSHIPS" in plans["F-REL"]["instruction"]
    assert "endorsed" not in plans["F-REL"]["instruction"].casefold()
    assert "SUPPLEMENTARY GRAPH-DERIVED DESCRIPTIONS" not in plans["F-REL"]["instruction"]
    assert "SUPPLEMENTARY GRAPH-DERIVED DESCRIPTIONS" in plans["F-DESC"]["instruction"]
    assert "generated gloss" in plans["F-DESC"]["instruction"]
    assert "generated gloss" not in " ".join(plans["F-DESC"]["grounding_texts"]).casefold()
    assert "VERIFIED RELATIONSHIPS" not in plans["F-DESC"]["instruction"]
    assert "MULTI-HOP EVIDENCE PATHS" in plans["F-HOP"]["instruction"]
    assert "one directly supported edge" not in plans["F-HOP"]["instruction"]
    assert "separately supported edges" in plans["F-HOP"]["instruction"]
    assert "SUPPLEMENTARY GRAPH-DERIVED DESCRIPTIONS" not in plans["F-HOP"]["instruction"]
    assert "VERIFIED RELATIONSHIPS" not in plans["F-HOP"]["instruction"]
    for medium, base in BUDGETS.items():
        budget = experiment_budget(medium)
        assert budget.relationships == budget.paths == budget.descriptions
        equal_graph = min(base.relationships, base.paths, base.descriptions)
        assert budget.relationships <= equal_graph
        assert budget.primary <= base.primary
        if medium in {"blog", "talk"}:
            assert budget.primary == 557
            assert budget.relationships == 83
    assert plans["F-SOURCE"]["token_source"] == "caller"
    assert plans["F-SOURCE"]["tokens_by_channel"]["primary_source"] > 0
    assert plans["F-SOURCE"]["tokens_by_channel"]["source_supported_relationship"] == 0


def test_generation_payload_keeps_descriptions_after_hit_normalization():
    from host_finetune.model_f_experiment.retrieval import generation_payload

    combined = generation_payload(_payload(), _catalog())
    assert combined["hits"]
    assert all(hit.get("provenance") != "entity_description" for hit in combined["hits"])
    assert "SaaS company" in combined["raw_entities"][0]["description"]
    assert "SaaS company" in combined["data"]["entities"][0]["description"]


def test_descriptions_are_not_source_hits():
    raw = _payload()
    normalized = {"hits": [{"id": "gd_a", "text": SOURCE_A, "provenance": "source_chunk"}]}
    attached = attach_raw_entities(normalized, raw)
    assert attached["raw_entities"][0]["description"].startswith("Company X is a SaaS company")
    assert attached["hits"] == normalized["hits"]
    assert len(entity_rows(raw)) == 1
    with pytest.raises(ValueError):
        attach_raw_entities({"hits": [{"provenance": "entity_description"}]}, raw)


def test_resume_skips_completed_cells(tmp_path: Path):
    questions = [{"question_id": "q1", "query": "Who?", "phenomenon": "factual"}]
    write_manifest(tmp_path, questions, snapshot="snapshots/finished")
    freeze_retrieval(tmp_path, questions[0], {"data": {"chunks": [], "entities": [], "relationships": []}})
    write_generation(tmp_path, {"question_id": "q1", "arm": "F-SOURCE", "medium": "blog", "answer": "done"})
    assert is_complete(tmp_path / "raw" / "generations" / "q1__F-SOURCE__blog.json")
    first = pending(tmp_path, questions)
    assert all(item["question_id"] != "q1" or not (item["arm"] == "F-SOURCE" and item["medium"] == "blog") for item in first)
    calls = []

    def generate_cell(question, arm, medium, payload):
        calls.append((arm, medium))
        return {"answer": "x", "query": question["query"], "retrieved_source_ids": [], "evidence": [], "latency": {}, "tokens": {}}

    report = run_pending(tmp_path, questions, generate_cell)
    assert ("F-SOURCE", "blog") not in calls
    assert report["skipped_complete"] == 1
    assert report["generated"] == 15
    again = run_pending(tmp_path, questions, generate_cell)
    assert again["generated"] == 0
    assert again["skipped_complete"] == 16


def test_snapshot_refuses_a_lock_and_the_live_store(tmp_path: Path):
    source = tmp_path / "workdir"
    source.mkdir()
    (source / "indexing.lock").write_text("1", encoding="utf-8")
    with pytest.raises(SnapshotError):
        copy_snapshot(source, tmp_path / "copy")
    (source / "indexing.lock").unlink()
    (source / "kv_store_doc_status.json").write_text(json.dumps({"a": {"status": "processing"}}), encoding="utf-8")
    with pytest.raises(SnapshotError):
        copy_snapshot(source, tmp_path / "copy")
    (source / "kv_store_doc_status.json").write_text("{}", encoding="utf-8")
    (source / "note.txt").write_text("ok", encoding="utf-8")
    report = copy_snapshot(source, tmp_path / "copy")
    assert report["files"] >= 1
    assert (tmp_path / "copy" / "snapshot_hashes.json").is_file()
    live = tmp_path / "storage_expand_v1"
    live.mkdir()
    with pytest.raises(SnapshotError):
        copy_snapshot(live, tmp_path / "elsewhere")


def test_verify_checks_dimension_and_hashes(tmp_path: Path):
    from host_finetune.model_f_experiment.snapshot import verify_snapshot

    source = tmp_path / "finished"
    source.mkdir()
    (source / "catalog.jsonl").write_text('{"document_id":"gd_a"}\n', encoding="utf-8")
    (source / "graph_chunk_entity_relation.graphml").write_text("<graphml/>", encoding="utf-8")
    (source / "vdb_entities.json").write_text("{}", encoding="utf-8")
    (source / "vdb_relationships.json").write_text("{}", encoding="utf-8")
    (source / "vdb_chunks.json").write_text(
        json.dumps({"embedding_dim": 768, "data": [{"vector": [0.0] * 768}]}),
        encoding="utf-8",
    )
    destination = tmp_path / "snapshot"
    copy_snapshot(source, destination)
    report = verify_snapshot(destination)
    assert report["embedding_dim"] == 768
    assert report["hash_match"] is True


def test_audit_export_reads_a_copy_and_does_not_recover(tmp_path: Path):
    from host_finetune.model_f_experiment.graph_audit import export_audit_records

    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    (snapshot / "graph_chunk_entity_relation.graphml").write_text(
        """<?xml version="1.0" encoding="UTF-8"?>
<graphml xmlns="http://graphml.graphdrawing.org/xmlns">
  <key id="d0" for="node" attr.name="description" attr.type="string"/>
  <key id="d1" for="edge" attr.name="source_id" attr.type="string"/>
  <graph edgedefault="undirected">
    <node id="You"><data key="d0">A gloss without a source.</data></node>
    <node id="Company X"/>
    <edge source="You" target="Company X"><data key="d1">gd_a</data></edge>
  </graph>
</graphml>
""",
        encoding="utf-8",
    )
    (snapshot / "kv_store_llm_response_cache.json").write_text(
        json.dumps({"a": {"cache_type": "extract", "return": "relationship<|#|>A<|#|>B<|#|>only"}}),
        encoding="utf-8",
    )
    exported = export_audit_records(snapshot)
    report = audit_records(exported)
    kinds = {item["kind"] for item in report["findings"]}
    assert "weak_entity_in_graph" in kinds
    assert "description_missing_provenance" in kinds
    assert report["recovery_applied"] is False
    assert not (snapshot / "graph_audit_export.json").exists()


def test_execute_refuses_the_live_store_before_loading_a_model(tmp_path: Path):
    from host_finetune.model_f_experiment.runner import execute_four_arms

    live = tmp_path / "storage_expand_v1"
    live.mkdir()
    with pytest.raises(SnapshotError):
        execute_four_arms(tmp_path / "out", [], live)


def test_audit_classifies_without_editing(tmp_path: Path):
    report = audit_records({
        "entity_names": ["You", "Company X", "Company  X"],
        "edges": [
            {"source": "Company X", "target": "Executive Y"},
            {"source": "Executive Y", "target": "Company X", "source_ids": ["gd_a-chunk-0"]},
        ],
        "descriptions": [{"name": "Company X", "text": "A generated gloss."}],
        "vector_ids": ["chunk-orphan"],
        "graph_chunk_ids": ["chunk-known"],
        "extraction_lines": ['relationship<|#|>A<|#|>B<|#|>keywords only'],
    })
    kinds = {item["kind"] for item in report["findings"]}
    assert "weak_entity_in_graph" in kinds
    assert "alias_group" in kinds
    assert "edge_missing_provenance" in kinds
    assert "description_missing_provenance" in kinds
    assert "orphan_vector" in kinds
    assert "omitted_description" in kinds or "four_fields_ambiguous" in kinds
    assert report["recovery_applied"] is False
    live = tmp_path / "storage_expand_v1"
    live.mkdir()
    with pytest.raises(SnapshotError):
        audit_directory(live)


def test_cell_record_keeps_the_claims_the_scorer_needs(tmp_path: Path):
    trace = {
        "medium": "blog",
        "answer": "The board wanted a sales leader.",
        "seed": 42,
        "accepted_attempt": 1,
        "max_new_tokens": 768,
        "prompt_tokens": 300,
        "generated_tokens_including_stop": 12,
        "latency_seconds": 2.5,
        "verified_claims": [{"claim": "The board wanted a sales leader.", "source_id": "gd_a"}],
        "retrieval_hits": [{"id": "gd_a", "text": SOURCE_A, "source_id": "gd_a", "split": "train"}],
        "model_f_context": {
            "profile": "source",
            "selected_ids": ["src:gd_a"],
            "grounding_texts": ["The board wanted a sales leader."],
            "tokens_used": 8,
            "tokens_by_channel": {"primary_source": 8},
            "token_estimate": "caller_tokenizer",
            "budget": {"total": 1000},
            "passages_retained": ["gd_a"],
        },
    }
    question = {"question_id": "q1", "query": "Who was hired?", "abstention_expected": False}
    record = cell_record(question, "F-SOURCE", trace)
    assert record["claims"][0]["source_id"] == "gd_a"
    assert record["evidence"][0]["text"] == SOURCE_A
    assert record["grounding_texts"] == ["The board wanted a sales leader."]
    assert record["latency"]["generation_seconds"] == 2.5
    assert record["tokens"]["output"] == 12
    assert record["generation_config"]["accepted_seed"] == 42
    freeze_evidence(tmp_path, "q1", trace)
    assert load_evidence(tmp_path, "q1")["verified"][0]["source_id"] == "gd_a"
    (tmp_path / "raw" / "generations").mkdir(parents=True)
    torn = tmp_path / "raw" / "generations" / "q1__F-SOURCE__blog.json"
    torn.write_text("{", encoding="utf-8")
    assert is_complete(torn) is False


def test_frozen_claims_come_from_evidence_and_are_not_replaced(tmp_path: Path):
    question = {"question_id": "q1", "query": "Who was hired?", "abstention_expected": False}
    trace = {
        "medium": "blog",
        "answer": "A draft that must not become the reference claim.",
        "seed": 42,
        "accepted_attempt": 1,
        "verified_claims": [{"claim": "The board wanted a sales leader.", "source_id": "gd_a"}],
        "retrieval_hits": [{"id": "gd_a", "text": SOURCE_A, "source_id": "gd_a"}],
        "model_f_context": {"grounding_texts": ["The board wanted a sales leader."], "budget": {}},
    }
    freeze_evidence(tmp_path, "q1", trace)
    later = dict(trace)
    later["answer"] = "A different arm draft."
    later["verified_claims"] = [{"claim": "This second extraction must not replace the first.", "source_id": "gd_b"}]
    later["model_f_context"] = {"grounding_texts": ["Company X — hired — Executive Y"], "budget": {}}
    freeze_evidence(tmp_path, "q1", later)
    saved = load_evidence(tmp_path, "q1")
    assert saved["verified"][0]["source_id"] == "gd_a"
    assert "answer" not in saved
    assert "draft" not in json.dumps(saved).casefold()
    first = cell_record(question, "F-SOURCE", trace)
    second = cell_record(question, "F-REL", later)
    assert first["claims"] == [{"claim": "The board wanted a sales leader.", "source_id": "gd_a"}]
    assert first["evidence"][0]["text"] == second["evidence"][0]["text"]
    assert first["grounding_texts"] != second["grounding_texts"]


def test_context_preflight_fits_every_planned_cell():
    from host_finetune.compare_adapters import require_context_room
    from host_finetune.model_f_experiment.context_window import cell_requirement, load_tokenizer, preflight

    questions = load_questions(Path("host_finetune/benchmarks/dev_retrieval_benchmark.jsonl"))
    tokenizer = load_tokenizer()
    report = preflight(questions, tokenizer)
    assert report["cells"] == 576
    assert report["overflowing"] == []
    assert report["max_required"] <= 2048
    longest = max(questions, key=lambda row: len(row["query"]))
    fitted = cell_requirement(tokenizer, longest, "F-REL", "blog")
    previous = fitted["total_required"] - fitted["evidence_allowance"] + 800 + 120
    assert fitted["fits"]
    assert previous > 2048
    with pytest.raises(RuntimeError, match="Evidence was not truncated"):
        require_context_room(1500, 768)


def test_evaluation_does_not_score_shared_retrieval(tmp_path: Path):
    questions = load_questions(Path("host_finetune/benchmarks/dev_retrieval_benchmark.jsonl"))
    assert len(questions) == 36
    assert fingerprint([row["question_id"] for row in questions]) == fingerprint([row["question_id"] for row in questions])
    evidence = [{"id": "gd_a", "text": SOURCE_A, "split": "train", "provenance": "fixture"}]
    for arm, answer in (("F-SOURCE", "The company hired the executive after that round."), ("F-REL", "The company hired the executive after that round.")):
        write_generation(tmp_path, {
            "question_id": "q_brex_price",
            "query": "What price did Capital One announce for acquiring Brex?",
            "arm": arm,
            "medium": "blog",
            "answer": answer,
            "claims": [{"claim": "Company X hired Executive Y after the Series B.", "source_id": "gd_a"}],
            "evidence": evidence,
            "retrieved_source_ids": ["gd_a"],
            "grounding_texts": [],
            "latency": {"generation_seconds": 1.0},
            "tokens": {"prompt": 10, "output": 8, "source": "fixture", "budget_total": 100},
            "generation_config": {"temperature": 0.7, "arm": arm},
        })
    report = evaluate_directory(tmp_path)
    assert report["composite_score"] is None
    assert report["not_annotated"]["retrieval_precision_recall_mrr_ndcg"] == "not_scored_without_human_grades"
    assert report["metrics"]["arms"]["F-SOURCE"]["precision_at_4_mean"] is None
    pair = report["paired_versus_source"]["F-REL"]
    assert pair["retrieval_difference"] is None
    assert pair["n"] == 1
