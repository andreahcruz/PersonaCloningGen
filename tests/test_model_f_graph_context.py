"""CPU tests for Model F graph context. No index writes and no model calls."""

import inspect
import json
import sys
from pathlib import Path

import pytest

from host_finetune.compare_adapters import MEDIUMS, generate, parse_args
from host_finetune.evidence_capsule import final_evidence_instruction
from host_finetune.factual_reliability import instruction_leak
from host_finetune.lightrag_backend import normalize_lightrag_data
from host_finetune.model_f_context import SCHEMA_VERSION, graph_context_callback, prepare_context
from host_finetune.model_f_context.budget import ChannelBudget, allocate_context, estimate_tokens
from host_finetune.model_f_context.leakage import (
    EVALUATION_CONFIGURATIONS,
    LeakageError,
    LeakagePolicy,
    assert_iterative_selection_corpus,
    assert_same_eligible_corpus,
    generation_observation,
    paired_comparison,
    screen_documents,
    withhold_graph_context,
)
from host_finetune.model_f_context.multihop import select_paths
from host_finetune.model_f_context.normalize import normalize_graph_context
from host_finetune.model_f_context.pipeline import context_media_jobs, render_prepared
from host_finetune.model_f_context.ranking import rank_evidence
from host_finetune.model_f_context.schema import RelationshipEvidence
from host_finetune.numeric_grounding import evaluate_numeric_grounding

ROOT = Path(__file__).resolve().parents[1]
FROZEN_HYBRID = (
    ROOT
    / "experiments"
    / "EXP-20261031-001-graph-relationship-conditioning"
    / "raw"
    / "frozen_hybrid.jsonl"
)
SOURCE_A = "Company X hired Executive Y after the Series B. The board wanted a sales leader."
SOURCE_B = "Executive Y implemented Strategy Z across the sales team in 2024."


def _catalog():
    return {
        "gd_a": {
            "graph_document_id": "gd_a",
            "parent_graph_document_id": "gd_a",
            "text": SOURCE_A,
            "split": "train",
            "medium": "blog",
            "group_id": "group_a",
            "attribution_status": "publication_page",
            "author_position": "not_verified",
        },
        "gd_b": {
            "graph_document_id": "gd_b",
            "parent_graph_document_id": "gd_b",
            "text": SOURCE_B,
            "split": "train",
            "medium": "blog",
            "group_id": "group_b",
            "attribution_status": "publication_page",
            "author_position": "not_verified",
        },
    }


def _edge(subject, predicate, object_, document, *, status="direct", decision="accepted"):
    return RelationshipEvidence(
        relationship_id=f"{subject}<SEP>{object_}",
        subject=subject,
        object=object_,
        predicate=predicate if status == "direct" else None,
        supporting_document_ids=[document] if document else [],
        supporting_chunk_ids=[f"{document}-chunk-0"] if document else [],
        graph_source_ids=[f"{document}-chunk-0"] if document else [],
        verbatim_evidence=f"{subject} {predicate} {object_}." if status == "direct" else None,
        support_status=status,
        validation_decision=decision,
        validation_reason=None if status == "direct" else "unsupported",
    )


def _raw_payload():
    return {
        "status": "success",
        "message": "Query executed successfully",
        "query": "What strategy followed after Company X hired Executive Y?",
        "metadata": {"query_mode": "hybrid", "keywords": {"high_level": ["hiring"], "low_level": ["Company X"]}},
        "data": {
            "chunks": [
                {"content": SOURCE_A, "file_path": "gd_a", "chunk_id": "gd_a-chunk-0"},
                {"content": SOURCE_B, "file_path": "gd_b", "chunk_id": "gd_b-chunk-0"},
                {"content": "not in either source", "file_path": "gd_a", "chunk_id": "gd_a-chunk-9"},
                {"content": "", "file_path": "gd_b", "chunk_id": "missing"},
            ],
            "entities": [
                {
                    "entity_name": "Company X",
                    "entity_type": "organization",
                    "description": "Company X is a SaaS company<SEP>A generated gloss that is not a source sentence.",
                    "source_id": "gd_a-chunk-0<SEP>gd_missing-chunk-0",
                    "file_path": "gd_a",
                },
                {"entity_name": "", "description": "missing name"},
                {
                    "entity_name": "You",
                    "entity_type": "person",
                    "description": "A generic reader mentioned in passing.",
                    "file_path": "gd_a",
                    "source_id": "gd_a-chunk-0",
                },
            ],
            "relationships": [
                {
                    "src_id": "Company X",
                    "tgt_id": "Executive Y",
                    "description": "Company X hired Executive Y after the Series B.",
                    "keywords": "hired after",
                    "source_id": "gd_a-chunk-0",
                    "file_path": "gd_a",
                    "weight": 2.0,
                },
                {
                    "src_id": "Executive Y",
                    "tgt_id": "Strategy Z",
                    "description": "Executive Y implemented Strategy Z across the sales team in 2024.",
                    "keywords": "implemented strategy",
                    "source_id": "gd_b-chunk-0",
                    "file_path": "gd_b",
                    "weight": 1.0,
                },
                {
                    "src_id": "Company X",
                    "tgt_id": "Strategy Z",
                    "description": "Company X endorsed Strategy Z as company policy.",
                    "keywords": "endorsed",
                    "source_id": "gd_a-chunk-0<SEP>gd_b-chunk-0",
                    "file_path": "gd_a<SEP>gd_b",
                    "weight": 9.0,
                },
                {
                    "src_id": "Company X",
                    "tgt_id": "",
                    "description": "Company X somehow relates.",
                    "keywords": "relates",
                    "file_path": "gd_a",
                },
                {
                    "src_id": "Company X",
                    "tgt_id": "Executive Y",
                    "description": "Company X hired Executive Y after the Series B.",
                    "keywords": "hired after",
                    "source_id": "gd_a-chunk-0",
                    "file_path": "gd_a",
                    "weight": 2.0,
                },
            ],
        },
    }


def test_channels_keep_passages_edges_and_descriptions_apart():
    normalized = normalize_graph_context(_raw_payload(), _catalog())
    bundle_primary = [item.passage for item in normalized["primary"]]
    assert SOURCE_A in bundle_primary
    assert SOURCE_B in bundle_primary
    assert all(item.channel == "primary_source" for item in normalized["primary"])
    assert "endorsed" not in " ".join(bundle_primary)
    hired = next(item for item in normalized["relationships"] if item.object == "Executive Y")
    assert hired.support_status == "direct"
    assert hired.validation_decision == "accepted"
    assert hired.predicate
    assert hired.verbatim_evidence
    assert "gd_a-chunk-0" in hired.supporting_chunk_ids
    assert hired.retrieval_score == 2.0
    assert hired.score_type == "lightrag_weight"
    inferred = next(item for item in normalized["relationships"] if item.object == "Strategy Z" and item.subject == "Company X")
    assert inferred.support_status == "inferred"
    assert inferred.validation_decision == "quarantined"
    assert inferred.predicate is None
    malformed = next(item for item in normalized["relationships"] if item.validation_reason == "malformed_relationship")
    assert malformed.predicate is None
    assert sum(item.relationship_id == hired.relationship_id for item in normalized["relationships"]) == 1
    company = next(item for item in normalized["descriptions"] if item.identity == "Company X")
    assert company.classification == "derived_context"
    assert company.confidence is None
    assert company.validation_status is None
    assert company.entity_type == "organization"
    assert company.clauses[0] == "Company X is a SaaS company"
    assert company.channel == "derived_description"
    assert company.source_reference_status == "resolved"
    encoded = json.dumps(company.to_dict(), sort_keys=True)
    assert json.loads(encoded)["description_id"] == company.description_id


def test_missing_source_ids_and_naive_mode_do_not_invent_facts():
    payload = {
        "status": "success",
        "metadata": {"query_mode": "naive"},
        "data": {
            "chunks": [{"content": SOURCE_A, "file_path": "gd_a", "chunk_id": "c0"}],
            "entities": [],
            "relationships": [{
                "src_id": "Company X",
                "tgt_id": "Executive Y",
                "description": "Company X hired Executive Y after the Series B.",
                "keywords": "hired",
                "file_path": "",
                "source_id": "",
            }],
        },
    }
    normalized = normalize_graph_context(payload, _catalog())
    assert normalized["mode"] == "naive"
    assert normalized["relationships"][0].validation_reason == "missing_source_document"
    assert normalized["relationships"][0].predicate is None
    assert normalized["relationships"][0].support_status == "unsupported"
    stale = normalize_graph_context(
        {
            "status": "success",
            "metadata": {"query_mode": "local"},
            "data": {
                "chunks": [],
                "entities": [{
                    "entity_name": "Company X",
                    "description": "A generated note.",
                    "file_path": "gd_missing",
                    "source_id": "gd_missing-chunk-0",
                }],
                "relationships": [],
            },
        },
        _catalog(),
    )
    assert stale["descriptions"][0].source_reference_status == "unresolved"
    assert stale["descriptions"][0].classification == "derived_context"
    assert stale["primary"] == []


def test_two_hop_path_keeps_both_sources_and_does_not_endorse_the_endpoint():
    prepared = prepare_context(
        _raw_payload(),
        profile="multihop",
        query="What strategy followed after Company X hired Executive Y?",
        catalog=_catalog(),
    )
    paths = prepared["paths"]["paths"]
    chain = next(item for item in paths if item.hops == 2 and item.nodes == ["Company X", "Executive Y", "Strategy Z"])
    assert chain.explicit_endpoint_claim is False
    assert chain.endpoint_claim_note == "path_is_not_a_single_source_claim"
    assert [edge["document_ids"] for edge in chain.edge_provenance] == [["gd_a"], ["gd_b"]]
    assert "hired" in chain.display
    assert "implemented" in chain.display
    assert "endorsed" not in chain.display
    rendered = render_prepared(
        prepared,
        MEDIUMS[0],
        "What strategy followed after Company X hired Executive Y?",
        [{"claim": "Company X hired Executive Y after the Series B.", "source_id": "gd_a", "support_span": SOURCE_A}],
    )
    prompt = rendered["instruction"]
    assert "MULTI-HOP EVIDENCE PATHS" in prompt
    assert "endorsed" not in prompt
    assert "gd_a" not in prompt
    assert "gd_b" not in prompt
    assert SOURCE_A not in prompt
    assert "Company X explicitly endorsed Strategy Z" not in prompt
    assert "PRIMARY SOURCE-GROUNDED CLAIMS" in prompt
    assert "SUPPLEMENTARY GRAPH-DERIVED DESCRIPTIONS" in prompt
    assert all("endorsed" not in text for text in rendered["grounding_texts"])


def test_cycles_duplicate_paths_and_unsupported_edges_are_rejected():
    first = _edge("Company X", "hired", "Executive Y", "gd_a")
    back = _edge("Executive Y", "left", "Company X", "gd_a")
    second = _edge("Executive Y", "implemented", "Strategy Z", "gd_b")
    missing = _edge("Strategy Z", "replaced", "Company X", "gd_b", status="unsupported", decision="rejected")
    result = select_paths([first, back, second, missing, first], max_hops=2, max_paths=24)
    identities = [tuple(item.edge_ids) for item in result["paths"]]
    assert identities.count((first.relationship_id,)) == 1
    assert (first.relationship_id, back.relationship_id) not in identities
    assert all(missing.relationship_id not in item.edge_ids for item in result["paths"])
    assert any(item["reason"] == "cycle" for item in result["rejected"])
    assert any(item["reason"] == "duplicate_path" for item in result["rejected"])
    assert any(item["relationship_id"] == missing.relationship_id for item in result["withheld_edges"])
    one_hop = select_paths([first, second], max_hops=1)
    assert one_hop["paths"]
    assert all(item.hops == 1 for item in one_hop["paths"])
    limited = select_paths([first, second], max_hops=2, max_paths=1)
    assert len(limited["paths"]) == 1
    assert any(item["reason"] == "candidate_limit" for item in limited["rejected"])


def test_ranking_prefers_direct_specific_edges_and_does_not_blend_scores():
    kept = _edge("Brex", "raised prices", "Capital One", "gd_a")
    kept.retrieval_score = 1.0
    kept.score_type = "lightrag_weight"
    heavier = _edge("Brex", "raised prices", "Ramp", "gd_b")
    heavier.retrieval_score = 5.0
    heavier.score_type = "lightrag_weight"
    weak = _edge("Monday", "related to", "You", "gd_a")
    ranked = rank_evidence(
        query="What did Brex do to prices?",
        primary=[],
        relationships=[kept, heavier, weak],
        descriptions=[],
        paths=[],
        profile="relationships",
    )
    selected = [
        unit for unit in ranked["grouped"]["source_supported_relationship"]
        if not unit["exclude_reasons"]
    ]
    assert [unit["record"].object for unit in selected] == ["Ramp", "Capital One"]
    weak_unit = next(unit for unit in ranked["units"] if unit["record"] is weak)
    assert "weak_entity" in weak_unit["exclude_reasons"]
    assert all(unit["score_type"] != "combined_probability" for unit in ranked["units"])
    assert "chroma_distance" not in {unit["score_type"] for unit in selected}


def test_budget_keeps_primary_claims_ahead_of_descriptions():
    primary = {
        "evidence_id": "claim:1",
        "channel": "primary_source",
        "text": "Company X hired Executive Y after the Series B.",
        "exclude_reasons": [],
    }
    description = {
        "evidence_id": "desc:1",
        "channel": "derived_description",
        "text": "A generated description " * 40,
        "exclude_reasons": [],
    }
    huge_claim = {
        "evidence_id": "claim:2",
        "channel": "primary_source",
        "text": "This claim names one hundred days and must stay whole.",
        "exclude_reasons": [],
    }

    def counter(text: str) -> int:
        if "hundred" in text:
            return 500
        if text.startswith("A generated"):
            return 80
        return estimate_tokens(text)

    allocation = allocate_context(
        {
            "primary_source": [primary, huge_claim],
            "source_supported_relationship": [],
            "multi_hop_path": [],
            "derived_description": [description],
        },
        "blog",
        count_tokens=counter,
        budget=ChannelBudget(total=40, primary=30, relationships=0, paths=0, descriptions=30),
    )
    selected_ids = [unit["evidence_id"] for unit in allocation["selected"]["primary_source"]]
    assert selected_ids == ["claim:1"]
    assert "claim:2" not in selected_ids
    assert allocation["selected"]["derived_description"] == []
    reasons = {
        row["evidence_id"]: row["reasons"]
        for row in allocation["audit"]
        if row["decision"] == "excluded"
    }
    assert reasons["claim:2"] == ["budget_item_does_not_fit"]
    assert "budget_item_does_not_fit" in reasons["desc:1"]
    assert "hundred" not in " ".join(unit["text"] for unit in allocation["selected"]["primary_source"])


def test_prompt_separates_channels_and_blocks_marker_and_number_leaks():
    prepared = prepare_context(
        _raw_payload(),
        profile="multihop",
        query="What strategy followed after Company X hired Executive Y?",
        catalog=_catalog(),
    )
    claims = [{
        "claim": "Company X hired Executive Y after the Series B.",
        "source_id": "gd_a",
        "support_span": SOURCE_A,
        "quoted_speakers": [],
    }]
    jobs = context_media_jobs(
        "What strategy followed after Company X hired Executive Y?",
        [{"id": "gd_a", "text": SOURCE_A}, {"id": "gd_b", "text": SOURCE_B}],
        claims,
        "verified",
        prepared,
    )
    assert len(jobs) == 4
    blog = jobs[0]["instruction"]
    post = jobs[2]["instruction"]
    assert "USER WRITING REQUEST" not in blog
    assert blog.startswith("Write a blog post")
    assert "FACTUALITY AND ATTRIBUTION" in blog
    assert "FINAL WRITING INSTRUCTION" in blog
    assert "VERIFIED RELATIONSHIPS" in blog
    assert SOURCE_A not in blog
    assert "gd_a" not in blog
    for description in (
        "Company X is a SaaS company",
        "A generated gloss that is not a source sentence.",
    ):
        if description in blog:
            assert blog.index(description) > blog.index("SUPPLEMENTARY GRAPH-DERIVED DESCRIPTIONS")
    assert estimate_tokens(blog) >= estimate_tokens(post)
    leaked = "The draft copied SUPPLEMENTARY GRAPH-DERIVED DESCRIPTIONS and [D1]."
    assert instruction_leak(leaked) is True
    grounded = evaluate_numeric_grounding(
        "Company X hired Executive Y after the Series B.",
        "What strategy followed after Company X hired Executive Y?",
        claims,
        enabled=True,
        extra_texts=jobs[0]["grounding_texts"],
    )
    assert grounded["pass"] is True
    invented = evaluate_numeric_grounding(
        "The company raised $9.99 billion in 2024.",
        "What strategy followed after Company X hired Executive Y?",
        claims,
        enabled=True,
        extra_texts=jobs[0]["grounding_texts"],
    )
    assert invented["pass"] is False
    assert jobs[0]["model_f_context"]["profile"] == "multihop"
    assert jobs[0]["model_f_context"]["passages_retained"]


def test_passthrough_descriptions_reach_f_desc_and_absent_rows_stay_counted():
    from host_finetune.compare_adapters import MEDIUMS
    from host_finetune.model_f_experiment.derive import derive_arm

    source = "Capital One announced it would acquire Brex."
    query = "What price did Capital One announce for acquiring Brex?"
    description = "Brex is described as a finance company valued at $12.3 billion."
    raw = {
        "status": "success",
        "query": query,
        "metadata": {"query_mode": "hybrid"},
        "data": {
            "chunks": [{"content": source, "file_path": "gd_a", "chunk_id": "c1"}],
            "entities": [{
                "entity_name": "Brex",
                "entity_type": "Organization",
                "description": description,
                "source_id": "chunk-c1",
                "file_path": "gd_a",
            }],
            "relationships": [],
        },
    }
    catalog = {"gd_a": {**_catalog()["gd_a"], "text": source}}
    normalized = normalize_lightrag_data(raw, catalog, mode="hybrid", evidence_budget=4, max_per_source=1)
    project = {
        "query": query,
        "mode": "hybrid",
        "metadata": {"query_mode": "hybrid"},
        "hits": normalized["hits"],
        "excluded": normalized["excluded"],
        "relationship_candidates": normalized["relationship_candidates"],
        "relationships": {
            "contract": "source-verified-relationship-v1",
            "accepted": [],
            "quarantined": [],
            "rejected": [],
        },
        "entity_descriptions": normalized["entity_descriptions"],
    }
    assert "data" not in project
    prepared = prepare_context(project, profile="descriptions", query=query, catalog=catalog)
    assert prepared["bundle"]["diagnostics"]["entity_descriptions_not_retained"] == 0
    entity_descriptions = [
        item for item in prepared["bundle"]["descriptions"] if item["kind"] == "entity"
    ]
    assert entity_descriptions[0]["description"] == description
    assert entity_descriptions[0]["classification"] == "derived_context"
    assert entity_descriptions[0]["entity_type"] == "Organization"
    claims = [{"claim": source, "source_id": "gd_a", "support_span": source}]
    plans = {
        arm: derive_arm(project, query, arm, claims, MEDIUMS[3], catalog=catalog)
        for arm in ("F-SOURCE", "F-REL", "F-DESC", "F-HOP")
    }
    assert "SUPPLEMENTARY GRAPH-DERIVED DESCRIPTIONS" not in plans["F-SOURCE"]["instruction"]
    assert description not in plans["F-SOURCE"]["instruction"]
    assert description not in plans["F-REL"]["instruction"]
    assert "VERIFIED RELATIONSHIPS" not in plans["F-REL"]["instruction"]
    assert description in plans["F-DESC"]["instruction"]
    assert plans["F-DESC"]["instruction"].index(description) > plans["F-DESC"]["instruction"].index(
        "SUPPLEMENTARY GRAPH-DERIVED DESCRIPTIONS"
    )
    assert "Graph context contract: model-f-graph-context-v1" in plans["F-DESC"]["instruction"]
    assert "ALLOWED FACTUAL EVIDENCE" not in plans["F-DESC"]["instruction"]
    assert description not in " ".join(plans["F-DESC"]["grounding_texts"])
    assert "VERIFIED RELATIONSHIPS" not in plans["F-DESC"]["instruction"]
    assert "MULTI-HOP EVIDENCE PATHS" not in plans["F-DESC"]["instruction"]
    assert description not in plans["F-HOP"]["instruction"]
    grounded = evaluate_numeric_grounding(
        "The announcement valued Brex at $12.3 billion.",
        query,
        claims,
        enabled=True,
        extra_texts=plans["F-DESC"]["grounding_texts"],
    )
    assert grounded["pass"] is False
    stripped = {key: value for key, value in project.items() if key != "entity_descriptions"}
    missing = prepare_context(stripped, profile="descriptions", query=query, catalog=catalog)
    assert missing["bundle"]["diagnostics"]["entity_descriptions_not_retained"] == 1
    assert all(item.get("description") != description for item in missing["bundle"]["descriptions"])
    assert "entity_descriptions_were_removed_before_this_payload_and_were_not_invented" in missing["bundle"]["diagnostics"]["notes"]


def test_saved_hybrid_payload_does_not_promote_stripped_entity_descriptions():
    payload = json.loads(FROZEN_HYBRID.open(encoding="utf-8").readline())
    prepared = prepare_context(payload, profile="relationships", query=payload["query"])
    assert prepared["schema_version"] == SCHEMA_VERSION
    assert prepared["bundle"]["mode"] == "hybrid"
    assert len(prepared["bundle"]["primary"]) == 2
    assert prepared["bundle"]["diagnostics"]["entity_descriptions_not_retained"] == 37
    assert all(item["channel"] == "primary_source" for item in prepared["bundle"]["primary"])
    assert all(item["classification"] == "derived_context" for item in prepared["bundle"]["descriptions"])
    assert all(item["confidence"] is None for item in prepared["bundle"]["descriptions"])
    direct = [
        item for item in prepared["bundle"]["relationships"]
        if item["support_status"] == "direct" and item["validation_decision"] == "accepted"
    ]
    assert direct
    assert all(item["predicate"] for item in direct)
    assert "raw_payload" not in prepared["bundle"]
    assert len(prepared["bundle"]["raw_payload_sha256"]) == 64


def test_leakage_screen_blocks_held_out_targets_and_keeps_arms_separate():
    policy = LeakagePolicy(
        blocked_document_ids={"gd_blocked"},
        blocked_groups={"group_val"},
        gold_overlap_groups={"gold_family"},
        blocked_families={"dup_1"},
        target_document_ids={"gd_target"},
    )
    documents = [
        {"graph_document_id": "gd_ok", "split": "train", "group_id": "group_ok"},
        {"graph_document_id": "gd_target", "split": "train", "group_id": "group_ok", "duplicate_family": "dup_1"},
        {"graph_document_id": "gd_val", "split": "validation", "group_id": "group_val"},
        {"graph_document_id": "gd_test", "split": "test", "group_id": "group_test"},
        {"graph_document_id": "gd_gold", "split": "train", "group_id": "gold_family"},
        {"graph_document_id": "gd_blocked", "split": "train", "group_id": "group_ok"},
    ]
    before = json.dumps(documents, sort_keys=True)
    screened = screen_documents(documents, policy)
    assert json.dumps(documents, sort_keys=True) == before
    assert [item["graph_document_id"] for item in screened["kept"]] == ["gd_ok"]
    reasons = {item["id"]: item["reason"] for item in screened["excluded"]}
    assert reasons["gd_target"] == "exact_target"
    assert reasons["gd_val"] == "held_out_split"
    assert reasons["gd_test"] == "held_out_split"
    assert reasons["gd_gold"] == "gold_overlap_family"
    assert reasons["gd_blocked"] == "blocked_document"
    eligible = [documents[0]]
    assert_same_eligible_corpus(eligible, [dict(documents[0])])
    report = assert_iterative_selection_corpus(eligible, purpose="iterative_model_selection")
    assert report["test_set_accessed"] is False
    try:
        assert_iterative_selection_corpus(documents, purpose="iterative_model_selection")
    except LeakageError as exc:
        assert "held-out" in str(exc)
    else:
        raise AssertionError("held-out corpus was accepted")
    comparison = paired_comparison(
        "q_hire",
        {
            "E-DENSE": {"backend": "dense", "hits": [documents[0]], "seconds": 0.2},
            "F-SOURCE": {"backend": "lightrag", "hits": [documents[0], documents[3]], "seconds": 0.4, "draft": "Company X hired Executive Y."},
        },
        policy,
    )
    arms = {item["id"]: item for item in comparison["arms"]}
    assert [item["id"] for item in comparison["arms"]] == [
        "E-DENSE", "F-OFF", "F-SOURCE", "F-REL", "F-DESC", "F-HOP",
    ]
    assert arms["E-DENSE"]["context_profile"] is None
    assert arms["F-OFF"]["context_profile"] == "off"
    assert arms["F-DESC"]["context_profile"] == "descriptions"
    assert arms["F-HOP"]["context_profile"] == "multihop"
    assert arms["E-DENSE"]["retrieval"]["contamination"] is False
    assert arms["E-DENSE"]["generation"] is None
    assert arms["F-SOURCE"]["retrieval"]["contamination"] is True
    assert arms["F-SOURCE"]["retrieval"]["retrieved_source_ids"] == ["gd_ok"]
    assert arms["F-SOURCE"]["generation"]["style"] == "not_scored"
    assert comparison["retrieval_separated_from_generation"] is True
    observation = generation_observation(
        "SUPPLEMENTARY GRAPH-DERIVED DESCRIPTIONS say the price was $9 billion.",
        claims=[{"claim": "Company X hired Executive Y."}],
        hits=[{"id": "gd_ok", "text": "Company X hired Executive Y."}],
    )
    assert observation["marker_copy"] is True
    assert observation["factual_pass"] is False
    withheld = withhold_graph_context([
        {"channel": "source_supported_relationship", "decision": "excluded", "reasons": ["weak_entity"]},
        {"channel": "derived_description", "decision": "excluded", "reasons": ["unrelated_to_request"]},
    ])
    assert "weak_entity" in withheld
    assert [item["id"] for item in EVALUATION_CONFIGURATIONS] == [
        "E-DENSE", "F-OFF", "F-SOURCE", "F-REL", "F-DESC", "F-HOP",
    ]
    assert [item["context_profile"] for item in EVALUATION_CONFIGURATIONS] == [
        None, "off", "source", "relationships", "descriptions", "multihop",
    ]


def test_existing_model_e_and_model_f_defaults_stay_unchanged():
    argv = sys.argv
    sys.argv = ["compare_adapters"]
    try:
        args = parse_args()
    finally:
        sys.argv = argv
    assert args.retrieval_backend == "dense"
    assert args.graph_relationships is False
    assert args.model_f_context == "off"
    assert inspect.signature(generate).parameters["graph_context_for"].default is None
    assert inspect.signature(generate).parameters["relationship_for"].default is None
    assert graph_context_callback("off") is None
    claims = [{"claim": "Keep the price.", "source_id": "train_1", "support_span": "Keep the price."}]
    plain = final_evidence_instruction(MEDIUMS[0], "Hiring", claims)
    assert "ALLOWED FACTUAL EVIDENCE" in plain
    assert "MULTI-HOP EVIDENCE PATHS" not in plain
    assert "SUPPLEMENTARY GRAPH-DERIVED DESCRIPTIONS" not in plain
    payload = {
        "status": "success",
        "data": {
            "chunks": [{"content": "Keep the price.", "file_path": "gd_a", "chunk_id": "c1"}],
            "entities": [{"entity_name": "Price", "description": "A generated summary.", "file_path": "gd_a"}],
            "relationships": [],
        },
    }
    catalog = {"gd_a": {**_catalog()["gd_a"], "text": "Keep the price."}}
    normalized = normalize_lightrag_data(payload, catalog, mode="hybrid", evidence_budget=4, max_per_source=1)
    assert normalized["hits"][0]["text"] == "Keep the price."
    assert "generated summary" not in normalized["hits"][0]["text"].lower()
    assert any(item["reason"] == "generated_summary_not_a_source_span" for item in normalized["excluded"])
    source_only = prepare_context(payload, profile="source", query="Keep the price", catalog=catalog)
    rendered = render_prepared(source_only, MEDIUMS[0], "Keep the price", claims)
    assert "VERIFIED RELATIONSHIPS" not in rendered["instruction"]
    assert "A generated summary" not in rendered["instruction"]


def test_support_span_inside_the_topic_or_another_claim_is_kept():
    topic = "What price did Capital One announce for acquiring Brex?"
    claims = [
        {
            "claim": "The announced price was $5.15 billion.",
            "source_id": "gd_a",
            "support_span": "Capital One",
        },
        {
            "claim": "Capital One agreed to acquire Brex.",
            "source_id": "gd_a",
            "support_span": "The announced price was $5.15 billion.",
        },
    ]
    prepared = prepare_context(_raw_payload(), profile="source", query=topic, catalog=_catalog())
    jobs = context_media_jobs(topic, [], claims, "verified", prepared)
    assert "The announced price was $5.15 billion." in jobs[0]["instruction"]
    assert "Capital One agreed to acquire Brex." in jobs[0]["instruction"]


def test_support_span_outside_the_claims_and_topic_still_raises():
    topic = "What is the generated gloss for Company X?"
    claims = [{
        "claim": "The announced price was $5.15 billion.",
        "source_id": "gd_a",
        "support_span": "A generated gloss that is not a source sentence.",
    }]
    prepared = prepare_context(_raw_payload(), profile="descriptions", query=topic, catalog=_catalog())
    with pytest.raises(ValueError, match="support span leaked into the final prompt"):
        context_media_jobs(topic, [], claims, "verified", prepared)


def test_graph_items_past_the_context_window_are_omitted_whole():
    from host_finetune.model_f_experiment.arms import experiment_budget

    topic = "What is the generated gloss for Company X?"
    claims = [{
        "claim": "The sources do not describe that halftime show.",
        "source_id": "gd_a",
        "support_span": "The sources do not describe that halftime show.",
    }]
    prepared = prepare_context(_raw_payload(), profile="descriptions", query=topic, catalog=_catalog())

    def window_tokens(instruction: str) -> int:
        if "SUPPLEMENTARY GRAPH-DERIVED DESCRIPTIONS" in instruction:
            return 1300
        return 1209

    prepared["window_tokens"] = window_tokens
    before = experiment_budget("talk")
    jobs = context_media_jobs(topic, [], claims, "verified", prepared)
    talk = next(job for job in jobs if job["medium"]["medium"] == "talk")
    assert "SUPPLEMENTARY GRAPH-DERIVED DESCRIPTIONS" not in talk["instruction"]
    assert claims[0]["claim"] in talk["instruction"]
    assert talk["model_f_context"]["window_omitted"]
    assert talk["model_f_context"]["tokens_by_channel"]["primary_source"] > 0
    assert talk["model_f_context"]["tokens_by_channel"]["derived_description"] == 0
    after = experiment_budget("talk")
    assert (after.primary, after.descriptions, after.relationships, after.paths) == (
        before.primary,
        before.descriptions,
        before.relationships,
        before.paths,
    )
    assert (after.primary, after.descriptions) == (557, 83)
