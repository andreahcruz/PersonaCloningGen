"""Normalization is installed in front of the LightRAG 1.5.7 parser."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import lightrag.operate as operate

from host_finetune.indexing_extraction_audit import (
    install_parser_normalizer,
    normalize_extraction_result,
    uninstall_parser_normalizer,
)
from host_finetune.relation_recovery import build_candidates, run_recovery


FIVE = "relation<|#|>HubSpot<|#|>Salesforce<|#|>Partnership<|#|>HubSpot competes with Salesforce for the same buyers."
SIX = (
    "relation<|#|>SaaStr<|#|>Okta<|#|>relationship_keywords<|#|>provides insights<|#|>"
    "The SaaStr community relies on Okta data to understand adoption patterns."
)
SEVEN = (
    "relation<|#|>Salesforce<|#|>Procurement<|#|>relationship_keywords<|#|>automate, discounting<|#|>"
    "relationship_description<|#|>Salesforce automates discounting by marking prices up 20 percent."
)
COMMA = "relation<|#|>SDR<|#|>AE<|#|>Collaboration, SDRs work closely with AEs to learn what great looks like."
AMBIGUOUS = "relation<|#|>Freemium<|#|>ChatGPT<|#|>Product offering, conversational AI chatbot"
EIGHT = (
    "relation<|#|>Interest rates<|#|>ZIRP<|#|>Post-ZIRP<|#|>High interest rates<|#|>Maybe<|#|>"
    "Does the downfall in SaaS multiples correlate with the rise of interest rates?<|#|>Maybe"
)
ENTITY = "entity<|#|>Smith, John<|#|>Person<|#|>A founder named Smith, John who started the company."


def _parse(text: str, chunk_id: str = "gd_demo-chunk-000", file_path: str = "blog.jsonl"):
    counters: dict[str, int] = {}
    install_parser_normalizer(counters)
    try:
        nodes, edges = asyncio.run(operate._process_extraction_result(
            text, chunk_id, 1, file_path,
        ))
    finally:
        uninstall_parser_normalizer()
    return nodes, edges, counters


def test_valid_five_field_record_is_unchanged_and_reaches_the_parser():
    rewritten, counts = normalize_extraction_result(FIVE + "\n<|COMPLETE|>")
    assert rewritten == FIVE + "\n<|COMPLETE|>"
    assert counts["valid_original"] == 1
    _nodes, edges, counters = _parse(FIVE + "\n<|COMPLETE|>")
    assert counters["valid_original"] == 1
    edge = edges[("HubSpot", "Salesforce")][0]
    assert edge["keywords"] == "Partnership"
    assert edge["file_path"] == "blog.jsonl"
    assert edge["source_id"] == "gd_demo-chunk-000"


def test_schema_label_rows_reach_five_field_validation():
    _nodes, edges, counters = _parse(SIX + "\n" + SEVEN + "\n<|COMPLETE|>")
    assert counters["repaired_schema_label"] == 2
    assert edges[("SaaStr", "Okta")][0]["keywords"] == "provides insights"
    assert "relationship_keywords" not in edges[("SaaStr", "Okta")][0]["description"]
    salesforce = edges[("Salesforce", "Procurement")][0]
    assert salesforce["description"].startswith("Salesforce automates")
    assert salesforce["file_path"] == "blog.jsonl"


def test_comma_packed_row_is_repaired_and_ambiguous_rows_stay_rejected():
    text = "\n".join([COMMA, AMBIGUOUS, EIGHT, ENTITY, "<|COMPLETE|>"])
    rewritten, counts = normalize_extraction_result(text)
    assert "Collaboration<|#|>SDRs work closely" in rewritten
    assert AMBIGUOUS in rewritten
    assert EIGHT in rewritten
    assert ENTITY in rewritten
    assert counts["repaired_comma_packed"] == 1
    assert counts["rejected_missing_description"] == 1
    assert counts["rejected_unsupported"] == 1
    _nodes, edges, _counters = _parse(text)
    assert ("SDR", "AE") in edges
    assert ("Freemium", "ChatGPT") not in edges
    assert ("Interest rates", "ZIRP") not in edges
    assert _nodes["Smith, John"][0]["entity_name"] == "Smith, John"


def test_duplicate_gleaning_lines_collapse_to_one_edge_key():
    text = COMMA + "\n" + COMMA + "\n<|COMPLETE|>"
    _nodes, edges, counters = _parse(text)
    assert counters["repaired_comma_packed"] == 2
    assert list(edges) == [("SDR", "AE")]


def test_five_field_description_keeps_punctuation_and_delimiter_like_words():
    line = "relation<|#|>Rates<|#|>ZIRP<|#|>Monetary policy<|#|>Rates fell, then rose, and the text mentions <|> as a sign."
    rewritten, counts = normalize_extraction_result(line)
    assert rewritten == line
    assert counts["valid_original"] == 1


def test_recovery_dry_run_is_idempotent_and_leaves_the_cache_unchanged(tmp_path: Path):
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    chunk = "The SaaStr community relies on Okta data to understand adoption patterns across the market."
    cache = {
        "default:extract:one": {
            "cache_type": "extract",
            "chunk_id": "gd_demo-chunk-000",
            "create_time": 1,
            "return": SIX + "\n" + SIX + "\n<|COMPLETE|>",
        },
        "default:extract:two": {
            "cache_type": "extract",
            "chunk_id": "gd_demo-chunk-000",
            "create_time": 2,
            "return": SIX + "\n<|COMPLETE|>",
        },
    }
    (snapshot / "kv_store_llm_response_cache.json").write_text(json.dumps(cache), encoding="utf-8")
    (snapshot / "kv_store_text_chunks.json").write_text(
        json.dumps({"gd_demo-chunk-000": {"content": chunk}}),
        encoding="utf-8",
    )
    before = (snapshot / "kv_store_llm_response_cache.json").read_bytes()
    output = tmp_path / "recovery"
    first = run_recovery(snapshot, output, apply=False, authorize=False)
    second = run_recovery(snapshot, output, apply=False, authorize=False)
    assert (snapshot / "kv_store_llm_response_cache.json").read_bytes() == before
    assert first["candidates"] == 1
    assert second["idempotent_replay"] is True
    assert first["cache_unchanged"] is True
    row = json.loads((output / "recovery_ledger.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert row["changes_document_status"] is False
    assert row["action"] in {"insert_candidate", "needs_review", "already_in_graph"}


def test_names_without_a_predicate_stay_in_review(tmp_path: Path):
    entries = [{
        "cache_type": "extract",
        "chunk_id": "gd_demo-chunk-000",
        "create_time": 1,
        "return": "relation<|#|>SaaStr<|#|>Okta<|#|>relationship_keywords<|#|>unrelated token<|#|>A sentence that does not repeat the source wording at all today.\n<|COMPLETE|>",
    }]
    rows = build_candidates(entries, {"gd_demo-chunk-000": "SaaStr mentioned Okta once."}, set())
    assert rows[0]["action"] == "needs_review"


def test_apply_refuses_the_live_store_name(tmp_path: Path):
    snapshot = tmp_path / "storage_expand_v1"
    snapshot.mkdir()
    (snapshot / "kv_store_llm_response_cache.json").write_text("{}", encoding="utf-8")
    try:
        run_recovery(snapshot, tmp_path / "out", apply=True, authorize=True)
        raised = False
    except SystemExit as exc:
        raised = "storage_expand_v1" in str(exc)
    assert raised
