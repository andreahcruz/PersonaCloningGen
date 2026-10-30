"""Quote attribution, residual diagnosis, and the smoke-index guard."""

import json

import pytest

from host_finetune.entity_audit import entity_name_flags, false_employer_descriptions
from host_finetune.graph_ready import lead_and_interior_gaps
from host_finetune.graph_ready_v2 import annotate_document, marked_chunk_text, write_derivative
from host_finetune.lightrag_index import assert_isolated_workdir
from host_finetune.quote_attribution import find_third_party_quotes, quote_attribution

ANNEKA = (
    '"AI means eveything in SaaS needs to be a personalized experience. '
    "A workflow that surfaces the job that needs to get done -- and just does it. "
    'That\'s the Future of Enterprise SaaS" Rubrik CPO Anneka Gupta New -> '
    '"Going Multi-Product in the Age of AI with CPOs of Webflow, Rubrik, Zoom, and Productboard" '
    "https://lnkd.in/gQFeDYeT"
)


def test_anneka_gupta_quote_is_not_lemkin():
    quotes = find_third_party_quotes(ANNEKA)
    assert [item["speaker"] for item in quotes] == ["Anneka Gupta"]
    assert quotes[0]["role"] == "Rubrik CPO"
    assert "Webflow" not in quotes[0]["span"]
    decision = quote_attribution(ANNEKA, "named_author", "account_or_name_attributed")
    assert decision["attribution_status"] == "third_party_quotation"
    assert decision["author_position"] == "quoted_third_party"


def test_title_case_headings_are_not_speakers():
    samples = [
        'No sugarcoating."\n\nJust clarity. Step 3: The 30-Day Plan — Maybe 60.',
        'a warning."\nThe $1 Billion Proof Point and the 100 Million Lead Revolution',
        'your meeting."\nMarket Timing Concerns\nThe concentration of value',
        'don’t make the hire.\nGiving Reps Too Much Time. Sales reps don’t have to scale.',
        '"A huge warning sign." Deep Dives and Product Advocacy followed.',
    ]
    for sample in samples:
        assert find_third_party_quotes(sample) == []


def test_lemkin_quote_is_not_relabelled():
    text = '"Hire slower than you want," Jason Lemkin said after the round.'
    assert find_third_party_quotes(text) == []
    decision = quote_attribution(text, "author_account", "account_or_name_attributed")
    assert decision["attribution_status"] == "author_account"


def test_v2_keeps_source_text_and_changes_id_only_when_wrapped():
    original = {
        "graph_document_id": "gd_parent",
        "text": ANNEKA,
        "attribution_status": "named_author",
        "author_position": "account_or_name_attributed",
        "flags": [],
        "policy_version": "graph-ready-v1",
        "split": "train",
        "included": True,
    }
    derived = annotate_document(original)
    assert derived["source_text"] == ANNEKA
    assert ANNEKA in derived["text"]
    assert derived["text"].startswith("[Attribution note:")
    assert derived["graph_document_id"] != "gd_parent"
    assert derived["parent_graph_document_id"] == "gd_parent"
    assert derived["included"] is True


def test_v2_refuses_to_overwrite_v1(tmp_path):
    with pytest.raises(SystemExit):
        write_derivative(tmp_path, tmp_path)


def test_contiguous_chunks_have_no_interior_gap_and_keep_the_lead_out():
    first = "Section two stays in the eligible chunk."
    second = "Section three stays in the eligible chunk."
    lead = "Author lead. Embedded note from Someone Else (@someone)."
    cleaned = f"{lead} {first} {second}"
    measured = lead_and_interior_gaps(cleaned, [first, second])
    assert measured["interior_gap_words"] == [0]
    assert measured["lead_words"] > 0
    assert "Someone Else" not in first
    assert "Someone Else" not in second


def test_interior_gap_is_measured_when_sections_skip_text():
    first = "Alpha section is present."
    second = "Later section is present."
    cleaned = f"{first} omitted middle words live here {second}"
    measured = lead_and_interior_gaps(cleaned, [first, second])
    assert measured["interior_gap_words"] == [5]
    assert measured["lead_words"] == 0


def test_section_marker_names_the_gap_and_does_not_copy_the_omitted_words():
    marked = marked_chunk_text(
        ["Alpha section is present.", "Later section is present."],
        [5],
    )
    assert marked is not None
    assert "5 words" in marked
    assert "omitted middle" not in marked
    assert marked_chunk_text(["Only one."], []) is None


def test_weak_entity_names_from_the_smoke_graph():
    assert "weekday" in entity_name_flags("Tuesday")
    assert "month" in entity_name_flags("December")
    assert "numeric_name" in entity_name_flags("93%-95% Correct")
    assert "generic_noun" in entity_name_flags("Basic")
    assert entity_name_flags("Salesforce Ventures") == []
    descriptions = ["Anneka Gupta is also the Chief Product Officer at Webflow."]
    assert false_employer_descriptions(descriptions, "Anneka Gupta", "Webflow") == descriptions


def test_smoke_workdir_rejects_a_larger_index():
    from host_finetune.lightrag_backend import DEFAULT_WORKDIR

    with pytest.raises(SystemExit):
        assert_isolated_workdir(DEFAULT_WORKDIR, 160)
    assert_isolated_workdir(DEFAULT_WORKDIR, 3)


def test_chroma_hit_uses_the_pilot_parent_when_the_row_is_in_the_sample():
    from host_finetune.pilot_retrieval import annotate_chroma_hits, row_source_index, sheet_row

    documents = [{
        "graph_document_id": "gd_new",
        "parent_graph_document_id": "gd_parent",
        "experiment_row_ids": ["train_4"],
        "attribution_status": "publication_page",
        "quoted_speakers": [],
        "split": "train",
        "medium": "blog",
    }]
    payload = annotate_chroma_hits(
        "What price did Capital One announce?",
        [{
            "id": "train_4",
            "text": "Capital One announced a price of $5.15 billion.",
            "distance": 0.2,
            "split": "train",
            "medium": "blog",
        }],
        row_source_index(documents),
        0.05,
    )
    row = sheet_row(
        {
            "id": "q_brex_price",
            "kind": "simple_fact",
            "query": "What price did Capital One announce?",
            "review_source_ids": ["gd_parent"],
            "abstain": False,
        },
        payload,
    )
    assert row["hit_source_ids"] == ["gd_parent"]
    assert row["designated_sources_retrieved"] == ["gd_parent"]
    assert row["designated_source_covered"] is True
    assert row["precision_recall"].startswith("not measured")


def test_chroma_hit_outside_the_pilot_keeps_its_train_id():
    from host_finetune.pilot_retrieval import annotate_chroma_hits, sheet_row

    payload = annotate_chroma_hits(
        "Olympic ceremony",
        [{"id": "train_9", "text": "A different source.", "distance": 0.4, "split": "train"}],
        {},
        0.02,
    )
    row = sheet_row(
        {
            "id": "q_olympics_ceremony",
            "kind": "unsupported",
            "query": "Olympic ceremony",
            "review_source_ids": [],
            "abstain": True,
        },
        payload,
    )
    assert row["hit_source_ids"] == ["train_9"]
    assert row["designated_source_covered"] is None
    assert payload["backend"] == "chroma_v2"


def test_review_keys_are_not_part_of_the_retrieval_query():
    from host_finetune.pilot_retrieval import load_questions
    from pathlib import Path

    rows = load_questions(Path(__file__).resolve().parents[1] / "host_finetune" / "pilot_questions.json")
    kinds = {row["kind"] for row in rows}
    assert "unsupported" in kinds
    assert "conflicting_statements" in kinds
    for row in rows:
        for source_id in row["review_source_ids"]:
            assert source_id not in row["query"]


def test_recorded_collection_follows_the_path_when_omitted(tmp_path):
    from host_finetune.compare_adapters import write_parameters

    write_parameters(
        tmp_path,
        retrieval=True,
        embed_model="nomic-embed-text",
        retrieval_k=4,
        chroma_path="host_finetune/output/chroma_lemkin_train_only",
    )
    payload = json.loads((tmp_path / "config" / "run_parameters.json").read_text(encoding="utf-8"))
    assert payload["retrieval"]["collection"] == "lemkin_train_only"
    other = tmp_path / "explicit"
    write_parameters(
        other,
        retrieval=True,
        embed_model="nomic-embed-text",
        retrieval_k=4,
        chroma_path="host_finetune/output/chroma_lemkin_train_only_v2",
        collection="lemkin_train_only_v2_shadow",
    )
    explicit = json.loads((other / "config" / "run_parameters.json").read_text(encoding="utf-8"))
    assert explicit["retrieval"]["collection"] == "lemkin_train_only_v2_shadow"
