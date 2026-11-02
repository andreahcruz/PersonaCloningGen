"""Offline extraction diagnostics. No store, model, or graph writes."""

from __future__ import annotations

import asyncio

from host_finetune.indexing_extraction_audit import (
    alias_candidates,
    classify_relation_fields,
    endpoint_cooccurrence,
    numeric_fact_gaps,
    recover_relation,
    retrieve_task_exceptions,
    summarize_responses,
)


def test_omitted_keywords_are_syntactic_and_not_yet_semantic():
    fields = ["relation", "Freemium", "AI Products", "Freemium is the pricing model used for the AI products."]
    verdict = classify_relation_fields(fields)
    assert verdict["kind"] == "omitted_keywords"
    repair = recover_relation(fields)
    assert repair["keywords"] == ""
    assert repair["semantic_status"] == "unverified"
    assert endpoint_cooccurrence("Freemium", "AI Products", "The AI products use a freemium plan.")
    assert not endpoint_cooccurrence("Freemium", "AI Products", "The essay discusses pricing.")


def test_keywords_without_a_description_are_not_recovered():
    fields = ["relation", "SaaStr", "Okta", "Partnership"]
    assert classify_relation_fields(fields)["recoverable"] is False
    assert recover_relation(fields) is None


def test_schema_labels_are_removed_and_arbitrary_tails_are_not_joined():
    labeled = [
        "relation", "SaaStr", "Okta", "relationship_keywords", "provides insights",
        "The SaaStr community relies on Okta's data to understand adoption.",
    ]
    repair = recover_relation(labeled)
    assert repair["repair"] == "schema_labels"
    assert repair["keywords"] == "provides insights"
    assert "relationship_keywords" not in repair["description"]
    seven = [
        "relation", "Salesforce", "Procurement", "relationship_keywords", "automate, discounting",
        "relationship_description", "Salesforce automates discounting by marking prices up 20%.",
    ]
    assert recover_relation(seven)["description"].startswith("Salesforce automates")
    zirp = [
        "relation", "Interest rates", "ZIRP", "Post-ZIRP", "High interest rates",
        "Maybe", "Does the downfall in SaaS multiples correlate with the rise of interest rates?", "Maybe",
    ]
    assert recover_relation(zirp) is None


def test_ambiguous_extra_fields_are_not_joined():
    fields = ["relation", "Company", "Customer", "The company sells software to buyers every quarter", "to the customer", "in 2022", "through Salesforce"]
    assert classify_relation_fields(fields)["recoverable"] is False
    assert recover_relation(fields) is None


def test_self_relation_and_duplicates_are_counted_once_per_identity():
    initial = (
        "relation<|#|>SaaS<|#|>SaaS<|#|>Identity<|#|>A product referring to itself.\n"
        "relation<|#|>SDR<|#|>Quota<|#|>SDR owns the quota and reports it weekly.\n"
        "relation<|#|>SDR<|#|>Quota<|#|>SDR owns the quota and reports it weekly.\n"
        "<|COMPLETE|>"
    )
    gleaning = (
        "relation<|#|>SDR<|#|>Quota<|#|>SDR owns the quota and reports it weekly.\n"
        "<|COMPLETE|>"
    )
    summary = summarize_responses([
        {"chunk_id": "gd_demo-chunk-000", "create_time": 1, "original_prompt": "Extract entities and relationships", "return": initial},
        {"chunk_id": "gd_demo-chunk-000", "create_time": 2, "original_prompt": "Based on the last extraction task", "return": gleaning},
    ])
    assert summary["relation_rows_by_kind"]["self_relation_dropped"] == 1
    assert summary["repeated_rows_inside_one_response"] == 1
    assert summary["gleaning_repeats_of_initial_malformed"] == 1
    assert summary["unique_syntactic_repairs"] == 1
    assert summary["chunks_whose_relation_rows_were_all_rejected"] == 1


def test_aliases_and_numbers_are_reported_without_merging():
    groups = alias_candidates(["HubSpot", "Hubspot", "MongoDB", "SDR"])
    assert groups == [{"key": "hubspot", "names": ["HubSpot", "Hubspot"]}]
    gaps = numeric_fact_gaps("130%", "A rate mentioned in the essay.")
    assert "named_entity_in_description" in gaps
    assert numeric_fact_gaps("Salesforce", "A CRM company.") == []


def test_gather_retrieves_a_cancellation_exception_that_wait_would_drop():
    class PipelineCancelled(Exception):
        pass

    async def raise_cancel():
        raise PipelineCancelled("User cancelled during chunk processing")

    async def scenario():
        failed = asyncio.create_task(raise_cancel())
        sleeper = asyncio.create_task(asyncio.sleep(30))
        await asyncio.wait([failed])
        # wait() marks the task done and does not retrieve the exception.
        assert failed.done() and not failed.cancelled()
        return await retrieve_task_exceptions([failed, sleeper])

    retrieved = asyncio.run(scenario())
    assert len(retrieved) == 1
    assert "chunk processing" in str(retrieved[0])
