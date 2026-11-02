"""What the current questions cover. Synthetic rows are not ground truth.

Counts were read from field names and question labels on 2026-11-01. Gold
reference text was not copied into this module.
"""

from __future__ import annotations

GOLD_DEV_SET = {
    "path": "data/openai_ft/lemkin_gold_eval.jsonl",
    "rows": 30,
    "format": "blog_draft",
    "fields": [
        "audience",
        "cta",
        "expected_facts",
        "format",
        "goal",
        "gold_reference",
        "id",
        "k",
        "rubric_must_haves",
        "topic",
    ],
    "role": "Generation development set already used for prompt work. Not the corpus test split. Not a graded retrieval qrel.",
}

REVIEW_KINDS = {
    "host_finetune/pilot_questions.json": [
        "simple_fact",
        "author_position",
        "multi_document",
        "conditional_advice",
        "conflicting_statements",
        "attribution",
        "unsupported",
    ],
    "host_finetune/graph_relationship_questions.json": [
        "simple_fact",
        "relational",
        "multi_document",
        "conflicting_statements",
        "conditional_advice",
        "attribution",
        "unsupported",
    ],
}

COVERAGE = (
    {
        "phenomenon": "entity_identification",
        "covered_by": "relational review questions name entities, but there is no entity-linking label",
        "gap": "No graded entity-identification set",
    },
    {
        "phenomenon": "entity_relationships",
        "covered_by": "graph_relationship_questions.json relational rows",
        "gap": "Review notes are not qrels. Overlap with a preselected id is not recall",
    },
    {
        "phenomenon": "multi_hop",
        "covered_by": "multi_document questions retrieve more than one post about one topic",
        "gap": "No labeled two-hop chain where the endpoint claim is unsupported",
    },
    {
        "phenomenon": "temporal_reasoning",
        "covered_by": "q_boom_promotions mentions 2021 inside one document",
        "gap": "No comparison of two dated claims across documents",
    },
    {
        "phenomenon": "numerical_claims",
        "covered_by": "q_brex_price and q_five_percent_share",
        "gap": "Too few graded numeric items to estimate a rate",
    },
    {
        "phenomenon": "attribution",
        "covered_by": "q_personalized_saas_speaker",
        "gap": "One speaker control",
    },
    {
        "phenomenon": "multiple_sources",
        "covered_by": "q_first_vp_sales and q_vp_sales_tension",
        "gap": "Preselected ids, not a complete relevant set",
    },
    {
        "phenomenon": "abstention",
        "covered_by": "unsupported questions in both review files",
        "gap": "The generation length gates treat a short abstention as a failure. This harness does not.",
    },
)

# Fixture questions for the scorer. They are not human-validated answers.
SYNTHETIC_QUESTIONS = (
    {
        "question_id": "syn_hop",
        "phenomenon": "multi_hop",
        "question": "What strategy did Company X adopt by hiring Executive Y?",
        "ground_truth": "not_human_validated",
        "note": "A correct system may show both edges and must not claim one source stated the endpoint.",
    },
    {
        "question_id": "syn_abstain",
        "phenomenon": "abstention",
        "question": "What was Company X's ARR in 2019?",
        "abstention_expected": True,
        "ground_truth": "not_human_validated",
        "note": "The fixture has no supporting figure. Inventing one fails the numeric check.",
    },
    {
        "question_id": "syn_temporal",
        "phenomenon": "temporal_comparison",
        "question": "Did the 2022 valuation exceed the 2024 price?",
        "ground_truth": "not_human_validated",
        "note": "No reference number is supplied here, so this row cannot be scored as correct.",
    },
)
