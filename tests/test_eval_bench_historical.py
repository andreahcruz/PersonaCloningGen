"""Replay saved pilot traces. No retrieval and no generation."""

from __future__ import annotations

import json
from pathlib import Path

from host_finetune.eval_bench.historical import (
    CELLS,
    crosswalk_to_profile,
    judge_pair,
    load_generation_records,
    load_retrieval_records,
    replay,
)
from host_finetune.eval_bench.matrix import profile_alignment

SOURCE = Path("experiments/EXP-20261031-001-graph-relationship-conditioning")
QUESTIONS = Path("host_finetune/graph_relationship_questions.json")
EXPERIMENT = "EXP-20261101-008-eval-bench-historical-replay"


def test_five_arms_match_model_f_profiles_and_not_the_october_cells():
    alignment = profile_alignment()
    assert alignment["every_profile_has_one_arm"] is True
    assert alignment["descriptions_is_its_own_arm"] is True
    assert alignment["model_f_profiles"] == ["off", "source", "relationships", "descriptions", "multihop"]
    assert alignment["leakage_has_separate_descriptions_arm"] is True
    assert alignment["registries_agree"] is True
    assert alignment["model_e_is_not_the_off_profile"] is True
    assert alignment["multihop_id"] == "F-HOP"
    assert alignment["historical_october_31_cells_are_these_profiles"] is False
    assert crosswalk_to_profile("D_blog", "descriptions")["status"] == "rejected"
    assert crosswalk_to_profile("A", "source")["comparable"] is False


def test_saved_cells_keep_their_own_names():
    records = load_generation_records(SOURCE, EXPERIMENT)
    assert len(records) == 65
    assert {row["architecture"] for row in records} == {spec["architecture"] for spec in CELLS.values()}
    assert all(row["provenance"][0]["profile_match"] is False for row in records)
    by_cell = {}
    for row in records:
        if row["question_id"] == "q_brex_price" and row["medium"] == "blog":
            by_cell[row["provenance"][0]["historical_cell"]] = row
    ablation = judge_pair(by_cell["A"], by_cell["B"])
    assert ablation["comparable"] is True
    assert ablation["interpretable_as_profile_comparison"] is False
    dense = judge_pair(by_cell["A"], by_cell["C"])
    assert dense["comparable"] is False
    assert "passages_not_held_fixed" in dense["reasons"]
    pad = judge_pair(by_cell["B"], by_cell["D_blog"])
    assert pad["comparable"] is False
    assert "token_pad_is_a_length_control" in pad["reasons"]


def test_replay_reconciles_without_inventing_ranked_scores(tmp_path: Path):
    report = replay(SOURCE, QUESTIONS, tmp_path / "replay", EXPERIMENT)
    generation = report["generation_reconciliation"]
    retrieval = report["retrieval_reconciliation"]
    assert generation["missing_historical_rows"] == []
    for field in ("factual_pass", "numeric_pass", "word_count"):
        counted = generation["agreements"][field] + sum(
            1 for item in generation["discrepancies"] if item["field"] == field
        )
        assert counted == 65
        assert all(item["calculation_changed"] is False for item in generation["discrepancies"])
    assert retrieval["rows"] == 48
    assert retrieval["agreements"] + len(retrieval["discrepancies"]) == 48
    template = json.loads((tmp_path / "replay" / "annotation_template.json").read_text(encoding="utf-8"))
    grades = [
        document["grade"]
        for question in template["questions"]
        for document in question["documents"]
    ]
    assert grades
    assert set(grades) == {None}
    assert template["grades_are_human_assigned"] is False
    retrieval_records = load_retrieval_records(SOURCE, QUESTIONS, EXPERIMENT)
    brex = [
        row for row in retrieval_records
        if row["question_id"] == "q_brex_price" and row["retrieval_config"]["mode"] == "hybrid"
    ]
    assert brex[0]["review_source_ids"] == ["gd_22550cc95a46ded2d9a37c99"]
    olympics = [row for row in retrieval_records if row["question_id"] == "q_olympics_ceremony"]
    assert olympics
    assert all(row["review_source_ids"] == [] for row in olympics)
