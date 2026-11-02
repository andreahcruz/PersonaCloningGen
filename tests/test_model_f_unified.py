"""CPU checks for the 120-prompt panel. No retrieval, generation, or judge."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from host_finetune.model_f_experiment.arms import ARMS, EXPERIMENT_ID
from host_finetune.model_f_experiment.runner import cell_record, write_generation
from host_finetune.model_f_experiment.snapshot import SnapshotError
from host_finetune.model_f_experiment.unified import (
    GOLD_SHA256,
    PROTECTED_EXPERIMENTS,
    UNIFIED_EXPERIMENT_ID,
    comparison_matrix,
    file_sha256,
    load_gold_topics,
    panel_cells,
    record_overlap,
    require_profile,
    saved_architecture_matrix,
    score_draft,
    score_panel,
    write_plan,
)
from host_finetune.score_saved_experiment import EXPECTED_GENERATION_SHA
from host_finetune.unified_eval import SCHEMA, CopyIndex


def test_panel_is_the_frozen_thirty_topics():
    topics = load_gold_topics()
    cells = panel_cells(topics)
    assert file_sha256(Path("data/openai_ft/lemkin_gold_eval.jsonl")) == GOLD_SHA256
    assert [topic["id"] for topic in topics] == [f"gold_{index:03d}" for index in range(1, 31)]
    assert len(cells) == 120
    assert {cell["medium"] for cell in cells} == {"blog", "linkedin", "x", "talk"}
    assert len({(cell["id"], cell["medium"]) for cell in cells}) == 120


def test_profile_must_be_named_and_the_four_arm_run_stays_separate():
    assert require_profile("F-REL") == "F-REL"
    with pytest.raises(ValueError):
        require_profile("E-DENSE")
    assert EXPERIMENT_ID == "EXP-20261101-009-model-f-four-arm"
    assert ARMS == ("F-SOURCE", "F-REL", "F-DESC", "F-HOP")
    assert UNIFIED_EXPERIMENT_ID not in PROTECTED_EXPERIMENTS
    assert EXPECTED_GENERATION_SHA.startswith("266507aca9")


def test_development_questions_do_not_share_panel_ids():
    report = record_overlap()
    assert report["dev_questions"] == 36
    assert report["gold_topics"] == 30
    assert report["shared_ids"] == []
    assert report["wording_overlaps"] == []
    assert report["single_term_overlaps"]
    assert report["same_task"] is False
    assert report["panel_used_for_profile_selection"] is False
    assert report["gold_sha256"] == GOLD_SHA256


def test_plan_refuses_the_live_store_and_historical_directories(tmp_path: Path):
    with pytest.raises(SnapshotError):
        write_plan(tmp_path / "panel", "F-SOURCE", tmp_path / "storage_expand_v1")
    with pytest.raises(SnapshotError):
        write_plan(tmp_path / "EXP-20261005-003", "F-SOURCE", tmp_path / "snapshot_copy")
    plan = write_plan(tmp_path / "panel", "F-SOURCE", tmp_path / "snapshot_copy")
    assert plan["status"] == "prepared_not_run"
    assert plan["cells"] == 120
    assert plan["profile_selected_by_this_panel"] is False
    assert plan["judge_overlay"] == "not_applied"
    assert not (tmp_path / "panel" / "raw").exists()


def test_saved_cell_scores_without_filling_missing_judge_metrics(tmp_path: Path):
    question = {
        "question_id": "gold_001",
        "query": "A synthetic topic",
        "abstention_expected": False,
    }
    trace = {
        "medium": "x",
        "answer": "Bring a concrete plan. Then do the work.",
        "prompt": "Write an X post.",
        "retrieval_hits": [{"id": "src:gd_a", "text": "Bring a concrete plan before the interview."}],
        "latency_seconds": 1.25,
        "model_f_context": {},
        "verified_claims": ["Bring a concrete plan."],
    }
    write_generation(tmp_path, cell_record(question, "F-SOURCE", trace))
    coverage = score_panel(tmp_path, "F-SOURCE", {}, CopyIndex([]), {})
    assert coverage["scored"] == 1
    assert coverage["missing_metrics_are_zero"] is False
    assert len(coverage["missing"]) == 119
    row = json.loads((tmp_path / "metrics" / "rows.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert row["schema"] == SCHEMA == "unified_eval.v1"
    assert row["writing_quality"]["mean"] is None
    assert row["persona_utility"] is None
    assert row["judge_overlay"] == "not_applied"
    assert row["historical_scores"] is None
    matrix = comparison_matrix({"Model F": [row]})
    assert matrix["composite_score"] is None
    assert matrix["rank"] is None
    overall = next(item for item in matrix["rows"] if item["medium"] == "all")
    assert overall["metrics"]["writing_quality"]["mean"] is None
    assert overall["metrics"]["writing_quality"]["missing"] == 1
    assert overall["metrics"]["persona_utility"]["mean"] is None


def test_direct_score_keeps_judge_fields_null():
    gold = {
        "id": "gold_001",
        "topic": "A synthetic topic",
        "audience": "founders",
        "goal": "give one lesson",
        "expected_facts": [],
        "rubric_must_haves": [],
        "gold_reference": "A synthetic reference.",
    }
    cell = {
        "medium": "x",
        "format": "x_thread",
        "answer": "Bring a concrete plan. Then do the work.",
        "prompt": "Write an X post.",
        "evidence": [{"text": "Bring a concrete plan before the interview."}],
        "latency": {"generation_seconds": 1.25},
        "profile": "F-SOURCE",
    }
    row = score_draft(cell, gold, {}, CopyIndex([]), {})
    assert row["persona_utility"] is None
    assert row["writing_quality"]["available"] is False


def test_saved_rows_do_not_invent_a_model_f_result():
    report = saved_architecture_matrix()
    assert report["model_f"] is None
    assert report["composite_score"] is None
    assert report["rank"] is None
    counted = {item["architecture"]: item["drafts"] for item in report["rows"] if item["medium"] == "all"}
    assert counted["Relabel + RAG"] == 120
    assert counted["Base"] == 120
    assert "Model F" not in counted
