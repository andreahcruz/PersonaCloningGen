"""CPU checks for the 32-cell pilot and the 120-prompt unified adapter."""

from __future__ import annotations

import inspect
import json
from pathlib import Path

import pytest

from host_finetune.model_f_context.prompt import GRAPH_CONTEXT_CONTRACT
from host_finetune.model_f_experiment.arms import ARMS, PREPARE_PROFILE
from host_finetune.model_f_experiment.diagnostics import summarize_graph, write_model_f_reports
from host_finetune.model_f_experiment.guards import ModelFGuardError, require_lightrag_trace
from host_finetune.model_f_experiment.pilot import (
    PILOT_CELLS,
    pilot_pending,
    score_pilot,
    select_arm,
    write_pilot_manifest,
)
from host_finetune.model_f_experiment.retrieval import generation_payload, prepare_query_workdir
from host_finetune.model_f_experiment.snapshot import SnapshotError
from host_finetune.model_f_experiment.runner import (
    cell_record,
    freeze_retrieval,
    run_pending,
    write_generation,
)
from host_finetune.model_f_experiment.unified import (
    PANEL_SIZE,
    judge_saved_panel,
    panel_cells,
    score_panel,
    write_plan,
)
from host_finetune.score_saved_experiment import EXPECTED_GENERATION_SHA
from host_finetune.unified_eval import SCHEMA, CopyIndex

BENCH = Path("host_finetune/benchmarks/dev_retrieval_benchmark.jsonl")
GOLD_ROWS = Path("docs/evaluation/examples/data/relabel_rag_on_rows.jsonl")
CLAIM = "The board wanted a sales leader."


def _manifest(tmp_path: Path) -> dict:
    snapshot = tmp_path / "storage_expand_v1_snapshot"
    snapshot.mkdir()
    manifest = write_pilot_manifest(tmp_path / "pilot", BENCH, str(snapshot))
    assert list(snapshot.iterdir()) == []
    return manifest


def _trace(answer: str, profile: str, channels: dict | None = None) -> dict:
    return {
        "medium": "blog",
        "format": "blog_draft",
        "answer": answer,
        "prompt": "Write a blog post.",
        "seed": 42,
        "accepted_attempt": 1,
        "max_new_tokens": 768,
        "prompt_tokens": 80,
        "generated_tokens_including_stop": 10,
        "latency_seconds": 2.0,
        "extraction_seconds": 1.0,
        "retrieval_backend": "lightrag",
        "retrieval_mode": "hybrid",
        "graph_context_contract": GRAPH_CONTEXT_CONTRACT,
        "verified_claims": [{"claim": CLAIM, "source_id": "gd_a"}],
        "retrieval_hits": [{
            "id": "gd_a",
            "text": CLAIM,
            "source_id": "gd_a",
            "split": "train",
            "provenance": {"origin": "source_chunk"},
        }],
        "model_f_context": {
            "profile": profile,
            "selected_ids": ["src:gd_a"],
            "grounding_texts": [CLAIM],
            "tokens_used": 8,
            "tokens_by_channel": channels or {"primary_source": 8},
            "token_estimate": "caller_tokenizer",
            "budget": {"total": 1000},
        },
    }


def _write_arm(directory: Path, question: dict, arm: str, answer: str, channels: dict | None = None) -> None:
    trace = _trace(answer, PREPARE_PROFILE[arm], channels)
    trace["medium"] = question["pilot_medium"]
    write_generation(directory, cell_record(question, arm, trace))


def test_pilot_plan_is_thirty_two_paired_cells(tmp_path: Path):
    before = BENCH.read_bytes()
    manifest = _manifest(tmp_path)
    assert BENCH.read_bytes() == before
    assert manifest["benchmark_sha256"]
    assert manifest["cells"] == PILOT_CELLS == 32
    assert manifest["questions_selected"] == 8
    assert manifest["media_counts"] == {"blog": 2, "linkedin": 2, "x": 2, "talk": 2}
    assert manifest["judge"] is None
    assert manifest["retrieval"]["backend"] == "lightrag"
    assert manifest["retrieval"]["mode"] == "hybrid"
    assert manifest["adapter"] == "host_finetune/output/lemkin_lora_relabel"
    assert manifest["decoding"]["temperature"] == 0.7
    by_question: dict[str, list[dict]] = {}
    for cell in manifest["plan"]:
        by_question.setdefault(cell["question_id"], []).append(cell)
    assert len(by_question) == 8
    phenomena = set()
    for question in manifest["questions"]:
        cells = by_question[question["question_id"]]
        assert len(cells) == 4
        assert {cell["arm"] for cell in cells} == set(ARMS)
        assert {cell["medium"] for cell in cells} == {question["pilot_medium"]}
        assert {cell["retrieval_key"] for cell in cells} == {question["question_id"]}
        assert {cell["context_profile"] for cell in cells} == set(PREPARE_PROFILE.values())
        phenomena.add(question["phenomenon"])
    assert phenomena == {"factual", "direct_relationship", "multi_hop", "unanswerable"}
    first = manifest["questions"][:4]
    assert len({row["pilot_medium"] for row in first}) == 4
    assert len({row["phenomenon"] for row in first}) == 4
    preflight = manifest["runtime_preflight"]
    assert preflight["full_estimate"]["measured"] is False
    assert preflight["full_estimate"]["estimated_minutes"] > 60
    assert preflight["fits_budget"] is False
    assert preflight["recommended_cells"] == 16
    assert preflight["conditions_unchanged"] is True
    assert preflight["recommended_estimate"]["estimated_minutes"] <= 60
    remaining = pilot_pending(tmp_path / "pilot")
    assert len(remaining) == 32
    question = manifest["questions"][0]
    write_generation(
        tmp_path / "pilot",
        {"question_id": question["question_id"], "arm": "F-SOURCE", "medium": question["pilot_medium"], "answer": "done"},
    )
    assert len(pilot_pending(tmp_path / "pilot")) == 31


def test_runtime_stop_leaves_incomplete_groups_out(tmp_path: Path):
    questions = [
        {"question_id": "q1", "query": "One?", "pilot_medium": "x"},
        {"question_id": "q2", "query": "Two?", "pilot_medium": "blog"},
    ]
    for question in questions:
        freeze_retrieval(tmp_path, question, {"data": {}})
    produced = {"n": 0}

    def generate_cell(question, arm, medium, payload):
        produced["n"] += 1
        return {"answer": question["question_id"], "retrieval_backend": "lightrag", "retrieval_mode": "hybrid"}

    def should_stop():
        return produced["n"] >= 2

    report = run_pending(tmp_path, questions, generate_cell, should_stop=should_stop)
    assert report["stopped"] is True
    assert report["generated"] == 2
    again = run_pending(tmp_path, questions, generate_cell)
    assert again["skipped_complete"] == 2
    assert again["generated"] == 6


def test_graph_context_is_not_enough_to_win(tmp_path: Path):
    directory = tmp_path / "pilot"
    manifest = _manifest(tmp_path)
    question = dict(manifest["questions"][0])
    bad = "The board wanted a sales leader and the price was $999."
    for arm in ARMS:
        channels = {"primary_source": 8, "derived_description": 40} if arm == "F-DESC" else {"primary_source": 8}
        _write_arm(directory, question, arm, bad, channels)
    freeze_retrieval(directory, question, {"data": {"entities": [], "relationships": [], "chunks": []}}, graph_summary={
        "status": "recorded",
        "accepted_relationships": 3,
        "entity_description_available": True,
        "two_hop_path_available": False,
    })
    score_pilot(directory)
    chosen = select_arm(directory)
    assert chosen["selected_arm"] == "F-SOURCE"
    assert chosen["status"] == "fallback"
    assert chosen["graph_context_availability_used_as_score"] is False
    comparison = json.loads((directory / "arm_comparison.json").read_text(encoding="utf-8"))
    assert comparison["graph_evidence_is_a_score"] is False
    assert comparison["complete_question_ids"] == [question["question_id"]]
    text = (directory / "arm_selection.json").read_text(encoding="utf-8")
    _write_arm(directory, question, "F-REL", CLAIM, {"primary_source": 8, "source_supported_relationship": 20})
    assert select_arm(directory)["selected_arm"] == "F-SOURCE"
    assert (directory / "arm_selection.json").read_text(encoding="utf-8") == text


def test_a_factual_gain_can_select_a_graph_arm(tmp_path: Path):
    directory = tmp_path / "gain"
    snapshot = tmp_path / "storage_expand_v1_snapshot"
    snapshot.mkdir()
    manifest = write_pilot_manifest(directory, BENCH, str(snapshot))
    question = dict(manifest["questions"][0])
    bad = "The board wanted a sales leader and the price was $999."
    for arm in ARMS:
        answer = CLAIM if arm == "F-REL" else bad
        _write_arm(directory, question, arm, answer)
    freeze_retrieval(directory, question, {"data": {}})
    score_pilot(directory)
    metrics = json.loads((directory / "deterministic_metrics.json").read_text(encoding="utf-8"))
    assert metrics["judge"] is None
    assert metrics["ranked_retrieval"] == "not_scored_without_human_grades"
    by_arm = {cell["arm"]: cell for cell in metrics["cells"] if cell["question_id"] == question["question_id"]}
    assert by_arm["F-REL"]["generation"]["faithfulness"]["pass"] is True
    assert by_arm["F-SOURCE"]["generation"]["faithfulness"]["pass"] is False
    chosen = select_arm(directory)
    assert chosen["selected_arm"] == "F-REL"
    assert chosen["status"] == "selected"
    assert "persona" in chosen["does_not_measure"]


def test_incomplete_groups_do_not_choose_a_winner(tmp_path: Path):
    directory = tmp_path / "partial"
    snapshot = tmp_path / "storage_expand_v1_snapshot"
    snapshot.mkdir()
    manifest = write_pilot_manifest(directory, BENCH, str(snapshot))
    question = dict(manifest["questions"][0])
    for arm in ARMS:
        if arm == "F-HOP":
            continue
        _write_arm(directory, question, arm, CLAIM)
    chosen = select_arm(directory)
    assert chosen["selected_arm"] == "F-SOURCE"
    assert chosen["status"] == "fallback_incomplete"
    assert question["question_id"] in chosen["excluded_incomplete_question_ids"]
    assert chosen["complete_question_ids"] == []


def test_guards_reject_dense_retrieval_and_context_off():
    base = {
        "retrieval_backend": "lightrag",
        "retrieval_mode": "hybrid",
        "graph_context_contract": GRAPH_CONTEXT_CONTRACT,
        "model_f_context": {"profile": "source"},
    }
    require_lightrag_trace(base, "F-SOURCE")
    dense = dict(base)
    dense["retrieval_backend"] = "dense"
    with pytest.raises(ModelFGuardError):
        require_lightrag_trace(dense, "F-SOURCE")
    naive = dict(base)
    naive["retrieval_mode"] = "naive"
    with pytest.raises(ModelFGuardError):
        require_lightrag_trace(naive, "F-SOURCE")
    off = dict(base)
    off["graph_context_contract"] = None
    off["model_f_context"] = {"profile": "relationships"}
    with pytest.raises(ModelFGuardError):
        require_lightrag_trace(off, "F-REL")
    source_as_graph = dict(base)
    with pytest.raises(ModelFGuardError):
        require_lightrag_trace(source_as_graph, "F-REL")


def test_query_copy_stays_outside_the_snapshot(tmp_path: Path):
    snapshot = tmp_path / "storage_expand_v1_snapshot"
    snapshot.mkdir()
    (snapshot / "catalog.jsonl").write_text('{"graph_document_id": "gd_a"}\n', encoding="utf-8")
    workdir = prepare_query_workdir(snapshot, tmp_path / "exp" / "raw" / "lightrag_query")
    (workdir / "kv_store_llm_response_cache.json").write_text("{}", encoding="utf-8")
    assert sorted(path.name for path in snapshot.iterdir()) == ["catalog.jsonl"]
    assert (workdir / "kv_store_llm_response_cache.json").is_file()
    with pytest.raises(SnapshotError):
        prepare_query_workdir(snapshot, snapshot / "inside")


def test_generation_payload_records_backend_and_observed_mode():
    payload = generation_payload(
        {"metadata": {"query_mode": "naive"}, "data": {"chunks": [], "entities": [], "relationships": []}},
        {},
    )
    assert payload["backend"] == "lightrag"
    assert payload["mode"] == "naive"
    assert payload["mode_source"] == "payload"
    missing = generation_payload({"data": {}}, {})
    assert missing["mode"] == "hybrid"
    assert missing["mode_source"] == "experiment_query"


def test_summary_does_not_invent_missing_graph_counts():
    summary = summarize_graph({"data": {"entities": [{"entity_name": "A"}]}}, {"entity_descriptions": []}, None)
    assert summary["raw_entity_count"] == 1
    assert summary["accepted_relationships"] is None
    assert summary["relationship_validation_status"] == "not_recorded"
    assert summary["two_hop_path_count"] is None


def test_pilot_and_official_diagnostics_use_one_scorer(tmp_path: Path):
    import host_finetune.model_f_experiment.diagnostics as diagnostics
    import host_finetune.model_f_experiment.pilot as pilot

    source = inspect.getsource(pilot) + inspect.getsource(diagnostics)
    assert "apply_judge_overlay" not in source
    assert "judge_json" not in source
    question = {
        "question_id": "q1",
        "query": "Who was hired?",
        "pilot_medium": "x",
        "abstention_expected": False,
    }
    _write_arm(tmp_path, question, "F-SOURCE", CLAIM)
    left = tmp_path / "pilot-out"
    right = tmp_path / "official-out"
    write_model_f_reports(tmp_path, left, "pilot")
    write_model_f_reports(tmp_path, right, "official")
    left_metrics = json.loads((left / "deterministic_metrics.json").read_text(encoding="utf-8"))
    right_metrics = json.loads((right / "deterministic_metrics.json").read_text(encoding="utf-8"))
    assert left_metrics["cells"][0]["generation"] == right_metrics["cells"][0]["generation"]
    assert (left / "graph_diagnostics.json").is_file()
    assert "graph_context_tokens" not in left_metrics["cells"][0]


def test_unified_rows_match_historical_schema_and_keep_diagnostics_separate(tmp_path: Path):
    assert len(panel_cells()) == PANEL_SIZE == 120
    snapshot = tmp_path / "storage_expand_v1_snapshot"
    snapshot.mkdir()
    plan = write_plan(tmp_path / "panel", "F-SOURCE", snapshot)
    assert plan["cells"] == 120
    assert plan["protocol"] == "unified_eval.v1"
    assert (tmp_path / "panel" / "generation_manifest.json").is_file()
    assert list(snapshot.iterdir()) == []
    question = {"question_id": "gold_001", "query": "A synthetic topic", "abstention_expected": False}
    trace = _trace("Bring a concrete plan. Then do the work.", "source")
    trace["medium"] = "x"
    trace["retrieval_hits"] = [{"id": "gd_a", "text": "Bring a concrete plan before the interview.", "split": "train"}]
    trace["verified_claims"] = ["Bring a concrete plan."]
    trace["model_f_context"]["grounding_texts"] = ["Bring a concrete plan."]
    write_generation(tmp_path / "panel", cell_record(question, "F-SOURCE", trace))
    score_panel(tmp_path / "panel", "F-SOURCE", {}, CopyIndex([]), {})
    official = (tmp_path / "panel" / "unified_eval" / "rows.jsonl").read_text(encoding="utf-8")
    assert official == (tmp_path / "panel" / "metrics" / "rows.jsonl").read_text(encoding="utf-8")
    row = json.loads(official.splitlines()[0])
    assert row["schema"] == SCHEMA == "unified_eval.v1"
    assert not any("graph" in key or key.startswith("model_f") for key in row)
    diagnostic = json.loads((tmp_path / "panel" / "diagnostics" / "graph_diagnostics.json").read_text(encoding="utf-8"))
    assert diagnostic["join"]["official_rows"] == "unified_eval/rows.jsonl"
    with GOLD_ROWS.open(encoding="utf-8") as handle:
        historical = json.loads(handle.readline())
    assert historical["schema"] == "unified_eval.v1"
    for key in ("gates", "writing_quality", "persona_style", "perspective_fidelity", "task_adherence", "grounding", "diagnostics"):
        assert key in row
        assert key in historical
    frozen = tmp_path / "arm_selection.json"
    frozen.write_text(json.dumps({
        "frozen": True,
        "selected_arm": "F-SOURCE",
        "do_not_revise_after_unified_eval": True,
    }) + "\n", encoding="utf-8")
    with pytest.raises(ValueError):
        write_plan(tmp_path / "other", "F-REL", snapshot, frozen)


def test_judge_waits_for_120_cells_and_the_historical_gate_stays(tmp_path: Path):
    assert EXPECTED_GENERATION_SHA == "266507aca9effe008596ca64dc199d2e0de32bafbd7a9765d738adb1d34ec550"
    with pytest.raises(RuntimeError, match="120"):
        judge_saved_panel(tmp_path / "not-run", "F-SOURCE")


_PILOT_MEDIA = [
    ("q_brex_price", "blog"),
    ("q_dev_hop_valuation_and_price", "linkedin"),
    ("q_cvc_second_check", "x"),
    ("q_dev_unans_halftime", "talk"),
    ("q_five_percent_share", "linkedin"),
    ("q_exec_craft", "talk"),
    ("q_first_vp_sales", "blog"),
    ("q_olympics_ceremony", "x"),
]
_MISSING_CELLS = {
    ("q_brex_price", "F-SOURCE", "blog"),
    ("q_brex_price", "F-REL", "blog"),
    ("q_brex_price", "F-DESC", "blog"),
    ("q_brex_price", "F-HOP", "blog"),
    ("q_dev_unans_halftime", "F-DESC", "talk"),
}


def _pilot_questions(directory: Path) -> list[dict]:
    questions = []
    for question_id, medium in _PILOT_MEDIA:
        question = {
            "question_id": question_id,
            "query": f"Question {question_id}?",
            "pilot_medium": medium,
        }
        freeze_retrieval(directory, question, {"data": {}})
        questions.append(question)
    return questions


def _mark_complete(directory: Path, question_id: str, arm: str, medium: str, marker: str) -> str:
    path = write_generation(directory, {
        "question_id": question_id,
        "arm": arm,
        "medium": medium,
        "answer": marker,
    })
    return path.read_text(encoding="utf-8")


def test_resume_writes_only_the_five_missing_cells(tmp_path: Path):
    questions = _pilot_questions(tmp_path)
    saved = {}
    for question_id, medium in _PILOT_MEDIA:
        for arm in ARMS:
            if (question_id, arm, medium) in _MISSING_CELLS:
                continue
            saved[(question_id, arm, medium)] = _mark_complete(
                tmp_path, question_id, arm, medium, f"kept {question_id} {arm}"
            )
    called = []

    def generate_cell(question, arm, medium, payload):
        called.append((question["question_id"], arm, medium))
        return {"answer": "new", "retrieval_backend": "lightrag", "retrieval_mode": "hybrid"}

    report = run_pending(tmp_path, questions, generate_cell)
    assert report["generated"] == 5
    assert report["skipped_complete"] == 27
    assert report["unsaved"] == []
    assert set(called) == _MISSING_CELLS
    for key, text in saved.items():
        question_id, arm, medium = key
        path = tmp_path / "raw" / "generations" / f"{question_id}__{arm}__{medium}.json"
        assert path.read_text(encoding="utf-8") == text


def test_partial_question_group_runs_only_the_open_arm(tmp_path: Path):
    question = {
        "question_id": "q_dev_unans_halftime",
        "query": "What did Jason Lemkin write about the choreography of the 2026 Super Bowl halftime show?",
        "pilot_medium": "talk",
    }
    freeze_retrieval(tmp_path, question, {"data": {}})
    kept = {}
    for arm in ("F-SOURCE", "F-REL", "F-HOP"):
        kept[arm] = _mark_complete(tmp_path, question["question_id"], arm, "talk", f"kept {arm}")
    called = []

    def generate_cell(question, arm, medium, payload):
        called.append((arm, medium))
        return {"answer": "desc", "retrieval_backend": "lightrag", "retrieval_mode": "hybrid"}

    report = run_pending(tmp_path, [question], generate_cell)
    assert called == [("F-DESC", "talk")]
    assert report["generated"] == 1
    assert report["skipped_complete"] == 3
    for arm, text in kept.items():
        path = tmp_path / "raw" / "generations" / f"{question['question_id']}__{arm}__talk.json"
        assert path.read_text(encoding="utf-8") == text


def test_a_planned_cell_that_is_not_saved_is_reported(tmp_path: Path):
    questions = _pilot_questions(tmp_path)
    for question_id, medium in _PILOT_MEDIA:
        for arm in ARMS:
            if (question_id, arm, medium) == ("q_brex_price", "F-SOURCE", "blog"):
                continue
            if (question_id, arm, medium) == ("q_dev_unans_halftime", "F-DESC", "talk"):
                continue
            _mark_complete(tmp_path, question_id, arm, medium, "kept")

    def generate_cell(question, arm, medium, payload):
        if question["question_id"] == "q_brex_price":
            raise ValueError("support span leaked into the final prompt")
        raise RuntimeError(
            "refusing to generate: prompt tokens 1300 + max_new_tokens 768 exceed 2048. "
            "Evidence was not truncated."
        )

    report = run_pending(tmp_path, questions, generate_cell)
    assert report["generated"] == 0
    assert report["skipped_complete"] == 30
    reasons = {(row["question_id"], row["arm"]): row["reason"] for row in report["unsaved"]}
    assert reasons[("q_brex_price", "F-SOURCE")] == "ValueError: support span leaked into the final prompt"
    assert reasons[("q_dev_unans_halftime", "F-DESC")].startswith("RuntimeError: refusing to generate")
    assert not (tmp_path / "raw" / "generations" / "q_brex_price__F-SOURCE__blog.json").exists()
    assert not (tmp_path / "raw" / "generations" / "q_dev_unans_halftime__F-DESC__talk.json").exists()
