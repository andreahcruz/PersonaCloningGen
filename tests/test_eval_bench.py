"""Synthetic checks for the offline scorer. No model and no live index."""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from host_finetune.eval_bench.__main__ import main
from host_finetune.eval_bench.metrics import score_retrieval
from host_finetune.eval_bench.report import refuse_live_store, write_report
from host_finetune.eval_bench.runner import paired_faithfulness, score_batch
from host_finetune.eval_bench.schema import RecordError, validate_record
from host_finetune.eval_bench.stats import paired_mean_difference, wilson_interval


def _record(**overrides) -> dict:
    base = {
        "experiment_id": "EXP-SYN-001",
        "architecture": "model_f_source",
        "model_version": "relabel_qlora",
        "question_id": "syn_hire",
        "question": "Who did Company X hire?",
        "medium": "blog",
        "uses_graph": False,
        "retrieval_config": {"backend": "lightrag", "mode": "hybrid", "evidence_budget": 4},
        "generation_config": {"temperature": 0.7, "top_p": 0.9},
        "retrieved_source_ids": ["gd_a", "gd_b"],
        "evidence": [
            {"id": "gd_a", "text": "Company X hired Executive Y after the Series B.", "split": "train", "provenance": "fixture"},
            {"id": "gd_b", "text": "The board approved the search.", "split": "train", "provenance": "fixture"},
        ],
        "qrels": [
            {"doc_id": "gd_a", "grade": 2},
            {"doc_id": "gd_b", "grade": 0},
            {"doc_id": "gd_c", "grade": 2},
            {"doc_id": "gd_d", "grade": 0},
        ],
        "claims": [{"claim": "Company X hired Executive Y after the Series B.", "source_id": "gd_a"}],
        "answer": "The company hired the executive after that round.",
        "reference_answer": None,
        "latency": {"retrieval_seconds": 0.2, "generation_seconds": 1.0, "end_to_end_seconds": 1.2},
        "tokens": {"prompt": 100, "output": 20, "source": "fixture"},
        "context_budget_tokens": 400,
    }
    base.update(overrides)
    return base


def test_schema_rejects_a_missing_field():
    record = _record()
    del record["answer"]
    with pytest.raises(RecordError):
        validate_record(record)


def test_dense_and_graph_records_share_one_schema():
    dense = validate_record(_record(architecture="model_e", uses_graph=False, retrieval_config={"backend": "dense"}))
    graph = validate_record(_record(architecture="model_f_multihop", uses_graph=True))
    assert dense["uses_graph"] is False
    assert graph["uses_graph"] is True
    assert "relationship" not in dense["retrieval_config"]


def test_ranked_metrics_match_an_independent_formula():
    record = _record(evidence=[
        {"id": "gd_a", "text": "Company X hired Executive Y after the Series B.", "split": "train", "provenance": "fixture"},
        {"id": "gd_b", "text": "The board approved the search.", "split": "train", "provenance": "fixture"},
        {"id": "gd_c", "text": "Executive Y joined the leadership team.", "split": "train", "provenance": "fixture"},
        {"id": "gd_d", "text": "A later note.", "split": "train", "provenance": "fixture"},
    ], retrieved_source_ids=["gd_a", "gd_b", "gd_c", "gd_d"])
    scored = score_retrieval(record)
    grades = [2, 0, 2, 0]
    dcg = sum((2 ** grade - 1) / math.log2(index + 2) for index, grade in enumerate(grades))
    ideal = [2, 2, 0, 0]
    idcg = sum((2 ** grade - 1) / math.log2(index + 2) for index, grade in enumerate(ideal))
    assert scored["precision_at_4"] == 0.5
    assert scored["recall_at_4"] == 1.0
    assert scored["mrr"] == 1.0
    assert scored["ndcg_at_4"] == pytest.approx(dcg / idcg)


def test_missing_qrels_leave_ranked_metrics_null():
    record = _record()
    del record["qrels"]
    scored = score_retrieval(record)
    assert scored["precision_at_4"] is None
    assert scored["mrr"] is None
    assert scored["relevance_scope"] == "none"


def test_missing_reference_does_not_invent_an_overlap_score():
    batch = score_batch([_record(reference_answer=None)])
    generation = batch["results"][0]["generation"]
    assert generation["reference_used"] is False
    assert "bleu" not in generation
    assert "rouge" not in generation


def test_empty_retrieval_is_a_failure_and_not_a_zero_precision():
    record = _record(evidence=[], retrieved_source_ids=[], claims=[])
    batch = score_batch([record])
    assert batch["results"][0]["retrieval"]["empty"] is True
    assert batch["results"][0]["retrieval"]["precision_at_4"] is None
    assert batch["failures"][0]["error_class"] == "empty_retrieval"


def test_duplicate_prediction_is_excluded_from_the_aggregate():
    batch = score_batch([_record(), _record(answer="A second draft of the same prediction.")])
    assert len(batch["results"]) == 1
    assert batch["failures"][0]["error_class"] == "duplicate_prediction"


def test_invalid_citation_marker_is_a_leak_not_a_citation_score():
    batch = score_batch([_record(answer="The company hired the executive after that round. [E1]")])
    generation = batch["results"][0]["generation"]
    assert generation["citation_marker_leak"] is True
    assert generation["citations"]["status"] == "not_emitted"
    assert generation["citations"]["precision"] is None


def test_explicit_wrong_citation_has_zero_precision():
    batch = score_batch([_record(citations=["gd_missing"])])
    citations = batch["results"][0]["generation"]["citations"]
    assert citations["status"] == "scored"
    assert citations["precision"] == 0.0


def test_abstention_is_not_failed_for_having_no_claim():
    record = _record(
        question_id="syn_abstain",
        claims=[],
        evidence=[],
        retrieved_source_ids=[],
        abstention_expected=True,
        forbidden_assertions=["$9 million"],
        answer="The available notes do not state that figure.",
    )
    generation = score_batch([record])["results"][0]["generation"]
    assert generation["faithfulness"]["status"] == "scored_as_abstention"
    assert generation["faithfulness"]["pass"] is True


def test_invented_quantity_fails_even_when_the_prose_is_fluent():
    batch = score_batch([_record(answer="Company X reached $9 billion after the round.")])
    generation = batch["results"][0]["generation"]
    assert generation["numeric_pass"] is False
    assert generation["unsupported_numeric_count"] == 1
    assert generation["faithfulness"]["pass"] is False


def test_held_out_evidence_is_not_scored_as_a_success():
    record = _record(evidence=[
        {"id": "gd_secret", "text": "A validation passage.", "split": "test"},
    ])
    batch = score_batch([record])
    assert batch["results"] == []
    assert batch["failures"][0]["error_class"] == "held_out_evidence"


def test_two_runs_produce_the_same_aggregate():
    rows = [_record(), _record(question_id="syn_second", architecture="model_e", uses_graph=False)]
    first = json.dumps(score_batch(rows)["metrics"], sort_keys=True)
    second = json.dumps(score_batch(rows)["metrics"], sort_keys=True)
    assert first == second


def test_pairing_keeps_each_medium():
    rows = [
        _record(question_id="q1", medium="blog", architecture="model_e"),
        _record(question_id="q1", medium="blog", architecture="model_f_source", answer="The price was $9 billion."),
        _record(question_id="q1", medium="linkedin", architecture="model_e", answer="The price was $9 billion."),
        _record(question_id="q1", medium="linkedin", architecture="model_f_source"),
    ]
    batch = score_batch(rows)
    paired = paired_faithfulness(batch["results"], "model_e", "model_f_source", seed=298)
    assert paired["n"] == 2
    assert paired["mean_difference"] == 0.0
    assert paired["paired_on"] == "question_id_and_medium"


def test_retrieval_only_scope_does_not_invent_a_faithfulness_rate():
    batch = score_batch([_record(scopes=["retrieval"], answer="")])
    arm = batch["metrics"]["arms"]["model_f_source"]
    assert arm["faithfulness_rate"] is None
    assert batch["results"][0]["generation"]["status"] == "not_requested"


def test_paired_interval_is_deterministic():
    rows = [
        _record(question_id="q1", architecture="model_e"),
        _record(question_id="q1", architecture="model_f_source", answer="The price was $9 billion."),
        _record(question_id="q2", architecture="model_e"),
        _record(question_id="q2", architecture="model_f_source"),
    ]
    batch = score_batch(rows)
    first = paired_faithfulness(batch["results"], "model_e", "model_f_source", seed=298)
    second = paired_faithfulness(batch["results"], "model_e", "model_f_source", seed=298)
    assert first["n"] == 2
    assert first["mean_difference"] == 0.5
    assert first["ci95"] == second["ci95"]
    assert batch["metrics"]["composite_score"] is None


def test_wilson_and_bootstrap_reject_bad_input():
    assert wilson_interval(0, 0) is None
    low, high = wilson_interval(8, 10)
    assert 0 <= low <= 0.8 <= high <= 1
    with pytest.raises(ValueError):
        paired_mean_difference([1.0], [1.0, 0.0])


def test_report_files_and_live_store_refusal(tmp_path: Path):
    batch = score_batch([_record()])
    write_report(tmp_path / "report", batch, experiment_id="EXP-SYN-001")
    written = {path.name for path in (tmp_path / "report").iterdir()}
    assert written == {
        "results.jsonl",
        "metrics.json",
        "per_question_analysis.jsonl",
        "failures.jsonl",
        "experiment_manifest.json",
        "report.md",
    }
    manifest = json.loads((tmp_path / "report" / "experiment_manifest.json").read_text(encoding="utf-8"))
    assert manifest["live_index_queried"] is False
    assert manifest["composite_score"] is None
    with pytest.raises(RuntimeError):
        refuse_live_store(Path("artifacts") / "lightrag" / "model_f" / "storage_expand_v1" / "report")


def test_cli_scores_a_saved_file(tmp_path: Path):
    source = tmp_path / "predictions.jsonl"
    source.write_text(json.dumps(_record()) + "\n", encoding="utf-8")
    main([
        "--predictions", str(source),
        "--out", str(tmp_path / "out"),
        "--experiment-id", "EXP-SYN-001",
        "--pair-left", "model_e",
        "--pair-right", "model_f_source",
    ])
    report = (tmp_path / "out" / "report.md").read_text(encoding="utf-8")
    assert "no composite score" in report
    assert "not_comparable" in report or "Paired faithfulness" in report
