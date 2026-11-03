"""CPU checks for the F-DESC versus F-REL stage. No model and no index writes."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from host_finetune.model_f_experiment.desc_vs_rel import (
    STAGE_ARMS,
    STAGE_EXPERIMENT_ID,
    compare_stage,
    select_stage_questions,
    write_stage_manifest,
)
from host_finetune.model_f_experiment.pilot import PILOT_EXPERIMENT_ID
from host_finetune.model_f_experiment.runner import freeze_retrieval, run_pending, write_generation

BENCH = Path("host_finetune/benchmarks/dev_retrieval_benchmark.jsonl")
PILOT = Path("experiments/EXP-model-f-quick-pilot/pilot_manifest.json")


def _exclude(path: Path, ids: list[str]) -> Path:
    path.write_text(json.dumps({
        "experiment_id": PILOT_EXPERIMENT_ID,
        "execution_order": ids,
    }), encoding="utf-8")
    return path


def _write_benchmark(path: Path, ids: list[str]) -> None:
    lines = []
    for question_id in ids:
        lines.append(json.dumps({
            "question_id": question_id,
            "phenomenon": "factual",
            "query": f"What is {question_id}?",
        }))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_requested_eight_per_medium_does_not_fit_the_development_pool():
    from host_finetune.model_f_experiment.runner import load_questions

    questions = load_questions(BENCH)
    excluded = json.loads(PILOT.read_text(encoding="utf-8"))["execution_order"]
    assert len(questions) == 36
    assert len(excluded) == 8
    selected, report = select_stage_questions(questions, excluded, per_medium=8)
    assert report["available_after_exclusion"] == 28
    assert report["shortfall_questions"] == 4
    assert report["per_medium"] == 7
    assert report["selected_questions"] == 28
    assert report["design_status"] == "reduced_to_largest_balanced_alternative"
    assert report["duplicated_questions"] == 0
    assert report["gold_questions_used"] == 0
    assert report["media_counts"] == {"blog": 7, "linkedin": 7, "x": 7, "talk": 7}
    ids = [row["question_id"] for row in selected]
    assert len(ids) == len(set(ids))
    assert not set(ids) & set(excluded)
    assert "q_brex_price" not in ids
    assert ids[0] == "q_dev_fact_acquirer"
    assert selected[0]["pilot_medium"] == "blog"


def test_stage_manifest_records_the_reduced_design(tmp_path: Path):
    snapshot = tmp_path / "storage_expand_v1_snapshot"
    snapshot.mkdir()
    manifest = write_stage_manifest(
        tmp_path / "stage",
        BENCH,
        str(snapshot),
        exclude_manifest=PILOT,
        per_medium=8,
    )
    assert manifest["experiment_id"] == STAGE_EXPERIMENT_ID
    assert manifest["arms"] == list(STAGE_ARMS)
    assert manifest["cells"] == 56
    assert manifest["questions_selected"] == 28
    assert {cell["arm"] for cell in manifest["plan"]} == set(STAGE_ARMS)
    assert manifest["decoding"]["seed_base"] == 42
    assert manifest["questions"][0]["seed"] == 42
    assert manifest["plan"][0]["seed"] == manifest["plan"][1]["seed"]
    assert manifest["snapshot"]["snapshot_modified"] is False
    assert manifest["frozen_pilot_unchanged"] is True
    assert not (tmp_path / "stage" / "arm_selection.json").exists()
    status = json.loads((tmp_path / "stage" / "cell_status.json").read_text(encoding="utf-8"))
    assert status["planned"] == 56
    assert status["completed"] == 0
    assert status["unsaved"] == 56
    assert status["failed"] == 0


def test_stage_plan_refuses_the_frozen_pilot_directory(tmp_path: Path):
    pilot = tmp_path / "pilot"
    pilot.mkdir()
    (pilot / "pilot_manifest.json").write_text(json.dumps({
        "experiment_id": PILOT_EXPERIMENT_ID,
        "execution_order": ["q_brex_price"],
    }), encoding="utf-8")
    bench = tmp_path / "bench.jsonl"
    _write_benchmark(bench, ["q_a", "q_b", "q_c", "q_d"])
    exclude = _exclude(tmp_path / "exclude.json", [])
    with pytest.raises(RuntimeError, match="frozen quick pilot"):
        write_stage_manifest(pilot, bench, str(tmp_path / "storage_expand_v1_snapshot"), exclude_manifest=exclude)


def test_resume_runs_only_missing_desc_and_rel_cells(tmp_path: Path):
    questions = [
        {"question_id": "q1", "query": "One?", "pilot_medium": "blog"},
        {"question_id": "q2", "query": "Two?", "pilot_medium": "x"},
    ]
    for question in questions:
        freeze_retrieval(tmp_path, question, {"data": {}})
    kept = write_generation(tmp_path, {
        "question_id": "q1",
        "arm": "F-REL",
        "medium": "blog",
        "answer": "kept",
    }).read_text(encoding="utf-8")
    called = []

    def generate_cell(question, arm, medium, payload):
        called.append((question["question_id"], arm, medium))
        return {"answer": "new", "retrieval_backend": "lightrag", "retrieval_mode": "hybrid"}

    report = run_pending(tmp_path, questions, generate_cell, arms=STAGE_ARMS)
    assert report["generated"] == 3
    assert report["skipped_complete"] == 1
    assert set(called) == {
        ("q1", "F-DESC", "blog"),
        ("q2", "F-REL", "x"),
        ("q2", "F-DESC", "x"),
    }
    assert "F-SOURCE" not in {arm for _question, arm, _medium in called}
    assert (tmp_path / "raw" / "generations" / "q1__F-REL__blog.json").read_text(encoding="utf-8") == kept
    assert not (tmp_path / "raw" / "generations" / "q1__F-SOURCE__blog.json").exists()


def _pair(directory: Path, question_id: str, query: str, medium: str, answers: dict, scored: dict, tokens: dict):
    manifest = {
        "experiment_id": STAGE_EXPERIMENT_ID,
        "arms": list(STAGE_ARMS),
        "execution_order": [question_id],
        "questions": [{
            "question_id": question_id,
            "query": query,
            "phenomenon": "factual",
            "pilot_medium": medium,
        }],
    }
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "stage2_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    cells = []
    for arm in STAGE_ARMS:
        write_generation(directory, {
            "question_id": question_id,
            "arm": arm,
            "medium": medium,
            "answer": answers[arm],
            "tokens": tokens[arm],
            "window_omitted": tokens[arm].get("omitted") or [],
            "latency": {"generation_seconds": 3.0},
        })
        cells.append({
            "question_id": question_id,
            "arm": arm,
            "medium": medium,
            "generation": scored[arm],
        })
    (directory / "deterministic_metrics.json").write_text(
        json.dumps({"cells": cells, "failures": []}),
        encoding="utf-8",
    )


def test_an_echo_does_not_win_and_extra_tokens_do_not_break_a_tie(tmp_path: Path):
    query = "What price did Capital One announce for acquiring Brex?"
    _pair(
        tmp_path / "echo",
        "q_price",
        query,
        "blog",
        {"F-DESC": query, "F-REL": "Capital One announced a price."},
        {
            "F-DESC": {"faithfulness": {"pass": True}, "copying": {"pass": True}, "citation_marker_leak": False, "numeric_pass": True},
            "F-REL": {"faithfulness": {"pass": True}, "copying": {"pass": True}, "citation_marker_leak": False, "numeric_pass": True},
        },
        {
            "F-DESC": {"prompt": 400, "context_by_channel": {"primary_source": 100, "derived_description": 80}},
            "F-REL": {"prompt": 300, "context_by_channel": {"primary_source": 100, "source_supported_relationship": 12}},
        },
    )
    echo = compare_stage(tmp_path / "echo")
    assert echo["status"] == "selected"
    assert echo["selected_arm"] == "F-REL"
    assert echo["echo_is_not_a_win"] is True
    assert not (tmp_path / "echo" / "arm_selection.json").exists()

    _pair(
        tmp_path / "tie",
        "q_tie",
        "Which company announced an acquisition of Brex?",
        "x",
        {"F-DESC": "Capital One announced the acquisition.", "F-REL": "The announcement names Capital One."},
        {
            "F-DESC": {"faithfulness": {"pass": True}, "copying": {"pass": True}, "citation_marker_leak": False, "numeric_pass": True},
            "F-REL": {"faithfulness": {"pass": True}, "copying": {"pass": True}, "citation_marker_leak": False, "numeric_pass": True},
        },
        {
            "F-DESC": {"prompt": 900, "context_by_channel": {"primary_source": 100, "derived_description": 200}, "omitted": ["desc:1"]},
            "F-REL": {"prompt": 200, "context_by_channel": {"primary_source": 100, "source_supported_relationship": 10}},
        },
    )
    tie = compare_stage(tmp_path / "tie")
    assert tie["status"] == "inconclusive"
    assert tie["selected_arm"] is None
    assert tie["context_tokens_are_not_a_score"] is True
    body = json.loads((tmp_path / "tie" / "desc_vs_rel_comparison.json").read_text(encoding="utf-8"))
    assert body["arms"]["F-DESC"]["mean_graph_context_tokens"] > body["arms"]["F-REL"]["mean_graph_context_tokens"]
    assert body["arms"]["F-DESC"]["window_omissions"] == 1


def test_a_factual_advantage_selects_the_leading_arm(tmp_path: Path):
    _pair(
        tmp_path / "lead",
        "q_lead",
        "Which company announced an acquisition of Brex?",
        "linkedin",
        {"F-DESC": "Capital One announced the acquisition.", "F-REL": "The draft adds a price of $999."},
        {
            "F-DESC": {"faithfulness": {"pass": True}, "copying": {"pass": True}, "citation_marker_leak": False, "numeric_pass": True},
            "F-REL": {"faithfulness": {"pass": False}, "copying": {"pass": True}, "citation_marker_leak": False, "numeric_pass": False, "unsupported_numeric_count": 1},
        },
        {
            "F-DESC": {"prompt": 100, "context_by_channel": {"derived_description": 4}},
            "F-REL": {"prompt": 100, "context_by_channel": {"source_supported_relationship": 4}},
        },
    )
    chosen = compare_stage(tmp_path / "lead")
    assert chosen["selected_arm"] == "F-DESC"
    assert chosen["status"] == "selected"
    body = json.loads((tmp_path / "lead" / "desc_vs_rel_comparison.json").read_text(encoding="utf-8"))
    assert body["per_medium"]["linkedin"]["wins"]["F-DESC"] == 1
    assert body["arms"]["F-REL"]["numeric_failure_rate"] == 1


def test_failed_stage2_questions_allow_a_verbatim_claim_and_reject_it_elsewhere():
    from host_finetune.evidence_capsule import (
        extraction_user_prompt,
        fit_extraction_hits,
        freeze_hits,
        unauthorized_raw_passage,
    )
    from host_finetune.lightrag_backend import read_catalog, read_failed_document_ids
    from host_finetune.model_f_context.pipeline import context_media_jobs
    from host_finetune.model_f_experiment.context_window import load_tokenizer, prompt_tokens
    from host_finetune.model_f_experiment.derive import arm_callback
    from host_finetune.model_f_experiment.retrieval import generation_payload

    root = Path("experiments/EXP-model-f-desc-vs-rel-32")
    tokenizer = load_tokenizer()
    catalog = read_catalog(root / "raw" / "lightrag_query" / "catalog.jsonl")
    failed_ids = read_failed_document_ids(root / "raw" / "lightrag_query")
    questions = {
        "q_investor_update": "talk",
        "q_dev_hop_update_and_cap_table": "x",
    }
    for question_id, medium_name in questions.items():
        stored = json.loads((root / "raw" / "retrieval" / f"{question_id}.json").read_text(encoding="utf-8"))
        payload = generation_payload(stored["payload"], catalog, failed_document_ids=failed_ids)
        hits = [hit for hit in payload.get("hits") or [] if (hit.get("text") or "").strip()]
        short = next(hit for hit in hits if hit.get("id") == "gd_4eba33d02cb019ab6f7191ce")
        passage = short["text"].strip()
        query = stored["query"]
        kept, omitted = fit_extraction_hits(
            query,
            freeze_hits(hits),
            lambda user: prompt_tokens(tokenizer, user),
        )
        assert omitted == []
        assert prompt_tokens(tokenizer, extraction_user_prompt(query, kept)) <= 2048
        claim = {
            "claim": passage,
            "source_id": short["id"],
            "support_span": passage,
            "claim_type": "fact",
        }
        for arm in STAGE_ARMS:
            prepared = arm_callback(
                arm,
                catalog=catalog,
                failed_document_ids=failed_ids,
                count_tokens=lambda text: len(tokenizer.encode(text, add_special_tokens=False)),
                window_tokens=lambda instruction: prompt_tokens(tokenizer, instruction),
            )({"topic": query}, payload)
            jobs = context_media_jobs(
                query,
                hits,
                [claim],
                "verified",
                prepared,
                count_tokens=prepared.get("count_tokens"),
                budget=prepared.get("experiment_budget"),
            )
            job = next(item for item in jobs if item["medium"]["medium"] == medium_name)
            assert passage in job["instruction"]
            before_claims = job["instruction"].split("PRIMARY SOURCE-GROUNDED CLAIMS", 1)[0]
            assert passage not in before_claims
        allowed = "PRIMARY SOURCE-GROUNDED CLAIMS\n\n[E1] " + passage + "\n\nFINAL WRITING INSTRUCTION\n"
        assert unauthorized_raw_passage(allowed, [short], [claim]) is None
        leaked = allowed + "\n" + passage
        reason = unauthorized_raw_passage(leaked, [short], [claim])
        assert reason is not None and "outside_verified_claims" in reason
        longer = next(hit for hit in hits if hit.get("id") != short["id"])
        long_text = longer["text"].strip()
        long_reason = unauthorized_raw_passage(allowed + long_text, hits, [claim])
        assert long_reason is not None and longer["id"] in long_reason
