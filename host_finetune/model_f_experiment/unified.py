"""Put one selected Model F profile on the existing 120-prompt panel.

The 36-question run chooses the profile. This module does not look at the
panel to make that choice. It does not query the live LightRAG store.
"""

from __future__ import annotations

import json
from pathlib import Path

from host_finetune.compare_adapters import MEDIUMS
from host_finetune.model_f_experiment.arms import ARMS
from host_finetune.model_f_experiment.snapshot import LIVE_STORE_NAME, SnapshotError

GOLD_PATH = Path("data/openai_ft/lemkin_gold_eval.jsonl")
DEV_PATH = Path("host_finetune/benchmarks/dev_retrieval_benchmark.jsonl")
PANEL_SIZE = 120
PROFILE_SOURCE = "36-question development experiment"
_STOP = {
    "about", "after", "again", "from", "have", "that", "this", "with", "your",
    "what", "when", "where", "which", "would", "should", "their", "there",
}


def load_gold_topics(path: Path = GOLD_PATH) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            rows.append({
                "id": str(row["id"]),
                "topic": str(row["topic"]),
                "audience": row.get("audience") or "",
                "goal": row.get("goal") or "",
                "expected_facts": list(row.get("expected_facts") or []),
                "rubric_must_haves": list(row.get("rubric_must_haves") or []),
                "gold_reference": row.get("gold_reference") or "",
            })
    if len(rows) != 30:
        raise ValueError(f"expected 30 gold topics, found {len(rows)}")
    if len({row["id"] for row in rows}) != 30:
        raise ValueError("gold topic ids are not unique")
    return rows


def panel_cells(topics: list[dict] | None = None) -> list[dict]:
    """Thirty topics times blog, LinkedIn, X, and talk."""
    chosen = topics if topics is not None else load_gold_topics()
    cells = []
    for index, topic in enumerate(chosen):
        for medium in MEDIUMS:
            cells.append({
                "id": topic["id"],
                "topic_index": index,
                "topic": topic["topic"],
                "query": topic["topic"],
                "medium": medium["medium"],
                "format": medium["format"],
                "max_new_tokens": medium["max_new_tokens"],
                "audience": topic["audience"],
                "goal": topic["goal"],
                "expected_facts": list(topic["expected_facts"]),
                "rubric_must_haves": list(topic["rubric_must_haves"]),
                "gold_reference": topic["gold_reference"],
            })
    if len(cells) != PANEL_SIZE:
        raise ValueError(f"expected {PANEL_SIZE} panel cells, found {len(cells)}")
    return cells


def require_profile(profile: str) -> str:
    """The caller names the profile. This panel does not select one."""
    if profile not in ARMS:
        raise ValueError(
            f"profile {profile!r} is not one of {list(ARMS)}. "
            "Choose it from the 36-question experiment before this panel."
        )
    return profile


def _terms(text: str) -> set[str]:
    return {
        token for token in "".join(char.lower() if char.isalnum() else " " for char in text).split()
        if len(token) >= 5 and token not in _STOP
    }


def panel_overlap(dev_questions: list[dict], topics: list[dict]) -> dict:
    """Record shared ids and wording. A shared word is not a shared task."""
    dev_ids = {str(row["question_id"]) for row in dev_questions}
    gold_ids = {row["id"] for row in topics}
    pairs = []
    single = []
    for question in dev_questions:
        left = _terms(str(question.get("query") or ""))
        for topic in topics:
            right = _terms(topic["topic"])
            shared = sorted(left & right)
            item = {
                "question_id": question["question_id"],
                "gold_id": topic["id"],
                "shared_terms": shared,
            }
            if len(shared) >= 2:
                pairs.append(item)
            elif len(shared) == 1:
                single.append(item)
    return {
        "shared_ids": sorted(dev_ids & gold_ids),
        "dev_questions": len(dev_questions),
        "gold_topics": len(topics),
        "wording_overlaps": pairs,
        "single_term_overlaps": single,
        "same_task": False,
        "note": (
            "Development questions ask for a retrieved fact. Gold topics ask for a Lemkin-style draft. "
            "Shared wording does not make the 120-prompt panel an untouched test, and it does not "
            "make the development set a substitute for that panel."
        ),
    }


def trace_input(cell: dict, gold: dict) -> dict:
    """One saved draft in the shape ``evaluate_row`` already accepts."""
    hits = []
    for item in cell.get("evidence") or []:
        if isinstance(item, dict) and (item.get("text") or "").strip():
            hits.append({"text": item["text"]})
    return {
        "id": gold["id"],
        "medium": cell["medium"],
        "format": cell.get("format") or gold.get("format") or "",
        "topic": gold["topic"],
        "query": gold["topic"],
        "audience": gold.get("audience") or "",
        "goal": gold.get("goal") or "",
        "expected_facts": list(gold.get("expected_facts") or []),
        "rubric_must_haves": list(gold.get("rubric_must_haves") or []),
        "gold_reference": gold.get("gold_reference") or "",
        "answer": cell.get("answer") or "",
        "prompt": cell.get("prompt") or "",
        "retrieval": True,
        "retrieval_hits": hits,
        "latency_seconds": (cell.get("latency") or {}).get("generation_seconds"),
    }


def score_draft(cell: dict, gold: dict, pack: dict, copy_index, specs: dict) -> dict:
    """Deterministic gates only. Judge fields stay absent until the overlay runs."""
    from host_finetune.unified_eval import evaluate_row

    row = evaluate_row(trace_input(cell, gold), pack, copy_index, specs)
    row["model"] = cell.get("profile")
    row["source_experiment"] = cell.get("experiment_id")
    row["judge_overlay"] = "not_applied"
    row["historical_scores"] = None
    if row.get("persona_utility") is None:
        row["persona_utility_reason"] = "not_scored_until_the_same_judge_overlay_is_applied"
    return row


def _gate_rate(rows: list[dict], name: str) -> dict:
    """A gate that was not scored stays missing. A failure is not imputed as zero."""
    known = []
    for row in rows:
        gate = ((row.get("gates") or {}).get(name) or {})
        if "pass" not in gate:
            continue
        if gate.get("available") is False:
            continue
        if name == "grounding" and gate.get("applicable") is False:
            continue
        known.append(bool(gate["pass"]))
    passed = sum(known)
    return {
        "pass_rate": round(passed / len(known), 6) if known else None,
        "n": len(known),
        "failures": (len(known) - passed) if known else None,
        "missing": len(rows) - len(known),
    }


def comparison_matrix(arms: dict[str, list[dict]]) -> dict:
    """Means by architecture and medium. A missing number stays null."""
    from eval_tools.rows import METRIC_PATHS, metric_value

    media = ("all", "blog", "linkedin", "x", "talk")
    table = []
    for name, rows in arms.items():
        for medium in media:
            chosen = [row for row in rows if medium == "all" or row.get("medium") == medium]
            metrics = {}
            for metric in (
                "writing_quality", "task_adherence", "stance_consistency", "style_distance",
                "persona_utility", "claim_support", "word_count", "latency_seconds",
            ):
                if metric not in METRIC_PATHS:
                    metrics[metric] = None
                    continue
                values = [metric_value(row, metric) for row in chosen]
                present = [value for value in values if value is not None]
                metrics[metric] = {
                    "mean": round(sum(present) / len(present), 6) if present else None,
                    "n": len(present),
                    "missing": len(chosen) - len(present),
                }
            eligible = sum(1 for row in chosen if row.get("eligible_for_ranking") is True)
            table.append({
                "architecture": name,
                "medium": medium,
                "drafts": len(chosen),
                "eligible": eligible,
                "gates": {
                    name: _gate_rate(chosen, name)
                    for name in ("completion", "format", "copying_memorization", "grounding")
                },
                "metrics": metrics,
            })
    return {"rows": table, "composite_score": None, "rank": None}


def assert_unified_snapshot(path: Path) -> None:
    if LIVE_STORE_NAME in Path(path).resolve().parts:
        raise SnapshotError("the 120-prompt run must use a verified snapshot, not the live store")


UNIFIED_EXPERIMENT_ID = "EXP-20261102-001-model-f-unified-panel"


def panel_experiment_id(directory: Path) -> str:
    """Use a versioned output directory name when this panel is a separate run.

    The original F-REL directory keeps its existing id. A later directory such
    as the F-SOURCE panel records its own name and does not rewrite that run.
    """
    name = Path(directory).name
    if name.startswith("EXP-"):
        return name
    return UNIFIED_EXPERIMENT_ID


GOLD_SHA256 = "d7c4f997899e28f28eb3ea9aa3417857da6f0666acbf72fd8ac31422034e520e"
JUDGE = {
    "model": "qwen2.5:14b",
    "temperature": 0,
    "blinded": True,
    "candidate_family": "llama",
    "overlay": "host_finetune.unified_eval.apply_judge_overlay",
}
# These directories already hold measured results. The panel writes a new one.
PROTECTED_EXPERIMENTS = (
    "EXP-20261001-010",
    "EXP-20261004-003",
    "EXP-20261004-005",
    "EXP-20261004-006",
    "EXP-20261004-008",
    "EXP-20261005-003",
    "EXP-20261030-012",
    "EXP-20261031-001",
    "EXP-20261101-008",
    "EXP-20261101-009",
)
SAVED_ROWS = {
    "Base": Path("docs/evaluation/examples/data/architecture/kevin_base_llama31_rows.jsonl"),
    "PersonaRAG": Path("docs/evaluation/examples/data/architecture/colab_personarag_seed42_rows.jsonl"),
    "Relabel RAG off": Path("docs/evaluation/examples/data/relabel_rag_off_rows.jsonl"),
    "Relabel + RAG": Path("docs/evaluation/examples/data/relabel_rag_on_rows.jsonl"),
}


def file_sha256(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def assert_writable_panel(directory: Path) -> None:
    parts = set(Path(directory).parts)
    blocked = parts & set(PROTECTED_EXPERIMENTS)
    if blocked:
        raise SnapshotError(
            f"refusing to write the 120-prompt panel into {sorted(blocked)}; "
            "those directories keep their historical results"
        )


def load_dev_questions(path: Path = DEV_PATH) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def record_overlap(
    dev_path: Path = DEV_PATH,
    gold_path: Path = GOLD_PATH,
) -> dict:
    report = panel_overlap(load_dev_questions(dev_path), load_gold_topics(gold_path))
    report["gold_sha256"] = file_sha256(gold_path)
    report["profile_selected_on"] = "36-question development experiment"
    report["panel_used_for_profile_selection"] = False
    return report


def load_frozen_arm(path: Path) -> dict:
    """Read the pilot freeze. Scoring the 120-prompt panel does not rewrite it."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("frozen") is not True:
        raise ValueError(f"arm selection is not frozen: {path}")
    if payload.get("do_not_revise_after_unified_eval") is not True:
        raise ValueError(f"arm selection is not locked against the 120-prompt scores: {path}")
    arm = payload.get("selected_arm")
    if arm not in ARMS:
        raise ValueError(f"frozen arm {arm!r} is not a Model F profile")
    return payload


def assert_frozen_selection(profile: str, selection: Path | None) -> None:
    if selection is None:
        return
    frozen = load_frozen_arm(selection)
    if frozen["selected_arm"] != profile:
        raise ValueError(
            f"profile {profile!r} does not match the frozen pilot arm {frozen['selected_arm']!r}"
        )


def write_plan(directory: Path, profile: str, snapshot: Path, selection: Path | None = None) -> dict:
    """Record the panel. This does not retrieve, generate, or score."""
    profile = require_profile(profile)
    snapshot = Path(snapshot)
    directory = Path(directory)
    assert_writable_panel(directory)
    assert_frozen_selection(profile, selection)
    assert_unified_snapshot(snapshot)
    topics = load_gold_topics()
    cells = panel_cells(topics)
    gold_sha = file_sha256(GOLD_PATH)
    if gold_sha != GOLD_SHA256:
        raise ValueError("the gold file hash does not match the frozen 120-prompt panel")
    plan = {
        "experiment_id": panel_experiment_id(directory),
        "protocol": "unified_eval.v1",
        "status": "prepared_not_run",
        "profile": profile,
        "profile_source": PROFILE_SOURCE,
        "profile_selected_by_this_panel": False,
        "gold_path": str(GOLD_PATH),
        "gold_sha256": gold_sha,
        "topic_ids": [topic["id"] for topic in topics],
        "media": [item["medium"] for item in MEDIUMS],
        "cells": len(cells),
        "adapter": "host_finetune/output/lemkin_lora_relabel",
        "base_model": "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit",
        "quantization": "4bit",
        "decoding": {
            "temperature": 0.7,
            "top_p": 0.9,
            "repetition_penalty": 1.15,
            "seed_base": 42,
            "seed": "42 + gold_index * 4 + medium_index",
        },
        "retrieval": {
            "backend": "lightrag",
            "mode": "hybrid",
            "evidence_budget": 4,
            "top_k": 20,
            "num_ctx": 16384,
            "snapshot": str(snapshot),
            "live_store_queried": False,
        },
        "prompt_family": "model-f-graph-context-v1",
        "max_seq_length": 2048,
        "judge": dict(JUDGE),
        "judge_overlay": "not_applied",
        "four_arm_experiment": "EXP-20261101-009-model-f-four-arm",
        "four_arm_unchanged": True,
        "frozen_selection": None if selection is None else str(selection),
        "outputs": {
            "raw": "raw/",
            "unified_eval": "unified_eval/",
            "diagnostics": "diagnostics/",
        },
    }
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "unified_plan.json").write_text(
        json.dumps(plan, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (directory / "generation_manifest.json").write_text(
        json.dumps(plan, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return plan


def score_panel(directory: Path, profile: str, pack: dict, copy_index, specs: dict) -> dict:
    """Score saved cells with ``evaluate_row``. The judge overlay is a later step."""
    profile = require_profile(profile)
    directory = Path(directory)
    assert_writable_panel(directory)
    from host_finetune.model_f_experiment.runner import MEDIA, cell_path, is_complete

    topics = load_gold_topics()
    rows = []
    missing = []
    for topic in topics:
        for medium in MEDIA:
            path = cell_path(directory, topic["id"], profile, medium)
            if not is_complete(path):
                missing.append({"id": topic["id"], "medium": medium})
                continue
            cell = json.loads(path.read_text(encoding="utf-8"))
            cell["profile"] = profile
            cell["medium"] = medium
            cell["experiment_id"] = UNIFIED_EXPERIMENT_ID
            rows.append(score_draft(cell, topic, pack, copy_index, specs))
    metrics = directory / "metrics"
    metrics.mkdir(parents=True, exist_ok=True)
    (metrics / "rows.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    coverage = {
        "schema": "unified_eval.v1",
        "scored": len(rows),
        "missing": missing,
        "judge_overlay": "not_applied",
        "missing_metrics_are_zero": False,
    }
    (metrics / "coverage.json").write_text(
        json.dumps(coverage, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    official = directory / "unified_eval"
    official.mkdir(parents=True, exist_ok=True)
    (official / "rows.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    (official / "coverage.json").write_text(
        json.dumps(coverage, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    from host_finetune.model_f_experiment.diagnostics import write_model_f_reports

    write_model_f_reports(directory, directory / "diagnostics", UNIFIED_EXPERIMENT_ID)
    return coverage


def saved_architecture_matrix() -> dict:
    """Means of the saved 120-row files. Model F is absent until its panel is run."""
    from eval_tools.rows import load_rows

    arms = {}
    for name, path in SAVED_ROWS.items():
        if path.is_file():
            arms[name] = load_rows(path)
    report = comparison_matrix(arms)
    report["model_f"] = None
    report["model_f_reason"] = "no 120-prompt drafts have been generated"
    report["source"] = "saved unified_eval.v1 rows; not a new measurement"
    report["paired_intervals"] = "eval_tools.compare on the same row files"
    return report


def score_saved_panel(directory: Path, profile: str) -> dict:
    """Deterministic ``evaluate_row`` pass. It does not call the writing judge."""
    require_profile(profile)
    directory = Path(directory)
    assert_writable_panel(directory)
    from host_finetune.eval_gold_rag import load_format_specs
    from host_finetune.rescore_unified import base_controls
    from host_finetune.score_saved_experiment import SPECS
    from host_finetune.train_only_index import (
        DEFAULT_ASSIGNMENTS,
        DEFAULT_DATASET,
        DEFAULT_GOLD_DISPOSITIONS,
        load_jsonl,
    )
    from host_finetune.unified_eval import CopyIndex, build_reference_pack
    from host_finetune.voice_distance import iter_train_texts

    dataset = load_jsonl(DEFAULT_DATASET)
    assignments = load_jsonl(DEFAULT_ASSIGNMENTS)
    dispositions = load_jsonl(DEFAULT_GOLD_DISPOSITIONS)
    pack = build_reference_pack(dataset, assignments, dispositions, base_controls())
    return score_panel(
        directory,
        profile,
        pack,
        CopyIndex(iter_train_texts(dataset, assignments)),
        load_format_specs(SPECS),
    )


def record_posthoc_arm(directory: Path, profile: str) -> dict:
    """Record an engineering choice without revising the frozen pilot or stage 2.

    The 120-prompt panel is not an input to this choice. The frozen pilot file
    and the stage-2 recommendation are not opened or rewritten.
    """
    profile = require_profile(profile)
    directory = Path(directory)
    assert_writable_panel(directory)
    payload = {
        "artifact": "posthoc_engineering_selection",
        "selected_arm": profile,
        "frozen": False,
        "uses_120_prompt_panel_for_selection": False,
        "does_not_replace": "experiments/EXP-model-f-quick-pilot/arm_selection.json",
        "does_not_modify": "experiments/EXP-model-f-desc-vs-rel-32/desc_vs_rel_recommendation.json",
        "frozen_pilot_arm": "F-SOURCE",
        "frozen_pilot_unchanged": True,
        "later_engineering_selection": "F-REL",
        "later_engineering_selection_directory": "experiments/EXP-20261102-001-model-f-unified-panel",
        "this_run_profile": profile,
        "stage2_recommendation_unchanged": True,
        "reason": (
            f"This directory runs {profile}. "
            "The frozen pilot selection remains F-SOURCE. "
            "The later engineering selection remains F-REL. "
            "This panel does not revise either decision."
        ),
    }
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "posthoc_arm_selection.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return payload


def panel_status(directory: Path, profile: str) -> dict:
    """Count saved cells. This does not score or judge."""
    from host_finetune.model_f_experiment.runner import MEDIA, cell_path, is_complete

    profile = require_profile(profile)
    directory = Path(directory)
    topics = load_gold_topics()
    completed = []
    open_cells = []
    for topic in topics:
        for medium in MEDIA:
            item = {"id": topic["id"], "medium": medium}
            if is_complete(cell_path(directory, topic["id"], profile, medium)):
                completed.append(item)
            else:
                open_cells.append(item)
    reasons: dict[tuple[str, str], str] = {}
    failure_path = directory / "raw" / "failures.jsonl"
    if failure_path.is_file():
        for line in failure_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("arm") not in (None, profile):
                continue
            reason = str(row.get("reason") or row.get("error") or "unspecified")
            for medium in row.get("media") or []:
                reasons[(str(row.get("question_id")), str(medium))] = reason
    failed = [
        {**item, "reason": reasons[(item["id"], item["medium"])]}
        for item in open_cells
        if (item["id"], item["medium"]) in reasons
    ]
    progress = {}
    progress_path = directory / "progress.json"
    if progress_path.is_file():
        progress = json.loads(progress_path.read_text(encoding="utf-8"))
    return {
        "profile": profile,
        "cells": len(topics) * len(MEDIA),
        "completed": len(completed),
        "failed": len(failed),
        "remaining": len(open_cells),
        "not_started": len(open_cells) - len(failed),
        "current_question": progress.get("current_question"),
        "failed_cells": failed,
        "judge_overlay": "not_applied",
    }


def write_progress(directory: Path, profile: str, question_id: str | None, index: int, total: int) -> dict:
    status = panel_status(directory, profile)
    payload = {
        "profile": profile,
        "current_question": question_id,
        "question_index": index,
        "questions": total,
        "completed": status["completed"],
        "failed": status["failed"],
        "remaining": status["remaining"],
        "not_started": status["not_started"],
        "judge_overlay": "not_applied",
    }
    Path(directory).mkdir(parents=True, exist_ok=True)
    (Path(directory) / "progress.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return payload


def execute_selected_profile(
    directory: Path,
    profile: str,
    snapshot: Path,
    selection: Path | None = None,
) -> dict:
    """Generate the 120 cells for one profile already chosen on the 36-question set.

    Tests do not call this. It refuses the live store before it loads a model.
    """
    profile = require_profile(profile)
    snapshot = Path(snapshot)
    directory = Path(directory)
    write_plan(directory, profile, snapshot, selection)
    record_posthoc_arm(directory, profile)
    from host_finetune.model_f_experiment.snapshot import assert_snapshot_path, verify_snapshot

    assert_snapshot_path(snapshot)
    verify_snapshot(snapshot)
    topics = load_gold_topics()
    questions = [
        {"question_id": topic["id"], "query": topic["topic"], "abstention_expected": False}
        for topic in topics
    ]
    import asyncio
    import time

    from host_finetune.compare_adapters import generate, load_generator
    from host_finetune.lightrag_backend import read_catalog, read_failed_document_ids
    from host_finetune.model_f_experiment.arms import ADAPTER, DECODING
    from host_finetune.model_f_experiment.context_window import assert_preflight
    from host_finetune.model_f_experiment.derive import arm_callback
    from host_finetune.model_f_experiment.diagnostics import summarize_graph
    from host_finetune.model_f_experiment.guards import require_lightrag_trace
    from host_finetune.model_f_experiment.retrieval import (
        generation_payload,
        prepare_query_workdir,
        retrieve_raw,
    )
    from host_finetune.model_f_experiment.runner import (
        MEDIA,
        cell_path,
        cell_record,
        freeze_evidence,
        freeze_retrieval,
        is_complete,
        load_evidence,
        note_unsaved,
        retrieval_path,
    )
    from host_finetune.model_f_experiment.snapshot import SnapshotError as Refused

    assert_preflight(questions)
    workdir = prepare_query_workdir(snapshot, directory / "raw" / "lightrag_query")
    catalog = read_catalog(workdir / "catalog.jsonl")
    failed_ids = read_failed_document_ids(workdir)
    if not catalog:
        raise Refused(f"snapshot catalog is empty: {snapshot}")
    model, tokenizer, load_report = load_generator(Path(ADAPTER), "4bit")

    def count_tokens(text: str) -> int:
        return len(tokenizer.encode(text, add_special_tokens=False))

    def window_tokens(instruction: str) -> int:
        from host_finetune.model_f_experiment.context_window import prompt_tokens

        return prompt_tokens(tokenizer, instruction)

    evidence_cache: dict = {}
    generated = 0
    unsaved: list[dict] = []
    try:
        for index, topic in enumerate(topics):
            question = questions[index]
            progress = write_progress(directory, profile, question["question_id"], index + 1, len(topics))
            print(
                f"question {index + 1}/{len(topics)} {question['question_id']} "
                f"completed {progress['completed']} failed {progress['failed']} "
                f"remaining {progress['remaining']}",
                flush=True,
            )
            stored = retrieval_path(directory, question["question_id"])
            if not is_complete(stored):
                started = time.perf_counter()
                raw = asyncio.run(retrieve_raw(question["query"], workdir))
                retrieval_seconds = round(time.perf_counter() - started, 3)
                payload = generation_payload(raw, catalog, failed_document_ids=failed_ids)
                try:
                    summary = summarize_graph(raw, payload, catalog, failed_ids, question["query"])
                except Exception as exc:
                    summary = {"status": "not_recorded", "reason": f"{type(exc).__name__}: {exc}"}
                freeze_retrieval(
                    directory,
                    question,
                    raw,
                    retrieval_seconds=retrieval_seconds,
                    graph_summary=summary,
                )
            else:
                stored_body = json.loads(stored.read_text(encoding="utf-8"))
                raw = stored_body["payload"]
                retrieval_seconds = stored_body.get("retrieval_seconds")
                payload = generation_payload(raw, catalog, failed_document_ids=failed_ids)
            question = dict(question)
            question["retrieval_seconds"] = retrieval_seconds

            def retriever(topic_text: str, comparison_medium: str, payload=payload) -> dict:
                return payload

            gold = [{"id": topic["id"], "topic": topic["topic"]}]
            saved = load_evidence(directory, question["question_id"])
            if saved is not None:
                evidence_cache[question["question_id"]] = saved
            missing = [
                medium
                for medium in MEDIA
                if not is_complete(cell_path(directory, question["question_id"], profile, medium))
            ]
            if not missing:
                continue

            def on_trace(trace: dict, question=question) -> None:
                require_lightrag_trace(trace, profile)
                freeze_evidence(directory, question["question_id"], trace)
                write_record = cell_record(question, profile, trace)
                from host_finetune.model_f_experiment.runner import write_generation

                write_generation(directory, write_record)

            try:
                generate(
                    Path(ADAPTER),
                    gold,
                    retriever=retriever,
                    seed_base=DECODING["seed_base"],
                    topic_rows=[(index, gold[0])],
                    loaded=(model, tokenizer, load_report),
                    unload=False,
                    evidence_cache=evidence_cache,
                    media_allow=set(missing),
                    graph_context_for=arm_callback(
                        profile,
                        catalog=catalog,
                        failed_document_ids=failed_ids,
                        count_tokens=count_tokens,
                        window_tokens=window_tokens,
                    ),
                    on_trace=on_trace,
                )
            except Exception as exc:
                still_missing = [
                    medium
                    for medium in missing
                    if not is_complete(cell_path(directory, question["question_id"], profile, medium))
                ]
                unsaved.append(note_unsaved(
                    directory,
                    question["question_id"],
                    profile,
                    still_missing or list(missing),
                    f"{type(exc).__name__}: {exc}",
                ))
                continue
            for medium in missing:
                if is_complete(cell_path(directory, question["question_id"], profile, medium)):
                    generated += 1
                    continue
                unsaved.append(note_unsaved(
                    directory,
                    question["question_id"],
                    profile,
                    [medium],
                    "generation finished without a complete cell",
                ))
    finally:
        del model
        del tokenizer
    write_progress(directory, profile, None, len(topics), len(topics))
    plan_path = directory / "unified_plan.json"
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    plan["status"] = "generated_not_judged"
    plan["generated_cells"] = generated
    plan["model_load"] = load_report
    plan["posthoc_engineering_selection"] = "posthoc_arm_selection.json"
    plan["graph_window_policy"] = (
        "whole graph items are omitted until the chat prompt fits the 2048 window; "
        "omitted ids are stored on window_omitted"
    )
    plan["extraction_input_policy"] = (
        "whole trailing passages are omitted until the extraction prompt fits 2048; "
        "a prompt that still does not fit is refused. Passage text is not truncated. "
        "Stage-2 development prompts were allowed to exceed 2048, so those cells are not mixed into this panel."
    )
    encoded = json.dumps(plan, indent=2, sort_keys=True) + "\n"
    plan_path.write_text(encoded, encoding="utf-8")
    (directory / "generation_manifest.json").write_text(encoded, encoding="utf-8")
    status = panel_status(directory, profile)
    return {
        "generated": generated,
        "completed": status["completed"],
        "failed": status["failed"],
        "remaining": status["remaining"],
        "unsaved": unsaved,
        "judge_overlay": "not_applied",
    }


def collect_judge_traces(directory: Path, profile: str) -> tuple[list[dict], list[dict]]:
    """Saved drafts in the shape ``evaluate_row`` already accepts."""
    from host_finetune.model_f_experiment.runner import MEDIA, cell_path, is_complete

    profile = require_profile(profile)
    topics = load_gold_topics()
    traces = []
    missing = []
    for topic in topics:
        for medium in MEDIA:
            path = cell_path(directory, topic["id"], profile, medium)
            if not is_complete(path):
                missing.append({"id": topic["id"], "medium": medium})
                continue
            cell = json.loads(path.read_text(encoding="utf-8"))
            cell["medium"] = medium
            cell["profile"] = profile
            traces.append(trace_input(cell, topic))
    return traces, missing


def judge_saved_panel(directory: Path, profile: str) -> dict:
    """Run the original Qwen overlay on 120 saved cells. Historical files stay put.

    This does not call ``score_saved_experiment``. That command is pinned to the
    October 5 Relabel generation hash and must keep refusing every other file.
    """
    import os
    import platform
    import subprocess
    import sys
    from datetime import datetime, timezone

    profile = require_profile(profile)
    directory = Path(directory)
    assert_writable_panel(directory)
    experiment_id = panel_experiment_id(directory)
    plan_path = directory / "unified_plan.json"
    if plan_path.is_file():
        recorded = json.loads(plan_path.read_text(encoding="utf-8")).get("experiment_id")
        if recorded:
            experiment_id = str(recorded)
    traces, missing = collect_judge_traces(directory, profile)
    if missing or len(traces) != PANEL_SIZE:
        raise RuntimeError(
            f"refusing to judge: {len(traces)} complete cells and {len(missing)} missing; "
            "unified_eval.v1 scores 120 cells"
        )
    from host_finetune.claim_judge import (
        EvidenceIndex,
        claim_prompt,
        evidence_documents,
        evidence_report,
        parse_claims,
        rag_grounding_report,
        stance_report,
    )
    from host_finetune.eval_gold_rag import load_format_specs
    from host_finetune.judge_client import DEFAULT_JUDGE_MODEL, judge_family
    from host_finetune.rescore_evidence import (
        calibration_samples,
        judge_json,
        load_cache,
        passages_for,
        stance_control,
        writing_record,
    )
    from host_finetune.rescore_unified import base_controls, sha256
    from host_finetune.score_saved_experiment import SPECS
    from host_finetune.train_only_index import (
        DEFAULT_ASSIGNMENTS,
        DEFAULT_DATASET,
        DEFAULT_GOLD_DISPOSITIONS,
        load_jsonl,
    )
    from host_finetune.unified_eval import (
        SCHEMA,
        UTILITY_WEIGHTS,
        CopyIndex,
        apply_judge_overlay,
        build_reference_pack,
        evaluate_row,
        summarize_model,
    )
    from host_finetune.voice_distance import iter_train_texts
    from host_finetune.writing_quality import writing_prompt, writing_quality_gate

    official = directory / "unified_eval"
    official.mkdir(parents=True, exist_ok=True)
    trace_path = directory / "raw" / "judge_inputs.jsonl"
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    trace_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in traces),
        encoding="utf-8",
    )
    generation_sha = sha256(trace_path)
    model = DEFAULT_JUDGE_MODEL
    family = judge_family(model)
    base_url = os.environ.get("OLLAMA_BASE", "http://127.0.0.1:11434")
    cache_path = official / "judge_cache.jsonl"
    cache = load_cache(cache_path)
    dataset = load_jsonl(DEFAULT_DATASET)
    assignments = load_jsonl(DEFAULT_ASSIGNMENTS)
    dispositions = load_jsonl(DEFAULT_GOLD_DISPOSITIONS)
    documents = evidence_documents(dataset, assignments, dispositions)
    index = EvidenceIndex(documents)
    samples = calibration_samples(documents, model, base_url, cache, cache_path)
    writing_gate = writing_quality_gate(samples)
    writing_gate["judge_model"] = model
    writing_gate["judge_family"] = family
    writing_gate["candidate_family"] = "llama"
    stance_gate = stance_control(index, model, base_url, cache, cache_path)
    pack = build_reference_pack(dataset, assignments, dispositions, base_controls())
    pack["writing_quality_calibration"] = {
        "calibrated": writing_gate["calibrated"],
        "role": writing_gate["role"],
    }
    copy_index = CopyIndex(iter_train_texts(dataset, assignments))
    specs = load_format_specs(SPECS)
    rows_path = official / "rows.jsonl"
    finished = {}
    if rows_path.is_file():
        kept = []
        for line in rows_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            previous = json.loads(line)
            if previous.get("judge_overlay") == "not_applied":
                continue
            finished[(previous.get("id"), previous.get("medium"))] = previous
            kept.append(line)
        rows_path.write_text("\n".join(kept) + ("\n" if kept else ""), encoding="utf-8")
    scored_rows = []
    with rows_path.open("a", encoding="utf-8") as handle:
        for trace in traces:
            key = (trace.get("id"), trace.get("medium"))
            prior = finished.get(key)
            if prior is not None:
                scored_rows.append(prior)
                continue
            base = evaluate_row(trace, pack, copy_index, specs)
            draft = (trace.get("answer") or "")[:1800]
            topic = trace.get("topic") or ""
            medium = trace.get("medium") or ""
            writing_parsed = judge_json(
                writing_prompt(draft, topic, medium),
                model, base_url, 220, cache, cache_path, "writing",
            )
            retrieved = passages_for(trace)
            evidence_hits = index.search(f"{topic}\n{draft[:900]}", medium, k=4)
            evidence_text = [hit["text"][:650] for hit in evidence_hits]
            claim_parsed = judge_json(
                claim_prompt(draft, topic, retrieved, evidence_text),
                model, base_url, 520, cache, cache_path, "claims",
            )
            claims = parse_claims(claim_parsed, retrieved is not None)
            writing = writing_record(writing_parsed, writing_gate["role"], writing_gate["calibrated"], model)
            row = apply_judge_overlay(
                base,
                writing,
                rag_grounding_report(claims, retrieved is not None),
                evidence_report(claims, len(evidence_text)),
                stance_report(claims),
                writing_role=writing_gate["role"],
                writing_calibrated=writing_gate["calibrated"],
                stance_role=stance_gate["role"],
            )
            row["model"] = profile
            row["source_experiment"] = experiment_id
            row["generation_sha256"] = generation_sha
            row["schema"] = SCHEMA
            scored_rows.append(row)
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            handle.flush()
    (directory / "metrics").mkdir(parents=True, exist_ok=True)
    (directory / "metrics" / "rows.jsonl").write_text(rows_path.read_text(encoding="utf-8"), encoding="utf-8")
    summary = summarize_model(scored_rows, f"{experiment_id}/{profile}", experiment_id)
    payload = {
        "schema": SCHEMA,
        "utility_weights": UTILITY_WEIGHTS,
        "judge_model": model,
        "judge_family": family,
        "candidate_family": "llama",
        "temperature": 0,
        "blinded": True,
        "generation_sha256": generation_sha,
        "n_traces": len(traces),
        "writing_quality_calibration": {
            "calibrated": writing_gate["calibrated"],
            "role": writing_gate["role"],
            "reason": writing_gate["reason"],
        },
        "stance_calibration": stance_gate,
        "model": summary,
    }
    (official / "summary.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=Path.cwd(), capture_output=True, text=True, check=False)
    dirty = subprocess.run(["git", "status", "--porcelain"], cwd=Path.cwd(), capture_output=True, text=True, check=False)
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "command": " ".join(sys.argv),
        "git": {"commit": commit.stdout.strip(), "dirty": bool(dirty.stdout.strip())},
        "judge_model": model,
        "judge_config": {"family": family, "temperature": 0, "blinded": True, "candidate_family": "llama"},
        "generation_sha256": generation_sha,
        "python": platform.python_version(),
        "schema": SCHEMA,
        "utility_weights": UTILITY_WEIGHTS,
    }
    (official / "evaluation_run.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    from eval_tools.rows import load_rows

    arms = {}
    for name, path in SAVED_ROWS.items():
        if path.is_file():
            arms[name] = load_rows(path)
    arms["Model F"] = scored_rows
    matrix = comparison_matrix(arms)
    matrix["schema"] = SCHEMA
    matrix["judge_model"] = model
    matrix["rows_excluded"] = []
    matrix["reweighted"] = False
    (official / "comparison.json").write_text(
        json.dumps(matrix, indent=2) + "\n",
        encoding="utf-8",
    )
    return {"schema": SCHEMA, "scored": len(scored_rows), "judge_model": model, "comparison_rows": len(matrix["rows"])}
