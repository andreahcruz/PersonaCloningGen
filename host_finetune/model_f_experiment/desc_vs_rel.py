"""Second-stage comparison of F-DESC and F-REL.

The quick pilot stays frozen. This module does not generate F-SOURCE or
F-HOP, does not open the gold-eval file as a question source, and does not
duplicate a development question to fill a medium quota.
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

from host_finetune.compare_adapters import MEDIUMS
from host_finetune.model_f_experiment.arms import ADAPTER, DECODING, PREPARE_PROFILE, PROMPT_FAMILY, RETRIEVAL
from host_finetune.model_f_experiment.diagnostics import write_model_f_reports
from host_finetune.model_f_experiment.pilot import (
    PILOT_EXPERIMENT_ID,
    _sha256,
    plan_cells,
    questions_from_manifest,
)
from host_finetune.model_f_experiment.runner import (
    cell_path,
    is_complete,
    load_questions,
    media_for_question,
)
from host_finetune.model_f_experiment.snapshot import assert_snapshot_path
from host_finetune.model_f_experiment.unified import GOLD_PATH

STAGE_EXPERIMENT_ID = "EXP-20261102-003-model-f-desc-vs-rel"
STAGE_ARMS = ("F-REL", "F-DESC")
MEDIA = ("blog", "linkedin", "x", "talk")
REQUESTED_PER_MEDIUM = 8
FROZEN_PILOT_MANIFEST = Path("experiments/EXP-model-f-quick-pilot/pilot_manifest.json")
MANIFEST_NAME = "stage2_manifest.json"

COMPARISON_RULE = (
    "Compare only questions where F-REL and F-DESC are both complete. "
    "Primary key: echo-adjusted faithfulness. A cell counts as a pass only when "
    "the existing deterministic faithfulness check passes and the saved answer is "
    "not a question echo. Echoes use repeats_topic and embeds_topic and are not "
    "a factual success. "
    "Secondary key: source-copy or citation-leak failure rate from the existing scorer. "
    "An arm wins only when its primary rate is strictly higher, or the primary rates "
    "tie and its secondary failure rate is strictly lower. "
    "Equal primary and secondary rates are inconclusive. "
    "Numeric failures, prompt length, latency, graph-context tokens, and omission "
    "counts are reported and are not used to choose an arm. "
    "A higher context-token count is not higher factual quality."
)


def excluded_question_ids(path: Path | None = None) -> list[str]:
    """Question ids from the frozen quick-pilot manifest. The file is only read."""
    source = Path(path) if path is not None else FROZEN_PILOT_MANIFEST
    body = json.loads(source.read_text(encoding="utf-8"))
    if body.get("experiment_id") != PILOT_EXPERIMENT_ID:
        raise ValueError(f"{source} is not the frozen quick pilot")
    return list(body["execution_order"])


def _gold_ids() -> set[str]:
    if not GOLD_PATH.is_file():
        return set()
    found = set()
    for line in GOLD_PATH.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        found.add(str(row.get("id") or row.get("question_id") or ""))
    found.discard("")
    return found


def select_stage_questions(
    questions: list[dict],
    exclude_ids: list[str],
    *,
    per_medium: int = REQUESTED_PER_MEDIUM,
) -> tuple[list[dict], dict]:
    """File order, one medium each, no duplicates and no excluded ids.

    When the pool cannot fill the requested quota, keep the largest multiple
    of the four media. Held-out gold ids are rejected if they appear.
    """
    if per_medium < 1:
        raise ValueError("per_medium must be at least 1")
    excluded = set(exclude_ids)
    seen: set[str] = set()
    pool = []
    for question in questions:
        question_id = question["question_id"]
        if question_id in excluded or question_id in seen:
            continue
        seen.add(question_id)
        pool.append(question)
    requested = per_medium * len(MEDIA)
    actual = min(per_medium, len(pool) // len(MEDIA))
    chosen = pool[: actual * len(MEDIA)]
    overlap = sorted({row["question_id"] for row in chosen} & _gold_ids())
    if overlap:
        raise ValueError("stage 2 selected held-out gold questions: " + ", ".join(overlap))
    assigned = []
    for index, question in enumerate(chosen):
        row = dict(question)
        row["pilot_medium"] = MEDIA[index % len(MEDIA)]
        row["selection_index"] = index
        assigned.append(row)
    counts: dict[str, int] = {}
    phenomena: dict[str, int] = {}
    for row in assigned:
        counts[row["pilot_medium"]] = counts.get(row["pilot_medium"], 0) + 1
        phenomena[str(row.get("phenomenon"))] = phenomena.get(str(row.get("phenomenon")), 0) + 1
    shortfall = requested - len(assigned)
    status = "requested" if shortfall == 0 else "reduced_to_largest_balanced_alternative"
    reason = "the development pool fills the requested medium quota"
    if shortfall:
        reason = (
            f"The development pool has {len(pool)} questions after exclusions. "
            f"{requested} questions at {per_medium} per medium do not fit. "
            f"The largest balanced alternative is {actual} per medium, "
            f"{len(assigned)} questions, {len(assigned) * len(STAGE_ARMS)} generations. "
            "No question was duplicated and no gold-eval question was added."
        )
    return assigned, {
        "requested_per_medium": per_medium,
        "requested_questions": requested,
        "available_after_exclusion": len(pool),
        "selected_questions": len(assigned),
        "per_medium": actual,
        "media_counts": counts,
        "phenomenon_counts": phenomena,
        "shortfall_questions": shortfall,
        "excluded_question_ids": list(exclude_ids),
        "design_status": status,
        "reason": reason,
        "gold_questions_used": 0,
        "duplicated_questions": 0,
    }


def _medium_index(medium: str) -> int:
    for index, item in enumerate(MEDIUMS):
        if item["medium"] == medium:
            return index
    raise ValueError(f"unknown medium {medium}")


def paired_seed(topic_index: int, medium: str) -> int:
    """Same seed the generator uses for this question and medium on every arm."""
    return int(DECODING["seed_base"]) + topic_index * len(MEDIUMS) + _medium_index(medium)


def snapshot_identity(snapshot: str | Path) -> dict:
    """Read the recorded hash file. This does not rehash or copy the snapshot."""
    path = Path(snapshot)
    assert_snapshot_path(path)
    record = path / "snapshot_hashes.json"
    if not record.is_file():
        return {
            "path": str(path),
            "hash_of_hashes": None,
            "status": "not_recorded",
            "snapshot_modified": False,
        }
    body = json.loads(record.read_text(encoding="utf-8"))
    files = body.get("files") or {}
    digest = hashlib.sha256(json.dumps(files, sort_keys=True).encode("utf-8")).hexdigest()
    return {
        "path": str(path),
        "algorithm": body.get("algorithm") or "sha256",
        "files": len(files),
        "hash_of_hashes": digest,
        "status": "recorded_from_snapshot_hashes",
        "snapshot_modified": False,
    }


def build_stage_manifest(
    questions_path: Path,
    snapshot: str,
    *,
    exclude_manifest: Path | None = None,
    per_medium: int = REQUESTED_PER_MEDIUM,
) -> dict:
    source = Path(questions_path)
    exclude_path = Path(exclude_manifest) if exclude_manifest is not None else FROZEN_PILOT_MANIFEST
    selected, feasibility = select_stage_questions(
        load_questions(source),
        excluded_question_ids(exclude_path),
        per_medium=per_medium,
    )
    if not selected:
        raise ValueError(feasibility["reason"])
    cells = plan_cells(selected, STAGE_ARMS)
    for cell in cells:
        cell["seed"] = paired_seed(cell["topic_index"], cell["medium"])
        cell["context_profile"] = PREPARE_PROFILE[cell["arm"]]
    return {
        "experiment_id": STAGE_EXPERIMENT_ID,
        "status": "prepared_not_run",
        "comparison": "F-DESC versus F-REL",
        "benchmark_path": str(source),
        "benchmark_sha256": _sha256(source),
        "benchmark_modified": False,
        "exclude_manifest": str(exclude_path),
        "frozen_pilot_experiment_id": PILOT_EXPERIMENT_ID,
        "frozen_pilot_unchanged": True,
        "questions_selected": len(selected),
        "cells": len(cells),
        "arms": list(STAGE_ARMS),
        "profiles": {arm: PREPARE_PROFILE[arm] for arm in STAGE_ARMS},
        "feasibility": feasibility,
        "one_medium_per_question": True,
        "retrieval_once_per_question": True,
        "evidence_once_per_question": True,
        "snapshot": snapshot_identity(snapshot),
        "adapter": ADAPTER,
        "quantization": "4bit",
        "decoding": dict(DECODING),
        "max_new_tokens_by_medium": {item["medium"]: item["max_new_tokens"] for item in MEDIUMS},
        "retrieval": dict(RETRIEVAL),
        "prompt_family": PROMPT_FAMILY,
        "judge": None,
        "unified_eval": "unchanged",
        "comparison_rule": COMPARISON_RULE,
        "questions": [
            {
                "question_id": row["question_id"],
                "query": row["query"],
                "phenomenon": row.get("phenomenon"),
                "pilot_medium": row["pilot_medium"],
                "abstention_expected": bool(row.get("abstention_expected")),
                "designated_source_ids": list(row.get("designated_source_ids") or []),
                "topic_index": index,
                "seed": paired_seed(index, row["pilot_medium"]),
            }
            for index, row in enumerate(selected)
        ],
        "execution_order": [row["question_id"] for row in selected],
        "plan": cells,
        "limitation": (
            "This comparison does not measure persona, stance, or writing quality. "
            "Question echoes are reported separately and are not a win."
        ),
    }


def _refuse_frozen_pilot(directory: Path) -> None:
    directory = Path(directory)
    if directory.resolve() == FROZEN_PILOT_MANIFEST.parent.resolve():
        raise RuntimeError("refusing to write stage 2 into the frozen quick pilot")
    manifest = directory / "pilot_manifest.json"
    if manifest.is_file():
        body = json.loads(manifest.read_text(encoding="utf-8"))
        if body.get("experiment_id") == PILOT_EXPERIMENT_ID:
            raise RuntimeError("refusing to write stage 2 into the frozen quick pilot")


def load_stage_manifest(directory: Path) -> dict:
    path = Path(directory) / MANIFEST_NAME
    if not path.is_file():
        raise FileNotFoundError(f"missing stage-2 manifest: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def write_stage_manifest(
    directory: Path,
    questions_path: Path,
    snapshot: str,
    *,
    exclude_manifest: Path | None = None,
    per_medium: int = REQUESTED_PER_MEDIUM,
) -> dict:
    directory = Path(directory)
    _refuse_frozen_pilot(directory)
    manifest = build_stage_manifest(
        questions_path,
        snapshot,
        exclude_manifest=exclude_manifest,
        per_medium=per_medium,
    )
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / MANIFEST_NAME
    if path.is_file():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing.get("execution_order") != manifest["execution_order"] or existing.get("arms") != manifest["arms"]:
            raise ValueError("refusing to replace an existing stage-2 question list or arm list")
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report = cell_report(directory, manifest)
    (directory / "cell_status.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def _failure_reasons(directory: Path) -> dict[tuple[str, str, str], str]:
    path = directory / "raw" / "failures.jsonl"
    found: dict[tuple[str, str, str], str] = {}
    if not path.is_file():
        return found
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        reason = str(row.get("reason") or row.get("error") or "unspecified")
        media = row.get("media") or []
        if isinstance(media, str):
            media = [media]
        for medium in media:
            found[(str(row.get("question_id")), str(row.get("arm")), str(medium))] = reason
    return found


def cell_report(directory: Path, manifest: dict | None = None) -> dict:
    """Planned, completed, failed, and not-yet-started cells."""
    directory = Path(directory)
    body = manifest if manifest is not None else load_stage_manifest(directory)
    questions = questions_from_manifest(body)
    arms = tuple(body["arms"])
    reasons = _failure_reasons(directory)
    planned = []
    completed = []
    unsaved = []
    for question in questions:
        for arm in arms:
            for medium in media_for_question(question):
                item = {
                    "question_id": question["question_id"],
                    "arm": arm,
                    "medium": medium,
                }
                planned.append(item)
                if is_complete(cell_path(directory, question["question_id"], arm, medium)):
                    completed.append(item)
                    continue
                reason = reasons.get((question["question_id"], arm, medium), "not_started")
                unsaved.append({**item, "reason": reason})
    failed = [row for row in unsaved if row["reason"] != "not_started"]
    return {
        "experiment_id": body["experiment_id"],
        "planned": len(planned),
        "completed": len(completed),
        "failed": len(failed),
        "unsaved": len(unsaved),
        "not_started": len(unsaved) - len(failed),
        "planned_cells": planned,
        "completed_cells": completed,
        "failed_cells": failed,
        "unsaved_cells": unsaved,
    }


def run_stage(directory: Path, snapshot: Path, *, max_runtime_minutes: float = 180) -> dict:
    """Generate pending F-REL and F-DESC cells. Completed cells are skipped."""
    from host_finetune.model_f_experiment.runner import execute_four_arms

    directory = Path(directory)
    _refuse_frozen_pilot(directory)
    manifest = load_stage_manifest(directory)
    if manifest["arms"] != list(STAGE_ARMS):
        raise ValueError(f"stage manifest arms are {manifest['arms']}")
    questions = questions_from_manifest(manifest)
    started = time.perf_counter()
    limit = float(max_runtime_minutes) * 60

    def should_stop() -> bool:
        return (time.perf_counter() - started) >= limit

    report = execute_four_arms(
        directory,
        questions,
        snapshot,
        should_stop=should_stop,
        arms=STAGE_ARMS,
    )
    status = cell_report(directory, manifest)
    (directory / "cell_status.json").write_text(
        json.dumps(status, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    body = {
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "max_runtime_minutes": max_runtime_minutes,
        "stopped_for_runtime": bool(report.get("stopped")),
        "generated": report.get("generated"),
        "skipped_complete": report.get("skipped_complete"),
        "unsaved": list(report.get("unsaved") or []),
        "planned": status["planned"],
        "completed": status["completed"],
        "failed": status["failed"],
        "not_started": status["not_started"],
        "decoding_unchanged": True,
        "evidence_budget_unchanged": True,
        "frozen_pilot_unchanged": True,
    }
    (directory / "timing.json").write_text(json.dumps(body, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return body


def score_stage(directory: Path) -> dict:
    """Existing deterministic scorer. It does not add echo flags to its rows."""
    directory = Path(directory)
    _refuse_frozen_pilot(directory)
    manifest = load_stage_manifest(directory)
    return write_model_f_reports(directory, directory, manifest["experiment_id"])


def _load_generation(directory: Path, question_id: str, arm: str, medium: str) -> dict:
    path = cell_path(directory, question_id, arm, medium)
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _echo(answer: str, topic: str) -> bool:
    from host_finetune.evidence_capsule import repeats_topic
    from host_finetune.factual_reliability import embeds_topic

    return repeats_topic(answer or "", topic or "") or embeds_topic(answer or "", topic or "")


def _adjusted(scorer_pass, echo: bool):
    if echo:
        return False
    if scorer_pass is None:
        return None
    return bool(scorer_pass)


def _rate(values: list) -> float | None:
    known = [value for value in values if value is not None]
    if not known:
        return None
    return sum(1 for value in known if value) / len(known)


def _mean(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def _arm_leads(candidate: dict, other: dict) -> bool:
    primary = candidate.get("echo_adjusted_pass_rate")
    baseline = other.get("echo_adjusted_pass_rate")
    if primary is None or baseline is None:
        return False
    if primary > baseline:
        return True
    if primary < baseline:
        return False
    leaks = candidate.get("copy_or_leak_failure_rate")
    other_leaks = other.get("copy_or_leak_failure_rate")
    if leaks is None or other_leaks is None:
        return False
    return leaks < other_leaks


def _question_winner(left: dict, right: dict) -> str:
    """Return left, right, or tie. Echo-adjusted factual is the first key."""
    factual_left = left["echo_adjusted_pass"]
    factual_right = right["echo_adjusted_pass"]
    if factual_left is None or factual_right is None:
        return "unscored"
    if factual_left != factual_right:
        return "left" if factual_left else "right"
    leak_left = left["copy_or_leak_failure"]
    leak_right = right["copy_or_leak_failure"]
    if leak_left is None or leak_right is None or leak_left == leak_right:
        return "tie"
    return "left" if leak_left is False else "right"


def compare_stage(directory: Path) -> dict:
    """Paired F-DESC versus F-REL report. Does not write arm_selection.json."""
    directory = Path(directory)
    _refuse_frozen_pilot(directory)
    manifest = load_stage_manifest(directory)
    metrics_path = directory / "deterministic_metrics.json"
    if not metrics_path.is_file():
        score_stage(directory)
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    graph_path = directory / "graph_diagnostics.json"
    graph = json.loads(graph_path.read_text(encoding="utf-8")) if graph_path.is_file() else {"cells": []}
    graph_by = {
        (cell.get("question_id"), cell.get("arm")): cell
        for cell in graph.get("cells") or []
    }
    scored = {
        (cell.get("question_id"), cell.get("arm")): cell
        for cell in metrics.get("cells") or []
    }
    queries = {row["question_id"]: row for row in manifest["questions"]}
    per_question = []
    usable = []
    for question_id in manifest["execution_order"]:
        question = queries[question_id]
        medium = question["pilot_medium"]
        pair = {}
        complete = True
        for arm in STAGE_ARMS:
            saved = _load_generation(directory, question_id, arm, medium)
            if saved.get("status") != "complete":
                complete = False
            metric = scored.get((question_id, arm)) or {}
            generation = metric.get("generation") or {}
            faith = (generation.get("faithfulness") or {}).get("pass")
            copying = generation.get("copying") or {}
            leak = generation.get("citation_marker_leak")
            copy_pass = copying.get("pass")
            if leak is None and copy_pass is None:
                copy_failed = None
            else:
                copy_failed = bool(leak) or copy_pass is False
            numeric_pass = generation.get("numeric_pass")
            unsupported = generation.get("unsupported_numeric_count")
            numeric_failure = None
            if numeric_pass is False or (isinstance(unsupported, int) and unsupported > 0):
                numeric_failure = True
            elif numeric_pass is True and not unsupported:
                numeric_failure = False
            answer = str(saved.get("answer") or "")
            echo = _echo(answer, question["query"]) if saved else None
            tokens = saved.get("tokens") or {}
            channels = tokens.get("context_by_channel") or {}
            graph_tokens = graph_by.get((question_id, arm), {}).get("graph_context_tokens")
            if graph_tokens is None:
                graph_tokens = sum(
                    value for name, value in channels.items()
                    if name != "primary_source" and isinstance(value, (int, float))
                )
            omitted = list(saved.get("window_omitted") or [])
            channel_name = "derived_description" if arm == "F-DESC" else "source_supported_relationship"
            supplied = channels.get(channel_name)
            pair[arm] = {
                "scorer_faithfulness_pass": None if faith is None else bool(faith),
                "echo": echo,
                "echo_adjusted_pass": None if echo is None else _adjusted(faith, bool(echo)),
                "copy_or_leak_failure": copy_failed,
                "citation_leak": None if leak is None else bool(leak),
                "copy_failure": None if copy_pass is None else copy_pass is False,
                "numeric_failure": numeric_failure,
                "prompt_tokens": tokens.get("prompt"),
                "graph_context_tokens": graph_tokens,
                "graph_channel_tokens": supplied if isinstance(supplied, (int, float)) else None,
                "graph_context_supplied": bool(supplied) if isinstance(supplied, (int, float)) else None,
                "window_omitted": len(omitted),
                "generation_seconds": (saved.get("latency") or {}).get("generation_seconds"),
            }
        winner = "incomplete"
        if complete and all(pair[arm]["echo_adjusted_pass"] is not None for arm in STAGE_ARMS):
            winner = _question_winner(pair["F-REL"], pair["F-DESC"])
            if winner == "left":
                winner = "F-REL"
            elif winner == "right":
                winner = "F-DESC"
            usable.append(question_id)
        per_question.append({
            "question_id": question_id,
            "medium": medium,
            "phenomenon": question.get("phenomenon"),
            "winner": winner,
            "arms": pair,
        })
    compared = [row for row in per_question if row["question_id"] in usable and row["winner"] != "unscored"]

    def summary(arm: str) -> dict:
        rows = [row["arms"][arm] for row in compared]
        return {
            "n": len(rows),
            "scorer_faithfulness_pass_rate": _rate([row["scorer_faithfulness_pass"] for row in rows]),
            "echo_rate": _rate([row["echo"] for row in rows]),
            "echo_adjusted_pass_rate": _rate([row["echo_adjusted_pass"] for row in rows]),
            "copy_or_leak_failure_rate": _rate([row["copy_or_leak_failure"] for row in rows]),
            "citation_leak_rate": _rate([row["citation_leak"] for row in rows]),
            "copy_failure_rate": _rate([row["copy_failure"] for row in rows]),
            "numeric_failure_rate": _rate([row["numeric_failure"] for row in rows]),
            "mean_prompt_tokens": _mean([row["prompt_tokens"] for row in rows if isinstance(row["prompt_tokens"], (int, float))]),
            "mean_graph_context_tokens": _mean([
                row["graph_context_tokens"] for row in rows if isinstance(row["graph_context_tokens"], (int, float))
            ]),
            "graph_context_supplied": sum(1 for row in rows if row["graph_context_supplied"] is True),
            "graph_context_absent": sum(1 for row in rows if row["graph_context_supplied"] is False),
            "window_omissions": sum(int(row["window_omitted"] or 0) for row in rows),
            "mean_generation_seconds": _mean([
                row["generation_seconds"] for row in rows if isinstance(row["generation_seconds"], (int, float))
            ]),
        }

    summaries = {arm: summary(arm) for arm in STAGE_ARMS}
    by_medium = {}
    for medium in MEDIA:
        rows = [row for row in compared if row["medium"] == medium]
        by_medium[medium] = {
            "n": len(rows),
            "wins": {arm: sum(1 for row in rows if row["winner"] == arm) for arm in STAGE_ARMS},
            "ties": sum(1 for row in rows if row["winner"] == "tie"),
        }
    wins = {arm: sum(1 for row in compared if row["winner"] == arm) for arm in STAGE_ARMS}
    ties = sum(1 for row in compared if row["winner"] == "tie")
    status = "incomplete"
    selected = None
    reason = "no question was complete and scored for both arms"
    if compared:
        rel = summaries["F-REL"]
        desc = summaries["F-DESC"]
        if _arm_leads(desc, rel):
            status = "selected"
            selected = "F-DESC"
            reason = "F-DESC leads on echo-adjusted faithfulness or, given a tie, on copy and leakage"
        elif _arm_leads(rel, desc):
            status = "selected"
            selected = "F-REL"
            reason = "F-REL leads on echo-adjusted faithfulness or, given a tie, on copy and leakage"
        else:
            status = "inconclusive"
            reason = "F-DESC and F-REL tie on echo-adjusted faithfulness and on copy and leakage"
    comparison = {
        "schema": "model-f-desc-vs-rel-v1",
        "experiment_id": manifest["experiment_id"],
        "rule": COMPARISON_RULE,
        "echo_detection": ["repeats_topic", "embeds_topic"],
        "echo_is_not_a_win": True,
        "context_tokens_are_not_a_score": True,
        "scorer_rows_unchanged": True,
        "compared_question_ids": [row["question_id"] for row in compared],
        "incomplete_question_ids": [
            row["question_id"] for row in per_question if row["winner"] == "incomplete"
        ],
        "per_question": per_question,
        "per_medium": by_medium,
        "question_wins": wins,
        "question_ties": ties,
        "arms": summaries,
        "graph_evidence_is_a_score": False,
    }
    recommendation = {
        "artifact": "desc_vs_rel_recommendation",
        "experiment_id": manifest["experiment_id"],
        "does_not_replace": "experiments/EXP-model-f-quick-pilot/arm_selection.json",
        "selected_arm": selected,
        "status": status,
        "reason": reason,
        "rule": COMPARISON_RULE,
        "compared_questions": len(compared),
        "question_wins": wins,
        "question_ties": ties,
        "echo_is_not_a_win": True,
        "context_tokens_are_not_a_score": True,
        "frozen_pilot_unchanged": True,
    }
    (directory / "desc_vs_rel_comparison.json").write_text(
        json.dumps(comparison, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (directory / "desc_vs_rel_recommendation.json").write_text(
        json.dumps(recommendation, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return recommendation
