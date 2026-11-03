"""Eight-question, four-arm pilot. One medium per question, 32 cells.

The original 36-question benchmark is read and not rewritten. Arm choice uses
only questions finished under all four arms. A tie or a missing benefit keeps
F-SOURCE. This module does not call a judge and does not change decoding.
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

from host_finetune.compare_adapters import MEDIUMS
from host_finetune.model_f_experiment.arms import ADAPTER, ARMS, DECODING, PREPARE_PROFILE, PROMPT_FAMILY, RETRIEVAL
from host_finetune.model_f_experiment.diagnostics import write_model_f_reports
from host_finetune.model_f_experiment.runner import is_complete, load_questions, media_for_question, pending

PILOT_EXPERIMENT_ID = "EXP-20261102-002-model-f-quick-pilot"
PILOT_CELLS = 32
BUCKETS = ("factual", "direct_relationship", "multi_hop", "unanswerable")
PER_BUCKET = 2
MEDIA_CYCLE = ("blog", "linkedin", "x", "talk")
GRAPH_ARMS = ("F-REL", "F-DESC", "F-HOP")

# One measured X cell and one 10-query batch from the post-index check.
# Model load was not timed on its own, so scaling the remainder is an estimate.
MEASURED_ANCHORS = {
    "label": "estimate_not_a_registered_experiment",
    "measured": False,
    "x_smoke_wall_seconds": 63.0,
    "x_extraction_seconds": 22.0,
    "x_output_tokens": 160,
    "retrieval_batch_wall_seconds": 50.0,
    "retrieval_batch_queries": 10,
    "source": "post-index check on 2026-11-02: one F-DESC X cell and ten LightRAG queries",
    "model_load_seconds": None,
    "note": (
        "The remainder of the X smoke includes model load and a copy retry. "
        "Load was not timed separately, so scaling that remainder by max_new_tokens "
        "is an upper-leaning estimate. Output budgets are not reduced to fit the clock."
    ),
}

SELECTION_RULE = (
    "Compare only questions where F-SOURCE, F-REL, F-DESC, and F-HOP are all complete. "
    "Primary key: faithfulness pass rate from the existing deterministic factual check. "
    "Secondary key: source-copy or prompt-leak failure rate. "
    "A graph arm is chosen only when it strictly beats F-SOURCE on the primary key, "
    "or ties that key and strictly beats F-SOURCE on the secondary key. "
    "Usable graph context is reported and is not a score. "
    "Retry rate and generation latency break ties only between graph arms that already beat F-SOURCE. "
    "A tie, an incomplete pilot, or no such benefit freezes F-SOURCE."
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _medium_tokens(medium: str) -> int:
    for item in MEDIUMS:
        if item["medium"] == medium:
            return int(item["max_new_tokens"])
    raise ValueError(f"unknown medium {medium}")


def select_pilot_questions(questions: list[dict]) -> list[dict]:
    """First two file-order questions from each required phenomenon, then a medium cycle."""
    chosen = []
    for bucket in BUCKETS:
        pool = [row for row in questions if row.get("phenomenon") == bucket]
        if len(pool) < PER_BUCKET:
            raise ValueError(f"{bucket} has {len(pool)} questions; the pilot needs {PER_BUCKET}")
        chosen.extend(pool[:PER_BUCKET])
    if len(chosen) != 8:
        raise ValueError(f"pilot selection produced {len(chosen)} questions")
    assigned = []
    for index, question in enumerate(chosen):
        row = dict(question)
        row["pilot_medium"] = MEDIA_CYCLE[index % len(MEDIA_CYCLE)]
        row["selection_index"] = index
        assigned.append(row)
    return assigned


def execution_order(questions: list[dict]) -> list[dict]:
    """One question per medium first, covering four phenomena, then the remaining four."""
    by_medium: dict[str, list[dict]] = {}
    for question in questions:
        by_medium.setdefault(question["pilot_medium"], []).append(question)
    first = []
    used: set[str] = set()
    for medium in MEDIA_CYCLE:
        pool = by_medium.get(medium) or []
        if not pool:
            raise ValueError(f"pilot is missing medium {medium}")
        pick = next((row for row in pool if row.get("phenomenon") not in used), pool[0])
        first.append(pick)
        used.add(str(pick.get("phenomenon")))
    chosen = {row["question_id"] for row in first}
    rest = [row for row in questions if row["question_id"] not in chosen]
    return first + rest


def plan_cells(questions: list[dict], arms: tuple[str, ...] | None = None) -> list[dict]:
    chosen_arms = ARMS if arms is None else tuple(arms)
    cells = []
    for index, question in enumerate(questions):
        medium = question["pilot_medium"]
        for arm in chosen_arms:
            cells.append({
                "question_id": question["question_id"],
                "topic_index": index,
                "arm": arm,
                "medium": medium,
                "max_new_tokens": _medium_tokens(medium),
                "retrieval_key": question["question_id"],
                "context_profile": PREPARE_PROFILE[arm],
            })
    return cells


def estimate_runtime(cells: list[dict], anchors: dict | None = None) -> dict:
    """Scale the one measured X cell. This is an estimate, not a measurement."""
    used = dict(MEASURED_ANCHORS if anchors is None else anchors)
    remainder = float(used["x_smoke_wall_seconds"]) - float(used["x_extraction_seconds"])
    base_tokens = float(used["x_output_tokens"])
    questions = {cell["question_id"] for cell in cells}
    generation = sum(remainder * (int(cell["max_new_tokens"]) / base_tokens) for cell in cells)
    extraction = float(used["x_extraction_seconds"]) * len(questions)
    retrieval = float(used["retrieval_batch_wall_seconds"]) * (
        len(questions) / float(used["retrieval_batch_queries"])
    )
    total = generation + extraction + retrieval
    return {
        "measured": False,
        "estimated_seconds": round(total, 1),
        "estimated_minutes": round(total / 60, 1),
        "generation_seconds": round(generation, 1),
        "extraction_seconds": round(extraction, 1),
        "retrieval_seconds": round(retrieval, 1),
        "cells": len(cells),
        "questions": len(questions),
        "basis": "x_smoke_remainder_scaled_by_max_new_tokens",
        "anchors": used,
    }


def balanced_prefixes(questions: list[dict]) -> list[list[dict]]:
    """Prefixes where every medium occurs the same number of times."""
    counts: dict[str, int] = {}
    found = []
    for index, question in enumerate(questions, start=1):
        medium = question["pilot_medium"]
        counts[medium] = counts.get(medium, 0) + 1
        if len(counts) == len(MEDIA_CYCLE) and len(set(counts.values())) == 1:
            found.append(questions[:index])
    return found


def recommend_subset(questions: list[dict], max_minutes: float, anchors: dict | None = None) -> dict:
    """Largest balanced prefix whose estimate fits. The planned 32 cells stay intact."""
    full = plan_cells(questions)
    full_estimate = estimate_runtime(full, anchors)
    choice = questions
    reason = "the full 32-cell estimate fits the budget"
    if full_estimate["estimated_minutes"] > max_minutes:
        choice = []
        reason = "no balanced subset fits without changing decoding or evidence budgets"
        for prefix in balanced_prefixes(questions):
            estimate = estimate_runtime(plan_cells(prefix), anchors)
            if estimate["estimated_minutes"] <= max_minutes:
                choice = prefix
                reason = "largest balanced paired prefix at or under the budget"
    estimate = estimate_runtime(plan_cells(choice), anchors) if choice else None
    return {
        "max_runtime_minutes": max_minutes,
        "full_plan_cells": len(full),
        "full_estimate": full_estimate,
        "fits_budget": full_estimate["estimated_minutes"] <= max_minutes,
        "recommended_question_ids": [row["question_id"] for row in choice],
        "recommended_cells": len(choice) * len(ARMS),
        "recommended_estimate": estimate,
        "reason": reason,
        "conditions_unchanged": True,
    }


def build_manifest(
    questions_path: Path,
    snapshot: str,
    *,
    max_runtime_minutes: float = 60,
) -> dict:
    source = Path(questions_path)
    digest = _sha256(source)
    selected = execution_order(select_pilot_questions(load_questions(source)))
    cells = plan_cells(selected)
    if len(cells) != PILOT_CELLS:
        raise ValueError(f"expected {PILOT_CELLS} pilot cells, found {len(cells)}")
    preflight = recommend_subset(selected, max_runtime_minutes)
    media_counts: dict[str, int] = {}
    for question in selected:
        media_counts[question["pilot_medium"]] = media_counts.get(question["pilot_medium"], 0) + 1
    return {
        "experiment_id": PILOT_EXPERIMENT_ID,
        "status": "prepared_not_run",
        "benchmark_path": str(source),
        "benchmark_sha256": digest,
        "benchmark_modified": False,
        "questions_selected": len(selected),
        "cells": len(cells),
        "arms": list(ARMS),
        "media_counts": media_counts,
        "one_medium_per_question": True,
        "snapshot": snapshot,
        "snapshot_modified": False,
        "adapter": ADAPTER,
        "quantization": "4bit",
        "decoding": dict(DECODING),
        "max_new_tokens_by_medium": {item["medium"]: item["max_new_tokens"] for item in MEDIUMS},
        "retrieval": dict(RETRIEVAL),
        "prompt_family": PROMPT_FAMILY,
        "judge": None,
        "selection_rule": SELECTION_RULE,
        "questions": [
            {
                "question_id": row["question_id"],
                "query": row["query"],
                "phenomenon": row.get("phenomenon"),
                "pilot_medium": row["pilot_medium"],
                "abstention_expected": bool(row.get("abstention_expected")),
                "designated_source_ids": list(row.get("designated_source_ids") or []),
                "topic_index": index,
            }
            for index, row in enumerate(selected)
        ],
        "execution_order": [row["question_id"] for row in selected],
        "plan": cells,
        "runtime_preflight": preflight,
        "limitation": "The pilot does not measure persona, stance, or writing quality.",
    }


def write_pilot_manifest(directory: Path, questions_path: Path, snapshot: str, *, max_runtime_minutes: float = 60) -> dict:
    manifest = build_manifest(questions_path, snapshot, max_runtime_minutes=max_runtime_minutes)
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "pilot_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def questions_from_manifest(manifest: dict) -> list[dict]:
    rows = []
    for question in manifest["questions"]:
        rows.append({
            "question_id": question["question_id"],
            "query": question["query"],
            "phenomenon": question.get("phenomenon"),
            "pilot_medium": question["pilot_medium"],
            "abstention_expected": bool(question.get("abstention_expected")),
            "designated_source_ids": list(question.get("designated_source_ids") or []),
        })
    return rows


def load_manifest(directory: Path) -> dict:
    path = Path(directory) / "pilot_manifest.json"
    if not path.is_file():
        raise FileNotFoundError(f"missing pilot manifest: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def run_pilot(directory: Path, snapshot: Path, *, max_runtime_minutes: float = 60) -> dict:
    """Generate pending pilot cells. Stops between cells when the clock expires."""
    from host_finetune.model_f_experiment.runner import execute_four_arms

    directory = Path(directory)
    manifest = load_manifest(directory)
    if manifest["cells"] != PILOT_CELLS:
        raise ValueError(f"pilot manifest has {manifest['cells']} cells; expected {PILOT_CELLS}")
    questions = questions_from_manifest(manifest)
    started = time.perf_counter()
    limit = float(max_runtime_minutes) * 60

    def should_stop(now: float = None) -> bool:
        clock = time.perf_counter() if now is None else now
        return (clock - started) >= limit

    report = execute_four_arms(directory, questions, snapshot, should_stop=should_stop)
    elapsed = round(time.perf_counter() - started, 3)
    groups = group_status(directory, questions)
    body = {
        "elapsed_seconds": elapsed,
        "max_runtime_minutes": max_runtime_minutes,
        "stopped_for_runtime": bool(report.get("stopped")),
        "generated": report.get("generated"),
        "skipped_complete": report.get("skipped_complete"),
        "unsaved": list(report.get("unsaved") or []),
        "complete_question_ids": groups["complete"],
        "incomplete_question_ids": groups["incomplete"],
        "decoding_unchanged": True,
        "evidence_budget_unchanged": True,
    }
    (directory / "timing.json").write_text(json.dumps(body, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return body


def group_status(directory: Path, questions: list[dict]) -> dict:
    """A question counts only when its one medium is complete for every arm."""
    from host_finetune.model_f_experiment.runner import cell_path

    complete = []
    incomplete = []
    for question in questions:
        medium = media_for_question(question)[0]
        done = all(
            is_complete(cell_path(directory, question["question_id"], arm, medium))
            for arm in ARMS
        )
        (complete if done else incomplete).append(question["question_id"])
    return {"complete": complete, "incomplete": incomplete}


def score_pilot(directory: Path) -> dict:
    """Deterministic scores only. The same writer serves the official diagnostics."""
    directory = Path(directory)
    manifest = load_manifest(directory)
    return write_model_f_reports(directory, directory, manifest["experiment_id"])


def _rate(values: list[bool | None]) -> float | None:
    known = [value for value in values if value is not None]
    if not known:
        return None
    return sum(1 for value in known if value) / len(known)


def _mean(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def _cell_bits(cell: dict) -> dict:
    generation = cell.get("generation") or {}
    faith = (generation.get("faithfulness") or {}).get("pass")
    factual = None if faith is None else bool(faith)
    copying = generation.get("copying") or {}
    leak = generation.get("citation_marker_leak")
    copy_pass = copying.get("pass")
    if leak is None and copy_pass is None:
        failed = None
    else:
        failed = bool(leak) or copy_pass is False
    attempt = cell.get("accepted_attempt")
    retried = None if attempt is None else int(attempt) > 1
    latency = (cell.get("efficiency") or {}).get("generation_seconds")
    return {
        "factual_pass": factual,
        "copy_or_leak_failure": failed,
        "retried": retried,
        "generation_seconds": latency if isinstance(latency, (int, float)) else None,
        "numeric_pass": generation.get("numeric_pass"),
        "unsupported_numeric_count": generation.get("unsupported_numeric_count"),
        "role_unsupported_count": generation.get("role_unsupported_count"),
        "word_count": generation.get("word_count"),
    }


def _arm_summary(cells: list[dict]) -> dict:
    bits = [_cell_bits(cell) for cell in cells]
    latencies = [item["generation_seconds"] for item in bits if item["generation_seconds"] is not None]
    return {
        "n": len(cells),
        "factual_pass_rate": _rate([item["factual_pass"] for item in bits]),
        "copy_or_leak_failure_rate": _rate([item["copy_or_leak_failure"] for item in bits]),
        "retry_rate": _rate([item["retried"] for item in bits]),
        "mean_generation_seconds": _mean(latencies),
    }


def _beats(candidate: dict, baseline: dict) -> bool:
    left = candidate.get("factual_pass_rate")
    right = baseline.get("factual_pass_rate")
    if left is None or right is None:
        return False
    if left > right:
        return True
    if left < right:
        return False
    leak_left = candidate.get("copy_or_leak_failure_rate")
    leak_right = baseline.get("copy_or_leak_failure_rate")
    if leak_left is None or leak_right is None:
        return False
    return leak_left < leak_right


def _tie_key(summary: dict) -> tuple:
    retry = summary.get("retry_rate")
    latency = summary.get("mean_generation_seconds")
    return (
        0 if retry is None else retry,
        0 if latency is None else latency,
    )


def select_arm(directory: Path) -> dict:
    """Freeze the arm. An existing freeze is left untouched."""
    directory = Path(directory)
    destination = directory / "arm_selection.json"
    if destination.is_file():
        existing = json.loads(destination.read_text(encoding="utf-8"))
        if existing.get("frozen") is True:
            return existing
    manifest = load_manifest(directory)
    metrics_path = directory / "deterministic_metrics.json"
    if not metrics_path.is_file():
        score_pilot(directory)
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    questions = questions_from_manifest(manifest)
    groups = group_status(directory, questions)
    complete = set(groups["complete"])
    by_question: dict[str, dict[str, dict]] = {}
    for cell in metrics.get("cells") or []:
        if cell.get("question_id") not in complete:
            continue
        by_question.setdefault(cell["question_id"], {})[cell.get("arm")] = cell
    paired = {
        question_id: arms
        for question_id, arms in by_question.items()
        if all(arm in arms for arm in ARMS)
    }
    per_question = []
    for question_id in manifest["execution_order"]:
        if question_id not in paired:
            continue
        outcome = {"question_id": question_id}
        for arm in ARMS:
            outcome[arm] = _cell_bits(paired[question_id][arm])
        per_question.append(outcome)
    summaries = {
        arm: _arm_summary([paired[question_id][arm] for question_id in paired])
        for arm in ARMS
    }
    graph_path = directory / "graph_diagnostics.json"
    graph = json.loads(graph_path.read_text(encoding="utf-8")) if graph_path.is_file() else {}
    status = "fallback"
    selected = "F-SOURCE"
    reason = "no graph arm beat F-SOURCE on factual reliability or copy and leakage"
    if not paired:
        status = "fallback_incomplete"
        reason = "no question was complete for all four arms, so no winner was chosen"
    else:
        winners = [arm for arm in GRAPH_ARMS if _beats(summaries[arm], summaries["F-SOURCE"])]
        if len(winners) == 1:
            selected = winners[0]
            status = "selected"
            reason = f"{selected} beat F-SOURCE on the predefined factual or leakage rule"
        elif len(winners) > 1:
            ordered = sorted(winners, key=lambda arm: _tie_key(summaries[arm]))
            if _tie_key(summaries[ordered[0]]) == _tie_key(summaries[ordered[1]]):
                status = "fallback_ambiguous"
                reason = "more than one graph arm beat F-SOURCE and the remaining keys tied"
            else:
                selected = ordered[0]
                status = "selected"
                reason = f"{selected} beat F-SOURCE and led the other graph arms on retry or latency"
    comparison = {
        "rule": SELECTION_RULE,
        "complete_question_ids": [question_id for question_id in manifest["execution_order"] if question_id in paired],
        "excluded_incomplete_question_ids": groups["incomplete"],
        "per_question": per_question,
        "arms": summaries,
        "graph_evidence_fraction": graph.get("fraction_with_usable_graph_evidence"),
        "graph_evidence_is_a_score": False,
        "limitation": manifest.get("limitation"),
    }
    (directory / "arm_comparison.json").write_text(
        json.dumps(comparison, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    frozen = {
        "frozen": True,
        "experiment_id": manifest["experiment_id"],
        "selected_arm": selected,
        "status": status,
        "reason": reason,
        "rule": SELECTION_RULE,
        "complete_question_ids": comparison["complete_question_ids"],
        "excluded_incomplete_question_ids": groups["incomplete"],
        "do_not_revise_after_unified_eval": True,
        "does_not_measure": "persona style, stance, and writing quality",
        "graph_context_availability_used_as_score": False,
    }
    destination.write_text(json.dumps(frozen, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return frozen


def pilot_pending(directory: Path) -> list[dict]:
    questions = questions_from_manifest(load_manifest(directory))
    return pending(directory, questions)
