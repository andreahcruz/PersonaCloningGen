"""Fixture checks for the shared project-state validator. No model calls."""

from __future__ import annotations

import json
from pathlib import Path

from project_pm.check import (
    _allowed,
    _porcelain_path,
    check_manifest_text,
    check_repository,
    hook_session_start,
    hook_stop,
)

COMPLETE_MANIFEST = """
experiment_id: EXP-20261108-001-demo
timestamp: 2026-11-08T00:00:00Z
execution_commands:
  - python -m demo
git:
  commit: abc
  dirty: false
data:
  source_manifest: n/a
  source_hashes: {}
  processed_dataset_hash: abc
  split_manifest_hash: abc
  gold_eval_hash: abc
rag:
  persona_profile_hash: n/a
  chroma_collection: n/a
  index_version_or_snapshot: n/a
  embedding_model: n/a
model:
  model_name: fixture
  model_or_adapter_hash: fixture
  config: {}
  random_seed: 0
evaluation:
  evaluator_commit: abc
  judge_model: n/a
  judge_config: n/a
environment:
  python_version: "3.11"
  package_snapshot: fixture
  hardware: none
outputs:
  raw: raw/
  metrics: metrics/
  warnings_and_failures: none
"""


def _write(root: Path, *, task_status: str = "in_progress", verification: str = "", decision_id: str = "") -> None:
    project = root / "docs" / "project"
    project.mkdir(parents=True)
    (root / "docs").mkdir(exist_ok=True)
    state = {
        "schema": "project-state.v1",
        "updated": "2026-11-07",
        "git": {
            "branch": "fixture",
            "commit": "abc123",
            "known_dirty_prefixes": ["docs/project/"],
        },
        "next_task": "TASK-100",
        "evaluation": {"role": "fixture benchmark, not a final test"},
        "experiments": {
            "validate_id_date_on_or_after": "20261108",
            "known_unregistered": ["experiments/EXP-20261103-009-old"],
        },
        "evidence": ["docs/DECISION_LOG.md"],
    }
    tasks = {
        "tasks": [
            {
                "id": "TASK-100",
                "objective": "Exercise the checker",
                "status": task_status,
                "owns_files": ["docs/project/"],
                "acceptance": ["checker returns the expected code"],
                "verification": verification,
                "decision_id": decision_id,
            }
        ]
    }
    (project / "state.json").write_text(json.dumps(state), encoding="utf-8")
    (project / "tasks.json").write_text(json.dumps(tasks), encoding="utf-8")
    (project / "HANDOFF.md").write_text(
        "\n".join(
            [
                "# Handoff",
                "- updated: 2026-11-07",
                "- git_commit: abc123",
                "- git_branch: fixture",
                "- active_task: TASK-100",
                "- next_action: Continue TASK-100",
                "",
            ]
        ),
        encoding="utf-8",
    )
    (root / "docs" / "DECISION_LOG.md").write_text("# Decisions\n\n## DEC-011 — Fixture\n", encoding="utf-8")
    old = root / "experiments" / "EXP-20261103-009-old"
    old.mkdir(parents=True)


def _codes(root: Path) -> set[str]:
    report = check_repository(root, live_git=False)
    return {item.code for item in report.findings}


def test_new_and_partial_task_is_accepted(tmp_path: Path) -> None:
    _write(tmp_path, task_status="in_progress")
    assert "task_unverified" not in _codes(tmp_path)
    assert not check_repository(tmp_path, live_git=False).errors


def test_done_without_verification_fails(tmp_path: Path) -> None:
    _write(tmp_path, task_status="done")
    assert "task_unverified" in _codes(tmp_path)


def test_done_with_verification_and_decision_passes(tmp_path: Path) -> None:
    _write(tmp_path, task_status="done", verification="pytest passed", decision_id="DEC-011")
    assert not check_repository(tmp_path, live_git=False).errors


def test_blocked_task_keeps_a_failure_record(tmp_path: Path) -> None:
    _write(tmp_path, task_status="blocked", verification="pytest failed: 1 error")
    report = check_repository(tmp_path, live_git=False)
    assert not report.errors
    assert "task_unverified" not in {item.code for item in report.findings}


def test_missing_decision_is_an_error(tmp_path: Path) -> None:
    _write(tmp_path, task_status="done", verification="noted", decision_id="DEC-999")
    assert "decision_missing" in _codes(tmp_path)


def test_stale_handoff_commit_fails(tmp_path: Path) -> None:
    _write(tmp_path)
    handoff = tmp_path / "docs" / "project" / "HANDOFF.md"
    handoff.write_text(handoff.read_text(encoding="utf-8").replace("abc123", "old"), encoding="utf-8")
    assert "handoff_stale_commit" in _codes(tmp_path)


def test_interrupted_session_is_a_warning(tmp_path: Path) -> None:
    _write(tmp_path)
    handoff = tmp_path / "docs" / "project" / "HANDOFF.md"
    text = handoff.read_text(encoding="utf-8") + "- session: interrupted\n"
    handoff.write_text(text, encoding="utf-8")
    report = check_repository(tmp_path, live_git=False)
    assert "session_interrupted" in {item.code for item in report.findings}
    assert not report.errors


def test_overlapping_active_tasks_fail(tmp_path: Path) -> None:
    _write(tmp_path)
    path = tmp_path / "docs" / "project" / "tasks.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    data["tasks"].append(
        {
            "id": "TASK-101",
            "objective": "Edit the same snapshot",
            "status": "in_progress",
            "owns_files": ["docs/project/state.json"],
            "acceptance": ["no overlap"],
            "verification": "",
        }
    )
    path.write_text(json.dumps(data), encoding="utf-8")
    assert "task_overlap" in _codes(tmp_path)


def test_parallel_tasks_with_disjoint_files_are_allowed(tmp_path: Path) -> None:
    _write(tmp_path)
    path = tmp_path / "docs" / "project" / "tasks.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    data["tasks"].append(
        {
            "id": "TASK-101",
            "objective": "Edit a different module",
            "status": "in_progress",
            "owns_files": ["host_finetune/example.py"],
            "acceptance": ["disjoint ownership"],
            "verification": "",
        }
    )
    path.write_text(json.dumps(data), encoding="utf-8")
    assert "task_overlap" not in _codes(tmp_path)


def test_new_experiment_without_manifest_fails(tmp_path: Path) -> None:
    _write(tmp_path)
    (tmp_path / "experiments" / "EXP-20261108-001-demo").mkdir()
    assert "experiment_manifest_missing" in _codes(tmp_path)


def test_complete_manifest_passes_and_incomplete_manifest_warns() -> None:
    ok = check_manifest_text(COMPLETE_MANIFEST, experiment_id="EXP-20261108-001-demo")
    assert not ok.errors
    short = "experiment_id: EXP-20261108-001-demo\nstatus: incomplete\n"
    warned = check_manifest_text(short, experiment_id="EXP-20261108-001-demo")
    assert not warned.errors
    assert warned.findings[0].severity == "warning"
    assert warned.findings[0].code == "manifest_missing_fields"


def test_known_unregistered_experiment_warns(tmp_path: Path) -> None:
    _write(tmp_path)
    assert "known_unregistered_experiment" in _codes(tmp_path)


def test_missing_evidence_path_fails(tmp_path: Path) -> None:
    _write(tmp_path)
    state_path = tmp_path / "docs" / "project" / "state.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    state["evidence"].append("docs/missing.md")
    state_path.write_text(json.dumps(state), encoding="utf-8")
    assert "missing_evidence" in _codes(tmp_path)


def test_worktree_path_rules() -> None:
    assert _porcelain_path(" M docs/project/state.json") == "docs/project/state.json"
    assert _porcelain_path("?? experiments/EXP-20261103-002-model-f-desc-unified-panel/") == (
        "experiments/EXP-20261103-002-model-f-desc-unified-panel/"
    )
    assert _allowed("docs/project/state.json", ["docs/project/"])
    assert not _allowed("host_finetune/train.py", ["docs/project/"])


def test_hooks_emit_json(tmp_path: Path) -> None:
    start = hook_session_start()
    assert "HANDOFF.md" in start["additional_context"]
    _write(tmp_path, task_status="done")
    stop = hook_stop(tmp_path)
    assert "task_unverified" in stop["followup_message"]


def test_live_repository_check() -> None:
    report = check_repository(Path(__file__).resolve().parents[1], live_git=True)
    assert report.to_dict()["error_count"] == 0, report.to_dict()
