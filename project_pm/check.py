"""Validate the operational project snapshot without scoring models."""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass, field
from pathlib import Path


REQUIRED_MANIFEST_PATHS = (
    "experiment_id",
    "timestamp",
    "execution_commands",
    "git.commit",
    "git.dirty",
    "data.source_manifest",
    "data.source_hashes",
    "data.processed_dataset_hash",
    "data.split_manifest_hash",
    "data.gold_eval_hash",
    "rag.persona_profile_hash",
    "rag.chroma_collection",
    "rag.index_version_or_snapshot",
    "rag.embedding_model",
    "model.model_name",
    "model.model_or_adapter_hash",
    "model.config",
    "model.random_seed",
    "evaluation.evaluator_commit",
    "evaluation.judge_model",
    "evaluation.judge_config",
    "environment.python_version",
    "environment.package_snapshot",
    "environment.hardware",
    "outputs.raw",
    "outputs.metrics",
    "outputs.warnings_and_failures",
)

HANDOFF_KEYS = (
    "updated",
    "git_commit",
    "git_branch",
    "active_task",
    "next_action",
)

TASK_STATUSES = {"proposed", "in_progress", "blocked", "done"}


@dataclass
class Finding:
    severity: str
    code: str
    message: str

    def to_dict(self) -> dict:
        return {"severity": self.severity, "code": self.code, "message": self.message}


@dataclass
class Report:
    findings: list[Finding] = field(default_factory=list)

    def add(self, severity: str, code: str, message: str) -> None:
        self.findings.append(Finding(severity, code, message))

    @property
    def errors(self) -> list[Finding]:
        return [item for item in self.findings if item.severity == "error"]

    def to_dict(self) -> dict:
        return {
            "ok": not self.errors,
            "error_count": len(self.errors),
            "warning_count": len(self.findings) - len(self.errors),
            "findings": [item.to_dict() for item in self.findings],
        }


def check_repository(root: Path, *, live_git: bool = False) -> Report:
    root = Path(root)
    report = Report()
    state = _read_json(root / "docs" / "project" / "state.json", report, "state_json")
    tasks_doc = _read_json(root / "docs" / "project" / "tasks.json", report, "tasks_json")
    handoff = _read_handoff(root / "docs" / "project" / "HANDOFF.md", report)
    if state is None or tasks_doc is None or handoff is None:
        return report

    _check_state_shape(state, report)
    tasks = _check_tasks(tasks_doc, report)
    _check_handoff(state, tasks, handoff, report)
    _check_decisions(root, tasks, report)
    _check_experiments(root, state, report)
    _check_references(root, state, report)
    if live_git:
        _check_git(root, state, tasks, report)
    return report


def check_manifest_text(text: str, *, experiment_id: str | None = None) -> Report:
    report = Report()
    present = yaml_key_paths(text)
    missing = [key for key in REQUIRED_MANIFEST_PATHS if key not in present]
    if missing:
        report.add("error", "manifest_missing_fields", "manifest is missing: " + ", ".join(missing))
    declared = _scalar_after(text, "experiment_id")
    if experiment_id and declared and declared != experiment_id:
        report.add(
            "error",
            "manifest_id_mismatch",
            f"manifest experiment_id is {declared}, directory is {experiment_id}",
        )
    if _scalar_after(text, "status") == "incomplete" and missing:
        report.findings = [
            Finding("warning", item.code, item.message) if item.code == "manifest_missing_fields" else item
            for item in report.findings
        ]
    return report


def yaml_key_paths(text: str) -> set[str]:
    """Collect dotted keys from a small nested YAML mapping. Not a full parser."""
    paths: set[str] = set()
    stack: list[tuple[int, str]] = []
    for raw in text.splitlines():
        if not raw.strip() or raw.lstrip().startswith("#"):
            continue
        indent = len(raw) - len(raw.lstrip(" "))
        stripped = raw.strip()
        if stripped.startswith("- "):
            stripped = stripped[2:].strip()
            if ":" not in stripped:
                continue
        if ":" not in stripped:
            continue
        key = stripped.split(":", 1)[0].strip()
        if not key or not key.replace("_", "").isalnum():
            continue
        while stack and stack[-1][0] >= indent:
            stack.pop()
        path = ".".join([name for _, name in stack] + [key])
        paths.add(path)
        stack.append((indent, key))
    return paths


def hook_session_start() -> dict:
    return {
        "additional_context": (
            "DATA 298B handoff: read docs/project/HANDOFF.md, then docs/project/state.json "
            "for the phase and next task. Follow AGENTS.md. Do not load docs/research/ "
            "unless the task is architectural, experimental, or a report change. "
            "Before finishing substantial work, run: python -m project_pm check"
        )
    }


def hook_stop(root: Path) -> dict:
    report = check_repository(root, live_git=True)
    if not report.errors:
        return {}
    lines = [f"{item.code}: {item.message}" for item in report.errors[:5]]
    return {
        "followup_message": (
            "Project-state check failed. Fix the operational files before treating "
            "this task as finished. Do not mark a task done without verification.\n"
            + "\n".join(lines)
        )
    }


def _read_json(path: Path, report: Report, code: str) -> dict | None:
    if not path.is_file():
        report.add("error", code, f"missing {path.as_posix()}")
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        report.add("error", code, f"{path.name} is not valid JSON: {exc}")
        return None
    if not isinstance(data, dict):
        report.add("error", code, f"{path.name} must be an object")
        return None
    return data


def _read_handoff(path: Path, report: Report) -> dict | None:
    if not path.is_file():
        report.add("error", "handoff_missing", "docs/project/HANDOFF.md is missing")
        return None
    fields: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.startswith("- "):
            continue
        body = line[2:]
        if ": " not in body:
            continue
        key, value = body.split(": ", 1)
        key = key.strip()
        if key.replace("_", "").isalpha() and key not in fields:
            fields[key] = value.strip()
    missing = [key for key in HANDOFF_KEYS if key not in fields]
    if missing:
        report.add("error", "handoff_fields", "HANDOFF.md is missing: " + ", ".join(missing))
        return None
    return fields


def _check_state_shape(state: dict, report: Report) -> None:
    git = state.get("git")
    if not isinstance(git, dict) or not git.get("commit") or not git.get("branch"):
        report.add("error", "state_git", "state.json git.branch and git.commit are required")
    if not state.get("updated"):
        report.add("error", "state_updated", "state.json updated is required")
    if not state.get("next_task"):
        report.add("error", "state_next_task", "state.json next_task is required")
    evaluation = state.get("evaluation")
    if not isinstance(evaluation, dict) or not evaluation.get("role"):
        report.add("error", "state_evaluation", "state.json evaluation.role is required")


def _check_tasks(tasks_doc: dict, report: Report) -> list[dict]:
    rows = tasks_doc.get("tasks")
    if not isinstance(rows, list):
        report.add("error", "tasks_list", "tasks.json tasks must be a list")
        return []
    seen: set[str] = set()
    active: list[dict] = []
    for task in rows:
        if not isinstance(task, dict):
            report.add("error", "task_shape", "each task must be an object")
            continue
        task_id = str(task.get("id") or "")
        if not task_id or task_id in seen:
            report.add("error", "task_id", f"duplicate or empty task id: {task_id or '<blank>'}")
            continue
        seen.add(task_id)
        status = task.get("status")
        if status not in TASK_STATUSES:
            report.add("error", "task_status", f"{task_id} has unknown status {status!r}")
            continue
        if not task.get("objective") or not task.get("acceptance"):
            report.add("error", "task_fields", f"{task_id} needs objective and acceptance")
        if status == "done" and not task.get("verification"):
            report.add("error", "task_unverified", f"{task_id} is done without verification")
        if status == "in_progress":
            active.append(task)
        owns = task.get("owns_files") or []
        if status == "in_progress" and not owns:
            report.add("error", "task_ownership", f"{task_id} is in progress without owns_files")
    _check_overlap(active, report)
    return rows


def _check_overlap(active: list[dict], report: Report) -> None:
    for index, left in enumerate(active):
        for right in active[index + 1 :]:
            if _paths_overlap(left.get("owns_files") or [], right.get("owns_files") or []):
                report.add(
                    "error",
                    "task_overlap",
                    f"{left.get('id')} and {right.get('id')} own overlapping paths",
                )


def _paths_overlap(left: list, right: list) -> bool:
    for raw_left in left:
        for raw_right in right:
            a = str(raw_left).rstrip("/")
            b = str(raw_right).rstrip("/")
            if a == b or a.startswith(b + "/") or b.startswith(a + "/"):
                return True
    return False


def _check_handoff(state: dict, tasks: list[dict], handoff: dict, report: Report) -> None:
    if handoff.get("updated") != state.get("updated"):
        report.add("warning", "handoff_stale_date", "HANDOFF.md updated does not match state.json")
    git = state.get("git") if isinstance(state.get("git"), dict) else {}
    if handoff.get("git_commit") != git.get("commit"):
        report.add("error", "handoff_stale_commit", "HANDOFF.md git_commit does not match state.json")
    if handoff.get("git_branch") != git.get("branch"):
        report.add("error", "handoff_stale_branch", "HANDOFF.md git_branch does not match state.json")
    active = handoff.get("active_task")
    ids = {task.get("id") for task in tasks if isinstance(task, dict)}
    if active not in ids:
        report.add("error", "handoff_unknown_task", f"active task {active} is not in tasks.json")
    elif active != state.get("next_task"):
        matching = next(task for task in tasks if task.get("id") == active)
        if matching.get("status") == "done":
            report.add("warning", "handoff_finished_task", "HANDOFF.md still names a done task")
    if handoff.get("session") == "interrupted":
        report.add("warning", "session_interrupted", "HANDOFF.md records an interrupted session")
    if not handoff.get("next_action"):
        report.add("error", "handoff_next", "HANDOFF.md next_action is empty")


def _check_decisions(root: Path, tasks: list[dict], report: Report) -> None:
    path = root / "docs" / "DECISION_LOG.md"
    text = path.read_text(encoding="utf-8") if path.is_file() else ""
    if not path.is_file():
        report.add("error", "decision_log_missing", "docs/DECISION_LOG.md is missing")
    for task in tasks:
        if not isinstance(task, dict) or task.get("status") != "done":
            continue
        decision_id = task.get("decision_id")
        if decision_id and f"## {decision_id} " not in text and f"## {decision_id} —" not in text:
            report.add("error", "decision_missing", f"{task.get('id')} requires {decision_id} in the decision log")


def _check_experiments(root: Path, state: dict, report: Report) -> None:
    experiments = state.get("experiments") if isinstance(state.get("experiments"), dict) else {}
    cutoff = str(experiments.get("validate_id_date_on_or_after") or "")
    warned = set(experiments.get("known_unregistered") or [])
    base = root / "experiments"
    if not base.is_dir() or not cutoff:
        return
    for path in sorted(base.glob("EXP-*")):
        if not path.is_dir():
            continue
        stamp = path.name[4:12]
        if not stamp.isdigit() or stamp < cutoff:
            continue
        manifest = path / "manifest.yaml"
        if not manifest.is_file():
            report.add("error", "experiment_manifest_missing", f"{path.name} has no manifest.yaml")
            continue
        result = check_manifest_text(manifest.read_text(encoding="utf-8"), experiment_id=path.name)
        for item in result.findings:
            report.add(item.severity, item.code, f"{path.name}: {item.message}")
    for relative in warned:
        manifest = root / relative / "manifest.yaml"
        if not manifest.is_file():
            report.add(
                "warning",
                "known_unregistered_experiment",
                f"{relative} has no manifest.yaml and is not a registered result",
            )


def _check_references(root: Path, state: dict, report: Report) -> None:
    for relative in state.get("evidence") or []:
        if not (root / str(relative)).exists():
            report.add("error", "missing_evidence", f"state.json evidence path does not exist: {relative}")


def _check_git(root: Path, state: dict, tasks: list[dict], report: Report) -> None:
    git = state.get("git") if isinstance(state.get("git"), dict) else {}
    try:
        head = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        branch = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        porcelain = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.splitlines()
    except (OSError, subprocess.CalledProcessError) as exc:
        report.add("warning", "git_unavailable", f"live git check skipped: {exc}")
        return
    if head != git.get("commit"):
        report.add("error", "stale_commit", f"HEAD is {head}, state.json says {git.get('commit')}")
    if branch != git.get("branch"):
        report.add("error", "stale_branch", f"branch is {branch}, state.json says {git.get('branch')}")
    allowed = [str(item).replace("\\", "/") for item in (git.get("known_dirty_prefixes") or [])]
    owned = []
    for task in tasks:
        if isinstance(task, dict) and task.get("status") == "in_progress":
            owned.extend(str(item).replace("\\", "/") for item in (task.get("owns_files") or []))
    for line in porcelain:
        path = _porcelain_path(line)
        if path is None:
            continue
        if _allowed(path, allowed) or _allowed(path, owned):
            continue
        report.add("error", "unlisted_worktree_change", f"worktree change is not listed in state or an active task: {path}")


def _porcelain_path(line: str) -> str | None:
    if len(line) < 4:
        return None
    path = line[3:].strip().replace("\\", "/")
    if " -> " in path:
        path = path.split(" -> ", 1)[1]
    return path.strip('"')


def _allowed(path: str, prefixes: list[str]) -> bool:
    for prefix in prefixes:
        cleaned = prefix.rstrip("/")
        if path == cleaned or path.startswith(cleaned + "/"):
            return True
    return False


def _scalar_after(text: str, key: str) -> str | None:
    prefix = key + ":"
    for raw in text.splitlines():
        stripped = raw.strip()
        if stripped.startswith(prefix):
            return stripped[len(prefix) :].strip().strip('"').strip("'") or None
    return None
