"""Command line for the shared project-state check."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from project_pm.check import check_manifest_text, check_repository, hook_session_start, hook_stop


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Check the DATA 298B operational project files.")
    parser.add_argument("--root", type=Path, default=Path.cwd())
    sub = parser.add_subparsers(dest="command", required=True)

    check = sub.add_parser("check", help="validate state, tasks, handoff, and live git")
    check.add_argument("--no-git", action="store_true", help="skip the live worktree comparison")

    manifest = sub.add_parser("check-manifest", help="validate one experiment manifest")
    manifest.add_argument("path", type=Path)

    sub.add_parser("cursor-session-start", help="print Cursor sessionStart hook JSON")
    sub.add_parser("cursor-stop", help="print Cursor stop hook JSON")
    args = parser.parse_args(argv)

    if args.command == "cursor-session-start":
        _consume_stdin()
        print(json.dumps(hook_session_start()))
        return 0
    if args.command == "cursor-stop":
        _consume_stdin()
        print(json.dumps(hook_stop(args.root)))
        return 0
    if args.command == "check-manifest":
        path = args.path if args.path.is_absolute() else args.root / args.path
        text = path.read_text(encoding="utf-8")
        experiment_id = path.parent.name if path.parent.name.startswith("EXP-") else None
        report = check_manifest_text(text, experiment_id=experiment_id)
    else:
        report = check_repository(args.root, live_git=not args.no_git)
    print(json.dumps(report.to_dict(), indent=2))
    return 1 if report.errors else 0


def _consume_stdin() -> None:
    if sys.stdin is None or sys.stdin.closed:
        return
    try:
        sys.stdin.read()
    except Exception:
        return


if __name__ == "__main__":
    raise SystemExit(main())
