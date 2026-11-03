"""Commands for the four-arm experiment. Retrieval and generation stay opt-in.

    host_finetune\\.venv\\Scripts\\python.exe -X utf8 -m host_finetune.model_f_experiment plan ^
        --questions host_finetune\\benchmarks\\dev_retrieval_benchmark.jsonl ^
        --out experiments\\EXP-20261101-009-model-f-four-arm ^
        --snapshot PATH\\TO\\SNAPSHOT

    host_finetune\\.venv\\Scripts\\python.exe -X utf8 -m host_finetune.model_f_experiment evaluate ^
        --out experiments\\EXP-20261101-009-model-f-four-arm
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from host_finetune.model_f_experiment.evaluate import evaluate_directory
from host_finetune.model_f_experiment.desc_vs_rel import (
    cell_report,
    compare_stage,
    run_stage,
    score_stage,
    write_stage_manifest,
)
from host_finetune.model_f_experiment.pilot import (
    run_pilot,
    score_pilot,
    select_arm,
    write_pilot_manifest,
)
from host_finetune.model_f_experiment.unified import (
    execute_selected_profile,
    judge_saved_panel,
    panel_status,
    score_saved_panel,
    write_plan,
)
from host_finetune.model_f_experiment.graph_audit import audit_directory, audit_records, export_audit_records
from host_finetune.model_f_experiment.runner import execute_four_arms, load_questions, pending, write_manifest
from host_finetune.model_f_experiment.snapshot import copy_snapshot, verify_snapshot


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    plan = sub.add_parser("plan")
    plan.add_argument("--questions", type=Path, required=True)
    plan.add_argument("--out", type=Path, required=True)
    plan.add_argument("--snapshot", required=True)
    evaluate = sub.add_parser("evaluate")
    evaluate.add_argument("--out", type=Path, required=True)
    audit = sub.add_parser("audit")
    audit.add_argument("--snapshot", type=Path, required=True)
    snap = sub.add_parser("snapshot")
    snap.add_argument("--source", type=Path, required=True)
    snap.add_argument("--dest", type=Path, required=True)
    snap.add_argument("--allow-expansion-source", action="store_true")
    verify = sub.add_parser("verify")
    verify.add_argument("--snapshot", type=Path, required=True)
    run = sub.add_parser("run")
    run.add_argument("--questions", type=Path, required=True)
    run.add_argument("--out", type=Path, required=True)
    run.add_argument("--snapshot", type=Path, required=True)
    unified = sub.add_parser("unified-plan")
    unified.add_argument("--profile", required=True)
    unified.add_argument("--snapshot", type=Path, required=True)
    unified.add_argument("--out", type=Path, required=True)
    unified_run = sub.add_parser("unified-run")
    unified_run.add_argument("--profile", required=True)
    unified_run.add_argument("--snapshot", type=Path, required=True)
    unified_run.add_argument("--out", type=Path, required=True)
    unified.add_argument("--selection", type=Path)
    unified_run.add_argument("--selection", type=Path)
    unified_score = sub.add_parser("unified-score")
    unified_score.add_argument("--profile", required=True)
    unified_score.add_argument("--out", type=Path, required=True)
    unified_judge = sub.add_parser("unified-judge")
    unified_judge.add_argument("--profile", required=True)
    unified_judge.add_argument("--out", type=Path, required=True)
    unified_status = sub.add_parser("unified-status")
    unified_status.add_argument("--profile", required=True)
    unified_status.add_argument("--out", type=Path, required=True)
    pilot_plan = sub.add_parser("pilot-plan")
    pilot_plan.add_argument("--questions", type=Path, required=True)
    pilot_plan.add_argument("--out", type=Path, required=True)
    pilot_plan.add_argument("--snapshot", required=True)
    pilot_plan.add_argument("--max-runtime-minutes", type=float, default=60)
    pilot_run = sub.add_parser("pilot-run")
    pilot_run.add_argument("--out", type=Path, required=True)
    pilot_run.add_argument("--snapshot", type=Path, required=True)
    pilot_run.add_argument("--max-runtime-minutes", type=float, default=60)
    pilot_score = sub.add_parser("pilot-score")
    pilot_score.add_argument("--out", type=Path, required=True)
    pilot_select = sub.add_parser("pilot-select")
    pilot_select.add_argument("--out", type=Path, required=True)
    stage_plan = sub.add_parser("stage2-plan")
    stage_plan.add_argument("--questions", type=Path, required=True)
    stage_plan.add_argument("--out", type=Path, required=True)
    stage_plan.add_argument("--snapshot", required=True)
    stage_plan.add_argument("--exclude-manifest", type=Path)
    stage_plan.add_argument("--per-medium", type=int, default=8)
    stage_run = sub.add_parser("stage2-run")
    stage_run.add_argument("--out", type=Path, required=True)
    stage_run.add_argument("--snapshot", type=Path, required=True)
    stage_run.add_argument("--max-runtime-minutes", type=float, default=180)
    stage_score = sub.add_parser("stage2-score")
    stage_score.add_argument("--out", type=Path, required=True)
    stage_compare = sub.add_parser("stage2-compare")
    stage_compare.add_argument("--out", type=Path, required=True)
    stage_status = sub.add_parser("stage2-status")
    stage_status.add_argument("--out", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    if args.command == "plan":
        questions = load_questions(args.questions)
        manifest = write_manifest(args.out, questions, snapshot=args.snapshot)
        remaining = pending(args.out, questions)
        print(
            f"fingerprint {manifest['fingerprint']} questions {len(questions)} pending cells {len(remaining)}",
            flush=True,
        )
        return
    if args.command == "evaluate":
        report = evaluate_directory(args.out)
        print(
            f"scored arms {sorted((report['metrics'].get('arms') or {}))} composite {report['composite_score']}",
            flush=True,
        )
        return
    if args.command == "audit":
        if (args.snapshot / "graph_audit_export.json").is_file():
            report = audit_directory(args.snapshot)
        else:
            report = audit_records(export_audit_records(args.snapshot))
            (args.snapshot.parent / f"{args.snapshot.name}_graph_audit.json").write_text(
                json.dumps(report, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
        print(json.dumps({"findings": len(report["findings"]), "recovery_applied": False}), flush=True)
        return
    if args.command == "verify":
        report = verify_snapshot(args.snapshot)
        print(json.dumps(report, sort_keys=True), flush=True)
        return
    if args.command == "run":
        questions = load_questions(args.questions)
        write_manifest(args.out, questions, snapshot=str(args.snapshot))
        report = execute_four_arms(args.out, questions, args.snapshot)
        print(json.dumps(report, sort_keys=True), flush=True)
        return
    if args.command == "unified-plan":
        plan = write_plan(args.out, args.profile, args.snapshot, args.selection)
        print(
            f"protocol {plan['protocol']} profile {plan['profile']} cells {plan['cells']} status {plan['status']}",
            flush=True,
        )
        return
    if args.command == "unified-run":
        report = execute_selected_profile(args.out, args.profile, args.snapshot, args.selection)
        print(json.dumps(report, sort_keys=True), flush=True)
        return
    if args.command == "unified-score":
        coverage = score_saved_panel(args.out, args.profile)
        print(json.dumps(coverage, sort_keys=True), flush=True)
        return
    if args.command == "unified-judge":
        report = judge_saved_panel(args.out, args.profile)
        print(json.dumps(report, sort_keys=True), flush=True)
        return
    if args.command == "unified-status":
        print(json.dumps(panel_status(args.out, args.profile), sort_keys=True), flush=True)
        return
    if args.command == "pilot-plan":
        manifest = write_pilot_manifest(
            args.out,
            args.questions,
            args.snapshot,
            max_runtime_minutes=args.max_runtime_minutes,
        )
        print(
            f"cells {manifest['cells']} questions {manifest['questions_selected']} "
            f"estimate_minutes {manifest['runtime_preflight']['full_estimate']['estimated_minutes']}",
            flush=True,
        )
        return
    if args.command == "pilot-run":
        report = run_pilot(args.out, args.snapshot, max_runtime_minutes=args.max_runtime_minutes)
        print(json.dumps(report, sort_keys=True), flush=True)
        return
    if args.command == "pilot-score":
        report = score_pilot(args.out)
        print(json.dumps(report, sort_keys=True), flush=True)
        return
    if args.command == "pilot-select":
        report = select_arm(args.out)
        print(json.dumps({"selected_arm": report["selected_arm"], "status": report["status"]}, sort_keys=True), flush=True)
        return
    if args.command == "stage2-plan":
        manifest = write_stage_manifest(
            args.out,
            args.questions,
            args.snapshot,
            exclude_manifest=args.exclude_manifest,
            per_medium=args.per_medium,
        )
        feasibility = manifest["feasibility"]
        print(
            f"design {feasibility['design_status']} questions {manifest['questions_selected']} "
            f"cells {manifest['cells']} per_medium {feasibility['per_medium']} "
            f"shortfall {feasibility['shortfall_questions']}",
            flush=True,
        )
        print(feasibility["reason"], flush=True)
        return
    if args.command == "stage2-run":
        report = run_stage(args.out, args.snapshot, max_runtime_minutes=args.max_runtime_minutes)
        print(json.dumps(report, sort_keys=True), flush=True)
        return
    if args.command == "stage2-score":
        report = score_stage(args.out)
        print(json.dumps({"cells": report.get("cells"), "failures": report.get("failures")}, sort_keys=True), flush=True)
        return
    if args.command == "stage2-compare":
        report = compare_stage(args.out)
        print(json.dumps(report, sort_keys=True), flush=True)
        return
    if args.command == "stage2-status":
        print(json.dumps(cell_report(args.out), sort_keys=True), flush=True)
        return
    if args.command == "snapshot":
        report = copy_snapshot(
            args.source,
            args.dest,
            allow_expansion_source=args.allow_expansion_source,
        )
        print(f"copied {report['files']} files hash {report['hash_of_hashes']}", flush=True)
        return
    raise SystemExit(f"unknown command {args.command}")


if __name__ == "__main__":
    main()
