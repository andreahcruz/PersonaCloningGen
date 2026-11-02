"""Write a scored batch to disk. The live graph store is refused."""

from __future__ import annotations

import json
from pathlib import Path

from host_finetune.eval_bench.coverage import COVERAGE, GOLD_DEV_SET, REVIEW_KINDS
from host_finetune.eval_bench.matrix import CONTROLS, EXPERIMENT_MATRIX
from host_finetune.eval_bench.metrics import latency_summary
from host_finetune.eval_bench.schema import SCHEMA_VERSION

LIVE_STORE_NAME = "storage_expand_v1"


def refuse_live_store(path: Path) -> None:
    if LIVE_STORE_NAME in path.resolve().parts:
        raise RuntimeError(f"refusing to read or write the live store: {path}")


def _git() -> dict:
    import subprocess

    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], text=True, stderr=subprocess.DEVNULL).strip())
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "dirty": None}
    return {"commit": commit, "dirty": dirty}


def _jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _fmt(value) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def render_markdown(batch: dict, manifest: dict) -> str:
    lines = [
        "# Evaluation report",
        "",
        f"Schema `{SCHEMA_VERSION}`. Experiment `{manifest['experiment_id']}`.",
        "",
        "Retrieval, faithfulness, numeric checks, and latency are separate. There is no composite score.",
        "",
        "## Arms",
        "",
        "| Architecture | N | P@4 | R@4 | MRR | nDCG@4 | Faithfulness | Numeric pass | Citation leak | Words |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, arm in sorted((batch["metrics"].get("arms") or {}).items()):
        lines.append(
            "| {architecture} | {n} | {p} | {r} | {mrr} | {ndcg} | {faith} | {numeric} | {leak} | {words} |".format(
                architecture=name,
                n=arm["n"],
                p=_fmt(arm["precision_at_4_mean"]),
                r=_fmt(arm["recall_at_4_mean"]),
                mrr=_fmt(arm["mrr_mean"]),
                ndcg=_fmt(arm["ndcg_at_4_mean"]),
                faith=_fmt(arm["faithfulness_rate"]),
                numeric=_fmt(arm["numeric_pass_rate"]),
                leak=_fmt(arm["citation_leak_rate"]),
                words=_fmt(arm["mean_word_count"]),
            )
        )
    lines.extend([
        "",
        "Null ranked metrics mean the questions had no graded qrels. They are not zeros.",
        "",
        "## Failures",
        "",
    ])
    if not batch["failures"]:
        lines.append("None.")
    else:
        counts: dict[str, int] = {}
        for row in batch["failures"]:
            counts[row["error_class"]] = counts.get(row["error_class"], 0) + 1
        for name, count in sorted(counts.items()):
            lines.append(f"- {name}: {count}")
    latency = latency_summary(batch["results"])
    lines.extend([
        "",
        "## Latency",
        "",
        f"- End-to-end p50: {_fmt((latency['end_to_end_seconds'] or {}).get('p50'))}",
        f"- End-to-end p95: {_fmt((latency['end_to_end_seconds'] or {}).get('p95'))}",
        f"- Throughput per second: {_fmt(latency['throughput_per_second'])}",
        "",
        "Throughput counts scored rows only. It is not a quality metric.",
        "",
    ])
    if batch.get("paired"):
        paired = batch["paired"]
        lines.extend(["## Paired faithfulness", ""])
        if paired.get("status") != "scored":
            lines.append(f"Status: {paired.get('status')}. {paired.get('reason', '')}")
        else:
            lines.extend([
                f"{paired['left']} minus {paired['right']}: {_fmt(paired.get('mean_difference'))}",
                f"95 percent bootstrap interval: {paired.get('ci95')}",
                f"Questions: {paired.get('n')}. Seed {paired.get('seed')}.",
            ])
        lines.append("")
    lines.append("Word count is a covariate. A longer answer is not scored as better.")
    return "\n".join(lines) + "\n"


def write_report(directory: Path, batch: dict, *, experiment_id: str, paired: dict | None = None) -> dict:
    refuse_live_store(directory)
    directory.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "experiment_id": experiment_id,
        "git": _git(),
        "scorer": "host_finetune.eval_bench",
        "live_index_queried": False,
        "gpu_inference": False,
        "composite_score": None,
        "matrix": list(EXPERIMENT_MATRIX),
        "controls": CONTROLS,
        "gold_dev_set": GOLD_DEV_SET,
        "review_kinds": REVIEW_KINDS,
        "coverage": list(COVERAGE),
        "n_results": len(batch["results"]),
        "n_failures": len(batch["failures"]),
    }
    payload = dict(batch)
    payload["paired"] = paired
    _jsonl(directory / "results.jsonl", batch["results"])
    _jsonl(directory / "per_question_analysis.jsonl", batch["per_question"])
    _jsonl(directory / "failures.jsonl", batch["failures"])
    (directory / "metrics.json").write_text(
        json.dumps(batch["metrics"], indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (directory / "experiment_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (directory / "report.md").write_text(render_markdown(payload, manifest), encoding="utf-8")
    return manifest
