"""Rescore saved generation traces. Does not train, generate, or edit source experiments.

Run from the repo root::

    .venv-test\\Scripts\\python.exe -m host_finetune.rescore_unified
"""
from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from host_finetune.eval_gold_rag import load_format_specs
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
    build_reference_pack,
    evaluate_row,
    rank_models,
    summarize_model,
)
from host_finetune.voice_distance import iter_train_texts

ROOT = Path(__file__).resolve().parents[1]
SPECS = ROOT / "formats" / "format_specs.yaml"
OUT = ROOT / "experiments" / "EXP-20261004-007-unified-rescore"
SOURCES = (
    ("EXP-20261001-007-mixed-medium-compare", ("cleaned", "balanced", "r32")),
    ("EXP-20261001-010-repaired-and-baselines", ("llama3.1", "lemkin-clone-v5", "lemkin-clone-v6", "repaired")),
    ("EXP-20261004-003-repaired-vs-relabel", ("repaired", "relabel")),
    ("EXP-20261004-004-repaired-vs-relabel-retrieval", ("repaired", "relabel")),
    ("EXP-20261004-005-repaired-vs-relabel-seed43", ("repaired", "relabel")),
    ("EXP-20261004-006-repaired-vs-relabel-seed44", ("repaired", "relabel")),
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_traces(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def base_controls() -> dict[str, list[str]]:
    path = ROOT / "experiments" / "EXP-20261001-010-repaired-and-baselines" / "raw" / "llama3.1.jsonl"
    grouped: dict[str, list[str]] = {}
    for row in load_traces(path):
        text = (row.get("answer") or "").strip()
        medium = row.get("medium") or ""
        if text and medium:
            grouped.setdefault(medium, []).append(text)
    return grouped


def public_media(pack: dict) -> dict:
    media = {}
    for medium, info in pack["media"].items():
        media[medium] = {
            "n_profile": info["n_profile"],
            "n_real_controls": info["n_real_controls"],
            "n_base_controls": info["n_base_controls"],
            "style_role": info["style_role"],
            "perspective_role": info["perspective_role"],
            "style_calibration": info["style_calibration"],
            "perspective_calibration": info["perspective_calibration"],
        }
    return media


def git_commit() -> tuple[str, bool]:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=False
    )
    dirty = subprocess.run(
        ["git", "status", "--porcelain"], cwd=ROOT, capture_output=True, text=True, check=False
    )
    return (commit.stdout.strip() or "unknown"), bool(dirty.stdout.strip())


def write_manifest(source_hashes: dict[str, str], warnings: list[str]) -> None:
    commit, dirty = git_commit()
    manifest = {
        "experiment_id": OUT.name,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "execution_commands": [f"{sys.executable} -m host_finetune.rescore_unified"],
        "git": {"commit": commit, "dirty": dirty},
        "data": {
            "source_manifest": "saved generation traces listed in config/sources.json",
            "source_hashes": source_hashes,
            "processed_dataset_hash": sha256(DEFAULT_DATASET),
            "split_manifest_hash": sha256(DEFAULT_ASSIGNMENTS),
            "gold_eval_hash": sha256(ROOT / "data" / "openai_ft" / "lemkin_gold_eval.jsonl"),
        },
        "rag": {
            "persona_profile_hash": "n/a: rescore does not build a profile file; held-out posts are the style reference",
            "chroma_collection": "n/a: no retrieval was executed in this rescore",
            "index_version_or_snapshot": "n/a",
            "embedding_model": "n/a",
        },
        "model": {
            "model_name": "n/a: no model was loaded",
            "model_or_adapter_hash": "n/a",
            "config": "evaluation only",
            "random_seed": "n/a: no sampling",
        },
        "evaluation": {
            "evaluator_commit": commit,
            "judge_model": "n/a: writing-quality judge was not run",
            "judge_config": "writing_quality.py schema only; role diagnostic_only until calibration",
        },
        "environment": {
            "python_version": platform.python_version(),
            "package_snapshot": "config/package_snapshot.txt",
            "hardware": platform.platform(),
        },
        "outputs": {
            "raw": "n/a: source traces were not copied",
            "metrics": "metrics/summary.json",
            "warnings_and_failures": warnings,
        },
    }
    (OUT / "manifest.yaml").write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    if OUT.exists() and any(OUT.iterdir()):
        raise SystemExit(f"{OUT} already exists; refusing to overwrite a registered experiment")
    for name in ("config", "metrics", "figures", "raw"):
        (OUT / name).mkdir(parents=True)
    print("loading held-out references", flush=True)
    dataset = load_jsonl(DEFAULT_DATASET)
    assignments = load_jsonl(DEFAULT_ASSIGNMENTS)
    dispositions = load_jsonl(DEFAULT_GOLD_DISPOSITIONS)
    pack = build_reference_pack(dataset, assignments, dispositions, base_controls())
    print("building copy index", flush=True)
    copy_index = CopyIndex(iter_train_texts(dataset, assignments))
    specs = load_format_specs(SPECS)
    source_hashes = {}
    models = []
    selections = {}
    warnings = [
        "Writing-quality judge was not run, so persona utility is unavailable.",
        "BERTScore was not computed.",
        "Early EOS is unavailable for EXP-20261001 traces that did not record stop metadata.",
        "gold_rubric.overall is brief/task agreement, not persona quality.",
        "Perspective fidelity is framing-rate distance, not claim-level stance entailment.",
    ]
    rows_path = OUT / "metrics" / "rows.jsonl"
    with rows_path.open("w", encoding="utf-8") as handle:
        for experiment, names in SOURCES:
            group = []
            for name in names:
                path = ROOT / "experiments" / experiment / "raw" / f"{name}.jsonl"
                if not path.is_file():
                    warnings.append(f"missing {path}")
                    continue
                source_hashes[str(path.relative_to(ROOT)).replace("\\", "/")] = sha256(path)
                traces = load_traces(path)
                print(f"scoring {experiment} {name} n={len(traces)}", flush=True)
                scored = []
                for trace in traces:
                    if "query" not in trace:
                        trace = {**trace, "query": trace.get("topic", "")}
                    row = evaluate_row(trace, pack, copy_index, specs)
                    row["model"] = name
                    row["source_experiment"] = experiment
                    scored.append(row)
                    handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                model_id = f"{experiment}/{name}"
                summary = summarize_model(scored, model_id, experiment)
                models.append(summary)
                group.append(summary)
            if group:
                selections[experiment] = rank_models(group)
    summary = {
        "schema": SCHEMA,
        "utility_weights": UTILITY_WEIGHTS,
        "weights_are_project_decision_not_universal": True,
        "limits": pack["limits"],
        "reference_counts": pack["reference_counts"],
        "blocked_group_count": pack["blocked_group_count"],
        "calibration": public_media(pack),
        "writing_quality_calibration": pack["writing_quality_calibration"],
        "selection_by_experiment": selections,
        "models": models,
        "not_recomputed": [
            "bertscore",
            "writing_quality judge scores",
            "latency except where a source row already stored latency_seconds",
            "early_eot where the source row has no stop metadata",
        ],
    }
    (OUT / "metrics" / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )
    (OUT / "config" / "sources.json").write_text(
        json.dumps({"sources": [{"experiment": exp, "models": list(names)} for exp, names in SOURCES]}, indent=2) + "\n",
        encoding="utf-8",
    )
    freeze = subprocess.run(
        [sys.executable, "-m", "pip", "freeze"], cwd=ROOT, capture_output=True, text=True, check=False
    )
    (OUT / "config" / "package_snapshot.txt").write_text(freeze.stdout, encoding="utf-8")
    notes = """# Unified rescore

Question: on the saved drafts, which models pass completion, copying, format, and applicable grounding gates, and how do the separate persona, perspective, writing, task, and diagnostic scores look?

Held fixed: saved answer text, prompts, splits, and adapters. No model was loaded and no source experiment summary was overwritten.

Writing quality and BERTScore were not recomputed because those judge and embedding outputs are not in the saved traces. Persona utility stays empty until writing quality is calibrated. Perspective fidelity is framing-rate distance against held-out Lemkin posts, not claim entailment.
"""
    (OUT / "notes.md").write_text(notes, encoding="utf-8")
    write_manifest(source_hashes, warnings)
    print(json.dumps(selections, indent=2), flush=True)


if __name__ == "__main__":
    main()
