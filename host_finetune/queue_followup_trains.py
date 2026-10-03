"""Wait for the current fine-tune, then run two follow-ups on one GPU.

Run 1 drops SaaStr multi-speaker talks and caps X so its row count matches
blog. Hyperparameters match the batch-1 baseline. Run 2 uses that same file
with a wider LoRA and a lower learning rate.

Create ``experiments/training_queue.STOP`` to cancel before the next launch.
"""
from __future__ import annotations

import json
import os
import subprocess
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STOP = ROOT / "experiments" / "training_queue.STOP"
QUEUE_LOG = ROOT / "experiments" / "training_queue.log"
PYTHON_TRAIN = ROOT / "host_finetune" / ".venv" / "Scripts" / "python.exe"
PYTHON_CPU = ROOT / ".venv-test" / "Scripts" / "python.exe"
SOURCE_DATASET = ROOT / "host_finetune" / "data" / "dataset_from_cleaned_sources_fit512.jsonl"
BALANCED_DATASET = ROOT / "host_finetune" / "data" / "dataset_fit512_no_saastr_xcap.jsonl"
BALANCED_SPLIT = ROOT / "experiments" / "EXP-20261001-004-balanced-sft-split"


def log(message: str) -> None:
    line = time.strftime("%Y-%m-%d %H:%M:%S") + " " + message
    print(line, flush=True)
    QUEUE_LOG.parent.mkdir(parents=True, exist_ok=True)
    with QUEUE_LOG.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def select_balanced_rows(rows: list[dict]) -> tuple[list[dict], dict]:
    """Drop SaaStr talks and keep whole X documents until X rows match blog."""
    blog_rows = sum(1 for row in rows if row.get("source") == "blog")
    x_docs: dict[tuple[str, str], list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        if row.get("source") != "x":
            continue
        key = (str(row.get("source_file") or ""), str(row.get("source_line") or ""))
        x_docs[key].append(index)
    chosen: set[int] = set()
    used = 0
    for key in sorted(x_docs):
        if used >= blog_rows:
            break
        for index in x_docs[key]:
            chosen.add(index)
        used += len(x_docs[key])
    kept = []
    for index, row in enumerate(rows):
        source = row.get("source")
        if source == "youtube_saastr":
            continue
        if source == "x" and index not in chosen:
            continue
        kept.append(row)
    stats = {
        "rows_in": len(rows),
        "rows_out": len(kept),
        "blog_rows": blog_rows,
        "x_rows_kept": used,
        "x_documents_kept": sum(1 for indexes in x_docs.values() if any(i in chosen for i in indexes)),
        "dropped_youtube_saastr": sum(1 for row in rows if row.get("source") == "youtube_saastr"),
    }
    return kept, stats


def finetune_pids() -> set[int]:
    script = (
        "Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | "
        "Where-Object { $_.CommandLine -like '*host_finetune.finetune*' } | "
        "ForEach-Object { $_.ProcessId }"
    )
    result = subprocess.run(
        ["powershell", "-NoProfile", "-Command", script],
        capture_output=True,
        text=True,
        check=False,
    )
    found = set()
    for line in result.stdout.splitlines():
        line = line.strip()
        if line.isdigit():
            found.add(int(line))
    return found


def wait_until_clear(block: set[int]) -> None:
    while True:
        if STOP.is_file():
            raise SystemExit(f"queue stopped by {STOP}")
        still = finetune_pids() & block
        if not still:
            return
        log(f"waiting for finetune pid {sorted(still)}")
        time.sleep(60)


def build_balanced() -> None:
    rows = []
    with SOURCE_DATASET.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    kept, stats = select_balanced_rows(rows)
    BALANCED_DATASET.parent.mkdir(parents=True, exist_ok=True)
    with BALANCED_DATASET.open("w", encoding="utf-8") as handle:
        for row in kept:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    log(f"balanced dataset {stats} -> {BALANCED_DATASET}")


def write_split() -> None:
    subprocess.run(
        [
            str(PYTHON_CPU),
            "-m",
            "host_finetune.split_groups",
            "--dataset",
            str(BALANCED_DATASET),
            "--out-dir",
            str(BALANCED_SPLIT),
        ],
        cwd=ROOT,
        check=True,
    )


def train(name: str, adapter: Path, log_path: Path, extra_env: dict[str, str]) -> None:
    if STOP.is_file():
        raise SystemExit(f"queue stopped by {STOP}")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.update({
        "PYTHONUNBUFFERED": "1",
        "NUM_EPOCHS": "2",
        "TRAIN_OPTIM": "adamw_8bit",
        "MAX_VRAM_FRACTION": "0.85",
        "PER_DEVICE_BATCH": "1",
        "GRAD_ACCUM": "16",
        "EVAL_STEPS": "400",
        "SAVE_STEPS": "400",
        "SAVE_TOTAL_LIMIT": "3",
        "FINETUNE_DATASET": str(BALANCED_DATASET),
        "SPLIT_MANIFEST": str(BALANCED_SPLIT / "raw" / "split_assignments.jsonl"),
        "ADAPTER_DIR": str(adapter),
    })
    env.update(extra_env)
    log(f"starting {name} -> {log_path}")
    with log_path.open("w", encoding="utf-8") as handle:
        proc = subprocess.Popen(
            [str(PYTHON_TRAIN), "-u", "-m", "host_finetune.finetune"],
            cwd=ROOT,
            env=env,
            stdout=handle,
            stderr=subprocess.STDOUT,
        )
        code = proc.wait()
    if code != 0:
        raise SystemExit(f"{name} exited {code}; see {log_path}")
    log(f"finished {name}")


def main() -> None:
    running = finetune_pids()
    log(f"queue armed; current finetune pids={sorted(running) or 'none'}")
    if running:
        wait_until_clear(running)
    log("gpu free; building the no-SaaStr X-capped file")
    build_balanced()
    write_split()
    train(
        "balanced-r16",
        ROOT / "host_finetune" / "output" / "lemkin_lora_balanced",
        ROOT / "experiments" / "EXP-20261001-004-balanced-qlora" / "raw" / "train.log",
        {"LORA_R": "16", "LORA_ALPHA": "32", "LEARNING_RATE": "0.0001"},
    )
    train(
        "balanced-r32",
        ROOT / "host_finetune" / "output" / "lemkin_lora_r32",
        ROOT / "experiments" / "EXP-20261001-005-lora-r32" / "raw" / "train.log",
        {"LORA_R": "32", "LORA_ALPHA": "64", "LEARNING_RATE": "0.00005"},
    )
    log("queue complete")


if __name__ == "__main__":
    main()
