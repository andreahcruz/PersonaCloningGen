#!/usr/bin/env python3
"""
Upload prepared JSONL files and create an OpenAI supervised fine-tuning job.

This script assumes the dataset has already been created by
openai_ft/prepare_nano_dataset.py.
"""

from __future__ import annotations

import argparse
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = "gpt-4.1-nano"


def main() -> None:
    parser = argparse.ArgumentParser(description="Create an OpenAI supervised fine-tuning job.")
    parser.add_argument(
        "--train-file",
        default="data/openai_ft/lemkin_blog_nano_train.jsonl",
        help="Training JSONL file.",
    )
    parser.add_argument(
        "--eval-file",
        default="data/openai_ft/lemkin_blog_nano_eval.jsonl",
        help="Validation JSONL file.",
    )
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Base model to fine-tune.")
    parser.add_argument("--suffix", default="lemkin-blog-nano", help="Job suffix.")
    parser.add_argument("--n-epochs", type=int, default=3, help="Number of supervised fine-tuning epochs.")
    parser.add_argument("--dry-run", action="store_true", help="Print the resolved inputs without calling OpenAI.")
    args = parser.parse_args()

    train_path = REPO_ROOT / args.train_file
    eval_path = REPO_ROOT / args.eval_file
    if not train_path.exists():
        raise SystemExit(f"Training file not found: {train_path}")
    if not eval_path.exists():
        raise SystemExit(f"Validation file not found: {eval_path}")

    if args.dry_run:
        print(f"train_file={train_path}")
        print(f"eval_file={eval_path}")
        print(f"model={args.model}")
        print(f"suffix={args.suffix}")
        print(f"n_epochs={args.n_epochs}")
        return

    try:
        from openai import OpenAI
    except ImportError as err:
        raise SystemExit(
            "OpenAI SDK not installed. Install requirements-openai.txt before running this script."
        ) from err

    client = OpenAI()

    with train_path.open("rb") as train_f:
        train_upload = client.files.create(file=train_f, purpose="fine-tune")
    with eval_path.open("rb") as eval_f:
        eval_upload = client.files.create(file=eval_f, purpose="fine-tune")

    job = client.fine_tuning.jobs.create(
        training_file=train_upload.id,
        validation_file=eval_upload.id,
        model=args.model,
        suffix=args.suffix,
        method={
            "type": "supervised",
            "supervised": {
                "hyperparameters": {"n_epochs": args.n_epochs},
            },
        },
    )

    print(f"training_file_id={train_upload.id}")
    print(f"validation_file_id={eval_upload.id}")
    print(f"fine_tuning_job_id={job.id}")
    if getattr(job, "status", None):
        print(f"status={job.status}")


if __name__ == "__main__":
    main()
