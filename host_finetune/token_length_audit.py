"""Audit *formatted* SFT examples for truncation before QLoRA training.

The audit applies the Llama chat template, so it counts the instruction and
chat headers as well as the answer.  Use the same tokenizer model as the
training job; this is the check missing from character-only chunking.

Example::

    python -m host_finetune.token_length_audit \
      --dataset host_finetune/data/dataset_from_cleaned_sources.jsonl \
      --tokenizer unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def _chat_text(tokenizer, row: dict) -> str:
    instruction = str(row.get("instruction") or "").strip()
    user = instruction
    if row.get("input"):
        user += "\n\n" + str(row["input"]).strip()
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": user}, {"role": "assistant", "content": str(row.get("output") or "")}],
        tokenize=False,
        add_generation_prompt=False,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--max-seq-length", type=int, default=512)
    parser.add_argument("--report", type=Path, default=None)
    args = parser.parse_args()

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
    counts: list[tuple[int, int, dict]] = []
    for index, line in enumerate(args.dataset.read_text(encoding="utf-8").splitlines()):
        if not line.strip():
            continue
        row = json.loads(line)
        n_tokens = len(tokenizer(_chat_text(tokenizer, row), add_special_tokens=False)["input_ids"])
        counts.append((n_tokens, index + 1, row))

    lengths = sorted(n for n, _, _ in counts)
    over = [(n, line, row) for n, line, row in counts if n > args.max_seq_length]
    report = {
        "dataset": str(args.dataset),
        "tokenizer": args.tokenizer,
        "max_seq_length": args.max_seq_length,
        "examples": len(counts),
        "max_tokens": max(lengths, default=0),
        "p50_tokens": lengths[len(lengths) // 2] if lengths else 0,
        "p95_tokens": lengths[min(len(lengths) - 1, int(len(lengths) * 0.95))] if lengths else 0,
        "over_limit": len(over),
        "over_limit_examples": [
            {"line": line, "tokens": n, "instruction": row.get("instruction", "")}
            for n, line, row in sorted(over, reverse=True)[:50]
        ],
    }
    target = args.report or args.dataset.with_suffix(".token_audit.json")
    target.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    if over:
        raise SystemExit(f"FAIL: {len(over)} examples exceed {args.max_seq_length} tokens; see {target}")
    print(f"PASS: no examples exceed {args.max_seq_length} tokens")


if __name__ == "__main__":
    main()
