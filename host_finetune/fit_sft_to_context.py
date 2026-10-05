"""Compatibility command for the frozen keep-breaks fit.

The transformation lives in ``canonical_fit``. This module keeps the historical
command line and the helper names existing tests import. It does not contain
a second packer.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from host_finetune.canonical_fit import (
    FitConfig,
    _n_tokens,
    _short_topic,
    _split_output,
    atomic_blocks,
    fit_rows,
    merge_lead_in_rows,
    move_dangling_lead_ins,
)

__all__ = [
    "_n_tokens",
    "_short_topic",
    "_split_output",
    "atomic_blocks",
    "merge_lead_in_rows",
    "move_dangling_lead_ins",
    "main",
]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Re-chunk SFT JSONL with the frozen Llama chat-template token budget."
    )
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--max-seq-length", type=int, default=512)
    parser.add_argument("--max-prompt-tokens", type=int, default=120)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
    loaded: list[dict] = []
    with args.dataset.open(encoding="utf-8") as source:
        for raw in source:
            if raw.strip():
                loaded.append(json.loads(raw))
    result = fit_rows(
        loaded,
        tokenizer,
        FitConfig(max_tokens=args.max_seq_length, prompt_cap=args.max_prompt_tokens),
    )
    target = args.out or args.dataset.with_name(args.dataset.stem + "_fit512.jsonl")
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as dest:
        for part in result.parts:
            dest.write(json.dumps(part.record, ensure_ascii=False) + "\n")
    print(
        f"wrote {len(result.parts)} rows from {result.merged_rows}; "
        f"shortened_titles={result.shortened_titles}; dropped={result.dropped}; out={target}"
    )


if __name__ == "__main__":
    main()
