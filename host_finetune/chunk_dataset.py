"""Expand long ``output`` fields into multiple SFT rows (local JSONL, no Spark).

Usage:
    python -m host_finetune.chunk_dataset host_finetune/data/dataset.jsonl
    python -m host_finetune.chunk_dataset in.jsonl out.jsonl --max-chars 1600

Uses ``SFT_CHUNK_OUTPUT_CHARS`` (default 1600 for MAX_SEQ_LENGTH 512) unless ``--max-chars`` is set.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from host_finetune.sft_chunk_utils import DEFAULT_CHUNK_OUTPUT_CHARS, expand_jsonl_text


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("input", type=Path, help="Input dataset.jsonl")
    p.add_argument(
        "output",
        type=Path,
        nargs="?",
        default=None,
        help="Output path (default: overwrite input)",
    )
    p.add_argument(
        "--max-chars",
        type=int,
        default=None,
        help=f"Max chars per output chunk (default env SFT_CHUNK_OUTPUT_CHARS or {DEFAULT_CHUNK_OUTPUT_CHARS})",
    )
    args = p.parse_args()
    inp = args.input
    out = args.output or inp
    max_chars = args.max_chars if args.max_chars is not None else DEFAULT_CHUNK_OUTPUT_CHARS

    lines = inp.read_text(encoding="utf-8").splitlines()
    new_lines, stats = expand_jsonl_text(lines, max_chars=max_chars)
    Path(out).write_text("\n".join(new_lines) + "\n", encoding="utf-8")
    print(
        f"rows_in={stats['rows_in']} rows_out={stats['rows_out']} "
        f"long_docs_split={stats['split_from_long']} max_chars={max_chars}"
    )


if __name__ == "__main__":
    main()
