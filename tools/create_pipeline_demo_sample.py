#!/usr/bin/env python3
"""Create a tiny, representative corpus for the EC2 pipeline proof run.

The sample intentionally uses the same five source files and schemas as the
cleaned corpus.  It is for demonstrating extract -> Spark clean/embed -> Chroma
on AWS; it must never replace the full collection used by PersonaRAG.
"""

from __future__ import annotations

import argparse
from pathlib import Path


SOURCE_FILES = (
    "jasonlemkin_blog.jsonl",
    "jasonlemkinlinkedin.jsonl",
    "jasonmlemkinyoutubetranscripts.jsonl",
    "saastryoutubetranscripts.jsonl",
    "jasonlk_originals.jsonl",
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=Path("data/cleaned"))
    parser.add_argument("--output", type=Path, default=Path("data/deploy/pipeline_sample"))
    parser.add_argument("--per-source", type=int, default=3)
    args = parser.parse_args()

    if args.per_source < 1:
        raise SystemExit("--per-source must be at least 1")
    args.output.mkdir(parents=True, exist_ok=True)

    total = 0
    for filename in SOURCE_FILES:
        source_path = args.source / filename
        rows: list[str] = []
        with source_path.open(encoding="utf-8") as source_file:
            for line in source_file:
                if line.strip():
                    rows.append(line.rstrip("\n"))
                if len(rows) == args.per_source:
                    break
        if len(rows) < args.per_source:
            raise SystemExit(f"{source_path} only has {len(rows)} nonempty rows")
        (args.output / filename).write_text("\n".join(rows) + "\n", encoding="utf-8")
        total += len(rows)

    print(f"Created {total} records in {args.output}")


if __name__ == "__main__":
    main()
