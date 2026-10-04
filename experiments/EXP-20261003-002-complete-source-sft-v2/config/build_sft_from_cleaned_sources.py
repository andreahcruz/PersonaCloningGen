"""Create QLoRA-style SFT rows from the checked-in ``cleaned/`` source corpus.

Writes ``dataset_from_cleaned_sources.jsonl`` rather than replacing
``dataset.jsonl``. X posts use a 12-word floor so short posts stay in the
corpus. Blog, LinkedIn, and YouTube keep a 50-word floor.

Run from the repository root::

    python -m host_finetune.build_sft_from_cleaned_sources \
      --cleaned-dir data/cleaned
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from host_finetune.sft_chunk_utils import instruction_for, split_long_document


SOURCES = {
    "jasonlemkin_blog.jsonl": ("blog", "content", "title"),
    "jasonlemkinlinkedin.jsonl": ("linkedin", "content", None),
    "jasonlk_originals.jsonl": ("x", "text", None),
    "jasonmlemkinyoutubetranscripts.jsonl": ("youtube_jason", "transcript_text", "video_title"),
    "saastryoutubetranscripts.jsonl": ("youtube_saastr", "transcript_text", "video_title"),
}

MIN_WORDS = {
    "x": 12,
}
DEFAULT_MIN_WORDS = 50

FALLBACK_TOPIC = {
    "blog": "a B2B SaaS topic",
    "linkedin": "a SaaS founder takeaway",
    "x": "a sharp SaaS observation",
    "youtube_jason": "a SaaStr lesson",
    "youtube_saastr": "a SaaStr lesson",
}


def _title(row: dict, explicit_key: str | None, source: str, text: str) -> str:
    title = (row.get(explicit_key) if explicit_key else "") or ""
    title = str(title).strip()
    if title:
        return title[:500]
    preview = " ".join(text.split())[:120].strip()
    return preview + ("..." if len(preview) == 120 else "") or FALLBACK_TOPIC[source]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cleaned-dir", type=Path, required=True)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("host_finetune/data/dataset_from_cleaned_sources.jsonl"),
    )
    parser.add_argument("--chunk-max-chars", type=int, default=1600)
    args = parser.parse_args()

    if not args.cleaned_dir.is_dir():
        raise SystemExit(f"missing cleaned directory: {args.cleaned_dir}")
    args.out.parent.mkdir(parents=True, exist_ok=True)

    stats: Counter[str] = Counter()
    with args.out.open("w", encoding="utf-8") as dest:
        for filename, (source, text_key, title_key) in SOURCES.items():
            path = args.cleaned_dir / filename
            if not path.is_file():
                print(f"skip missing {path}")
                continue
            floor = MIN_WORDS.get(source, DEFAULT_MIN_WORDS)
            # Iterate physical lines. str.splitlines() also breaks on U+2028,
            # which appears inside some X post bodies and would drop the row.
            with path.open(encoding="utf-8") as source_file:
                for line_no, line in enumerate(source_file, 1):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        row = json.loads(line)
                    except json.JSONDecodeError:
                        stats["bad_json"] += 1
                        continue
                    text = str(row.get(text_key) or "").strip()
                    if len(text.split()) < floor:
                        stats[f"dropped_short_{source}"] += 1
                        continue
                    title = _title(row, title_key, source, text)
                    pieces = split_long_document(text, args.chunk_max_chars)
                    if not pieces:
                        stats[f"dropped_empty_{source}"] += 1
                        continue
                    for part, (heading, chunk) in enumerate(pieces, 1):
                        topic = title
                        if heading and len(pieces) > 1:
                            topic = f"{title} — {heading}"
                            stats["section_headed_chunks"] += 1
                        if len(pieces) > 1:
                            topic = f"{topic} (part {part} of {len(pieces)})"
                        record = {
                            "instruction": instruction_for(source, topic),
                            "input": "",
                            "output": chunk,
                            "source": source,
                            "source_file": filename,
                            "source_line": line_no,
                        }
                        dest.write(json.dumps(record, ensure_ascii=False) + "\n")
                        stats[f"rows_{source}"] += 1
                    stats[f"documents_{source}"] += 1

    stats["rows_total"] = sum(v for k, v in stats.items() if k.startswith("rows_"))
    summary = args.out.with_suffix(".summary.json")
    summary.write_text(json.dumps(dict(sorted(stats.items())), indent=2) + "\n", encoding="utf-8")
    print(f"wrote {stats['rows_total']} rows to {args.out}")
    print(f"summary: {summary}")


if __name__ == "__main__":
    main()
