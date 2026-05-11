"""Count dataset.jsonl rows by upstream source (blog, linkedin, x, youtube_*).

dataset.jsonl does not store ``source``; this script rebuilds normalized bodies from
the same local JSONL files used by ``extract_to_minio``, applies Spark-equivalent
``strip_html``, then matches each training ``output`` (exact or truncated prefix).
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

from bs4 import BeautifulSoup

REPO = Path(__file__).resolve().parent
DATA = REPO / "data"
DATASET = REPO / "host_finetune" / "data" / "dataset.jsonl"

TRUNC_SUFFIX = "\n\n[Truncated for training.]"

FILES = {
    "blog": "jasonlemkin_blog.jsonl",
    "linkedin": "jasonlemkinlinkedin.jsonl",
    "x": "jasonlk_originals.jsonl",
    "youtube_jason": "jasonmlemkinyoutubetranscripts.jsonl",
    "youtube_saastr": "saastryoutubetranscripts.jsonl",
}


def strip_html(text: str | None) -> str:
    if not text:
        return ""
    return BeautifulSoup(text, "html.parser").get_text(separator=" ")


def norm_nl(s: str) -> str:
    return s.replace("\r\n", "\n").replace("\r", "\n")


def body_for(source: str, row: dict) -> str | None:
    """Mirror count_sources / DAG extraction prior to MinIO."""
    if source == "blog":
        return (row.get("content") or row.get("text") or "").strip() or None
    if source == "linkedin":
        return (row.get("content") or "").strip() or None
    if source == "x":
        if row.get("is_reply") or row.get("is_repost_or_quote"):
            return None
        t = (row.get("text") or "").strip()
        return t if len(t) >= 25 else None
    if source.startswith("youtube"):
        return (row.get("transcript_text") or "").strip() or None
    return None


def word_count(t: str) -> int:
    return len(t.split())


def load_catalog() -> list[tuple[str, str]]:
    """(source, spark_like_text) after strip_html, Spark-like filters (>=50 words)."""
    catalog: list[tuple[str, str]] = []
    for src, fname in FILES.items():
        path = DATA / fname
        if not path.is_file():
            continue
        with path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                raw_body = body_for(src, row)
                if not raw_body:
                    continue
                text = strip_html(raw_body).strip()
                if not text:
                    continue
                if word_count(text) < 50:
                    continue
                catalog.append((src, norm_nl(text)))
    return catalog


def strip_trunc_suffix(out: str) -> str:
    o = norm_nl(out)
    if o.endswith(TRUNC_SUFFIX):
        o = o[: -len(TRUNC_SUFFIX)].rstrip()
    return o


def match_output(o_clean: str, catalog: list[tuple[str, str]]) -> str | None:
    """Return source or None if no confident match."""
    if not o_clean:
        return None
    # Exact match (prefer first hit; duplicates rare)
    for src, full in catalog:
        if full == o_clean:
            return src
    # Label truncated in Spark / clean_dataset: output is a prefix of full body
    best_src: str | None = None
    best_full_len = -1
    for src, full in catalog:
        if full.startswith(o_clean) and len(full) > len(o_clean):
            if len(full) > best_full_len:
                best_full_len = len(full)
                best_src = src
    if best_src is not None:
        return best_src
    # Rare: entire doc shorter than trunc threshold but padding differs
    for src, full in catalog:
        if o_clean.startswith(full) and full == o_clean[: len(full)]:
            return src
    return None


def main() -> None:
    dataset_path = Path(sys.argv[1]) if len(sys.argv) > 1 else DATASET
    if not dataset_path.is_file():
        print(f"Missing dataset: {dataset_path}", file=sys.stderr)
        sys.exit(1)

    catalog = load_catalog()
    if not catalog:
        print(f"No catalog entries — check JSONL files under {DATA}", file=sys.stderr)
        sys.exit(1)

    counts: dict[str, int] = {s: 0 for s in FILES}
    counts["unmatched"] = 0

    n = 0
    with dataset_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            out = rec.get("output") or ""
            o_clean = strip_trunc_suffix(out)
            src = match_output(o_clean, catalog)
            if src is None:
                counts["unmatched"] += 1
            else:
                counts[src] += 1
            n += 1

    print(f"dataset: {dataset_path}")
    print(f"total_json_rows: {n}")
    print(f"catalog_documents_ge50_words_post_html_strip: {len(catalog)}")
    print()
    w = max(len("source"), max(len(k) for k in counts))
    for key in list(FILES.keys()) + ["unmatched"]:
        label = key if key != "unmatched" else "unmatched (no local match)"
        pct = (100.0 * counts[key] / n) if n else 0.0
        print(f"{label:<{w}} {counts[key]:>8}  ({pct:5.1f}%)")
    print(f"{'TOTAL':<{w}} {n:>8}  (100.0%)")
    if counts["unmatched"]:
        print(
            "\nUnmatched rows usually mean dataset.jsonl was built from a different MinIO "
            "snapshot than these local data/*.jsonl files, or text normalization differs "
            "(re-scrape / encoding).",
            file=sys.stderr,
        )


if __name__ == "__main__":
    main()
