"""Clean scraped Lemkin JSONL without modifying the raw files.

Reads the five source files under ``data/`` and writes cleaned copies to
``data/cleaned/`` plus ``cleaning_summary.json``. Gold-eval JSONL is not an input.

Run from the repo root::

    python -m host_finetune.clean_scraped
"""
from __future__ import annotations

import argparse
import html
import json
import re
import sys
from collections import Counter
from pathlib import Path

_SPARK_JOBS = Path(__file__).resolve().parents[1] / "spark_jobs"
if str(_SPARK_JOBS) not in sys.path:
    sys.path.insert(0, str(_SPARK_JOBS))
from corpus_footer_scrub import scrub_corpus_text

REPO = Path(__file__).resolve().parents[1]
DEFAULT_DATA = REPO / "data"
DEFAULT_OUT = REPO / "data" / "cleaned"

# Same minimum the Airflow X normalizer uses.
MIN_X_CHARS = 25

# filename, kind, text field, title field, extra scrub kwargs
_SOURCES = (
    ("jasonlemkin_blog.jsonl", "blog", "content", "title", {}),
    ("jasonlemkinlinkedin.jsonl", "linkedin", "content", None, {"linkedin": True}),
    ("jasonlk_originals.jsonl", "x", "text", None, {}),
    ("jasonmlemkinyoutubetranscripts.jsonl", "youtube", "transcript_text", "video_title", {"unescape": True}),
    ("saastryoutubetranscripts.jsonl", "youtube", "transcript_text", "video_title", {"unescape": True}),
)

_EXAMPLE_KINDS = (
    "dropped_event_promo",
    "dropped_linkedin_chrome",
    "blog_footer_removed",
    "repaired_url",
    "unescaped_html",
)


def _preview(text: str, limit: int = 280) -> str:
    collapsed = " ".join((text or "").split())
    if len(collapsed) <= limit:
        return collapsed
    return collapsed[: limit - 1] + "…"


def _tail(text: str, limit: int = 220) -> str:
    collapsed = " ".join((text or "").split())
    if len(collapsed) <= limit:
        return collapsed
    return "…" + collapsed[-limit:]


def _footer_window(text: str) -> str:
    match = re.search(r"Related Posts.{0,160}", text or "", re.IGNORECASE | re.DOTALL)
    if not match:
        return _tail(text)
    return " ".join(match.group(0).split())


def _url_window(text: str) -> str:
    match = re.search(r".{0,50}https?://(?:[ \t\r\n]+\S|\S+).{0,40}", text or "", re.IGNORECASE)
    if not match:
        return _preview(text)
    return match.group(0).replace("\r", "").replace("\n", "\\n")


def _entity_window(text: str) -> str:
    match = re.search(r".{0,40}&(?:#\d+|#x[0-9a-fA-F]+|[a-zA-Z]+);.{0,40}", text or "")
    if not match:
        return _preview(text)
    return " ".join(match.group(0).split())


def clean_row(kind: str, row: dict, text_field: str, title_field: str | None, scrub_kwargs: dict) -> tuple[dict | None, str | None, dict]:
    """Return ``(cleaned_row, drop_reason, flags)``."""
    flags = {"footer": False, "url": False, "hashtag": False, "html": False}
    if kind == "x" and (row.get("is_reply") or row.get("is_repost_or_quote")):
        return None, "reply_or_repost", flags
    raw = row.get(text_field)
    raw_text = raw if isinstance(raw, str) else ""
    if kind == "x":
        stripped = raw_text.strip()
        if not stripped:
            return None, "empty", flags
        if len(stripped) < MIN_X_CHARS:
            return None, "too_short", flags
    title = ""
    if title_field:
        title = row.get(title_field) or ""
        if not isinstance(title, str):
            title = str(title)
    cleaned, reason, flags = scrub_corpus_text(raw_text, title=title, **scrub_kwargs)
    if reason == "empty" and kind == "youtube":
        reason = "empty_transcript"
    if reason:
        return None, reason, flags
    out = dict(row)
    out[text_field] = cleaned
    return out, None, flags


def _maybe_example(examples: dict, kind: str, payload: dict) -> None:
    current = examples.get(kind)
    if current is None:
        examples[kind] = payload
        return
    # Prefer a caption entity that is visible after unescape (``&gt;``), not ``&nbsp;``.
    if kind == "unescaped_html" and "&gt;" not in current.get("before", "") and "&gt;" in payload.get("before", ""):
        examples[kind] = payload


def clean_file(path: Path, kind: str, text_field: str, title_field: str | None, scrub_kwargs: dict, out_path: Path) -> dict:
    rows_in = 0
    rows_out = 0
    bad_json = 0
    dropped: Counter = Counter()
    changed: Counter = Counter()
    examples: dict = {}

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("r", encoding="utf-8") as src, out_path.open("w", encoding="utf-8") as dst:
        for line in src:
            if not line.strip():
                continue
            rows_in += 1
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                bad_json += 1
                dropped["bad_json"] += 1
                continue
            if not isinstance(row, dict):
                bad_json += 1
                dropped["bad_json"] += 1
                continue
            original = row.get(text_field) if isinstance(row.get(text_field), str) else ""
            cleaned, reason, flags = clean_row(kind, row, text_field, title_field, scrub_kwargs)
            if reason:
                dropped[reason] += 1
                if reason == "event_promo":
                    title = row.get(title_field) if title_field else ""
                    _maybe_example(examples, "dropped_event_promo", {
                        "file": path.name,
                        "title": _preview(title if isinstance(title, str) else "", 140),
                        "before": _preview(original),
                    })
                elif reason == "linkedin_feed_chrome":
                    _maybe_example(examples, "dropped_linkedin_chrome", {
                        "file": path.name,
                        "before": _preview(original),
                    })
                continue
            for key, hit in flags.items():
                if hit:
                    changed[key] += 1
            if flags.get("footer"):
                _maybe_example(examples, "blog_footer_removed", {
                    "file": path.name,
                    "title": _preview(row.get("title") or "", 140),
                    "before_tail": _footer_window(original),
                    "after_tail": _tail(cleaned[text_field]),
                })
            if flags.get("url"):
                _maybe_example(examples, "repaired_url", {
                    "file": path.name,
                    "before": _url_window(original),
                    "after": _url_window(cleaned[text_field]),
                })
            if flags.get("html"):
                before_ent = _entity_window(original)
                _maybe_example(examples, "unescaped_html", {
                    "file": path.name,
                    "title": _preview(row.get("video_title") or "", 140),
                    "before": before_ent,
                    "after": html.unescape(before_ent),
                })
            dst.write(json.dumps(cleaned, ensure_ascii=False) + "\n")
            rows_out += 1

    return {
        "file": path.name,
        "rows_in": rows_in,
        "rows_out": rows_out,
        "dropped": dict(dropped),
        "changed": dict(changed),
        "bad_json": bad_json,
        "examples": examples,
    }


def clean_scraped(data_dir: Path, out_dir: Path) -> dict:
    files = []
    examples = {}
    for filename, kind, text_field, title_field, scrub_kwargs in _SOURCES:
        src = data_dir / filename
        if not src.is_file():
            files.append({"file": filename, "error": "missing"})
            continue
        stats = clean_file(src, kind, text_field, title_field, scrub_kwargs, out_dir / filename)
        for kind_name, payload in stats.pop("examples").items():
            # The broken ``https://`` + newline form shows up in the X export.
            if kind_name == "repaired_url" and payload.get("file") == "jasonlk_originals.jsonl":
                examples[kind_name] = payload
            else:
                _maybe_example(examples, kind_name, payload)
        files.append(stats)
    summary = {
        "data_dir": str(data_dir),
        "out_dir": str(out_dir),
        "files": files,
        "examples": {name: examples[name] for name in _EXAMPLE_KINDS if name in examples},
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "cleaning_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    summary = clean_scraped(args.data_dir, args.out_dir)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
