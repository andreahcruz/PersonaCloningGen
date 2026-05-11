"""Split long SFT labels into rows sized for ~MAX_SEQ_LENGTH tokens (16 GB VRAM).

Logic mirrors ``spark_jobs/clean_and_embed.py`` — keep both in sync when changing rules.

Default chunk ~1600 characters pairs with ``MAX_SEQ_LENGTH=512`` (same ratio as
3200/1024; ~350–425 tokens of ``output`` after Alpaca template + instruction).
"""
from __future__ import annotations

import json
import os
import re

MIN_OUTPUT_CHARS = 80

# Default aligns with ``host_finetune.config.SFT_CHUNK_OUTPUT_CHARS`` (1600 @ MAX_SEQ_LENGTH 512).
DEFAULT_CHUNK_OUTPUT_CHARS = int(os.environ.get("SFT_CHUNK_OUTPUT_CHARS", "1600"))

_TOPIC_RE = re.compile(
    r"^Write in the style of Jason Lemkin about:\s*(.+?)\s*$",
    re.DOTALL | re.IGNORECASE,
)


def _split_oversized_segment(chunk: str, max_chars: int) -> list[str]:
    """Split one segment into pieces each <= ``max_chars`` (paragraph cuts preferred)."""
    chunk = chunk.strip()
    if not chunk:
        return []
    if len(chunk) <= max_chars:
        return [chunk]
    out: list[str] = []
    rest = chunk
    while len(rest) > max_chars:
        window = rest[:max_chars]
        cut = window.rfind("\n\n")
        if cut < max_chars // 5:
            cut = max_chars
        piece = rest[:cut].strip()
        rest = rest[cut:].lstrip()
        if piece:
            out.append(piece)
    if rest.strip():
        out.append(rest.strip())
    return out


def split_long_body_for_sft(body: str, max_chars: int) -> list[str]:
    """Non-overlapping segments <= max_chars; prefer paragraph breaks.

    Never merge a short tail into the previous chunk if that would exceed
    ``max_chars``; oversized pieces are split again so labels stay within cap.
    """
    body = body.strip()
    if not body:
        return []
    if max_chars <= 0 or len(body) <= max_chars:
        return [body]

    parts: list[str] = []
    rest = body
    while rest:
        if len(rest) <= max_chars:
            p = rest.strip()
            if p:
                parts.append(p)
            break
        window = rest[:max_chars]
        cut = window.rfind("\n\n")
        if cut < max_chars // 5:
            cut = max_chars
        piece = rest[:cut].strip()
        rest = rest[cut:].lstrip()
        if piece:
            parts.append(piece)

    merged: list[str] = []
    for p in parts:
        if merged and len(p) < MIN_OUTPUT_CHARS:
            cand = merged[-1] + "\n\n" + p
            if len(cand) <= max_chars:
                merged[-1] = cand.strip()
                continue
        merged.append(p)

    capped: list[str] = []
    for m in merged:
        if len(m) <= max_chars:
            capped.append(m)
        else:
            capped.extend(_split_oversized_segment(m, max_chars))

    final: list[str] = []
    for x in capped:
        if len(x) >= MIN_OUTPUT_CHARS:
            final.append(x)
        elif final:
            cand = final[-1] + "\n\n" + x
            if len(cand) <= max_chars:
                final[-1] = cand.strip()
            else:
                final.append(x)
        else:
            final.append(x)

    return final if final else [body[:max_chars]]


def extract_base_topic(instruction: str) -> str:
    m = _TOPIC_RE.match((instruction or "").strip())
    if not m:
        return (instruction or "").strip() or "this topic"
    inner = m.group(1).strip()
    inner = re.sub(r"\s*\(part\s+\d+\s+of\s+\d+\)\s*$", "", inner, flags=re.I)
    return inner.strip() or "this topic"


def expand_record(
    record: dict,
    max_chars: int = DEFAULT_CHUNK_OUTPUT_CHARS,
) -> list[dict]:
    """Return one or more Alpaca records; splits long ``output`` into parts."""
    instr = record.get("instruction") or ""
    inp = record.get("input") or ""
    out = (record.get("output") or "").strip()
    if not out:
        return []
    base = extract_base_topic(instr)
    chunks = split_long_body_for_sft(out, max_chars)
    if len(chunks) <= 1:
        return [{"instruction": instr, "input": inp, "output": out}]
    rows: list[dict] = []
    n = len(chunks)
    for i, chunk in enumerate(chunks):
        topic = base if n == 1 else f"{base} (part {i + 1} of {n})"
        rows.append(
            {
                "instruction": f"Write in the style of Jason Lemkin about: {topic}",
                "input": inp,
                "output": chunk,
            }
        )
    return rows


def expand_jsonl_text(
    lines: list[str],
    max_chars: int = DEFAULT_CHUNK_OUTPUT_CHARS,
) -> tuple[list[str], dict]:
    stats = {"rows_in": 0, "rows_out": 0, "split_from_long": 0}
    out_lines: list[str] = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        stats["rows_in"] += 1
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        expanded = expand_record(rec, max_chars=max_chars)
        if len(expanded) > 1:
            stats["split_from_long"] += 1
        for r in expanded:
            out_lines.append(json.dumps(r, ensure_ascii=False))
            stats["rows_out"] += 1
    return out_lines, stats
