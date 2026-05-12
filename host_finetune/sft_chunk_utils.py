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


# ── Sentence-aware splitter (clean_dataset_v2.py path) ───────────────────
# Used by the v2 cleaner instead of the legacy paragraph-or-hardcut splitter
# above. Guarantees: never cuts mid-word; prefers sentence boundaries
# (``. ! ?`` + whitespace + capital letter), then paragraph breaks, then word
# boundaries. Required because the legacy ``split_long_body_for_sft`` falls
# back to ``cut = max_chars`` when there's no ``\n\n`` in the window, which
# slices unpunctuated transcripts mid-word (64% of rows in the pre-v2 dataset).
_SENT_BOUND_RE = re.compile(r"(?<=[.!?])\s+(?=[\"'\u201c\u2018(\[]?[A-Z])")
_PARA_BREAK_RE = re.compile(r"\n{2,}")


def _split_sentences(paragraph: str) -> list[str]:
    """Tokenize one paragraph into sentences. Empty paragraphs → []."""
    paragraph = paragraph.strip()
    if not paragraph:
        return []
    parts = _SENT_BOUND_RE.split(paragraph)
    return [p.strip() for p in parts if p.strip()]


def _pack_units(units: list[str], max_chars: int, joiner: str) -> list[str]:
    """Greedily pack ``units`` into chunks <= max_chars joined by ``joiner``."""
    chunks: list[str] = []
    cur = ""
    for u in units:
        if not u:
            continue
        if not cur:
            cur = u
            continue
        candidate = cur + joiner + u
        if len(candidate) <= max_chars:
            cur = candidate
        else:
            chunks.append(cur)
            cur = u
    if cur:
        chunks.append(cur)
    return chunks


def _split_long_sentence_on_words(sentence: str, max_chars: int) -> list[str]:
    """Last-resort: split a sentence longer than max_chars on word boundaries."""
    if len(sentence) <= max_chars:
        return [sentence]
    words = sentence.split()
    out: list[str] = []
    cur = ""
    for w in words:
        if not cur:
            cur = w
        elif len(cur) + 1 + len(w) <= max_chars:
            cur = cur + " " + w
        else:
            out.append(cur)
            cur = w
    if cur:
        out.append(cur)
    return out


def split_sentence_aware(body: str, max_chars: int) -> list[str]:
    """Sentence-aware splitter; never cuts mid-word.

    Order of preference for chunk boundaries:
      1. paragraph break (``\\n\\n``)
      2. sentence end (``.`` / ``!`` / ``?`` + whitespace + capital letter)
      3. word boundary (single space) — only when one sentence > max_chars

    Returns a list of non-empty chunks; never returns empty string entries.
    """
    body = (body or "").strip()
    if not body:
        return []
    if max_chars <= 0 or len(body) <= max_chars:
        return [body]

    paragraphs = [p.strip() for p in _PARA_BREAK_RE.split(body) if p.strip()]
    if len(paragraphs) <= 1:
        # No paragraph structure (transcripts / single-paragraph posts):
        # split on sentence boundaries directly.
        sentences = _split_sentences(body)
        if not sentences:
            return _split_long_sentence_on_words(body, max_chars)
        packed = _pack_units(sentences, max_chars, joiner=" ")
        oversized: list[str] = []
        for p in packed:
            if len(p) <= max_chars:
                oversized.append(p)
            else:
                oversized.extend(_split_long_sentence_on_words(p, max_chars))
        return oversized

    # Multi-paragraph body: first try to keep paragraphs whole (packed by
    # ``\n\n``); if a single paragraph exceeds max_chars, recurse on it via
    # the single-paragraph path above.
    packed_paragraphs: list[str] = []
    for p in paragraphs:
        if len(p) <= max_chars:
            packed_paragraphs.append(p)
        else:
            packed_paragraphs.extend(split_sentence_aware(p, max_chars))

    return _pack_units(packed_paragraphs, max_chars, joiner="\n\n")


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
    # Match ``clean_dataset_v2`` / RAG rebuild: sentence-aware cuts only.
    chunks = split_sentence_aware(out, max_chars)
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
