"""Re-chunk SFT JSONL with the real Llama chat-template token budget.

Counts the complete user+assistant training text. Shortens pathological
scraped titles and splits answers on paragraph and sentence boundaries until
every emitted record fits ``MAX_SEQ_LENGTH``. Newlines stay. A colon lead-in
is kept with the list that follows it. The medium named in the instruction
is preserved.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from host_finetune.sft_chunk_utils import (
    _split_sentences,
    extract_base_topic,
    style_prefix,
)


def _text(tokenizer, instruction: str, output: str) -> str:
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": instruction}, {"role": "assistant", "content": output}],
        tokenize=False,
        add_generation_prompt=False,
    )


def _n_tokens(tokenizer, instruction: str, output: str) -> int:
    return len(tokenizer(_text(tokenizer, instruction, output), add_special_tokens=False)["input_ids"])


def _short_topic(tokenizer, prefix: str, topic: str, max_prompt_tokens: int) -> str:
    """Keep a meaningful title, but never let scraped title junk consume context."""
    topic = " ".join((topic or "").split())
    if not topic:
        return "a SaaS founder lesson"

    def fits(value: str) -> bool:
        return _n_tokens(tokenizer, prefix + value, "") <= max_prompt_tokens

    if fits(topic):
        return topic
    words = topic.split()
    while len(words) > 4:
        candidate = " ".join(words).rstrip(" ,;:-") + "..."
        if fits(candidate):
            return candidate
        words.pop()
    return "a SaaS founder lesson"


_LIST_OR_HEADING_RE = re.compile(
    r"^(?:#{1,3}\s+\S+|[-*]|\d{1,2}[.)])\s+\S"
)


def _paragraphs(text: str) -> list[str]:
    return [part.strip() for part in re.split(r"\n{2,}", (text or "").strip()) if part.strip()]


def _ends_with_colon(text: str) -> bool:
    return text.rstrip().endswith(":")


def _is_list_or_heading(text: str) -> bool:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return bool(lines) and all(_LIST_OR_HEADING_RE.match(line) for line in lines)


def atomic_blocks(text: str) -> list[str]:
    """Keep a colon lead-in, a heading, and the list under it in one block."""
    paragraphs = _paragraphs(text)
    if not paragraphs:
        stripped = (text or "").strip()
        return [stripped] if stripped else []
    blocks: list[str] = []
    index = 0
    while index < len(paragraphs):
        block = paragraphs[index]
        index += 1
        started_list = False
        while index < len(paragraphs):
            nxt = paragraphs[index]
            if _ends_with_colon(block):
                block = block + "\n\n" + nxt
                index += 1
                started_list = True
                continue
            if _is_list_or_heading(nxt) and (started_list or _is_list_or_heading(block)):
                block = block + "\n\n" + nxt
                index += 1
                started_list = True
                continue
            break
        blocks.append(block)
    return blocks


def _segments(text: str) -> tuple[list[str], str]:
    """Split one block without throwing away its newlines."""
    if "\n\n" in text:
        return _paragraphs(text), "\n\n"
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if len(lines) > 1:
        return lines, "\n"
    sentences = _split_sentences(text)
    if len(sentences) > 1:
        return sentences, " "
    words = text.split()
    if len(words) > 1:
        return words, " "
    return [text], " "


def _split_trailing_lead_in(piece: str) -> tuple[str, str]:
    """Move a final colon lead-in onto the next training example."""
    if "\n\n" in piece:
        parts, sep = _paragraphs(piece), "\n\n"
    elif "\n" in piece:
        parts, sep = [line.strip() for line in piece.splitlines() if line.strip()], "\n"
    else:
        parts, sep = [piece], " "
    if not parts:
        return "", ""
    if _ends_with_colon(parts[-1]) and len(parts) > 1:
        return sep.join(parts[:-1]), parts[-1]
    if len(parts) == 1 and _ends_with_colon(parts[0]):
        return "", parts[0]
    return piece, ""


def move_dangling_lead_ins(pieces: list[str]) -> list[str]:
    """A training answer should not be only the words before a list."""
    kept: list[str] = []
    carry = ""
    for piece in pieces:
        if carry:
            piece = (carry + "\n\n" + piece).strip()
            carry = ""
        head, tail = _split_trailing_lead_in(piece)
        if head:
            kept.append(head)
        if tail:
            carry = tail
    if carry:
        if kept:
            kept[-1] = kept[-1] + "\n\n" + carry
        else:
            kept.append(carry)
    return [piece for piece in kept if piece.strip()]


def _flush_piece(pieces: list[str], current: str) -> str:
    """Close a piece. A trailing colon lead-in starts the next piece instead."""
    if not current:
        return ""
    head, tail = _split_trailing_lead_in(current)
    if head:
        pieces.append(head)
    return tail


def _pack_segments(
    tokenizer,
    prefix: str,
    topic: str,
    segments: list[str],
    joiner: str,
    limit: int,
) -> list[str]:
    instruction = prefix + topic
    pieces: list[str] = []
    current = ""
    for segment in segments:
        candidate = segment if not current else current + joiner + segment
        if candidate and _n_tokens(tokenizer, instruction, candidate) <= limit:
            current = candidate
            continue
        current = _flush_piece(pieces, current)
        candidate = segment if not current else current + joiner + segment
        if candidate and _n_tokens(tokenizer, instruction, candidate) <= limit:
            current = candidate
            continue
        if current and _n_tokens(tokenizer, instruction, current) <= limit:
            pieces.append(current)
        current = ""
        if _n_tokens(tokenizer, instruction, segment) <= limit:
            current = segment
            continue
        finer, fine_joiner = _segments(segment)
        if len(finer) == 1 and finer[0] == segment:
            current = segment
            continue
        sub = _pack_segments(tokenizer, prefix, topic, finer, fine_joiner, limit)
        if sub:
            pieces.extend(sub[:-1])
            current = sub[-1]
    current = _flush_piece(pieces, current)
    if current:
        pieces.append(current)
    return pieces


def _split_output(tokenizer, prefix: str, topic: str, output: str, max_tokens: int) -> list[str]:
    instruction = prefix + topic
    limit = max_tokens - 8  # leaves room for `(part k of N)` after packing
    blocks = atomic_blocks(output)
    pieces = _pack_segments(tokenizer, prefix, topic, blocks, "\n\n", limit)
    return [
        piece for piece in pieces
        if piece.strip() and _n_tokens(tokenizer, instruction, piece) <= max_tokens
    ]


def merge_lead_in_rows(rows: list[dict]) -> list[dict]:
    """Attach a colon lead-in to the next chunk of the same source document."""
    merged: list[dict] = []
    index = 0
    while index < len(rows):
        row = dict(rows[index])
        output = (row.get("output") or "").rstrip()
        nxt = rows[index + 1] if index + 1 < len(rows) else None
        same_document = (
            nxt is not None
            and row.get("source_file")
            and (row.get("source_file"), str(row.get("source_line")))
            == (nxt.get("source_file"), str(nxt.get("source_line")))
        )
        if output.endswith(":") and same_document:
            attached = dict(nxt)
            attached["output"] = output + "\n\n" + (attached.get("output") or "").lstrip()
            rows[index + 1] = attached
            index += 1
            continue
        row["output"] = output
        merged.append(row)
        index += 1
    return merged


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--max-seq-length", type=int, default=512)
    parser.add_argument("--max-prompt-tokens", type=int, default=120)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
    target = args.out or args.dataset.with_name(args.dataset.stem + "_fit512.jsonl")
    target.parent.mkdir(parents=True, exist_ok=True)
    source_rows = output_rows = shortened_titles = dropped = 0
    loaded: list[dict] = []
    with args.dataset.open(encoding="utf-8") as source:
        for raw in source:
            if raw.strip():
                loaded.append(json.loads(raw))
    loaded = merge_lead_in_rows(loaded)

    with target.open("w", encoding="utf-8") as dest:
        for row in loaded:
            source_rows += 1
            prefix = style_prefix(str(row.get("instruction") or ""))
            base_topic = extract_base_topic(str(row.get("instruction") or ""))
            topic = _short_topic(tokenizer, prefix, base_topic, args.max_prompt_tokens)
            shortened_titles += int(topic != base_topic)
            parts = _split_output(tokenizer, prefix, topic, str(row.get("output") or ""), args.max_seq_length)
            if not parts:
                dropped += 1
                continue
            for idx, output in enumerate(parts, 1):
                part_topic = topic if len(parts) == 1 else f"{topic} (part {idx} of {len(parts)})"
                instruction = prefix + part_topic
                if _n_tokens(tokenizer, instruction, output) > args.max_seq_length:
                    dropped += 1
                    continue
                record = {k: v for k, v in row.items() if k not in {"instruction", "output"}}
                record.update({"instruction": instruction, "output": output})
                dest.write(json.dumps(record, ensure_ascii=False) + "\n")
                output_rows += 1

    print(
        f"wrote {output_rows} rows from {source_rows}; "
        f"shortened_titles={shortened_titles}; dropped={dropped}; out={target}"
    )


if __name__ == "__main__":
    main()
