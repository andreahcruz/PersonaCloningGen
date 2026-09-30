"""Re-chunk SFT JSONL with the real Llama chat-template token budget.

Unlike the legacy 1,600-character rule, this utility counts the complete
user+assistant training text.  It shortens pathological scraped titles and
splits answers on sentence/word boundaries until every emitted record fits
``MAX_SEQ_LENGTH``.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from host_finetune.sft_chunk_utils import extract_base_topic, split_sentence_aware

PREFIX = "Write in the style of Jason Lemkin about: "


def _text(tokenizer, instruction: str, output: str) -> str:
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": instruction}, {"role": "assistant", "content": output}],
        tokenize=False,
        add_generation_prompt=False,
    )


def _n_tokens(tokenizer, instruction: str, output: str) -> int:
    return len(tokenizer(_text(tokenizer, instruction, output), add_special_tokens=False)["input_ids"])


def _short_topic(tokenizer, topic: str, max_prompt_tokens: int) -> str:
    """Keep a meaningful title, but never let scraped title junk consume context."""
    topic = " ".join((topic or "").split())
    if not topic:
        return "a SaaS founder lesson"

    def fits(value: str) -> bool:
        return _n_tokens(tokenizer, PREFIX + value, "") <= max_prompt_tokens

    if fits(topic):
        return topic
    words = topic.split()
    while len(words) > 4:
        candidate = " ".join(words).rstrip(" ,;:-") + "..."
        if fits(candidate):
            return candidate
        words.pop()
    return "a SaaS founder lesson"


def _units(text: str) -> list[str]:
    """Sentences first; long individual sentences become word-boundary units."""
    sentences = split_sentence_aware(text, max_chars=10_000_000)
    out: list[str] = []
    for sentence in sentences:
        if len(sentence) <= 120:
            out.append(sentence)
            continue
        words = sentence.split()
        current: list[str] = []
        for word in words:
            current.append(word)
            if len(" ".join(current)) >= 120:
                out.append(" ".join(current))
                current = []
        if current:
            out.append(" ".join(current))
    return out or [text.strip()]


def _split_output(tokenizer, topic: str, output: str, max_tokens: int) -> list[str]:
    instruction = PREFIX + topic
    limit = max_tokens - 8  # leaves room for `(part k of N)` after packing
    pieces: list[str] = []
    current = ""
    for unit in _units(output):
        candidate = (current + " " + unit).strip()
        if candidate and _n_tokens(tokenizer, instruction, candidate) <= limit:
            current = candidate
            continue
        if current:
            pieces.append(current)
            current = ""
        # A unit may still be too token-dense (emoji, URLs, code): pack words.
        words = unit.split()
        for word in words:
            candidate = (current + " " + word).strip()
            if candidate and _n_tokens(tokenizer, instruction, candidate) <= limit:
                current = candidate
            elif current:
                pieces.append(current)
                current = word
            else:
                # A single token-dense word cannot be safely trained; omit it.
                current = ""
    if current:
        pieces.append(current)
    return pieces


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

    with args.dataset.open(encoding="utf-8") as source, target.open("w", encoding="utf-8") as dest:
        for raw in source:
            if not raw.strip():
                continue
            source_rows += 1
            row = json.loads(raw)
            base_topic = extract_base_topic(str(row.get("instruction") or ""))
            topic = _short_topic(tokenizer, base_topic, args.max_prompt_tokens)
            shortened_titles += int(topic != base_topic)
            parts = _split_output(tokenizer, topic, str(row.get("output") or ""), args.max_seq_length)
            if not parts:
                dropped += 1
                continue
            for idx, output in enumerate(parts, 1):
                part_topic = topic if len(parts) == 1 else f"{topic} (part {idx} of {len(parts)})"
                instruction = PREFIX + part_topic
                # Defensive check: do not emit a silently truncated row.
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
