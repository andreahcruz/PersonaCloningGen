"""Frozen keep-breaks fit as a pure transformation.

This is the canonical owner of the historical algorithm. It does not write
files, load model weights, or open Chroma. ``fit_sft_to_context`` is the
compatibility command line. Do not import that module from here.

The packing rules are the frozen ones. This module does not change the token
limit, the lead-in rule, or the part label.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from host_finetune.canonical_ids import CONTEXT_FIT_VERSION
from host_finetune.sft_chunk_utils import _split_sentences, extract_base_topic, style_prefix

HISTORICAL_TOKENIZER_ID = "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit"
HISTORICAL_REVISION = None
COMPATIBILITY_REVISION = "f15c379fb32bb402fa06a7ae9aecb1febf4b79ec"
FIT_VERSION = CONTEXT_FIT_VERSION

_LIST_OR_HEADING_RE = re.compile(
    r"^(?:#{1,3}\s+\S+|[-*]|\d{1,2}[.)])\s+\S"
)


@dataclass(frozen=True)
class FitConfig:
    """Observed keep-breaks settings. Values match the frozen command defaults."""

    fit_version: str = FIT_VERSION
    max_tokens: int = 512
    prompt_cap: int = 120
    part_label_reserve: int = 8
    add_special_tokens: bool = False
    add_generation_prompt: bool = False
    historical_tokenizer_id: str = HISTORICAL_TOKENIZER_ID
    historical_revision: str | None = HISTORICAL_REVISION
    compatibility_revision: str = COMPATIBILITY_REVISION
    trust_remote_code: bool = True

    @property
    def packing_limit(self) -> int:
        return self.max_tokens - self.part_label_reserve


@dataclass(frozen=True)
class FitPart:
    """One emitted fit row and the parent indexes that produced it."""

    record: dict
    parent_indices: tuple[int, ...]
    relationship_type: str
    part_ordinal: int
    part_count: int
    fit_chunk_ordinal: int
    token_count_before: int
    token_count_after: int


@dataclass(frozen=True)
class FitResult:
    parts: tuple[FitPart, ...]
    input_rows: int
    merged_rows: int
    lead_ins_consumed: int
    shortened_titles: int
    dropped: int


def compatibility_snapshot_dir() -> Path:
    return (
        Path.home()
        / ".cache"
        / "huggingface"
        / "hub"
        / "models--unsloth--Meta-Llama-3.1-8B-Instruct-bnb-4bit"
        / "snapshots"
        / COMPATIBILITY_REVISION
    )


def load_compatibility_tokenizer(config: FitConfig | None = None):
    """Load the local snapshot that reproduces the frozen file. No download."""
    config = config or FitConfig()
    snapshot = compatibility_snapshot_dir()
    if not snapshot.is_dir():
        raise FileNotFoundError(f"compatibility tokenizer snapshot missing: {snapshot}")
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(
        str(snapshot),
        trust_remote_code=config.trust_remote_code,
        local_files_only=True,
    )


def _text(tokenizer, instruction: str, output: str, config: FitConfig) -> str:
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": instruction}, {"role": "assistant", "content": output}],
        tokenize=False,
        add_generation_prompt=config.add_generation_prompt,
    )


def count_tokens(tokenizer, instruction: str, output: str, config: FitConfig) -> int:
    rendered = _text(tokenizer, instruction, output, config)
    return len(tokenizer(rendered, add_special_tokens=config.add_special_tokens)["input_ids"])


def shorten_topic(tokenizer, prefix: str, topic: str, config: FitConfig) -> str:
    topic = " ".join((topic or "").split())
    if not topic:
        return "a SaaS founder lesson"

    def fits(value: str) -> bool:
        return count_tokens(tokenizer, prefix + value, "", config) <= config.prompt_cap

    if fits(topic):
        return topic
    words = topic.split()
    while len(words) > 4:
        candidate = " ".join(words).rstrip(" ,;:-") + "..."
        if fits(candidate):
            return candidate
        words.pop()
    return "a SaaS founder lesson"


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
    config: FitConfig,
) -> list[str]:
    instruction = prefix + topic
    pieces: list[str] = []
    current = ""
    for segment in segments:
        candidate = segment if not current else current + joiner + segment
        if candidate and count_tokens(tokenizer, instruction, candidate, config) <= limit:
            current = candidate
            continue
        current = _flush_piece(pieces, current)
        candidate = segment if not current else current + joiner + segment
        if candidate and count_tokens(tokenizer, instruction, candidate, config) <= limit:
            current = candidate
            continue
        if current and count_tokens(tokenizer, instruction, current, config) <= limit:
            pieces.append(current)
        current = ""
        if count_tokens(tokenizer, instruction, segment, config) <= limit:
            current = segment
            continue
        finer, fine_joiner = _segments(segment)
        if len(finer) == 1 and finer[0] == segment:
            current = segment
            continue
        sub = _pack_segments(tokenizer, prefix, topic, finer, fine_joiner, limit, config)
        if sub:
            pieces.extend(sub[:-1])
            current = sub[-1]
    current = _flush_piece(pieces, current)
    if current:
        pieces.append(current)
    return pieces


def split_to_budget(tokenizer, prefix: str, topic: str, output: str, config: FitConfig) -> list[str]:
    instruction = prefix + topic
    blocks = atomic_blocks(output)
    pieces = _pack_segments(tokenizer, prefix, topic, blocks, "\n\n", config.packing_limit, config)
    return [
        piece
        for piece in pieces
        if piece.strip() and count_tokens(tokenizer, instruction, piece, config) <= config.max_tokens
    ]


def merge_lead_in_rows(rows: list[dict]) -> list[dict]:
    """Attach a colon lead-in to the next chunk of the same source document.

    This preserves the historical mutation of the next row. ``fit_rows`` uses
    ``combine_leadins``, which does not mutate the caller.
    """
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


def combine_leadins(rows: list[dict]) -> list[tuple[dict, tuple[int, ...]]]:
    """Same lead-in rule as ``merge_lead_in_rows``, with every consumed index."""
    rows = [dict(row) for row in rows]
    merged: list[tuple[dict, tuple[int, ...]]] = []
    pending: list[int] = []
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
            pending.append(index)
            index += 1
            continue
        row["output"] = output
        merged.append((row, tuple(pending + [index])))
        pending = []
        index += 1
    return merged


def _relationship(parent_count: int, emitted: int, raw_parts: int, same_text: bool) -> str:
    if parent_count > 1 and emitted > 1:
        return "COMBINED_SPLIT"
    if parent_count > 1:
        return "COMBINED"
    if emitted > 1:
        return "SPLIT"
    if raw_parts > emitted or not same_text:
        return "REFLOWED"
    return "UNCHANGED"


def _record(row: dict, instruction: str, output: str) -> dict:
    record = {key: value for key, value in row.items() if key not in {"instruction", "output"}}
    record["instruction"] = instruction
    record["output"] = output
    return record


def fit_rows(rows: list[dict], tokenizer, config: FitConfig | None = None) -> FitResult:
    """Fit every input row. Does not mutate ``rows`` and does not write a file."""
    config = config or FitConfig()
    merged = combine_leadins(rows)
    parts: list[FitPart] = []
    shortened = 0
    dropped = 0
    seen: Counter[tuple] = Counter()
    for row, parents in merged:
        prefix = style_prefix(str(row.get("instruction") or ""))
        base_topic = extract_base_topic(str(row.get("instruction") or ""))
        topic = shorten_topic(tokenizer, prefix, base_topic, config)
        shortened += int(topic != base_topic)
        merged_output = str(row.get("output") or "")
        before_tokens = count_tokens(tokenizer, prefix + topic, merged_output, config)
        raw_parts = split_to_budget(tokenizer, prefix, topic, merged_output, config)
        if not raw_parts:
            dropped += 1
            continue
        emitted: list[tuple[str, str, int]] = []
        for index, output in enumerate(raw_parts, 1):
            part_topic = topic if len(raw_parts) == 1 else f"{topic} (part {index} of {len(raw_parts)})"
            instruction = prefix + part_topic
            after_tokens = count_tokens(tokenizer, instruction, output, config)
            if after_tokens > config.max_tokens:
                dropped += 1
                continue
            emitted.append((instruction, output, after_tokens))
        if not emitted:
            continue
        same_text = len(emitted) == 1 and emitted[0][1] == merged_output.rstrip()
        relationship = _relationship(len(parents), len(emitted), len(raw_parts), same_text)
        for part_ordinal, (instruction, output, after_tokens) in enumerate(emitted):
            record = _record(row, instruction, output)
            key = (record.get("source_file"), record.get("source_line"))
            ordinal = seen[key]
            seen[key] += 1
            parts.append(
                FitPart(
                    record=record,
                    parent_indices=parents,
                    relationship_type=relationship,
                    part_ordinal=part_ordinal,
                    part_count=len(emitted),
                    fit_chunk_ordinal=ordinal,
                    token_count_before=before_tokens,
                    token_count_after=after_tokens,
                )
            )
    return FitResult(
        parts=tuple(parts),
        input_rows=len(rows),
        merged_rows=len(merged),
        lead_ins_consumed=sum(len(parents) - 1 for _row, parents in merged),
        shortened_titles=shortened,
        dropped=dropped,
    )


def serialize_fit_records(records: list[dict], newline: bytes = b"\r\n") -> bytes:
    """Historical byte contract: UTF-8 JSON, explicit CRLF, final CRLF."""
    lines = [json.dumps(record, ensure_ascii=False).encode("utf-8") for record in records]
    if not lines:
        return newline
    return newline.join(lines) + newline


def _n_tokens(tokenizer, instruction: str, output: str) -> int:
    """Historical helper signature. Uses the frozen template flags."""
    return count_tokens(tokenizer, instruction, output, FitConfig())


def _short_topic(tokenizer, prefix: str, topic: str, max_prompt_tokens: int) -> str:
    """Historical helper signature. The fourth argument is the prompt cap."""
    return shorten_topic(tokenizer, prefix, topic, FitConfig(prompt_cap=max_prompt_tokens))


def _split_output(tokenizer, prefix: str, topic: str, output: str, max_tokens: int) -> list[str]:
    """Historical helper signature. ``max_tokens`` is the 512-token contract."""
    return split_to_budget(tokenizer, prefix, topic, output, FitConfig(max_tokens=max_tokens))


def frozen_parity_report(include_historical: bool = True) -> dict:
    """Compare this module with the frozen keep-breaks file. Does not write it."""
    import hashlib

    root = Path(__file__).resolve().parents[1]
    nopromo = root / "host_finetune" / "data" / "dataset_from_cleaned_sources_nopromo.jsonl"
    frozen_path = root / "host_finetune" / "data" / "dataset_from_cleaned_sources_fit512_keepbreaks.jsonl"
    lineage_path = root / "artifacts" / "refactor" / "provenance" / "fit_lineage_exact.preview.jsonl"
    expected_input = "55b9680657ed72a424d47ea7102214d9b2c6693a535572e02520e406d57f8241"
    expected_output = "bf00e53fd2401f2d6bea6ac39cb69829fc21761de3a4f7225fed7ecd75082fba"

    def sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    def load(path: Path) -> list[dict]:
        rows = []
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    rows.append(json.loads(line))
        return rows

    input_sha = sha256(nopromo)
    output_sha = sha256(frozen_path)
    if input_sha != expected_input or output_sha != expected_output:
        raise SystemExit(f"frozen hash mismatch input={input_sha} output={output_sha}")
    source = load(nopromo)
    frozen = load(frozen_path)
    lineage = load(lineage_path)
    tokenizer = load_compatibility_tokenizer()
    result = fit_rows(source, tokenizer)
    records = [part.record for part in result.parts]
    semantic = sum(left != right for left, right in zip(records, frozen))
    if len(records) != len(frozen):
        semantic += abs(len(records) - len(frozen))
    key_order = sum(list(left) != list(right) for left, right in zip(records, frozen))
    encoded = serialize_fit_records(records)
    parent_mismatches = 0
    ordinal_mismatches = 0
    relationship_mismatches = 0
    part_ordinal_mismatches = 0
    if len(result.parts) == len(lineage):
        for part, row in zip(result.parts, lineage):
            if list(part.parent_indices) != row["parent_sft_row_indices"]:
                parent_mismatches += 1
            if part.fit_chunk_ordinal != row["fit_chunk_ordinal"]:
                ordinal_mismatches += 1
            if part.relationship_type != row["relationship_type"]:
                relationship_mismatches += 1
            if part.part_ordinal != row["part_ordinal"]:
                part_ordinal_mismatches += 1
    else:
        parent_mismatches = ordinal_mismatches = relationship_mismatches = -1
    historical_mismatches = None
    if include_historical:
        from host_finetune.replay_fit_lineage import replay

        historical_records, historical_lineage, _accounting = replay(tokenizer, source)
        historical_mismatches = sum(left != right for left, right in zip(historical_records, records))
        if len(historical_records) != len(records):
            historical_mismatches += abs(len(historical_records) - len(records))
        for old, new in zip(historical_lineage, result.parts):
            if old["parents"] != list(new.parent_indices) or old["relationship_type"] != new.relationship_type:
                historical_mismatches += 1
    counts = Counter(part.relationship_type for part in result.parts)
    return {
        "input_rows": len(source),
        "output_rows": len(records),
        "semantic_mismatches": semantic,
        "key_order_mismatches": key_order,
        "byte_sha256": hashlib.sha256(encoded).hexdigest(),
        "frozen_sha256": output_sha,
        "lineage_rows": len(lineage),
        "parent_mismatches": parent_mismatches,
        "ordinal_mismatches": ordinal_mismatches,
        "relationship_mismatches": relationship_mismatches,
        "part_ordinal_mismatches": part_ordinal_mismatches,
        "historical_mismatches": historical_mismatches,
        "relationship_counts": dict(counts),
        "lead_ins_consumed": result.lead_ins_consumed,
        "merged_rows": result.merged_rows,
        "dropped": result.dropped,
    }


if __name__ == "__main__":
    print(json.dumps(frozen_parity_report(include_historical=False)))
