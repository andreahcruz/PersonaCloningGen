"""Filter and relabel the repaired SFT rows without refitting them.

Whole X posts and whole blog or LinkedIn posts keep their instructions.
Sentence-complete fragments, list lead-ins, and clean last slices become
continuation or opening instructions. Mid-sentence cuts and transcript
fragments are omitted. The prior dataset file is not modified.
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

from host_finetune.sft_chunk_utils import extract_base_topic
from host_finetune.split_groups import file_sha256

SENT_RE = re.compile(r"[.!?…][\"'”’)\]]*$")
COLON_RE = re.compile(r":\s*$")
HEADING_RE = re.compile(r"^(?:#{1,3}\s+\S+|\d{1,2}[.)]\s+\S|[-*]\s+\S)")
FULL_POST_RE = re.compile(
    r"^Write (?:a blog post|a LinkedIn post|an X post|a talk) in the style of Jason Lemkin about:",
    re.IGNORECASE,
)
MEDIUM_PHRASE = {
    "blog": "blog post",
    "linkedin": "LinkedIn post",
    "x": "X post",
    "youtube_jason": "talk",
}
# Train-split counts from the ending-audit classifier. Group 3 is 115
# above the summary table, and group 4's paragraph bucket is 115 below it.
# Those 115 end on the sentence regex, including an ellipsis, so they are
# relabeled rather than dropped. The build stops if a count moves.
EXPECTED_TRAIN_GROUPS = {
    "G1_whole_x_post": 10974,
    "G2_whole_document": 642,
    "G3_artificial_sentence_complete": 8430,
    "G4_mid_sentence": 1148,
    "G4_artificial_unfinished": 1332,
    "G5_structural_continuation": 99,
    "G6_whole_transcript_no_sentence_end": 87,
    "G6_transcript_tail": 103,
    "source_final_fragment": 2247,
}


def sentence_final(text: str) -> bool:
    return bool(SENT_RE.search((text or "").rstrip()))


def is_structural(text: str) -> bool:
    stripped = (text or "").rstrip()
    lines = stripped.splitlines()
    last = lines[-1].strip() if lines else ""
    return bool(COLON_RE.search(stripped)) or bool(last and HEADING_RE.match(last))


def is_word_cut(text: str, nxt: str | None) -> bool:
    if nxt is None or sentence_final(text):
        return False
    stripped = (text or "").rstrip()
    follower = (nxt or "").lstrip()
    return bool(stripped and follower and stripped[-1].isalnum() and follower[0].islower())


def classify(text: str, nxt: str | None, position: str, medium: str) -> str:
    """Same exclusive groups as the ending audit."""
    finished = sentence_final(text)
    continues = nxt is not None
    if is_word_cut(text, nxt):
        return "G4_mid_sentence"
    if continues and is_structural(text):
        return "G5_structural_continuation"
    if continues and finished:
        return "G3_artificial_sentence_complete"
    if continues:
        return "G4_artificial_unfinished"
    if position == "only" and medium == "x":
        return "G1_whole_x_post"
    if position == "only" and medium == "youtube_jason" and not finished:
        return "G6_whole_transcript_no_sentence_end"
    if position == "only":
        return "G2_whole_document"
    if medium == "youtube_jason" and not finished:
        return "G6_transcript_tail"
    return "source_final_fragment"


def action_for(group: str, text: str) -> str:
    if group in {"G1_whole_x_post", "G2_whole_document"}:
        return "keep"
    if group in {"G3_artificial_sentence_complete", "G5_structural_continuation"}:
        return "relabel"
    if group == "source_final_fragment":
        return "relabel" if sentence_final(text) else "drop"
    if group.startswith("G4_") or group.startswith("G6_"):
        return "drop"
    raise ValueError(f"unknown group {group}")


def previous_quote(text: str, max_chars: int = 280) -> str:
    flat = " ".join((text or "").split())
    if not flat:
        return ""
    parts = re.split(r"(?<=[.!?…])\s+", flat)
    quote = (parts[-1] if parts else flat).strip().replace('"', "'")
    if len(quote) > max_chars:
        quote = quote[-max_chars:].lstrip()
    return quote


def render_instruction(medium: str, topic: str, quote: str, opening: bool) -> str:
    phrase = MEDIUM_PHRASE.get(medium, "post")
    topic = (topic or "").strip().rstrip(".").strip() or "this topic"
    if opening:
        return (
            f"Write only the opening section, in the style of Jason Lemkin, "
            f"of a {phrase} about: {topic}."
        )
    if quote:
        return (
            f"Continue this {phrase} in the style of Jason Lemkin about: {topic}. "
            f'Previous section ending: "{quote}"'
        )
    return f"Continue this {phrase} in the style of Jason Lemkin about: {topic}."


def fit_instruction(
    medium: str,
    topic: str,
    previous: str | None,
    output: str,
    input_text: str,
    limit: int,
    n_tokens,
) -> str | None:
    """Shorten the quoted ending until the chat text fits. Never cut the target."""
    opening = not previous
    quote = "" if opening else previous_quote(previous or "")
    words = (topic or "").split() or ["this topic"]
    while True:
        instruction = render_instruction(medium, " ".join(words), quote, opening)
        if n_tokens(instruction, input_text, output) <= limit:
            return instruction
        if quote:
            quote = "" if len(quote) <= 24 else quote[max(1, len(quote) // 4) :].lstrip()
            continue
        if len(words) > 4:
            words = words[:-1]
            continue
        return None


def _positions(rows: list[dict]) -> tuple[dict[int, str], dict[int, str | None], dict[int, str | None]]:
    groups: dict[tuple, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        groups[(row.get("source_file"), str(row.get("source_line")))].append(index)
    position: dict[int, str] = {}
    nxt: dict[int, str | None] = {}
    prev: dict[int, str | None] = {}
    for indexes in groups.values():
        count = len(indexes)
        for offset, index in enumerate(indexes):
            if count == 1:
                position[index] = "only"
            elif offset == 0:
                position[index] = "first"
            elif offset == count - 1:
                position[index] = "final"
            else:
                position[index] = "middle"
            nxt[index] = rows[indexes[offset + 1]].get("output") if offset + 1 < count else None
            prev[index] = rows[indexes[offset - 1]].get("output") if offset else None
    return position, nxt, prev


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def load_assignments(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            if record.get("record_type") == "meta":
                continue
            rows.append(record)
    rows.sort(key=lambda row: row["row_index"])
    return rows


def transform_rows(rows: list[dict], assignments: list[dict], n_tokens, limit: int):
    if len(rows) != len(assignments):
        raise ValueError("dataset and assignment lengths differ")
    position, nxt, prev = _positions(rows)
    new_rows: list[dict] = []
    new_assignments: list[dict] = []
    dispositions: list[dict] = []
    train_groups: Counter[str] = Counter()
    actions: Counter[str] = Counter()
    for index, (row, assignment) in enumerate(zip(rows, assignments)):
        if assignment["row_index"] != index:
            raise ValueError(f"assignment row_index gap at {index}")
        medium = row.get("source") or ""
        text = row.get("output") or ""
        group = classify(text, nxt[index], position[index], medium)
        action = action_for(group, text)
        if assignment["split"] == "train":
            train_groups[group] += 1
        record = {
            "prior_row_index": index,
            "split": assignment["split"],
            "group": group,
            "action": action,
            "position": position[index],
            "source": medium,
        }
        if action == "drop":
            actions[f"{assignment['split']}:{group}:drop"] += 1
            dispositions.append(record)
            continue
        instruction = row.get("instruction") or ""
        role = "unchanged"
        if action == "relabel":
            fitted = fit_instruction(
                medium,
                extract_base_topic(instruction),
                prev[index],
                text,
                row.get("input") or "",
                limit,
                n_tokens,
            )
            if fitted is None:
                record["action"] = "drop_overflow"
                actions[f"{assignment['split']}:drop_overflow"] += 1
                dispositions.append(record)
                continue
            if FULL_POST_RE.match(fitted):
                raise RuntimeError(f"relabel still looks like a full post at row {index}")
            instruction = fitted
            role = "opening" if position[index] == "first" else "continuation"
        kept = {
            "instruction": instruction,
            "input": row.get("input") or "",
            "output": text,
            "source": medium,
            "source_file": row.get("source_file"),
            "source_line": row.get("source_line"),
            "sft_role": role,
            "prior_group": group,
        }
        new_index = len(new_rows)
        new_rows.append(kept)
        new_assignments.append(
            {
                "row_index": new_index,
                "split": assignment["split"],
                "group_id": assignment["group_id"],
                "base_title": assignment.get("base_title") or "",
                "part_index": assignment.get("part_index"),
                "part_count": assignment.get("part_count"),
                "source_platform": assignment.get("source_platform") or medium,
                "source_file": row.get("source_file") or "",
                "source_line": row.get("source_line") or "",
                "prior_row_index": index,
                "prior_group": group,
                "sft_role": role,
            }
        )
        record["new_row_index"] = new_index
        record["sft_role"] = role
        actions[f"{assignment['split']}:{role}"] += 1
        dispositions.append(record)
    stats = {
        "rows_in": len(rows),
        "rows_out": len(new_rows),
        "train_groups": dict(train_groups),
        "actions": dict(actions),
        "by_split": dict(Counter(row["split"] for row in new_assignments)),
    }
    return new_rows, new_assignments, dispositions, stats


def write_outputs(out_dir: Path, rows: list[dict], assignments: list[dict], dispositions: list[dict], stats: dict) -> str:
    raw = out_dir / "raw"
    metrics = out_dir / "metrics"
    raw.mkdir(parents=True, exist_ok=True)
    metrics.mkdir(parents=True, exist_ok=True)
    dataset_path = raw / "dataset.jsonl"
    with dataset_path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    digest = file_sha256(dataset_path)
    meta = {
        "record_type": "meta",
        "dataset_sha256": digest,
        "n_rows": len(assignments),
        "source_platform": "source_file+source_line",
        "policy": "keep G1-G2; relabel G3, G5, and sentence-final last slices; drop G4 and G6",
    }
    lines = [json.dumps(meta, ensure_ascii=False)]
    lines.extend(json.dumps(row, ensure_ascii=False) for row in assignments)
    (raw / "split_assignments.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")
    with (raw / "dispositions.jsonl").open("w", encoding="utf-8", newline="\n") as handle:
        for row in dispositions:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    (metrics / "relabel_report.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    return digest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--assignments", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--max-seq-length", type=int, default=512)
    args = parser.parse_args()
    from transformers import AutoTokenizer

    from host_finetune.llama_chat_format import training_text_from_row

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)

    def n_tokens(instruction: str, input_text: str, output: str) -> int:
        text = training_text_from_row(tokenizer, instruction, input_text, output)
        return len(tokenizer(text, add_special_tokens=False)["input_ids"])

    rows = load_jsonl(args.dataset)
    assignments = load_assignments(args.assignments)
    new_rows, new_assignments, dispositions, stats = transform_rows(
        rows, assignments, n_tokens, args.max_seq_length
    )
    if stats["train_groups"] != EXPECTED_TRAIN_GROUPS:
        raise SystemExit(f"train group counts changed: {stats['train_groups']}")
    digest = write_outputs(args.out, new_rows, new_assignments, dispositions, stats)
    print(json.dumps({"dataset_sha256": digest, **stats}, indent=2))


if __name__ == "__main__":
    main()
