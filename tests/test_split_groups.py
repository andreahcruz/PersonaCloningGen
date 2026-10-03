"""Grouped 80/10/10 split: titles stay together and near-duplicates cannot cross."""
from __future__ import annotations

import json

import pytest

from host_finetune.split_groups import (
    assert_manifest_matches,
    assign_groups,
    parse_instruction,
    write_assignment_file,
)


def _row(title: str, output: str, part: tuple[int, int] | None = None) -> dict:
    suffix = f" (part {part[0]} of {part[1]})" if part else ""
    return {
        "instruction": f"Write in the style of Jason Lemkin about: {title}{suffix}",
        "input": "",
        "output": output,
    }


def _unique_rows(n: int) -> list[dict]:
    return [_row(f"Title {i:02d}", f"unique{i:02d}token " * 40) for i in range(n)]


def test_parse_strips_part_suffix():
    title, part_index, part_count = parse_instruction(
        "Write in the style of Jason Lemkin about: Pricing (part 2 of 4)"
    )
    assert title == "Pricing"
    assert part_index == 2
    assert part_count == 4


def test_same_source_line_stays_together_even_when_titles_differ():
    left = _row("Alpha", "alpha body " * 20, (1, 2))
    right = _row("Beta", "beta body " * 20, (2, 2))
    left.update({"source": "blog", "source_file": "jasonlemkin_blog.jsonl", "source_line": 4})
    right.update({"source": "blog", "source_file": "jasonlemkin_blog.jsonl", "source_line": 4})
    rows = [left, right, *_unique_rows(8)]
    assignments, _pairs = assign_groups(rows)
    assert assignments[0]["group_id"] == assignments[1]["group_id"]
    assert assignments[0]["split"] == assignments[1]["split"]
    assert assignments[0]["source_file"] == "jasonlemkin_blog.jsonl"
    assert assignments[0]["source_line"] == 4
    assert assignments[0]["source_platform"] == "blog"


def test_medium_instruction_prefix_parses():
    title, part_index, part_count = parse_instruction(
        "Write an X post in the style of Jason Lemkin about: Pricing (part 2 of 4)"
    )
    assert title == "Pricing"
    assert part_index == 2
    assert part_count == 4


def test_same_title_parts_stay_together():
    rows = [
        _row("Alpha", "alpha part one text " * 20, (1, 2)),
        _row("Alpha", "alpha part two text " * 20, (2, 2)),
        *_unique_rows(8),
    ]
    assignments, _pairs = assign_groups(rows)
    assert assignments[0]["group_id"] == assignments[1]["group_id"]
    assert assignments[0]["split"] == assignments[1]["split"]
    assert assignments[0]["part_index"] == 1
    assert assignments[1]["part_count"] == 2


def test_jaccard_pair_cannot_cross_splits():
    shared = " ".join(f"tok{i}" for i in range(20))
    rows = [
        _row("Left post", shared + " extra"),
        _row("Right post", shared),
        *_unique_rows(8),
    ]
    assignments, pairs = assign_groups(rows)
    assert (0, 1) in pairs
    assert assignments[0]["group_id"] == assignments[1]["group_id"]
    assert assignments[0]["split"] == assignments[1]["split"]


def test_rerun_is_identical():
    rows = _unique_rows(10)
    first, pairs_a = assign_groups(rows)
    second, pairs_b = assign_groups(rows)
    assert first == second
    assert pairs_a == pairs_b


def test_row_count_matches_input_and_shares():
    rows = _unique_rows(10)
    assignments, _pairs = assign_groups(rows)
    assert len(assignments) == 10
    counts = {name: sum(row["split"] == name for row in assignments) for name in ("train", "validation", "test")}
    assert counts == {"train": 8, "validation": 1, "test": 1}
    assert {row["source_platform"] for row in assignments} == {"unavailable"}


def test_normalized_exact_outputs_stay_together():
    rows = [
        _row("One title", "Same sentence about pricing."),
        _row("Other title", "same   sentence about pricing."),
        *_unique_rows(8),
    ]
    assignments, _pairs = assign_groups(rows)
    assert assignments[0]["split"] == assignments[1]["split"]
    assert assignments[0]["group_id"] == assignments[1]["group_id"]


def test_each_medium_keeps_its_own_quota():
    rows = []
    for source, count in (("blog", 20), ("x", 40), ("linkedin", 10)):
        for index in range(count):
            row = _row(f"{source} title {index}", f"{source} unique{index:02d} token " * 12)
            row["source"] = source
            row["source_file"] = f"{source}.jsonl"
            row["source_line"] = index + 1
            rows.append(row)
    assignments, _pairs = assign_groups(rows)
    by_source: dict[str, dict[str, int]] = {}
    for row, assignment in zip(rows, assignments):
        counts = by_source.setdefault(row["source"], {"train": 0, "validation": 0, "test": 0})
        counts[assignment["split"]] += 1
    assert by_source["blog"] == {"train": 16, "validation": 2, "test": 2}
    assert by_source["x"] == {"train": 32, "validation": 4, "test": 4}
    assert by_source["linkedin"] == {"train": 8, "validation": 1, "test": 1}


def test_hash_mismatch_is_rejected(tmp_path):
    rows = _unique_rows(4)
    dataset = tmp_path / "dataset.jsonl"
    dataset.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    assignments, _pairs = assign_groups(rows)
    manifest = tmp_path / "split_assignments.jsonl"
    write_assignment_file(manifest, "0" * 64, assignments)
    with pytest.raises(RuntimeError, match="Refusing to train"):
        assert_manifest_matches(manifest, dataset)
