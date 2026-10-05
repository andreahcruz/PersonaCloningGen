"""Synthetic contracts and the frozen-byte check for the canonical keep-breaks fit."""

from pathlib import Path

from host_finetune.canonical_fit import (
    FitConfig,
    atomic_blocks,
    compatibility_snapshot_dir,
    count_tokens,
    fit_rows,
)
from host_finetune.sft_chunk_utils import extract_base_topic, style_prefix

INSTRUCTION = "Write a blog post in the style of Jason Lemkin about: Pricing"


class WordTokenizer:
    def __init__(self):
        self.add_generation_prompt = None
        self.add_special_tokens = None

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
        self.add_generation_prompt = add_generation_prompt
        self.tokenize = tokenize
        return "\n".join(message["content"] for message in messages)

    def __call__(self, text, add_special_tokens=False):
        self.add_special_tokens = add_special_tokens
        return {"input_ids": text.split()}


def _row(output: str, line: int = 1, source_file: str = "blog.jsonl") -> dict:
    return {
        "instruction": INSTRUCTION,
        "output": output,
        "source": "blog",
        "source_file": source_file,
        "source_line": line,
    }


def _instruction_tokens(tokenizer) -> int:
    prefix = style_prefix(INSTRUCTION)
    topic = extract_base_topic(INSTRUCTION)
    return count_tokens(tokenizer, prefix + topic, "", FitConfig())


def _words(count: int) -> str:
    return " ".join(f"w{index}" for index in range(count))


def test_unchanged_row_keeps_text_and_one_parent():
    result = fit_rows([_row("Ship the product.")], WordTokenizer())
    assert len(result.parts) == 1
    part = result.parts[0]
    assert part.relationship_type == "UNCHANGED"
    assert part.parent_indices == (0,)
    assert part.record["output"] == "Ship the product."
    assert part.record["source_file"] == "blog.jsonl"
    assert list(part.record)[-2:] == ["instruction", "output"]
    assert part.token_count_before == part.token_count_after


def test_reflow_changes_whitespace_without_changing_words():
    result = fit_rows([_row("Hello \n\nWorld")], WordTokenizer())
    part = result.parts[0]
    assert part.relationship_type == "REFLOWED"
    assert part.parent_indices == (0,)
    assert part.record["output"] != "Hello \n\nWorld"
    assert part.record["output"].split() == ["Hello", "World"]


def test_colon_lead_in_combines_with_the_next_same_document_row():
    rows = [_row("5 Interesting Learnings:"), _row("1. Price earlier.")]
    result = fit_rows(rows, WordTokenizer())
    assert len(result.parts) == 1
    part = result.parts[0]
    assert part.relationship_type == "COMBINED"
    assert part.parent_indices == (0, 1)
    assert part.record["output"].startswith("5 Interesting Learnings:")
    assert "1. Price earlier." in part.record["output"]
    assert rows[0]["output"] == "5 Interesting Learnings:"


def test_chained_lead_ins_keep_every_parent_and_do_not_cross_source_lines():
    rows = [
        _row("First:"),
        _row("Second:"),
        _row("Body."),
        _row("Other:"),
        _row("Elsewhere.", line=2),
    ]
    result = fit_rows(rows, WordTokenizer())
    assert [part.parent_indices for part in result.parts] == [(0, 1, 2), (3,), (4,)]
    assert result.parts[0].relationship_type == "COMBINED"
    assert "First:" in result.parts[0].record["output"]
    assert "Body." in result.parts[0].record["output"]
    assert result.parts[2].record["source_line"] == 2


def test_exact_packing_boundary_and_multi_part_splits():
    tokenizer = WordTokenizer()
    room = FitConfig().packing_limit - _instruction_tokens(tokenizer)
    below = fit_rows([_row(_words(room - 1))], tokenizer)
    exact = fit_rows([_row(_words(room))], tokenizer)
    two = fit_rows([_row(_words(room + 1))], tokenizer)
    three = fit_rows([_row(_words(room * 2 + 1))], tokenizer)
    four = fit_rows([_row(_words(room * 3 + 1))], tokenizer)
    assert below.parts[0].relationship_type == "UNCHANGED"
    assert exact.parts[0].relationship_type == "UNCHANGED"
    assert len(exact.parts) == 1
    assert len(two.parts) == 2
    assert {part.relationship_type for part in two.parts} == {"SPLIT"}
    assert [part.part_ordinal for part in two.parts] == [0, 1]
    assert len(three.parts) == 3
    assert len(four.parts) == 4
    for part in four.parts:
        assert part.token_count_after <= FitConfig().max_tokens
        assert part.parent_indices == (0,)
        assert "(part " in part.record["instruction"]


def test_part_label_overhead_can_reject_a_packed_piece():
    tokenizer = WordTokenizer()
    config = FitConfig(max_tokens=20, part_label_reserve=2)
    room = config.packing_limit - _instruction_tokens(tokenizer)
    assert room > 1
    result = fit_rows([_row(_words(room + 1))], tokenizer, config)
    assert result.dropped >= 1
    assert result.parts
    assert all(part.token_count_after <= config.max_tokens for part in result.parts)
    assert all(part.record["output"].split()[-1].startswith("w") for part in result.parts)


def test_sentence_and_paragraph_boundaries_stay_intact():
    tokenizer = WordTokenizer()
    config = FitConfig(max_tokens=30, part_label_reserve=8)
    text = "Alpha one two three four five. Beta one two three four five."
    result = fit_rows([_row(text)], tokenizer, config)
    assert len(result.parts) == 2
    assert result.parts[0].record["output"].endswith("five.")
    assert result.parts[1].record["output"].startswith("Beta")
    kept = fit_rows([_row("Alpha one.\n\nBeta two.")], tokenizer)
    assert kept.parts[0].relationship_type == "UNCHANGED"
    assert kept.parts[0].record["output"] == "Alpha one.\n\nBeta two."


def test_unicode_and_atomic_list_block():
    result = fit_rows([_row("The café in 東京 stays.")], WordTokenizer())
    assert result.parts[0].record["output"] == "The café in 東京 stays."
    text = "5 Interesting Learnings:\n\n1. Price earlier.\n\n2. Hire slower."
    assert atomic_blocks(text) == [text]


def test_template_flags_and_repeat_runs_are_stable():
    tokenizer = WordTokenizer()
    rows = [_row("Ship the product."), _row("Hello \n\nWorld")]
    first = fit_rows(rows, tokenizer)
    second = fit_rows(rows, tokenizer)
    assert [part.record for part in first.parts] == [part.record for part in second.parts]
    assert tokenizer.tokenize is False
    assert tokenizer.add_generation_prompt is False
    assert tokenizer.add_special_tokens is False


def test_combined_split_keeps_both_parents_on_every_part():
    tokenizer = WordTokenizer()
    room = FitConfig().packing_limit - _instruction_tokens(tokenizer)
    rows = [_row("Lead:"), _row(_words(room + 1))]
    result = fit_rows(rows, tokenizer)
    assert len(result.parts) >= 2
    assert {part.relationship_type for part in result.parts} == {"COMBINED_SPLIT"}
    assert all(part.parent_indices == (0, 1) for part in result.parts)


def test_compatibility_cli_writes_a_temp_file_and_not_the_frozen_artifact(tmp_path):
    import json
    import os
    import subprocess

    root = Path(__file__).resolve().parents[1]
    venv = root / "host_finetune" / ".venv" / "Scripts" / "python.exe"
    snapshot = compatibility_snapshot_dir()
    dataset = tmp_path / "sample.jsonl"
    out = tmp_path / "sample_fit.jsonl"
    dataset.write_text(
        json.dumps(
            {
                "instruction": "Write a blog post in the style of Jason Lemkin about: Pricing",
                "output": "Ship the product.",
                "source": "blog",
                "source_file": "blog.jsonl",
                "source_line": 1,
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
        newline="\n",
    )
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = ""
    env["PYTHONIOENCODING"] = "utf-8"
    completed = subprocess.run(
        [
            str(venv),
            "-m",
            "host_finetune.fit_sft_to_context",
            "--dataset",
            str(dataset),
            "--tokenizer",
            str(snapshot),
            "--out",
            str(out),
        ],
        cwd=root,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    raw = out.read_bytes()
    assert raw.endswith(b"\r\n")
    assert b"dataset_from_cleaned_sources_fit512_keepbreaks" not in str(out).encode()
    record = json.loads(raw.decode("utf-8"))
    assert list(record)[-2:] == ["instruction", "output"]
    assert record["output"] == "Ship the product."
    assert "wrote 1 rows from 1" in completed.stdout


def test_canonical_fit_matches_the_frozen_keepbreaks_file():
    import json
    import os
    import subprocess

    root = Path(__file__).resolve().parents[1]
    venv = root / "host_finetune" / ".venv" / "Scripts" / "python.exe"
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = ""
    env["PYTHONIOENCODING"] = "utf-8"
    completed = subprocess.run(
        [str(venv), "-m", "host_finetune.canonical_fit"],
        cwd=root,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    report = json.loads(completed.stdout)
    assert report["input_rows"] == 42771
    assert report["output_rows"] == 43012
    assert report["semantic_mismatches"] == 0
    assert report["key_order_mismatches"] == 0
    assert report["byte_sha256"] == "bf00e53fd2401f2d6bea6ac39cb69829fc21761de3a4f7225fed7ecd75082fba"
    assert report["parent_mismatches"] == 0
    assert report["ordinal_mismatches"] == 0
    assert report["relationship_mismatches"] == 0
    assert report["part_ordinal_mismatches"] == 0
    assert report["relationship_counts"] == {
        "UNCHANGED": 37517,
        "REFLOWED": 3072,
        "SPLIT": 1619,
        "COMBINED": 566,
        "COMBINED_SPLIT": 238,
    }
    assert report["lead_ins_consumed"] == 706
    assert report["merged_rows"] == 42065
    assert report["dropped"] == 0

