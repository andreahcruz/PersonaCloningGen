"""Training answers keep paragraph breaks and do not stop on a colon lead-in."""
from host_finetune.fit_sft_to_context import (
    _split_output,
    atomic_blocks,
    merge_lead_in_rows,
    move_dangling_lead_ins,
)

PREFIX = "Write a blog post in the style of Jason Lemkin about: "


class _WordTokenizer:
    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
        return "\n".join(message["content"] for message in messages)

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": text.split()}


def test_atomic_block_keeps_lead_in_with_its_list():
    text = "5 Interesting Learnings:\n\n1. Price earlier.\n\n2. Hire slower."
    assert atomic_blocks(text) == [text]


def test_newlines_survive_when_the_answer_fits():
    text = "Open with the point.\n\nThen give the number.\n\nClose on the action."
    pieces = _split_output(_WordTokenizer(), PREFIX, "pricing", text, max_tokens=80)
    assert pieces == [text]


def test_colon_lead_in_moves_onto_the_list_when_the_budget_breaks():
    preamble = " ".join(f"w{i}" for i in range(40))
    text = (
        preamble
        + "\n\n5 Interesting Learnings:\n\n1. Price earlier.\n\n2. Hire slower."
    )
    pieces = _split_output(_WordTokenizer(), PREFIX, "pricing", text, max_tokens=64)
    assert len(pieces) >= 2
    assert all(not piece.rstrip().endswith(":") for piece in pieces[:-1])
    assert any(piece.startswith("5 Interesting Learnings:") for piece in pieces)
    assert all("\n" in piece or not piece.endswith(":") for piece in pieces)


def test_same_document_lead_in_is_attached_to_the_next_chunk():
    rows = [
        {"output": "5 Interesting Learnings:", "source_file": "blog.jsonl", "source_line": 4},
        {"output": "1. Price earlier.", "source_file": "blog.jsonl", "source_line": 4},
        {"output": "A finished note:", "source_file": "other.jsonl", "source_line": 2},
    ]
    merged = merge_lead_in_rows(rows)
    assert len(merged) == 2
    assert merged[0]["output"].startswith("5 Interesting Learnings:")
    assert "1. Price earlier." in merged[0]["output"]
    assert merged[1]["output"] == "A finished note:"


def test_dangling_lead_in_is_not_its_own_middle_piece():
    moved = move_dangling_lead_ins([
        "The setup is long enough.",
        "5 Interesting Learnings:",
        "1. Price earlier.",
    ])
    assert moved[0] == "The setup is long enough."
    assert moved[1].startswith("5 Interesting Learnings:")
    assert "1. Price earlier." in moved[1]
