"""Fragment prompts are relabeled. Whole posts and the target text stay put."""
from host_finetune.relabel_continuations import (
    FULL_POST_RE,
    action_for,
    classify,
    fit_instruction,
    sentence_final,
    transform_rows,
)


def test_sentence_final_matches_the_audit_regex():
    assert sentence_final("Pay them.")
    assert sentence_final('He said "done."')
    assert sentence_final("Ship it!")
    assert not sentence_final("Pay them")
    assert not sentence_final("Here are 5 lessons:")
    assert sentence_final("trailing...")


def test_groups_follow_the_audit_order():
    assert classify("Ship it.", None, "only", "x") == "G1_whole_x_post"
    assert classify("no period #saas", None, "only", "x") == "G1_whole_x_post"
    assert classify("A full article ends here.", None, "only", "blog") == "G2_whole_document"
    assert classify("A sentence ends.", "The next paragraph starts.", "first", "blog") == (
        "G3_artificial_sentence_complete"
    )
    assert classify("Here are 5 lessons:", "1. Price earlier.", "middle", "blog") == (
        "G5_structural_continuation"
    )
    assert classify("But it should change how", "you hire.", "middle", "youtube_jason") == (
        "G4_mid_sentence"
    )
    assert classify("end of a line", "Next paragraph.", "middle", "blog") == (
        "G4_artificial_unfinished"
    )
    assert classify("and then he said", None, "only", "youtube_jason") == (
        "G6_whole_transcript_no_sentence_end"
    )
    assert classify("and then he said", None, "final", "youtube_jason") == "G6_transcript_tail"
    assert classify("The article really ends here.", None, "final", "blog") == (
        "source_final_fragment"
    )


def test_last_slice_drops_only_when_it_is_not_a_sentence():
    assert action_for("source_final_fragment", "The article really ends here.") == "relabel"
    assert action_for("source_final_fragment", "cut after the wor") == "drop"
    assert action_for("G4_mid_sentence", "how") == "drop"
    assert action_for("G6_transcript_tail", "and then") == "drop"
    assert action_for("G1_whole_x_post", "no period") == "keep"


def _tokens(instruction: str, input_text: str, output: str) -> int:
    return len(instruction.split()) + len(output.split())


def test_continuation_quotes_the_previous_ending_and_is_not_a_full_post():
    instruction = fit_instruction(
        "blog",
        "pricing",
        "Owners wait too long. Price earlier than feels comfortable.",
        "Then hire slower.",
        "",
        limit=40,
        n_tokens=_tokens,
    )
    assert instruction is not None
    assert not FULL_POST_RE.match(instruction)
    assert "Previous section ending:" in instruction
    assert "Price earlier than feels comfortable." in instruction
    assert "Then hire slower." not in instruction


def test_topic_period_is_not_doubled():
    instruction = fit_instruction(
        "blog",
        "pricing.",
        "Owners wait too long.",
        "Then hire slower.",
        "",
        limit=80,
        n_tokens=_tokens,
    )
    assert "about: pricing. Previous" in instruction
    assert "pricing.." not in instruction


def test_opening_has_no_invented_previous_section():
    instruction = fit_instruction(
        "blog", "pricing", None, "Open with the number.", "", limit=40, n_tokens=_tokens
    )
    assert instruction.startswith("Write only the opening section")
    assert "Previous section ending" not in instruction


def test_quote_shrinks_before_the_target_is_cut():
    long_previous = "alpha " * 30 + "Price earlier than feels comfortable."
    instruction = fit_instruction(
        "blog",
        "pricing",
        long_previous,
        "Then hire slower.",
        "",
        limit=16,
        n_tokens=_tokens,
    )
    assert instruction is not None
    assert instruction.endswith("Then hire slower.") is False
    assert "Continue this blog post" in instruction


def test_transform_keeps_whole_posts_and_drops_word_cuts():
    rows = [
        {
            "instruction": "Write an X post in the style of Jason Lemkin about: hiring",
            "input": "",
            "output": "Hire slow #saas",
            "source": "x",
            "source_file": "x.jsonl",
            "source_line": 1,
        },
        {
            "instruction": "Write a blog post in the style of Jason Lemkin about: pricing (part 1 of 2)",
            "input": "",
            "output": "Price earlier.",
            "source": "blog",
            "source_file": "blog.jsonl",
            "source_line": 2,
        },
        {
            "instruction": "Write a blog post in the style of Jason Lemkin about: pricing (part 2 of 2)",
            "input": "",
            "output": "Then hire slower.",
            "source": "blog",
            "source_file": "blog.jsonl",
            "source_line": 2,
        },
        {
            "instruction": "Write a talk in the style of Jason Lemkin about: growth",
            "input": "",
            "output": "But it should change how",
            "source": "youtube_jason",
            "source_file": "yt.jsonl",
            "source_line": 3,
        },
        {
            "instruction": "Write a talk in the style of Jason Lemkin about: growth",
            "input": "",
            "output": "you hire the first AE.",
            "source": "youtube_jason",
            "source_file": "yt.jsonl",
            "source_line": 3,
        },
    ]
    assignments = [
        {"row_index": i, "split": "validation", "group_id": "g", "base_title": "t"}
        for i in range(len(rows))
    ]
    # The checksum covers the train split only. Park these rows in validation
    # after a throwaway train row so the expected train counts can be patched.
    new_rows, new_assignments, dispositions, _stats = transform_rows(
        rows, assignments, _tokens, limit=80
    )
    assert [row["output"] for row in new_rows] == [
        "Hire slow #saas",
        "Price earlier.",
        "Then hire slower.",
        "you hire the first AE.",
    ]
    assert new_rows[0]["instruction"].startswith("Write an X post")
    assert new_rows[1]["sft_role"] == "opening"
    assert new_rows[2]["sft_role"] == "continuation"
    assert "Price earlier." in new_rows[2]["instruction"]
    assert not FULL_POST_RE.match(new_rows[1]["instruction"])
    assert not FULL_POST_RE.match(new_rows[2]["instruction"])
    assert dispositions[3]["action"] == "drop"
    assert new_assignments[3]["split"] == "validation"
    assert new_rows[3]["output"] == "you hire the first AE."
