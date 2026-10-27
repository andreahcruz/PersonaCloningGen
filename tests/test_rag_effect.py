"""Tests for eval_tools.rag_effect (PUK-34). All data here is small and made up, so nothing heavy runs.

    python -m pytest tests/test_rag_effect.py
"""
import json
import sys
from argparse import Namespace
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eval_tools import rag_effect  # noqa: E402

MEDIUMS = ("blog", "linkedin", "x", "talk")


def draft(topic, medium, **fields):
    """A raw draft the way the generation script saves it, with only the fields these tests need."""
    base = {"id": topic, "medium": medium, "answer": "one two three four", "prompt_tokens": 100,
            "generated_tokens_including_stop": 50, "stop_reason": "eos", "early_eos": False}
    base.update(fields)
    return base


def rag_draft(topic, medium, **fields):
    base = {"retrieval": True, "extraction_seconds": 10.0, "verification_seconds": 0.0,
            "numeric_grounding": {"applicable": True, "pass": True}, "rag_source_copying": {"flagged": False},
            "retrieval_hits": [{"medium": "blog", "distance": 0.2}] * 4,
            "proposed_atomic_claims": [1, 2, 3, 4], "verified_claims": [1, 2], "accepted_attempt": 1}
    base.update(fields)
    return draft(topic, medium, **base)


def write(path, rows):
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    return path


# ---------------------------------------------------------------- loading

def test_slim_keeps_retrieval_fields_only_when_retrieval_was_on():
    off = rag_effect.slim_draft(draft("t1", "blog"))
    on = rag_effect.slim_draft(rag_draft("t1", "blog"))
    assert "extraction_seconds" not in off and off["retrieval"] is False
    assert on["extraction_seconds"] == 10.0 and on["n_hits"] == 4
    assert on["n_proposed_claims"] == 4 and on["n_verified_claims"] == 2


def test_slim_counts_words_from_the_answer_when_no_count_is_saved():
    assert rag_effect.slim_draft(draft("t1", "blog", answer="a b c"))["answer_words"] == 3


def test_load_drafts_rejects_two_drafts_with_one_key(tmp_path):
    path = write(tmp_path / "d.jsonl", [draft("t1", "blog"), draft("t1", "blog")])
    with pytest.raises(ValueError, match="share the key"):
        rag_effect.load_drafts(path)


def test_slimming_a_slim_draft_changes_nothing():
    once = rag_effect.slim_draft(rag_draft("t1", "blog", numeric_grounding={"applicable": True, "pass": False},
                                           rag_source_copying={"flagged": True}))
    assert rag_effect.slim_draft(once) == once
    assert once["numeric_pass"] is False and once["copy_flagged"] is True


def test_report_from_slim_files_equals_report_from_raw_files(tmp_path):
    raw_a = write(tmp_path / "a.jsonl", [draft(f"t{i}", m) for i in range(4) for m in MEDIUMS])
    raw_b = write(tmp_path / "b.jsonl", [rag_draft(f"t{i}", m, numeric_grounding={"applicable": True, "pass": i % 2 == 0},
                                                   rag_source_copying={"flagged": i == 0}) for i in range(4) for m in MEDIUMS])
    slim_b = tmp_path / "b.slim.jsonl"
    rag_effect._cmd_slim(Namespace(raw=str(raw_b), out=str(slim_b)))
    from_raw = rag_effect.rag_only(rag_effect.load_drafts(raw_b))
    from_slim = rag_effect.rag_only(rag_effect.load_drafts(slim_b))
    assert from_raw == from_slim
    assert from_slim["overall"]["unsupported_numbers_rate"] == 0.5
    assert from_slim["overall"]["copy_flag_count"] == 4
    assert rag_effect.paired_cost(rag_effect.load_drafts(raw_a), rag_effect.load_drafts(raw_b), n_boot=100) == \
        rag_effect.paired_cost(rag_effect.load_drafts(raw_a), rag_effect.load_drafts(slim_b), n_boot=100)


def test_slim_command_writes_one_line_per_draft(tmp_path):
    raw = write(tmp_path / "raw.jsonl", [rag_draft(f"t{i}", "blog") for i in range(3)])
    out = tmp_path / "slim.jsonl"
    rag_effect._cmd_slim(Namespace(raw=str(raw), out=str(out)))
    assert len(out.read_text(encoding="utf-8").splitlines()) == 3
    assert set(rag_effect.load_drafts(out)) == {("t0", "blog"), ("t1", "blog"), ("t2", "blog")}


# ---------------------------------------------------------------- paired cost

def runs(extra_prompt=300):
    topics = [f"t{i}" for i in range(8)]
    a = {(t, m): rag_effect.slim_draft(draft(t, m)) for t in topics for m in MEDIUMS}
    b = {(t, m): rag_effect.slim_draft(rag_draft(t, m, prompt_tokens=100 + extra_prompt)) for t in topics for m in MEDIUMS}
    return a, b


def test_prompt_token_difference_is_exact_and_clear():
    a, b = runs(extra_prompt=300)
    result = {(r["metric"], r["scope"]): r for r in rag_effect.paired_cost(a, b, n_boot=300)}
    overall = result[("prompt_tokens", "all")]
    assert overall["mean_diff"] == 300 and overall["n"] == 32
    assert overall["mean_a"] == 100 and overall["mean_b"] == 400
    assert overall["ci_low"] > 0
    assert result[("prompt_tokens", "talk")]["n"] == 8


def test_share_of_drafts_cut_at_the_length_limit():
    a, b = runs()
    for key in list(b)[:8]:
        b[key]["stop_reason"] = "length"
    result = {(r["metric"], r["scope"]): r for r in rag_effect.paired_cost(a, b, n_boot=300)}
    assert result[("stopped_at_length", "all")]["mean_diff"] == 8 / 32


def test_pairs_with_a_missing_value_are_skipped_not_counted_as_zero():
    a, b = runs()
    a[("t0", "blog")]["prompt_tokens"] = None
    result = {(r["metric"], r["scope"]): r for r in rag_effect.paired_cost(a, b, n_boot=300)}
    assert result[("prompt_tokens", "all")]["n"] == 31


def test_only_drafts_present_in_both_runs_are_compared():
    a, b = runs()
    del b[("t0", "x")]
    result = {(r["metric"], r["scope"]): r for r in rag_effect.paired_cost(a, b, n_boot=300)}
    assert result[("answer_words", "all")]["n"] == 31


# ---------------------------------------------------------------- retrieval-only numbers

def test_no_retrieval_means_no_retrieval_numbers():
    a, _ = runs()
    assert rag_effect.rag_only(a) == {"n": 0}


def test_evidence_seconds_are_counted_once_per_topic_not_once_per_draft():
    # Two topics, four mediums each. The same extraction time is copied onto every medium of a topic.
    drafts = {}
    for topic, secs in (("t1", 10.0), ("t2", 30.0)):
        for m in MEDIUMS:
            drafts[(topic, m)] = rag_effect.slim_draft(rag_draft(topic, m, extraction_seconds=secs))
    overall = rag_effect.rag_only(drafts)["overall"]
    assert overall["topics"] == 2
    assert overall["evidence_seconds_total"] == 40          # not 160
    assert overall["evidence_seconds_per_topic_mean"] == 20
    assert overall["evidence_seconds_per_draft"] == 5        # 40 seconds spread over 8 drafts


def test_totals_are_not_reported_per_medium():
    drafts = {(f"t{i}", m): rag_effect.slim_draft(rag_draft(f"t{i}", m)) for i in range(3) for m in MEDIUMS}
    blog = rag_effect.rag_only(drafts)["by_medium"]["blog"]
    assert blog["evidence_seconds_total"] is None and blog["evidence_seconds_per_draft"] is None
    assert blog["evidence_seconds_per_topic_mean"] == 10


def test_verification_time_is_added_to_the_topic_total():
    drafts = {("t1", "blog"): rag_effect.slim_draft(rag_draft("t1", "blog", extraction_seconds=10.0, verification_seconds=3.0))}
    assert rag_effect.rag_only(drafts)["overall"]["evidence_seconds_total"] == 13


def test_passage_fit_depends_on_the_requested_medium():
    hits_blog = [{"medium": "blog", "distance": 0.2}] * 4
    drafts = {
        ("t1", "blog"): rag_effect.slim_draft(rag_draft("t1", "blog", retrieval_hits=hits_blog)),
        ("t1", "linkedin"): rag_effect.slim_draft(rag_draft("t1", "linkedin", retrieval_hits=hits_blog)),
        ("t1", "talk"): rag_effect.slim_draft(rag_draft("t1", "talk", retrieval_hits=hits_blog)),
    }
    effect = rag_effect.rag_only(drafts)
    assert effect["by_medium"]["blog"]["passage_fits_medium"] == 1.0
    assert effect["by_medium"]["linkedin"]["passage_fits_medium"] == 0.0
    assert effect["by_medium"]["talk"]["passage_fits_medium"] == 0.0
    assert effect["overall"]["passage_fits_medium"] == pytest.approx(4 / 12, abs=1e-4)


def test_talk_fits_transcript_passages():
    hits = [{"medium": "youtube_jason", "distance": 0.3}] * 2
    drafts = {("t1", "talk"): rag_effect.slim_draft(rag_draft("t1", "talk", retrieval_hits=hits))}
    assert rag_effect.rag_only(drafts)["overall"]["passage_fits_medium"] == 1.0


def test_unsupported_numbers_are_a_share_of_drafts_where_the_check_applies():
    drafts = {
        ("t1", "blog"): rag_effect.slim_draft(rag_draft("t1", "blog", numeric_grounding={"applicable": True, "pass": False})),
        ("t2", "blog"): rag_effect.slim_draft(rag_draft("t2", "blog", numeric_grounding={"applicable": True, "pass": True})),
        ("t3", "blog"): rag_effect.slim_draft(rag_draft("t3", "blog", numeric_grounding={"applicable": False})),
    }
    overall = rag_effect.rag_only(drafts)["overall"]
    assert overall["unsupported_numbers_rate"] == 0.5
    assert overall["unsupported_numbers_count"] == 1


def test_claim_acceptance_and_copy_flags_and_second_attempts():
    drafts = {
        ("t1", "blog"): rag_effect.slim_draft(rag_draft("t1", "blog", rag_source_copying={"flagged": True}, accepted_attempt=2)),
        ("t2", "blog"): rag_effect.slim_draft(rag_draft("t2", "blog")),
    }
    overall = rag_effect.rag_only(drafts)["overall"]
    assert overall["claim_acceptance"] == 0.5                # 4 of 8 proposed claims verified
    assert overall["copy_flag_count"] == 1 and overall["copy_flag_rate"] == 0.5
    assert overall["second_attempt_rate"] == 0.5


# ---------------------------------------------------------------- report text

def test_report_names_both_runs_and_says_how_to_read_the_fit_rows(tmp_path):
    a, b = runs()
    text = rag_effect.to_markdown(rag_effect.paired_cost(a, b, n_boot=200), rag_effect.rag_only(b), "Off", "On")
    assert "What On changed and what it cost, compared with Off" in text
    assert "counted once per topic" in text
    assert "Retrieved passages that fit the requested medium" in text
    assert "ignores the requested medium" in text


def test_report_without_retrieval_has_no_retrieval_section():
    a, _ = runs()
    text = rag_effect.to_markdown(rag_effect.paired_cost(a, a, n_boot=200), rag_effect.rag_only(a), "A", "B")
    assert "retrieval on" not in text
