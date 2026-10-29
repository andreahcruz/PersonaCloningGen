"""Tests for eval_tools.voice_fixed. A fake distance stands in for Kevin's style distance.

    python -m pytest tests/test_voice_fixed.py
"""
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eval_tools import compare, rows, voice_fixed  # noqa: E402


def words(n, prefix="w"):
    return " ".join(f"{prefix}{i}" for i in range(n))


def echo_length(text):
    """A fake distance: the number of words in the text it is given. It shows exactly what was scored."""
    return float(len(text.split()))


# ---------------------------------------------------------------- one draft

def test_only_the_first_n_words_are_scored():
    value, status = voice_fixed.fixed_length_distance(words(250), 100, echo_length)
    assert (value, status) == (100.0, "scored")


def test_a_draft_exactly_n_words_long_is_scored():
    assert voice_fixed.fixed_length_distance(words(100), 100, echo_length) == (100.0, "scored")


def test_a_draft_shorter_than_n_is_not_scored_and_the_distance_is_never_called():
    called = []
    value, status = voice_fixed.fixed_length_distance(words(99), 100, lambda t: called.append(t) or 1.0)
    assert (value, status) == (None, "too_short") and called == []


def test_a_distance_that_cannot_be_computed_is_reported_as_such():
    assert voice_fixed.fixed_length_distance(words(150), 100, lambda t: None) == (None, "no_distance")


def test_long_and_short_drafts_with_the_same_start_get_the_same_score():
    # The point of the method: length beyond N does not change the score.
    start = words(100)
    long_text = start + " " + words(400, "extra")
    d = lambda t: float(sum(len(w) for w in t.split()))  # depends on the content of the words
    assert voice_fixed.fixed_length_distance(long_text, 100, d) == voice_fixed.fixed_length_distance(start, 100, d)


# ---------------------------------------------------------------- many drafts

DISTANCES = {"blog": echo_length, "linkedin": echo_length}


def draft(topic, medium, n):
    return {"id": topic, "medium": medium, "answer": words(n)}


def test_blog_and_linkedin_are_scored_and_x_and_talk_are_not_supported():
    drafts = [draft("t1", "blog", 200), draft("t1", "linkedin", 150), draft("t1", "x", 400), draft("t1", "talk", 400)]
    by_medium = {r["medium"]: r for r in voice_fixed.score_drafts(drafts, DISTANCES)}
    assert by_medium["blog"]["style_distance_fixed"] == 100.0 and by_medium["blog"]["voice_fixed_status"] == "scored"
    assert by_medium["linkedin"]["style_distance_fixed"] == 100.0
    assert by_medium["x"]["style_distance_fixed"] is None and by_medium["x"]["voice_fixed_status"] == "not_supported"
    assert by_medium["talk"]["voice_fixed_status"] == "not_supported"


def test_each_medium_can_have_its_own_length():
    drafts = [draft("t1", "blog", 300), draft("t1", "linkedin", 300)]
    by_medium = {r["medium"]: r for r in voice_fixed.score_drafts(drafts, DISTANCES, {"blog": 150, "linkedin": 80})}
    assert by_medium["blog"]["style_distance_fixed"] == 150.0 and by_medium["blog"]["voice_fixed_words"] == 150
    assert by_medium["linkedin"]["style_distance_fixed"] == 80.0


def test_result_keeps_the_real_word_count_and_the_key_used_by_compare():
    [item] = voice_fixed.score_drafts([draft("g7", "blog", 123)], DISTANCES)
    assert item["id"] == "g7" and item["medium"] == "blog" and item["voice_words"] == 123


def test_empty_and_missing_text_count_as_too_short():
    out = voice_fixed.score_drafts([{"id": "a", "medium": "blog", "answer": ""}, {"id": "b", "medium": "blog"}], DISTANCES)
    assert [r["voice_fixed_status"] for r in out] == ["too_short", "too_short"]


def test_summary_says_how_many_drafts_were_dropped():
    drafts = [draft("t1", "blog", 200), draft("t2", "blog", 50), draft("t3", "blog", 120), draft("t1", "x", 30)]
    summary = voice_fixed.summarize(voice_fixed.score_drafts(drafts, DISTANCES))
    blog = summary["drafts"]["blog"]
    assert blog["n_drafts"] == 3 and blog["scored"] == 2 and blog["too_short"] == 1
    assert blog["share_scored"] == pytest.approx(2 / 3, abs=1e-4) and blog["mean_distance_fixed"] == 100.0
    assert summary["drafts"]["x"]["not_supported"] == 1 and summary["drafts"]["x"]["mean_distance_fixed"] is None


def test_real_post_reference_uses_only_posts_that_are_long_enough():
    real = voice_fixed.real_post_reference([words(50), words(120), words(300)], 100, echo_length)
    assert real == {"n_real_posts": 3, "n_at_least_n_words": 2, "fixed_words": 100, "mean_distance_fixed": 100.0}


# ---------------------------------------------------------------- plugging into compare

def test_the_metric_is_known_and_lower_is_better():
    assert "style_distance_fixed" in rows.METRIC_PATHS and "style_distance_fixed" in rows.LOWER_IS_BETTER


def test_compare_pairs_only_drafts_that_both_runs_scored(tmp_path):
    def build(values):
        data = {("t%d" % i, "blog"): {"id": "t%d" % i, "medium": "blog", "gates": {}} for i in range(8)}
        side = tmp_path / ("v%s.jsonl" % id(values))
        side.write_text("".join(json.dumps({"id": "t%d" % i, "medium": "blog", "style_distance_fixed": v}) + "\n" for i, v in enumerate(values)))
        assert rows.attach_extra(data, side) == 8
        return data

    a = build([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, None])   # the last draft was too short in run A
    b = build([0.5, 0.5, 0.5, 0.5, 0.5, 0.5, None, 0.5])   # and a different one in run B
    result = compare.compare_runs(a, b, "A", "B", ["style_distance_fixed"], n_boot=200, n_perm=200)
    overall = [r for r in result["views"]["all_drafts"]["results"] if r["scope"] == "all"][0]
    assert overall["n"] == 6 and overall["mean_diff"] == -0.5     # 6 topics have a score in both runs
