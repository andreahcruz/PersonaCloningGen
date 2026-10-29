"""Tests for eval_tools.grammar_lt. A fake LanguageTool is used, so nothing here needs Java or a download.

    python -m pytest tests/test_grammar_lt.py
"""
import json
import sys
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eval_tools import compare, grammar_lt, rows  # noqa: E402


def match(rule, category, text, context="some context", message="msg", offset=0):
    """What language_tool_python returns for one match, with only the fields we read."""
    return SimpleNamespace(rule_id=rule, category=category, matched_text=text, context=context, message=message, offset=offset)


class FakeTool:
    """Returns the matches registered for a text. Unknown texts have no matches."""

    def __init__(self, table=None):
        self.table = table or {}
        self.closed = False
        self.seen = []

    def check(self, text):
        self.seen.append(text)
        return self.table.get(text, [])

    def close(self):
        self.closed = True


def asdict(rule, category, text, context="ok"):
    return {"rule": rule, "category": category, "text": text, "context": context}


VOCAB = Counter({"overhire": 5, "typo": 1})


# ---------------------------------------------------------------- the filter

@pytest.mark.parametrize("category", ["GRAMMAR", "TYPOS", "CONFUSED_WORDS"])
def test_grammar_typo_and_confused_word_categories_are_kept(category):
    assert grammar_lt.keep_match(asdict("SOME_RULE", category, "x"), VOCAB)


@pytest.mark.parametrize("category", ["TYPOGRAPHY", "PUNCTUATION", "STYLE", "REPETITIONS_STYLE", "CASING", "MISC"])
def test_style_and_typography_categories_are_dropped(category):
    assert not grammar_lt.keep_match(asdict("SOME_RULE", category, "x"), VOCAB)


def test_a_vs_an_and_doubled_word_are_kept_although_languagetool_calls_them_misc():
    assert grammar_lt.keep_match(asdict("EN_A_VS_AN", "MISC", "a"), VOCAB)
    assert grammar_lt.keep_match(asdict("ENGLISH_WORD_REPEAT_RULE", "MISC", "for for"), VOCAB)


def test_disabled_rules_are_dropped_even_in_a_kept_category():
    assert "THE_SUPERLATIVE" in grammar_lt.DISABLED_RULES
    assert not grammar_lt.keep_match(asdict("THE_SUPERLATIVE", "GRAMMAR", "best"), VOCAB)


def test_a_match_touching_a_broken_character_is_dropped():
    broken = "it�s"
    assert not grammar_lt.keep_match(asdict("IT_ITS", "GRAMMAR", "it", context=f"when {broken} here"), VOCAB)
    assert not grammar_lt.keep_match(asdict("IT_ITS", "GRAMMAR", broken), VOCAB)


def test_spelling_flags_only_unknown_all_lowercase_words():
    spell = grammar_lt.SPELLING_RULE
    assert grammar_lt.keep_match(asdict(spell, "TYPOS", "decelaration"), VOCAB)       # a real typo
    assert not grammar_lt.keep_match(asdict(spell, "TYPOS", "Coatue"), VOCAB)         # a name
    assert not grammar_lt.keep_match(asdict(spell, "TYPOS", "SaaStr"), VOCAB)         # mixed case
    assert not grammar_lt.keep_match(asdict(spell, "TYPOS", "EMW"), VOCAB)            # an acronym
    assert not grammar_lt.keep_match(asdict(spell, "TYPOS", "b2b"), VOCAB)            # not only letters


def test_spelling_flag_is_dropped_when_lemkin_uses_the_word_often_enough():
    spell = grammar_lt.SPELLING_RULE
    assert not grammar_lt.keep_match(asdict(spell, "TYPOS", "overhire"), VOCAB)       # used 5 times in real posts
    assert grammar_lt.keep_match(asdict(spell, "TYPOS", "typo"), VOCAB)               # used once: still a typo
    assert grammar_lt.keep_match(asdict(spell, "TYPOS", "typo"), Counter({"typo": 2}))
    assert not grammar_lt.keep_match(asdict(spell, "TYPOS", "typo"), Counter({"typo": 3}))


def test_vocabulary_counts_lowercase_words_and_ignores_urls():
    vocabulary = grammar_lt.build_vocabulary(["Overhire is bad. overhire again, see https://x.com/overhire", "OVERHIRE"])
    assert vocabulary["overhire"] == 3
    assert vocabulary["bad"] == 1


# ---------------------------------------------------------------- the checker

def test_checker_returns_plain_dicts_and_applies_the_filter():
    tool = FakeTool({"He go home. dont.": [
        match("HE_VERB_AGR", "GRAMMAR", "go", context="He go home"),
        match("COMMA_PARENTHESIS_WHITESPACE", "TYPOGRAPHY", " ."),
    ]})
    found = grammar_lt.Checker(VOCAB, tool=tool).check_text("He go home. dont.")
    assert [f["rule"] for f in found] == ["HE_VERB_AGR"]
    assert found[0]["text"] == "go" and found[0]["category"] == "GRAMMAR" and "context" in found[0]


def test_empty_text_makes_no_call_to_languagetool():
    tool = FakeTool()
    assert grammar_lt.Checker(VOCAB, tool=tool).check_text("   ") == []
    assert grammar_lt.Checker(VOCAB, tool=tool).check_text(None) == []
    assert tool.seen == []


def test_checker_closes_the_tool_when_used_in_a_with_block():
    tool = FakeTool()
    with grammar_lt.Checker(VOCAB, tool=tool):
        pass
    assert tool.closed


# ---------------------------------------------------------------- rates and deviations

def checker_with(table):
    return grammar_lt.Checker(VOCAB, tool=FakeTool(table))


def test_error_rate_is_errors_per_100_words():
    text = " ".join(["word"] * 50)
    checker = checker_with({text: [match("EN_A_VS_AN", "MISC", "a"), match("EN_A_VS_AN", "MISC", "a")]})
    stats = grammar_lt.error_stats(text, checker)
    assert stats["errors"] == 2 and stats["words"] == 50 and stats["rate"] == 4.0
    assert stats["rules"] == {"EN_A_VS_AN": 2}


def test_pooled_rate_divides_total_errors_by_total_words():
    a, b = " ".join(["w"] * 100), " ".join(["w"] * 300)
    checker = checker_with({a: [match("EN_A_VS_AN", "MISC", "a")] * 2, b: []})
    pooled = grammar_lt.pooled_rate([a, b], checker)
    assert pooled["errors"] == 2 and pooled["words"] == 400 and pooled["rate"] == 0.5 and pooled["n_texts"] == 2


def test_deviation_is_draft_rate_minus_the_real_post_rate_of_the_same_medium():
    text = " ".join(["w"] * 100)
    checker = checker_with({text: [match("EN_A_VS_AN", "MISC", "a")] * 3})
    reference = {"blog": {"rate": 0.5}}
    [item] = grammar_lt.score_drafts([{"id": "g1", "medium": "blog", "answer": text}], reference, checker)
    assert item["grammar_lt_rate"] == 3.0 and item["grammar_lt_deviation"] == 2.5
    assert item["id"] == "g1" and item["medium"] == "blog"


def test_talk_has_no_reference_so_no_deviation_but_still_a_rate():
    text = " ".join(["w"] * 100)
    checker = checker_with({text: [match("EN_A_VS_AN", "MISC", "a")]})
    [item] = grammar_lt.score_drafts([{"id": "g1", "medium": "talk", "answer": text}], {"blog": {"rate": 0.5}}, checker)
    assert item["grammar_lt_rate"] == 1.0 and item["grammar_lt_deviation"] is None


def test_a_draft_with_no_words_has_no_rate():
    [item] = grammar_lt.score_drafts([{"id": "g1", "medium": "blog", "answer": ""}], {"blog": {"rate": 0.5}}, checker_with({}))
    assert item["grammar_lt_rate"] is None and item["grammar_lt_deviation"] is None


def test_summary_counts_drafts_with_an_error_and_the_excess_over_real_posts():
    t1, t2 = " ".join(["w"] * 100), " ".join(["v"] * 100)
    checker = checker_with({t1: [match("EN_A_VS_AN", "MISC", "a")] * 2})
    results = grammar_lt.score_drafts(
        [{"id": "a", "medium": "blog", "answer": t1}, {"id": "b", "medium": "blog", "answer": t2}], {"blog": {"rate": 0.5}}, checker
    )
    summary = grammar_lt.summarize(results, {"blog": {"rate": 0.5, "n_texts": 400, "words": 1000}})
    blog = summary["drafts"]["blog"]
    assert blog["n_drafts"] == 2 and blog["drafts_with_an_error"] == 1
    assert blog["rate_per_100_words"] == 1.0 and blog["excess_over_reference"] == 0.5
    assert blog["top_rules"] == {"EN_A_VS_AN": 2}


# ---------------------------------------------------------------- plugging into compare

def test_new_metrics_are_known_and_lower_is_better():
    for name in ("grammar_lt_rate", "grammar_lt_deviation", "style_distance_fixed"):
        assert name in rows.METRIC_PATHS and name in rows.LOWER_IS_BETTER and name in compare.DEFAULT_METRICS


def test_compare_reads_the_grammar_lt_side_file(tmp_path):
    def run(rates):
        data = {("t%d" % i, "blog"): {"id": "t%d" % i, "medium": "blog", "gates": {}} for i in range(8)}
        side = tmp_path / ("extra%s.jsonl" % id(rates))
        side.write_text("".join(json.dumps({"id": "t%d" % i, "medium": "blog", "grammar_lt_rate": r}) + "\n" for i, r in enumerate(rates)), encoding="utf-8")
        assert rows.attach_extra(data, side) == 8
        return data

    a, b = run([1.0] * 8), run([0.0] * 8)
    result = compare.compare_runs(a, b, "A", "B", ["grammar_lt_rate"], n_boot=200, n_perm=200)
    overall = [r for r in result["views"]["all_drafts"]["results"] if r["scope"] == "all"][0]
    assert overall["mean_diff"] == -1.0 and overall["lower_is_better"] is True
    assert overall["verdict"] == "B better"
