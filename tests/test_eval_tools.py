"""Tests for eval_tools. All data here is small and made up, so nothing heavy runs.

    python -m pytest tests/test_eval_tools.py
"""
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eval_tools import bootstrap_ci, compare, grammar, rows  # noqa: E402


# ---------------------------------------------------------------- bootstrap_ci

def test_all_positive_differences_give_a_clear_result():
    diffs = [1.0] * 12
    clusters = list(range(12))
    result = bootstrap_ci.summarize_paired(diffs, clusters, n_boot=500, n_perm=500)
    assert result["mean_diff"] == 1.0
    assert result["ci_low"] > 0
    assert result["p_value"] < 0.01


def test_symmetric_differences_are_not_clear():
    diffs = [1.0, -1.0] * 8
    clusters = list(range(16))
    result = bootstrap_ci.summarize_paired(diffs, clusters, n_boot=500, n_perm=500)
    assert result["mean_diff"] == 0.0
    assert result["p_value"] == 1.0
    assert result["ci_low"] <= 0 <= result["ci_high"]


def test_same_seed_gives_same_answer():
    diffs = [0.1, 0.5, -0.2, 0.4, 0.3, 0.0, 0.2, 0.6]
    a = bootstrap_ci.summarize_paired(diffs, None, n_boot=300, n_perm=300, seed=7)
    b = bootstrap_ci.summarize_paired(diffs, None, n_boot=300, n_perm=300, seed=7)
    assert a == b


def test_too_few_pairs_has_no_interval():
    result = bootstrap_ci.summarize_paired([0.5, 0.4], None)
    assert result["ci_low"] is None
    assert "too few" in result["note"]
    assert result["mean_diff"] == pytest.approx(0.45)


def test_empty_input_is_reported_not_crashed():
    assert bootstrap_ci.summarize_paired([])["note"] == "no paired drafts"


def test_clusters_are_resampled_together():
    # Four mediums per topic share one sign. Many draft level values, few independent topics.
    diffs = [1.0] * 4 + [-1.0] * 4 + [1.0] * 4
    clusters = [0] * 4 + [1] * 4 + [2] * 4
    result = bootstrap_ci.summarize_paired(diffs, clusters, n_boot=500, n_perm=500)
    assert result["n"] == 12 and result["n_clusters"] == 3
    assert result["ci_low"] is None  # only 3 topics: too few to claim an interval


def test_sign_flip_is_exact_for_few_clusters():
    # Two clusters, both +1: only 2 of 4 sign patterns reach |sum| = 2.
    assert bootstrap_ci.sign_flip_p([1.0, 1.0], [0, 1]) == pytest.approx(0.5)


def test_holm_adjustment():
    adjusted = bootstrap_ci.holm_adjust({"a": 0.01, "b": 0.04, "c": 0.03, "d": None})
    assert adjusted["a"] == pytest.approx(0.03)
    assert adjusted["c"] == pytest.approx(0.06)
    assert adjusted["b"] == pytest.approx(0.06)  # never below an earlier, smaller adjusted value
    assert adjusted["d"] is None


# ------------------------------------------------------------------------ rows

def make_row(topic, medium="blog", writing=3.0, style=0.8, words=200, passes=True, grounding=None, eligible=None):
    gates = {
        "completion": {"pass": passes},
        "copying_memorization": {"pass": True},
        "format": {"pass": True},
    }
    if grounding is not None:
        gates["grounding"] = {"pass": grounding, "applicable": True}
    else:
        gates["grounding"] = {"pass": True, "applicable": False}
    if eligible is None:
        eligible = passes and (grounding is not False)
    return {
        "id": topic,
        "medium": medium,
        "eligible_for_ranking": eligible,
        "gates": {**gates, "format": {"pass": True, "word_count": words}},
        "writing_quality": {"mean": writing},
        "persona_style": {"distance": style, "role": "selector"},
    }


def test_get_path_ignores_bad_values():
    row = {"a": {"b": 2}, "c": True, "d": float("nan"), "e": None, "f": "x"}
    assert rows.get_path(row, "a.b") == 2.0
    for path in ("c", "d", "e", "f", "missing.path"):
        assert rows.get_path(row, path) is None


def test_load_rows_filters_by_experiment_and_model(tmp_path):
    path = tmp_path / "rows.jsonl"
    items = [
        {"id": "g1", "medium": "blog", "source_experiment": "E1", "model": "relabel"},
        {"id": "g1", "medium": "blog", "source_experiment": "E1", "model": "repaired"},
        {"id": "g1", "medium": "blog", "source_experiment": "E2", "model": "EXP-9/relabel"},
    ]
    path.write_text("\n".join(json.dumps(i) for i in items), encoding="utf-8")
    assert len(rows.load_rows(path)) == 3
    assert len(rows.load_rows(path, source_experiment="E1", model="relabel")) == 1
    assert len(rows.load_rows(path, model="relabel")) == 2  # whole name or last path part


def test_duplicate_keys_are_refused():
    with pytest.raises(ValueError, match="share the key"):
        rows.index_rows([make_row("g1"), make_row("g1")])


def test_common_gates_ignore_grounding():
    failing_grounding = make_row("g1", grounding=False)
    assert failing_grounding["eligible_for_ranking"] is False
    assert rows.common_gates_pass(failing_grounding) is True
    assert rows.common_gates_pass(make_row("g2", passes=False)) is False


def test_attach_extra_merges_by_id_and_medium(tmp_path):
    indexed = rows.index_rows([make_row("g1"), make_row("g2")])
    extra = tmp_path / "extra.jsonl"
    extra.write_text(json.dumps({"id": "g1", "medium": "blog", "grammar_rate": 0.5}) + "\n", encoding="utf-8")
    assert rows.attach_extra(indexed, extra) == 1
    assert rows.metric_value(indexed[("g1", "blog")], "grammar_rate") == 0.5
    assert rows.metric_value(indexed[("g2", "blog")], "grammar_rate") is None


# --------------------------------------------------------------------- compare

def build_runs(n=12):
    a = rows.index_rows([make_row(f"g{i}", writing=3.0, style=0.9, words=200) for i in range(n)])
    b = rows.index_rows([make_row(f"g{i}", writing=3.0 + 0.5, style=0.9 - 0.2, words=150 + i) for i in range(n)])
    return a, b


def test_compare_reports_direction_with_lower_is_better():
    a, b = build_runs()
    result = compare.compare_runs(a, b, "A", "B", ["writing_quality", "style_distance", "word_count"], n_boot=300, n_perm=300)
    by_metric = {r["metric"]: r for r in result["views"]["all_drafts"]["results"] if r["scope"] == "all"}
    assert by_metric["writing_quality"]["verdict"] == "B better"      # higher is better
    assert by_metric["style_distance"]["verdict"] == "B better"       # lower is better
    assert by_metric["word_count"]["verdict"] == "B lower"            # neutral wording
    assert result["n_paired"] == 12


def test_compare_counts_unpaired_drafts():
    a, b = build_runs(10)
    del b[("g0", "blog")]
    b[("extra", "blog")] = make_row("extra")
    result = compare.compare_runs(a, b, "A", "B", ["writing_quality"], n_boot=100, n_perm=100)
    assert result["n_paired"] == 9
    assert result["only_in_a"] == 1 and result["only_in_b"] == 1


def test_extra_grounding_gate_does_not_hide_a_gain():
    # B passes more of the common gates, but its grounding gate fails two drafts, so its
    # "eligible" count looks worse. The common gate view must show the real picture.
    a = rows.index_rows(
        [make_row(f"g{i}", passes=(i >= 4)) for i in range(10)]   # 6 of 10 pass completion
    )
    b = rows.index_rows(
        [make_row(f"g{i}", passes=(i >= 2), grounding=(i >= 4 or None) if i < 4 else None) for i in range(10)]
    )
    b[("g2", "blog")] = make_row("g2", passes=True, grounding=False)
    b[("g3", "blog")] = make_row("g3", passes=True, grounding=False)
    result = compare.compare_runs(a, b, "A", "B", ["writing_quality"], n_boot=100, n_perm=100)
    arm_a, arm_b = result["arms"]["A"], result["arms"]["B"]
    assert arm_b["common_gates_pass"] > arm_a["common_gates_pass"]
    assert arm_b["eligible"] <= arm_b["common_gates_pass"] - 2
    assert arm_b["gate_failures"]["grounding"]["of"] == 2        # counted only where it applies
    assert arm_a["gate_failures"]["grounding"]["of"] == 0


def test_common_gate_view_keeps_only_drafts_passing_in_both():
    a = rows.index_rows([make_row("g1"), make_row("g2", passes=False), make_row("g3")])
    b = rows.index_rows([make_row("g1"), make_row("g2"), make_row("g3", passes=False)])
    result = compare.compare_runs(a, b, "A", "B", ["writing_quality"], n_boot=100, n_perm=100)
    assert result["views"]["all_drafts"]["n_pairs"] == 3
    assert result["views"]["common_gates"]["n_pairs"] == 1


def test_unknown_metric_is_rejected():
    a, b = build_runs()
    with pytest.raises(ValueError, match="Unknown metric"):
        compare.compare_runs(a, b, metrics=["vibes"])


def test_markdown_report_mentions_both_views_and_labels():
    a, b = build_runs()
    text = compare.to_markdown(compare.compare_runs(a, b, "RAG off", "RAG on", ["writing_quality"], n_boot=100, n_perm=100))
    assert "RAG on against RAG off" in text
    assert "All drafts" in text and "common gates" in text
    assert "B minus A" in text


# --------------------------------------------------------------------- grammar

@pytest.mark.parametrize(
    "text, rule",
    [
        ("We want the the best result for founders.", "repeated_word"),
        ("It was late.Then we left.", "missing_space_after_punct"),
        ("Hiring is hard , but worth it.", "space_before_punct"),
        ("That was the last deal. then we moved on.", "lowercase_sentence_start"),
        ("Honestly i think this is right.", "lowercase_i"),
        ("We tried it,, and it worked.", "double_punct"),
        ("She hired a employee last week.", "a_an"),
        ("He wrote an letter to the board.", "a_an"),
        ("They dont know what happened.", "missing_apostrophe"),
        ("You should of called sooner.", "modal_of"),
        ("They was ready for the launch.", "agreement"),
        ("We will definately raise next year.", "misspelling"),
        ("The text had a broken � character.", "broken_character"),
    ],
)
def test_each_rule_fires(text, rule):
    assert rule in {item["rule"] for item in grammar.check_text(text)}


@pytest.mark.parametrize(
    "text",
    [
        "We raised a Series A in March and a unit economics model after.",   # Series A, a unit
        "He drove a Uber to an hour long meeting with an MBA.",              # a Uber, an hour, an MBA
        "The net net result was very very good, and that that was fine.",    # deliberate repeats
        "Read more at https://example.com/page.Next and email me@site.io now.",
        "Use e.g. a short list (i) first, then (ii) second, etc. and so on.",       # abbreviations
        "## Heading\n\n- first point\n- second point\n\n**Bold** text and `code`.",
        "Wait... what? Yes... really. The year is 2026.Next year is different.".replace("2026.Next", "2026. Next"),
        "Revenue grew 3.5x. hashtag#SaaS was trending.".replace("x. hashtag", "x. Hashtag"),
    ],
)
def test_clean_text_is_not_flagged(text):
    assert grammar.check_text(text) == []


def test_markdown_and_links_are_not_counted_as_words_or_errors():
    text = "## Title\n\n- Visit https://a.b/c.Next now\n- @handle said hi"
    assert grammar.count_words(text) == len(grammar.WORD_RE.findall("Title Visit URL now HANDLE said hi"))


def test_error_rate_is_per_100_words():
    filler = "Alpha beta gamma delta epsilon zeta eta theta iota kappa lambda sigma omega rho tau phi chi psi"
    stats = grammar.error_stats("They dont know. " + filler)  # 21 words, 1 error
    assert stats["errors"] == 1
    assert stats["words"] == 21
    assert stats["rate"] == pytest.approx(100 / 21, rel=0.01)


def test_reference_texts_skip_replies_html_and_talk(tmp_path):
    (tmp_path / "jasonlemkin_blog.jsonl").write_text(
        json.dumps({"content": "<p>Real blog text that is long enough to count.</p>\nRelated Posts\nignore this"}) + "\n",
        encoding="utf-8",
    )
    (tmp_path / "jasonlk_originals.jsonl").write_text(
        json.dumps({"text": "A real tweet that is long enough to keep.", "is_reply": False}) + "\n"
        + json.dumps({"text": "A reply that is long enough but is a reply.", "is_reply": True}) + "\n",
        encoding="utf-8",
    )
    ref = grammar.reference_texts(tmp_path)
    assert "talk" not in ref
    assert len(ref["x"]) == 1
    assert "<p>" not in ref["blog"][0] and "ignore this" not in ref["blog"][0]


def test_deviation_is_none_without_a_reference():
    reference = {"blog": {"rate": 0.5, "n_texts": 10, "words": 1000}}
    drafts = [
        {"id": "g1", "medium": "blog", "answer": "They dont know what happened in the room today at all."},
        {"id": "g1", "medium": "talk", "answer": "They dont know what happened in the room today at all."},
    ]
    results = grammar.score_drafts(drafts, reference)
    assert results[0]["grammar_deviation"] is not None
    assert results[1]["grammar_deviation"] is None
    summary = grammar.summarize(results, reference)
    assert summary["drafts"]["talk"]["excess_over_reference"] is None


# ------------------------------------------------------- multiplicity and seeds

def _result(metric, scope, p, diff=0.5, low=0.1, high=0.9, lower_is_better=False):
    return {
        "metric": metric, "scope": scope, "p_value": p, "mean_diff": diff,
        "ci_low": low, "ci_high": high, "lower_is_better": lower_is_better,
    }


def test_verdict_has_three_outcomes():
    clear = {"mean_diff": 0.5, "ci_low": 0.1, "ci_high": 0.9, "p_value": 0.01, "p_adj": 0.02}
    assert compare.verdict(clear, False) == "B better"
    assert compare.verdict(clear, True) == "B worse"
    assert compare.verdict({**clear, "p_adj": 0.2}, False) == "B better (suggestive)"
    assert compare.verdict({**clear, "p_value": 0.2, "p_adj": 1.0}, False) == "no clear difference"
    assert compare.verdict({**clear, "ci_low": -0.1}, False) == "no clear difference"
    assert compare.verdict({**clear, "ci_low": None}, False) == "too few pairs"
    assert compare.verdict(clear, False, neutral=True) == "B higher"


def test_primary_rows_are_corrected_together_and_exploratory_separately():
    results = [
        _result("writing_quality", "all", 0.01),     # primary
        _result("style_distance", "blog", 0.02),     # primary
        _result("stance_consistency", "all", 0.5),   # primary
        _result("task_adherence", "all", 0.6),       # primary
        _result("word_count", "blog", 0.01),         # exploratory
        _result("word_count", "x", 0.4),             # exploratory
    ]
    compare.apply_multiplicity(results)
    by = {(r["metric"], r["scope"]): r for r in results}
    assert by[("writing_quality", "all")]["family"] == "primary"
    assert by[("word_count", "blog")]["family"] == "exploratory"
    assert by[("writing_quality", "all")]["p_adj"] == pytest.approx(0.04)    # 0.01 x 4
    assert by[("style_distance", "blog")]["p_adj"] == pytest.approx(0.06)    # 0.02 x 3
    assert by[("word_count", "blog")]["p_adj"] == pytest.approx(0.02)        # 0.01 x 2
    assert by[("writing_quality", "all")]["verdict"] == "B better"
    assert by[("style_distance", "blog")]["verdict"] == "B better (suggestive)"


def test_many_exploratory_tests_make_a_single_p_value_suggestive():
    results = [_result("word_count", f"s{i}", 0.5) for i in range(30)]
    results.append(_result("word_count", "blog", 0.03))
    compare.apply_multiplicity(results)
    blog = results[-1]
    assert blog["p_adj"] >= 0.05
    assert blog["verdict"].endswith("(suggestive)")


def test_seed_stability_reports_sign_and_intervals():
    from eval_tools import seeds

    def runs_for(shift):
        a = rows.index_rows([make_row(f"g{i}", writing=3.0 + (i % 3) * 0.1) for i in range(12)])
        b = rows.index_rows([make_row(f"g{i}", writing=3.0 + (i % 3) * 0.1 + shift) for i in range(12)])
        return a, b

    runs = [("42", *runs_for(0.5)), ("43", *runs_for(0.4)), ("44", *runs_for(0.6))]
    out = seeds.seed_stability(runs, ["writing_quality"], n_boot=300)
    item = out[0]
    assert item["same_sign"] is True
    assert item["seeds_with_interval_excluding_zero"] == 3
    assert [round(e["mean_diff"], 1) for e in item["per_seed"]] == [0.5, 0.4, 0.6]
    flipped = seeds.seed_stability([runs[0], ("43", *runs_for(-0.4))], ["writing_quality"], n_boot=300)
    assert flipped[0]["same_sign"] is False
    text = seeds.to_markdown(out, ["42", "43", "44"])
    assert "seed 42" in text and "Same sign" in text


# --------------------------------------------- false alarms found by labeling real posts
# Each text below was flagged by an earlier version of the grammar rules and is correct English
# or a scraping artifact. They are kept so the false alarms do not come back
# (see docs/evaluation/examples/grammar_precision_audit.csv).

@pytest.mark.parametrize(
    "text",
    [
        "WalkMe decided not to flip and become a U.S. company.",
        "Pricing starts at $900 a month on a usage-based model.",
        "All I can tell you is this is really hard, and rarely done.",
        "Jason Lemkin and I are breaking down how our team works.",
        "That Box would of course get to $1B in ARR given the momentum.",
        "You can upsell in the first 6-18 mos. without it being hard.",
        "You just lost a big deal.  and/or That great new initiative is not working.",
        "Customers prefer honest assessment over over-promising and under-delivering.",
        "Read http:// site.com/signs-your-vp-just-isnt-going-to-work-out/ for more.",
        "Streaming live today! saastraiday.com has the details.",
        "Is it worth it to respond? cc @someone on the thread.",
        "How can a founder raise prices? by Jason M. Lemkin",
        "We are far from the Go Go Days of 2021.",
        "Gross margins were up (33% cons. currency) for the quarter.",
        "You have to keep refreshing i\nt to make sure it works well.",
    ],
)
def test_known_false_alarms_stay_fixed(text):
    assert grammar.check_text(text) == []


@pytest.mark.parametrize(
    "text, rule",
    [
        ("It was an unproven space for a unproven team.", "a_an"),
        ("They was always late to the board meeting.", "agreement"),
        ("The deal closed.So we moved on.", "missing_space_after_punct"),
        ("He said it was fine, but i think it is not.", "lowercase_i"),
    ],
)
def test_real_errors_near_the_fixed_rules_are_still_caught(text, rule):
    assert rule in {item["rule"] for item in grammar.check_text(text)}


# --------------------------------------------------------------------- audit tool

def test_audit_coverage_set_has_forty_broad_mistakes_in_ten_categories():
    from eval_tools import grammar_audit

    cov = grammar_audit.coverage()
    assert cov["broad_total"] == 40
    assert len(cov["categories"]) == 10
    assert cov["rule_targeted_caught"] == cov["rule_targeted_total"] == 4   # the rules do fire on their own targets


def test_audit_coverage_accepts_a_stronger_checker():
    from eval_tools import grammar_audit

    cov = grammar_audit.coverage(check=lambda text: [{"rule": "x"}])   # a checker that flags everything
    assert cov["broad_share"] == 1.0


def test_audit_windows_are_clean_and_recall_is_structured():
    from eval_tools import grammar_audit

    texts = [
        "We definitely should hire earlier next year. It is because the market is hard to read, and it is slow. "
        "They don't know why the deal closed. I think the team did well, but we could have moved faster."
    ] * 3
    windows = grammar_audit.clean_windows(texts, limit=10)
    assert windows and all(grammar.check_text(w) == [] for w in windows)
    recall = grammar_audit.rule_recall(windows, per_rule=3)
    assert set(recall) == set(grammar_audit.INJECTORS)
    assert all(item["tried"] <= 3 for item in recall.values())
    assert recall["misspelling"]["recall"] == 1.0     # "definitely" -> "definately" is always caught
    assert recall["modal_of"]["recall"] == 1.0        # "could have" -> "could of" is always caught
