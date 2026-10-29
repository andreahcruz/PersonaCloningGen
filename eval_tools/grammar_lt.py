"""Grammar check with LanguageTool, tuned for Lemkin's informal writing (PUK-28).

``eval_tools.grammar`` is a small rule based check that only finds typos and punctuation slips
(0 of 40 ordinary grammar mistakes in the audit). This module uses LanguageTool instead, which
also finds agreement, wrong word forms and similar mistakes. Kevin chose it for the grammar axis.

It is optional. It needs Java 17 or newer and the Python package ``language_tool_python``
(``pip install language_tool_python``). The first run downloads LanguageTool (about 260 MB) and
then everything runs on this computer: no draft is sent to any web service. The other tools in
``eval_tools`` still need nothing but Python.

Why it is not used with its default rules
-----------------------------------------
Run on 400 real Lemkin posts per medium, the default rules flag 2 to 3 things per 100 words.
Nearly all of that is noise for this purpose: names and jargon the spell checker does not know
(Coatue, SaaStr, EMW), spacing and typography, comma style, and apostrophes that were broken in
the data (the replacement character). So this module keeps only:

* categories GRAMMAR, TYPOS and CONFUSED_WORDS (agreement, verb forms, its/it's, and so on), plus two
  rules from MISC that are plain typos (a doubled word, a/an);
* spelling only for a word that is all lowercase and that Lemkin's real posts use fewer than
  three times, so a name, an acronym or his own vocabulary is not called a typo;
* nothing that touches a broken character;
* none of a short list of rules (``DISABLED_RULES``) that, in the examples we read on real posts,
  fired on deliberate informal style such as "brand new customers" or "best case".

After that real posts flag about 0.1 to 0.4 things per 100 words. As with ``eval_tools.grammar``, the
useful number is the **deviation**: the draft's rate minus the rate of real posts of the same
medium. Near zero means "as clean as the real thing". Talk has no reference (auto captions), so it
is scored in absolute terms only.

    python -m eval_tools.grammar_lt --traces experiments/EXP-.../raw/relabel.jsonl --data-dir data \
        --out-jsonl grammar_lt_rows.jsonl --out-json grammar_lt_summary.json

The ``--out-jsonl`` file can be passed to ``eval_tools.compare`` with ``--extra-a`` or ``--extra-b``
and compared as ``grammar_lt_rate`` or ``grammar_lt_deviation``.

What it is not: it is not a full grammar score. LanguageTool misses some mistakes and flags a few
correct sentences, and the profile above was tuned on real posts, so it can miss a kind of error that
those posts happen to contain. See ``docs/evaluation/GRAMMAR_LANGUAGETOOL.md`` for the measured numbers.
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Iterable

from eval_tools.grammar import _read_jsonl, clean_for_checking, count_words, reference_texts

KEEP_CATEGORIES = ("GRAMMAR", "TYPOS", "CONFUSED_WORDS")
# Two rules that LanguageTool files under MISC but that are plain typos: a doubled word and a/an.
KEEP_RULES = frozenset({"ENGLISH_WORD_REPEAT_RULE", "EN_A_VS_AN"})
SPELLING_RULE = "MORFOLOGIK_RULE_EN_US"
MIN_VOCABULARY_COUNT = 3
BROKEN_CHARACTER = "�"

# Rules that fire on deliberate informal style or on line breaks in the real posts, not on mistakes.
# Each was disabled after reading its hits on real Lemkin posts (see the doc for examples).
DISABLED_RULES = frozenset(
    {
        "PHRASE_REPETITION",  # "are great are great" across a line break, repeated list headings
        "THE_SUPERLATIVE",  # "best case, it takes five years"
        "CA_BRAND_NEW",  # "brand new customers"
        "ANYMORE_ADVERB",  # "Nobody cares about down rounds anymore"
        "THESE_ONES",  # "Those ones will help you scale"
        "A_INFINITIVE",  # quoted phrases such as "more inbound revenue"
        "BEEN_PART_AGREEMENT",  # "be job #1", "isn't stage appropriate"
    }
)

_WORD = re.compile(r"[A-Za-z][A-Za-z'’]*")


def build_vocabulary(texts: Iterable[str]) -> Counter:
    """How often each lowercase word appears in real posts. A word used often enough is not a typo."""
    vocabulary: Counter = Counter()
    for text in texts:
        vocabulary.update(word.lower() for word in _WORD.findall(clean_for_checking(text)))
    return vocabulary


def keep_match(match: dict, vocabulary: Counter, min_count: int = MIN_VOCABULARY_COUNT) -> bool:
    """Decide whether one LanguageTool match counts as a mistake. ``match`` has rule, category, text, context."""
    if BROKEN_CHARACTER in (match.get("context") or "") or BROKEN_CHARACTER in (match.get("text") or ""):
        return False
    rule = match.get("rule")
    if rule in DISABLED_RULES:
        return False
    if rule == SPELLING_RULE:
        word = match.get("text") or ""
        return word.isalpha() and word.islower() and vocabulary[word.lower()] < min_count
    return rule in KEEP_RULES or match.get("category") in KEEP_CATEGORIES


class Checker:
    """LanguageTool wrapped to return plain dicts, with the filter above applied.

    ``tool`` can be passed in for tests. Otherwise a local LanguageTool server is started, so close
    the checker (or use it in a ``with`` block) when done.
    """

    def __init__(self, vocabulary: Counter, tool=None, language: str = "en-US"):
        self.vocabulary = vocabulary
        if tool is None:
            try:
                import language_tool_python  # type: ignore
            except ImportError as err:  # pragma: no cover - depends on the machine
                raise SystemExit(
                    "eval_tools.grammar_lt needs the language_tool_python package and Java 17 or newer:\n"
                    "    pip install language_tool_python\n"
                    "The rest of eval_tools does not need either."
                ) from err
            tool = language_tool_python.LanguageTool(language)
        self.tool = tool

    def __enter__(self) -> "Checker":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def close(self) -> None:
        close = getattr(self.tool, "close", None)
        if close:
            close()

    def check_text(self, text: str) -> list[dict]:
        clean = clean_for_checking(text or "")
        if not clean.strip():
            return []
        found = []
        for m in self.tool.check(clean):
            item = {
                "rule": m.rule_id,
                "category": m.category,
                "message": m.message,
                "text": m.matched_text,
                "context": m.context,
                "offset": m.offset,
            }
            if keep_match(item, self.vocabulary):
                found.append(item)
        return found


def error_stats(text: str, checker: Checker) -> dict:
    errors = checker.check_text(text)
    words = count_words(clean_for_checking(text or ""))
    rules = Counter(item["rule"] for item in errors)
    return {
        "errors": len(errors),
        "words": words,
        "rate": round(100.0 * len(errors) / words, 4) if words else None,
        "rules": dict(rules),
    }


def pooled_rate(texts: Iterable[str], checker: Checker) -> dict:
    """Errors per 100 words over many texts (total errors divided by total words)."""
    errors = words = n = 0
    rules: Counter = Counter()
    for text in texts:
        stats = error_stats(text, checker)
        errors += stats["errors"]
        words += stats["words"]
        rules.update(stats["rules"])
        n += 1
    return {
        "n_texts": n,
        "errors": errors,
        "words": words,
        "rate": round(100.0 * errors / words, 4) if words else None,
        "rules": dict(rules),
    }


def score_drafts(rows: list[dict], reference: dict[str, dict], checker: Checker, text_field: str = "answer") -> list[dict]:
    """One result per draft. ``grammar_lt_deviation`` is None where there is no reference (talk)."""
    results = []
    for row in rows:
        stats = error_stats(row.get(text_field) or "", checker)
        medium = str(row.get("medium"))
        ref_rate = (reference.get(medium) or {}).get("rate")
        deviation = None
        if stats["rate"] is not None and ref_rate is not None:
            deviation = round(stats["rate"] - ref_rate, 4)
        results.append(
            {
                "id": str(row.get("id")),
                "medium": medium,
                "grammar_lt_errors": stats["errors"],
                "grammar_lt_words": stats["words"],
                "grammar_lt_rate": stats["rate"],
                "grammar_lt_deviation": deviation,
                "grammar_lt_rules": stats["rules"],
            }
        )
    return results


def summarize(results: list[dict], reference: dict[str, dict]) -> dict:
    """Pooled error rate per medium for the drafts, next to the real post rate."""
    summary: dict = {
        "reference": {m: {"rate": r["rate"], "n_texts": r["n_texts"], "words": r["words"]} for m, r in reference.items()},
        "drafts": {},
    }
    for medium in sorted({r["medium"] for r in results}):
        items = [r for r in results if r["medium"] == medium]
        errors = sum(r["grammar_lt_errors"] for r in items)
        words = sum(r["grammar_lt_words"] for r in items)
        rate = round(100.0 * errors / words, 4) if words else None
        ref_rate = (reference.get(medium) or {}).get("rate")
        rules: Counter = Counter()
        for r in items:
            rules.update(r["grammar_lt_rules"])
        summary["drafts"][medium] = {
            "n_drafts": len(items),
            "drafts_with_an_error": sum(1 for r in items if r["grammar_lt_errors"]),
            "words": words,
            "errors": errors,
            "rate_per_100_words": rate,
            "reference_rate_per_100_words": ref_rate,
            "excess_over_reference": None if rate is None or ref_rate is None else round(rate - ref_rate, 4),
            "top_rules": dict(rules.most_common(5)),
        }
    return summary


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--traces", required=True, help="JSONL of drafts with id, medium and the text field.")
    p.add_argument("--text-field", default="answer")
    p.add_argument("--data-dir", default="data", help="Folder with the raw Lemkin JSONL files.")
    p.add_argument("--ref-limit", type=int, default=400, help="Real posts sampled per medium.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out-jsonl", help="Per draft results, readable by eval_tools.compare --extra-a/--extra-b.")
    p.add_argument("--out-json", help="Summary per medium.")
    args = p.parse_args()

    rows = list(_read_jsonl(Path(args.traces)))
    texts = reference_texts(args.data_dir, args.ref_limit, args.seed)
    if not texts:
        print(f"Warning: no reference posts found in {args.data_dir}; the vocabulary and deviation will be empty.")
    vocabulary = build_vocabulary(t for medium_texts in texts.values() for t in medium_texts)
    with Checker(vocabulary) as checker:
        reference = {medium: pooled_rate(medium_texts, checker) for medium, medium_texts in texts.items()}
        results = score_drafts(rows, reference, checker, args.text_field)
    summary = summarize(results, reference)
    if args.out_jsonl:
        with open(args.out_jsonl, "w", encoding="utf-8", newline="\n") as handle:
            for item in results:
                handle.write(json.dumps(item, ensure_ascii=False) + "\n")
    if args.out_json:
        Path(args.out_json).write_bytes(json.dumps(summary, indent=2).encode("utf-8"))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
