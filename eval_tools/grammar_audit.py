"""How good is the grammar tool? Two measurements with known answers.

1. Rule recall. Take real sentences the tool finds clean, inject one known error of each
   type the tool has a rule for, and count how often the tool catches it. This shows
   whether each rule works, not whether the tool covers grammar in general.
2. Coverage. A hand written set of 40 ordinary grammar mistakes in 10 categories that
   people and language models really make (tense, wrong word, missing article, ...), most
   of which the rules do not target on purpose. This shows how much of grammar the tool
   cannot see. It is the number to quote when someone asks "does it check grammar?".

    python -m eval_tools.grammar_audit --data-dir data --out-json audit.json --out-md audit.md
"""
from __future__ import annotations

import argparse
import json
import random
import re
from collections import defaultdict
from pathlib import Path

from eval_tools import grammar

# ---------------------------------------------------------------- 2. coverage set

# category -> sentences, each containing exactly the kind of error the category names.
BROAD_ERRORS: dict[str, list[str]] = {
    "subject_verb_agreement": [
        "The team of engineers are building the new pricing page.",
        "Each of the founders have a different view on equity.",
        "The number of customers who churn are rising every quarter.",
        "Neither the CEO nor the investors was happy with the board deck.",
    ],
    "verb_tense": [
        "Yesterday we close three enterprise deals and hire two reps.",
        "Last year our revenue grow fast and then it stop.",
        "When I was a founder I will never accept that offer.",
        "She has went to every board meeting this year.",
    ],
    "confused_words": [
        "Their going to raise a Series B next month.",
        "Your right that the churn number is too high.",
        "Its important to know when the company lost it's best rep.",
        "Hiring a VP early costs more then waiting.",
    ],
    "missing_words": [
        "We need to hire engineer before we launch product.",
        "She wrote report for board last week.",
        "It is important have clear quota for every rep.",
        "He asked me to join call tomorrow morning.",
    ],
    "preposition": [
        "This depends of how fast the team can ship.",
        "We are focused in growing the enterprise segment.",
        "She is good in sales but weak on hiring.",
        "He agreed to the plan despite of the risks.",
    ],
    "comma_splice_run_on": [
        "We shipped the feature on Monday, it broke on Tuesday.",
        "Revenue is flat the team is tired and the board wants answers.",
        "The deal closed we celebrated and then the customer left.",
        "I love this product it solves a real problem for founders.",
    ],
    "pronoun_case": [
        "Me and him closed the deal last night.",
        "The investors gave the founders and I a hard time.",
        "Between you and I, the plan will not work.",
        "Her and the CEO disagreed about the budget.",
    ],
    "plural_countability": [
        "We hired three engineer and two designer this quarter.",
        "The company has many customer but few advocate.",
        "He gave me a lot of advices about pricing.",
        "Several news about the funding round surprised us.",
    ],
    "double_negative": [
        "We don't have no budget for a new sales hire.",
        "Nobody never told me the quota was changing.",
        "She can't hardly wait for the launch.",
        "I didn't see nothing wrong with the contract.",
    ],
    "awkward_word_choice": [
        "The meeting was very much boring and long exhausting.",
        "He made a big loss on that investment since the market going down.",
        "We must to decide until Friday which vendor to choose.",
        "The software is enough easy for new users to learn.",
    ],
}

# Mistakes the rules are written for. A sanity check that the rules fire on plain examples.
RULE_TARGETED = [
    "We we need to move faster on hiring.",
    "They dont have enough pipeline this quarter.",
    "You should of hired earlier.",
    "She bought a apple laptop for the new rep.",
]


def coverage(check=grammar.check_text) -> dict:
    """Share of the broad mistakes the tool flags, per category, plus the sanity set."""
    categories = {}
    caught_total = 0
    total = 0
    for category, sentences in BROAD_ERRORS.items():
        caught = [s for s in sentences if check(s)]
        categories[category] = {"n": len(sentences), "caught": len(caught), "missed": [s for s in sentences if s not in caught]}
        caught_total += len(caught)
        total += len(sentences)
    sanity_caught = sum(1 for s in RULE_TARGETED if check(s))
    return {
        "broad_total": total,
        "broad_caught": caught_total,
        "broad_share": round(caught_total / total, 4) if total else None,
        "categories": categories,
        "rule_targeted_total": len(RULE_TARGETED),
        "rule_targeted_caught": sanity_caught,
    }


# ------------------------------------------------------------ 1. injection recall

MISSPELL_MAP = {
    "definitely": "definately", "separate": "seperate", "receive": "recieve", "occurred": "occured",
    "until": "untill", "which": "wich", "because": "becuase", "address": "adress", "government": "goverment",
    "environment": "enviroment", "successful": "succesful", "tomorrow": "tommorow", "beginning": "begining",
    "believe": "beleive", "friend": "freind", "their": "thier", "truly": "truely", "weird": "wierd",
    "necessary": "neccessary", "independent": "independant", "knowledge": "knowlege", "relevant": "relevent",
    "success": "sucess", "writing": "writting",
}
APOSTROPHE_MAP = {
    "don't": "dont", "doesn't": "doesnt", "didn't": "didnt", "isn't": "isnt", "aren't": "arent",
    "wasn't": "wasnt", "couldn't": "couldnt", "wouldn't": "wouldnt", "shouldn't": "shouldnt",
    "haven't": "havent", "hasn't": "hasnt", "hadn't": "hadnt", "you're": "youre", "they're": "theyre",
    "that's": "thats", "what's": "whats", "you've": "youve", "we've": "weve", "they've": "theyve",
}


def _pick(matches, rng):
    return rng.choice(matches) if matches else None


def inj_repeated_word(t, rng):
    ms = [m for m in re.finditer(r"\b[A-Za-z]{3,}\b", t) if m.group(0).lower() not in grammar.REPEAT_OK]
    m = _pick(ms, rng)
    return None if m is None else t[: m.end()] + " " + m.group(0) + t[m.end():]


def inj_missing_space(t, rng):
    m = _pick(list(re.finditer(r"([a-z]{2})([.!?]) ([A-Z][a-z])", t)), rng)
    return None if m is None else t[: m.start()] + m.group(1) + m.group(2) + m.group(3) + t[m.end():]


def inj_space_before(t, rng):
    m = _pick(list(re.finditer(r"(?<=[A-Za-z]), ", t)), rng)
    return None if m is None else t[: m.start()] + " , " + t[m.end():]


def inj_lowercase_start(t, rng):
    m = _pick(list(re.finditer(r"([a-z]{2,}[.!?]) ([A-Z])([a-z]+)", t)), rng)
    if m is None or m.group(1)[:-1] in grammar.ABBREVIATIONS:
        return None
    return t[: m.start()] + m.group(1) + " " + m.group(2).lower() + m.group(3) + t[m.end():]


def inj_lowercase_i(t, rng):
    m = _pick(list(re.finditer(r"(?<![A-Za-z0-9'])I(?![A-Za-z0-9.])", t)), rng)
    return None if m is None else t[: m.start()] + "i" + t[m.end():]


def inj_double_punct(t, rng):
    m = _pick(list(re.finditer(r",", t)), rng)
    return None if m is None else t[: m.start()] + ",," + t[m.end():]


def inj_a_an(t, rng):
    wrong_an = list(re.finditer(r"\ba ([bcdfgjklmnpqrstvwxyz][a-z]+)", t))
    wrong_a = list(re.finditer(r"\ban ([aeiou][a-z]+)", t))
    options = [("an", m) for m in wrong_an] + [("a", m) for m in wrong_a]
    pick = _pick(options, rng)
    if pick is None:
        return None
    article, m = pick
    return t[: m.start()] + f"{article} {m.group(1)}" + t[m.end():]


def inj_missing_apostrophe(t, rng):
    found = []
    for correct, wrong in APOSTROPHE_MAP.items():
        for m in re.finditer(r"\b" + re.escape(correct) + r"\b", t, re.IGNORECASE):
            found.append((m, wrong))
    pick = _pick(found, rng)
    return None if pick is None else t[: pick[0].start()] + pick[1] + t[pick[0].end():]


def inj_modal_of(t, rng):
    m = _pick(list(re.finditer(r"\b(should|could|would|must|might) have\b", t)), rng)
    return None if m is None else t[: m.start()] + m.group(1) + " of" + t[m.end():]


def inj_agreement(t, rng):
    found = []
    for m in re.finditer(r"\b(they|we|you) (are|were)\b", t):
        found.append((m, f"{m.group(1)} {'is' if m.group(2) == 'are' else 'was'}"))
    for m in re.finditer(r"\b(he|she|it) doesn't\b", t):
        found.append((m, f"{m.group(1)} don't"))
    pick = _pick(found, rng)
    return None if pick is None else t[: pick[0].start()] + pick[1] + t[pick[0].end():]


def inj_misspelling(t, rng):
    found = []
    for correct, wrong in MISSPELL_MAP.items():
        for m in re.finditer(r"\b" + correct + r"\b", t):
            found.append((m, wrong))
    pick = _pick(found, rng)
    return None if pick is None else t[: pick[0].start()] + pick[1] + t[pick[0].end():]


def inj_broken_character(t, rng):
    m = _pick(list(re.finditer(r"'", t)), rng)
    return None if m is None else t[: m.start()] + "�" + t[m.end():]


INJECTORS = {
    "repeated_word": inj_repeated_word,
    "missing_space_after_punct": inj_missing_space,
    "space_before_punct": inj_space_before,
    "lowercase_sentence_start": inj_lowercase_start,
    "lowercase_i": inj_lowercase_i,
    "double_punct": inj_double_punct,
    "a_an": inj_a_an,
    "missing_apostrophe": inj_missing_apostrophe,
    "modal_of": inj_modal_of,
    "agreement": inj_agreement,
    "misspelling": inj_misspelling,
    "broken_character": inj_broken_character,
}


def clean_windows(texts: list[str], limit: int = 600, seed: int = 0) -> list[str]:
    """Two sentence windows of real text that the tool finds clean, so any new flag comes from the injection."""
    rng = random.Random(seed)
    windows: list[str] = []
    for text in texts:
        text = grammar.clean_for_checking(text)
        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if 6 <= len(s.split()) <= 45]
        for i in range(len(sentences) - 1):
            window = sentences[i] + " " + sentences[i + 1]
            ascii_share = sum(1 for c in window if ord(c) < 128) / max(len(window), 1)
            if len(window) <= 420 and ascii_share > 0.99 and "URL" not in window and "HANDLE" not in window:
                windows.append(window)
    rng.shuffle(windows)
    return [w for w in windows if not grammar.check_text(w)][:limit]


def rule_recall(windows: list[str], per_rule: int = 60, seed: int = 0, check=grammar.check_text) -> dict:
    """For each rule: how many injected errors the tool flags with that rule."""
    rng = random.Random(seed)
    out = {}
    for rule, injector in INJECTORS.items():
        pool = windows[:]
        rng.shuffle(pool)
        tried = caught = 0
        missed: list[tuple[str, str]] = []
        for window in pool:
            mutated = injector(window, rng)
            if mutated is None or mutated == window:
                continue
            tried += 1
            if rule in {item["rule"] for item in check(mutated)}:
                caught += 1
            elif len(missed) < 3:
                missed.append((window[:90], mutated[:90]))
            if tried >= per_rule:
                break
        out[rule] = {"tried": tried, "caught": caught, "recall": round(caught / tried, 4) if tried else None, "examples_missed": missed}
    return out


def to_markdown(recall: dict, cov: dict, n_windows: int) -> str:
    lines = [
        "# Grammar tool audit",
        "",
        f"Base text: {n_windows} two sentence windows of real Lemkin writing that the tool finds clean.",
        "",
        "## 1. Rule recall (injected errors of the types the tool has rules for)",
        "",
        "| Rule | Injected | Caught | Recall |",
        "|---|---|---|---|",
    ]
    for rule, item in recall.items():
        share = "n/a" if item["recall"] is None else f"{item['recall']:.0%}"
        lines.append(f"| {rule} | {item['tried']} | {item['caught']} | {share} |")
    lines += [
        "",
        "Why two rules are below 100 percent. These are deliberate choices to cut false alarms on real posts, "
        "not bugs. `repeated_word` ignores repeats where both words are capitalized (\"Go Go Days\"), because "
        "those are usually names or titles. `agreement` ignores \"you is\" and \"I are\", because \"what I can tell "
        "you is this\" and \"Jason and I are\" are correct English. The injected errors include those cases, so the "
        "recall shows what the tool gives up for precision. Recall on the other rules is high partly by "
        "construction, because the injected errors are exactly the patterns the rules look for.",
        "",
        "## 2. Coverage of ordinary grammar mistakes",
        "",
        f"The tool flags {cov['broad_caught']} of {cov['broad_total']} hand written mistakes ({cov['broad_share']:.0%}).",
        f"On the sanity set of {cov['rule_targeted_total']} mistakes the rules are written for, it flags {cov['rule_targeted_caught']}.",
        "",
        "| Category | Mistakes | Flagged |",
        "|---|---|---|",
    ]
    for category, item in cov["categories"].items():
        lines.append(f"| {category} | {item['n']} | {item['caught']} |")
    lines += [
        "",
        "Read this as: the rules do what they say (section 1), and they cover only a small part of grammar (section 2).",
        "So the tool is a floor check for mechanical slips, not a grammar score. A stronger checker is needed for more.",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", default="data")
    p.add_argument("--posts-per-medium", type=int, default=400)
    p.add_argument("--windows", type=int, default=600)
    p.add_argument("--per-rule", type=int, default=60)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out-json")
    p.add_argument("--out-md")
    args = p.parse_args()

    refs = grammar.reference_texts(args.data_dir, args.posts_per_medium, args.seed)
    texts = [t for medium in ("blog", "linkedin", "x") for t in refs.get(medium, [])]
    if not texts:
        raise SystemExit(f"No reference posts found in {args.data_dir}.")
    windows = clean_windows(texts, args.windows, args.seed)
    recall = rule_recall(windows, args.per_rule, args.seed)
    cov = coverage()
    markdown = to_markdown(recall, cov, len(windows))
    if args.out_md:
        Path(args.out_md).write_text(markdown, encoding="utf-8")
    if args.out_json:
        Path(args.out_json).write_text(json.dumps({"windows": len(windows), "recall": recall, "coverage": cov}, indent=2), encoding="utf-8")
    print(markdown)


if __name__ == "__main__":
    main()
