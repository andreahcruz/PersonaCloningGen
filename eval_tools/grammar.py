"""A small rule based grammar and mechanics check, scored against real Lemkin posts.

Why this exists: the evaluation harness has no grammar metric. The only signal is the
judge's "fluency" item. See docs/evaluation/METRIC_MAP.md, gap G1.

What it does and does not do
* It counts mechanical errors only: doubled words, a missing space after punctuation,
  a space before punctuation, a lowercase sentence start, a lowercase "i", wrong a/an,
  missing apostrophes (dont, thats), "should of", a short list of common misspellings,
  some verb agreement slips, and broken characters.
* It does not understand meaning, tense or style. It is no match for LanguageTool or a
  human editor, and it needs no install. It can be replaced by a stronger checker later
  without changing anything else, because the output is just numbers per draft.
* An absolute error rate is misleading, because Lemkin writes informally on purpose.
  So the useful number is the deviation: the draft's errors per 100 words minus the
  rate for real Lemkin posts of the same medium. Near zero means "as clean or as messy
  as the real thing". Positive means more errors than the real posts.
* Talk is scored in absolute terms only. The real talk text is auto captions with no
  punctuation or capitals, so it cannot be a reference.

    python -m eval_tools.grammar --traces experiments/EXP-.../raw/relabel.jsonl \
        --data-dir data --out-jsonl grammar_rows.jsonl --out-json grammar_summary.json
"""
from __future__ import annotations

import argparse
import html
import json
import random
import re
from collections import Counter
from pathlib import Path
from typing import Iterable

WORD_RE = re.compile(r"[A-Za-z0-9À-ɏ]+(?:['’][A-Za-z]+)?")
URL_RE = re.compile(r"https?://\S+|www\.\S+")
URL_MARKERS = ("http", "www.", ".com", ".org", ".net", ".io", ".ai", ".co/", "@", "/", "#")

# Repeats that are usually deliberate or grammatical ("net net", "very very", "what it is is").
REPEAT_OK = {"had", "that", "very", "net", "no", "so", "ha", "bye", "yes", "really", "is", "more"}
ABBREVIATIONS = {
    "vs", "etc", "eg", "ie", "approx", "inc", "ltd", "co", "mr", "mrs", "ms", "dr", "st", "no", "fig", "est",
}
A_BEFORE_VOWEL_OK = ("uni", "use", "usu", "uti", "eu", "one", "once", "ubiq", "uber", "ufo", "ura", "uri")
AN_BEFORE_CONSONANT_OK = ("hour", "honest", "honor", "honour", "heir")
MISSING_APOSTROPHE = (
    "dont|doesnt|didnt|isnt|arent|wasnt|werent|couldnt|wouldnt|shouldnt|havent|hasnt|hadnt|"
    "youre|theyre|thats|whats|youve|weve|theyve"
)
MISSPELLINGS = {
    "definately", "seperate", "recieve", "occured", "untill", "wich", "becuase", "teh", "adress", "accomodate",
    "embarass", "goverment", "enviroment", "succesful", "succesfully", "occurence", "refered", "tommorow",
    "begining", "beleive", "calender", "collegue", "comming", "existance", "foriegn", "freind", "happend",
    "immediatly", "independant", "knowlege", "neccessary", "noticable", "persistant", "posession", "prefered",
    "publically", "reccomend", "relevent", "responsability", "sucess", "thier", "truely", "wierd", "writting",
    "alot", "acheive", "agressive", "apparant", "arguement", "basicly", "buisness", "catagory", "commited",
}

RULE_NAMES = (
    "repeated_word", "missing_space_after_punct", "space_before_punct", "lowercase_sentence_start",
    "lowercase_i", "double_punct", "a_an", "missing_apostrophe", "modal_of", "agreement", "misspelling",
    "broken_character",
)

# Rules checked on real Lemkin posts, with false alarms removed (see tests/test_eval_tools.py):
# * a/an looks at lowercase "a" and "an" only, so "Series A in" is not an error.
# * A double dot ".." is a habit of his, not a mistake, so it is not counted.
# * "hashtag" after a full stop is an export artifact of LinkedIn text.
# * Bracket and quote balance is not counted: lists like "1)" and truncated drafts make it noisy,
#   and truncation is already a completion gate.
_REPEAT = re.compile(r"\b([A-Za-z]{2,})[ \t]+\1\b", re.IGNORECASE)
_MISSING_SPACE = re.compile(r"[A-Za-z]{2}[.!?,;:][A-Z][a-z]")
_SPACE_BEFORE = re.compile(r"(?<=[A-Za-z0-9)])[ ]+[,;:!?](?=\s|$)")
_SENTENCE_LOWER = re.compile(r"\b([a-z]{2,})[.!?][ ]+([a-z][a-z']+)")
_LOWER_I = re.compile(r"(?<![A-Za-z0-9'’(\-/])i(?![A-Za-z0-9.)\-/])")
_DOUBLE_PUNCT = re.compile(r",{2,}|;{2,}|:{2,}|,\.(?!\.)")
_A_ARTICLE = re.compile(r"\b(a)[ ]+([A-Za-z]+)")
_AN_ARTICLE = re.compile(r"\b(an)[ ]+([A-Za-z]+)")
_APOSTROPHE = re.compile(rf"\b({MISSING_APOSTROPHE})\b", re.IGNORECASE)
_MODAL_OF = re.compile(r"\b(?:should|could|would|must|might)[ ]+of\b", re.IGNORECASE)
_AGREEMENT = re.compile(
    r"\b(?:he|she|it)[ ]+(?:don't|don’t|dont)\b|\b(?:they|we|you)[ ]+(?:is|was)\b|\bI[ ]+(?:is|are)\b",
    re.IGNORECASE,
)
_WORDISH = re.compile(r"[A-Za-z]+")


def clean_for_checking(text: str) -> str:
    """Strip markdown, links and handles so they are not counted as errors."""
    text = (text or "").replace("\xa0", " ").replace("​", "")
    text = URL_RE.sub(" URL ", text)
    text = re.sub(r"@\w+", " HANDLE ", text)
    text = re.sub(r"[*_`]+", "", text)
    text = re.sub(r"^[ \t]*#{1,6}[ \t]+", "", text, flags=re.MULTILINE)
    text = re.sub(r"^[ \t]*(?:[-*•]|\d+[.)])[ \t]+", "", text, flags=re.MULTILINE)
    return text


def count_words(text: str) -> int:
    return len(WORD_RE.findall(clean_for_checking(text)))


def _token_at(text: str, pos: int) -> str:
    start = pos
    while start > 0 and not text[start - 1].isspace():
        start -= 1
    end = pos
    while end < len(text) and not text[end].isspace():
        end += 1
    return text[start:end]


def check_text(text: str) -> list[dict]:
    """Return one dict per error found: ``{"rule": ..., "text": snippet}``."""
    text = clean_for_checking(text)
    found: list[dict] = []

    def add(rule: str, match_text: str) -> None:
        found.append({"rule": rule, "text": match_text.strip()[:60]})

    for m in _REPEAT.finditer(text):
        if m.group(1).lower() not in REPEAT_OK:
            add("repeated_word", m.group(0))
    for m in _MISSING_SPACE.finditer(text):
        token = _token_at(text, m.start()).lower()
        if not any(marker in token for marker in URL_MARKERS):
            add("missing_space_after_punct", m.group(0))
    for m in _SPACE_BEFORE.finditer(text):
        add("space_before_punct", text[max(0, m.start() - 8): m.end()])
    for m in _SENTENCE_LOWER.finditer(text):
        if m.group(1) not in ABBREVIATIONS and m.group(2) != "hashtag":
            add("lowercase_sentence_start", m.group(0))
    for m in _LOWER_I.finditer(text):
        add("lowercase_i", text[max(0, m.start() - 8): m.end() + 6])
    for m in _DOUBLE_PUNCT.finditer(text):
        add("double_punct", text[max(0, m.start() - 8): m.end() + 4])
    for m in _A_ARTICLE.finditer(text):
        word = m.group(2)
        low = word.lower()
        if low[0] in "aeiou" and not low.startswith(A_BEFORE_VOWEL_OK) and not (word.isupper() and len(word) > 1):
            add("a_an", m.group(0))
    for m in _AN_ARTICLE.finditer(text):
        word = m.group(2)
        low = word.lower()
        if len(word) > 1 and low[0] not in "aeiou" and not low.startswith(AN_BEFORE_CONSONANT_OK) and not word.isupper():
            add("a_an", m.group(0))
    for m in _APOSTROPHE.finditer(text):
        add("missing_apostrophe", m.group(0))
    for m in _MODAL_OF.finditer(text):
        add("modal_of", m.group(0))
    for m in _AGREEMENT.finditer(text):
        add("agreement", m.group(0))
    for m in _WORDISH.finditer(text):
        if m.group(0).lower() in MISSPELLINGS:
            add("misspelling", m.group(0))
    for _ in range(text.count("�")):
        add("broken_character", "�")
    return found


def error_stats(text: str) -> dict:
    errors = check_text(text)
    words = count_words(text)
    rules = Counter(item["rule"] for item in errors)
    return {
        "errors": len(errors),
        "words": words,
        "rate": round(100.0 * len(errors) / words, 4) if words else None,
        "rules": dict(rules),
    }


def pooled_rate(texts: Iterable[str]) -> dict:
    """Errors per 100 words over many texts (total errors divided by total words)."""
    errors = words = n = 0
    rules: Counter = Counter()
    for text in texts:
        stats = error_stats(text)
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


# ---- real Lemkin posts as the reference --------------------------------------------

def _read_jsonl(path: Path) -> Iterable[dict]:
    with open(path, encoding="utf-8-sig") as handle:
        for line in handle:
            line = line.strip()
            if line:
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(row, dict):
                    yield row


def _strip_html(text: str) -> str:
    text = re.sub(r"<[^>]+>", " ", text or "")
    text = html.unescape(text)
    cut = re.search(r"\n\s*Related Posts\s*\n", text, re.IGNORECASE)
    return text[: cut.start()] if cut else text


def reference_texts(data_dir: str | Path, limit: int = 400, seed: int = 0) -> dict[str, list[str]]:
    """Real Lemkin texts per medium, from the raw ``data/*.jsonl`` files. Talk has no reference."""
    data_dir = Path(data_dir)
    sources = {
        "blog": ("jasonlemkin_blog.jsonl", lambda r: _strip_html(r.get("content") or r.get("text") or "")),
        "linkedin": ("jasonlemkinlinkedin.jsonl", lambda r: (r.get("content") or "").strip()),
        "x": (
            "jasonlk_originals.jsonl",
            lambda r: ""
            if (r.get("is_reply") or r.get("is_repost_or_quote"))
            else (r.get("text") or "").strip(),
        ),
    }
    out: dict[str, list[str]] = {}
    for medium, (filename, getter) in sources.items():
        path = data_dir / filename
        if not path.is_file():
            continue
        texts = [t for t in (getter(r) for r in _read_jsonl(path)) if len(t) >= 25]
        rng = random.Random(seed)
        if len(texts) > limit:
            texts = rng.sample(texts, limit)
        out[medium] = texts
    return out


def reference_rates(data_dir: str | Path, limit: int = 400, seed: int = 0) -> dict[str, dict]:
    return {medium: pooled_rate(texts) for medium, texts in reference_texts(data_dir, limit, seed).items()}


# ---- scoring saved drafts -----------------------------------------------------------

def score_drafts(rows: list[dict], reference: dict[str, dict], text_field: str = "answer") -> list[dict]:
    """One result per draft. ``grammar_deviation`` is None where there is no reference."""
    results = []
    for row in rows:
        stats = error_stats(row.get(text_field) or "")
        medium = str(row.get("medium"))
        ref_rate = (reference.get(medium) or {}).get("rate")
        deviation = None
        if stats["rate"] is not None and ref_rate is not None:
            deviation = round(stats["rate"] - ref_rate, 4)
        results.append(
            {
                "id": str(row.get("id")),
                "medium": medium,
                "grammar_errors": stats["errors"],
                "grammar_words": stats["words"],
                "grammar_rate": stats["rate"],
                "grammar_deviation": deviation,
                "grammar_rules": stats["rules"],
            }
        )
    return results


def summarize(results: list[dict], reference: dict[str, dict]) -> dict:
    """Pooled error rate per medium for the drafts, next to the real post rate."""
    summary: dict = {"reference": {m: {"rate": r["rate"], "n_texts": r["n_texts"], "words": r["words"]} for m, r in reference.items()}, "drafts": {}}
    for medium in sorted({r["medium"] for r in results}):
        items = [r for r in results if r["medium"] == medium]
        errors = sum(r["grammar_errors"] for r in items)
        words = sum(r["grammar_words"] for r in items)
        rate = round(100.0 * errors / words, 4) if words else None
        ref_rate = (reference.get(medium) or {}).get("rate")
        rules: Counter = Counter()
        for r in items:
            rules.update(r["grammar_rules"])
        summary["drafts"][medium] = {
            "n_drafts": len(items),
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
    reference = reference_rates(args.data_dir, args.ref_limit, args.seed)
    if not reference:
        print(f"Warning: no reference posts found in {args.data_dir}; deviation will be empty.")
    results = score_drafts(rows, reference, args.text_field)
    summary = summarize(results, reference)
    if args.out_jsonl:
        with open(args.out_jsonl, "w", encoding="utf-8") as handle:
            for item in results:
                handle.write(json.dumps(item, ensure_ascii=False) + "\n")
    if args.out_json:
        Path(args.out_json).write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
