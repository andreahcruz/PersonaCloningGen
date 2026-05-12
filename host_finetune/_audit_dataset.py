"""One-shot dataset audit: classify rows + count quality issues.

Read-only. Prints a table; writes no files. Run:
    python -m host_finetune._audit_dataset
"""
from __future__ import annotations

import json
import re
import statistics
from collections import Counter
from pathlib import Path

import sys

PATH = Path(
    sys.argv[1] if len(sys.argv) > 1 else
    str(Path(__file__).resolve().parent / "data" / "dataset.jsonl")
)

# ── Regexes / helpers ───────────────────────────────────────────────
PART_RE = re.compile(r"\(part\s+(\d+)\s+of\s+(\d+)\)\s*$", re.IGNORECASE)
NON_ASCII_RE = re.compile(r"[^\x00-\x7f]")
CYRILLIC_OR_CJK_RE = re.compile(r"[\u0400-\u04FF\u3000-\u9FFF\uAC00-\uD7AF]")
EMOJI_RE = re.compile(
    r"[\U0001F300-\U0001FAFF\U00002700-\U000027BF\U0001F600-\U0001F64F]"
)
TAG_RE = re.compile(r"\[(?:music|applause|laughter|crosstalk|inaudible|.{0,15}?)\]", re.IGNORECASE)
URL_RE = re.compile(r"https?://\S+|pic\.twitter\.com/\S+", re.IGNORECASE)
HASHTAG_RE = re.compile(r"#\w+")
MENTION_RE = re.compile(r"@\w+")

INTRO_PHRASES = [
    "today's episode features",
    "this episode is brought to you by",
    "subscribe to",
    "thanks for listening",
    "before we dive in",
    "welcome back to",
]


def starts_with_lower_alpha(s: str) -> bool:
    s = s.lstrip()
    return bool(s) and s[0].isalpha() and s[0].islower()


def punctuation_rate(s: str) -> float:
    if not s:
        return 0.0
    return sum(s.count(p) for p in ".!?") / max(1, len(s.split()))


def cap_rate(s: str) -> float:
    if not s:
        return 0.0
    words = s.split()
    if not words:
        return 0.0
    return sum(1 for w in words if w[:1].isupper()) / len(words)


def classify(out: str) -> str:
    if not out:
        return "empty"
    n = len(out)
    has_caps = cap_rate(out) > 0.05
    has_punct = punctuation_rate(out) > 0.02
    has_emoji = bool(EMOJI_RE.search(out))
    has_hashtag = bool(HASHTAG_RE.search(out))
    if n < 280 and (has_emoji or has_hashtag or "\n" not in out and has_caps and has_punct):
        return "tweet_or_short_post"
    if has_caps and has_punct and "\n" in out:
        return "blog_post_chunk"
    if not has_caps and not has_punct:
        return "transcript_chunk"
    return "other"


def main() -> None:
    rows: list[dict] = []
    with open(PATH, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    n_total = len(rows)
    print(f"Total rows: {n_total}\n")

    # Class buckets
    by_class: Counter[str] = Counter()
    lengths: dict[str, list[int]] = {}
    multi_part = 0
    instr_prefix_eq_out = 0
    out_starts_mid_word = 0
    has_intro = 0
    has_tag = 0
    has_url = 0
    has_emoji = 0
    high_nonascii = 0
    has_cyrillic_cjk = 0
    speaker_introducer = 0
    instructions_truncated = 0
    duplicate_outputs = 0

    seen_outputs: set[str] = set()

    # Part-counts for transcript-style series
    part_titles: Counter[str] = Counter()

    instr_first_words: Counter[str] = Counter()

    for r in rows:
        instr = (r.get("instruction") or "").strip()
        out = (r.get("output") or "").strip()
        cls = classify(out)
        by_class[cls] += 1
        lengths.setdefault(cls, []).append(len(out))

        m = PART_RE.search(instr)
        if m:
            multi_part += 1
            base = PART_RE.sub("", instr).strip()
            part_titles[base] += 1

        if instr.endswith("..."):
            instructions_truncated += 1

        # instruction = output_prefix + "..."?
        if instr.endswith("..."):
            title_part = re.sub(
                r"^write in the style of jason lemkin about:\s*",
                "",
                instr,
                flags=re.IGNORECASE,
            ).rstrip(". ").strip()
            if title_part and len(title_part) > 40 and out.lower().startswith(title_part.lower()[:60]):
                instr_prefix_eq_out += 1

        if starts_with_lower_alpha(out):
            out_starts_mid_word += 1

        low = out.lower()
        if any(p in low for p in INTRO_PHRASES):
            has_intro += 1
        if TAG_RE.search(out):
            has_tag += 1
        if URL_RE.search(out):
            has_url += 1
        if EMOJI_RE.search(out):
            has_emoji += 1
        if NON_ASCII_RE.search(out):
            n_nonascii = sum(1 for c in out if ord(c) > 127)
            if n_nonascii / max(1, len(out)) > 0.02:
                high_nonascii += 1
        if CYRILLIC_OR_CJK_RE.search(out):
            has_cyrillic_cjk += 1

        # Speaker-introduction patterns (interview where a guest speaks)
        for needle in ("my name is", "i'm the ceo of", "i'm the founder of", "ceo of"):
            if needle in low[:200] and "jason" not in low[:200]:
                speaker_introducer += 1
                break

        if out in seen_outputs:
            duplicate_outputs += 1
        else:
            seen_outputs.add(out)

        if instr:
            first_words = " ".join(instr.split()[:3]).lower()
            instr_first_words[first_words] += 1

    print("Row classification:")
    for k in ["transcript_chunk", "blog_post_chunk", "tweet_or_short_post", "other", "empty"]:
        n = by_class.get(k, 0)
        if not n:
            continue
        lens = lengths.get(k, [])
        med = int(statistics.median(lens)) if lens else 0
        mx = max(lens) if lens else 0
        print(f"  {k:25s}  n={n:6d} ({100*n/n_total:5.1f}%)  median_chars={med:5d}  max={mx}")

    print()
    print("Issues across all rows:")
    print(f"  multi-part (part X of N) rows           {multi_part:6d} ({100*multi_part/n_total:5.1f}%)")
    print(f"  unique multi-part titles                {len(part_titles):6d}")
    print(f"  rows starting mid-word (lowercase start) {out_starts_mid_word:6d} ({100*out_starts_mid_word/n_total:5.1f}%)")
    print(f"  instructions ending with '...'          {instructions_truncated:6d} ({100*instructions_truncated/n_total:5.1f}%)")
    print(f"  instruction ~= output prefix (echo bug) {instr_prefix_eq_out:6d} ({100*instr_prefix_eq_out/n_total:5.1f}%)")
    print(f"  podcast-intro phrases in output         {has_intro:6d}")
    print(f"  caption-tags [music] etc.               {has_tag:6d}")
    print(f"  URLs / pic.twitter.com                  {has_url:6d}")
    print(f"  emoji                                   {has_emoji:6d}")
    print(f"  non-ASCII heavy (>2% non-ASCII chars)   {high_nonascii:6d}")
    print(f"  Cyrillic / CJK chars                    {has_cyrillic_cjk:6d}")
    print(f"  output starts with non-Lemkin speaker   {speaker_introducer:6d}")
    print(f"  duplicate-output rows                   {duplicate_outputs:6d}")

    print()
    print("Top 10 multi-part series (likely interview transcripts):")
    for title, count in part_titles.most_common(10):
        print(f"  {count:3d}x  {title[:90]}")


if __name__ == "__main__":
    main()
