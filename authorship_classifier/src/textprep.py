"""Text cleaning, fixed-length windows, and topic-neutral masking.

Why this file matters: a classifier will happily learn formatting leftovers
(footnote sections, curly quotes, markdown) instead of writing style. Every text,
Lemkin, other writer or model draft, goes through the same normalize() first.
"""
from __future__ import annotations

import re

# Closed-class English words: they carry style (how a person builds sentences)
# but almost no topic. Used by the "style" view; everything else is masked.
FUNCTION_WORDS = frozenset("""
a about above after again against all also am an and any are aren't as at be because been before being below
between both but by can can't cannot could couldn't did didn't do does doesn't doing don't down during each few
for from further had hadn't has hasn't have haven't having he he'd he'll he's her here here's hers herself him
himself his how how's i i'd i'll i'm i've if in into is isn't it it's its itself just let's like me more most
mustn't my myself no nor not now of off on once only or other ought our ours ourselves out over own really same
shan't she she'd she'll she's should shouldn't so some such than that that's the their theirs them themselves
then there there's these they they'd they'll they're they've this those through to too under until up very was
wasn't we we'd we'll we're we've were weren't what what's when when's where where's which while who who's whom
why why's will with won't would wouldn't you you'd you'll you're you've your yours yourself yourselves
also always never often still even ever yet already maybe perhaps rather quite almost actually however
though although whether either neither every many much few little less least lot lots one two three
get got gets make makes made going go goes went say says said know think thing things way want need
""".split())

_URL = re.compile(r"https?://\S+|www\.\S+")
_MD = re.compile(r"[*#_`~>|]+")
_SPACE = re.compile(r"\s+")
_TOKEN = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?|\d+(?:[.,]\d+)*|[^\w\s]")


def normalize(text: str) -> str:
    """One shared cleaning pass for every source."""
    t = text or ""
    t = (t.replace(" ", " ").replace("’", "'").replace("‘", "'")
          .replace("“", '"').replace("”", '"').replace("…", "..."))
    t = _URL.sub(" ", t)
    t = _MD.sub(" ", t)
    return _SPACE.sub(" ", t).strip()


def clean_pg_essay(text: str) -> str:
    """Paul Graham essays end with a Notes section and a thanks line. Cut both."""
    t = text or ""
    i = t.rfind("\n\nNotes\n")
    if i > len(t) * 0.5:
        t = t[:i]
    t = re.sub(r"\n\s*Thanks to [^\n]*", "", t)
    t = re.sub(r"\n\s*\(Thanks to [^\n]*", "", t)
    return t


def windows(text: str, size: int, max_windows: int | None = None) -> list[str]:
    """Non-overlapping windows of exactly ``size`` words. A short tail is dropped."""
    words = normalize(text).split()
    out = [" ".join(words[i:i + size]) for i in range(0, len(words) - size + 1, size)]
    return out[:max_windows] if max_windows else out


def first_window(text: str, size: int) -> str | None:
    """First ``size`` words, or None if the text is shorter (drafts and real posts alike)."""
    words = normalize(text).split()
    return " ".join(words[:size]) if len(words) >= size else None


NUMERIC_SYMBOLS = frozenset("$%+#")


def mask_to_style(text: str, drop_numbers: bool = False) -> str:
    """Topic-neutral view: keep function words and punctuation, hide everything else.

    Content words become '*', numbers become '#'. What is left is sentence
    skeleton: how the person strings words together, not what they talk about.
    """
    out = []
    for tok in _TOKEN.findall(text):
        low = tok.lower()
        if drop_numbers and (tok[0].isdigit() or tok in NUMERIC_SYMBOLS):
            continue                      # numbers and money symbols are invisible in this view
        if tok[0].isdigit():
            out.append("#")
        elif tok[0].isalpha():
            out.append(low if low in FUNCTION_WORDS else "*")
        else:
            out.append(tok)
    return " ".join(out)


def ngram_set(text: str, n: int = 8) -> set[tuple[str, ...]]:
    w = normalize(text).lower().split()
    return {tuple(w[i:i + n]) for i in range(len(w) - n + 1)}
