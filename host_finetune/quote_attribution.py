"""Mark quotations inside an otherwise Lemkin-attributed document.

The detector is conservative. It looks for a quoted span followed by a
personal name, and it refuses Jason Lemkin's own name. It does not decide
that every sentence on a SaaStr page is his.
"""

from __future__ import annotations

import re

_QUOTE = re.compile(r"[\"“]([^\"”]{20,700})[\"”]")
_NAME = re.compile(
    r"([A-Z][a-z]+(?:['’][A-Z][a-z]+)?(?:[ \t]+[A-Z][a-z]+(?:['’][A-Z][a-z]+)?){1,2}?)"
)
# Title-case headings that are not people. This list is the observed failures.
_NOT_A_NAME = {
    "actually", "advocacy", "agents", "alto", "background", "billion", "cloud",
    "concerns", "copilot", "creative", "day", "desk", "dives", "driven", "from",
    "fund", "founders", "giving", "growth", "hiring", "hub", "keep", "key",
    "led", "looks", "market", "palo", "plan", "point", "product", "proof",
    "raising", "reps", "street", "takeaways", "timing", "trade", "venture",
    "wall", "your", "zendesk",
}
_BAD_FIRST = {
    "the", "this", "that", "these", "those", "new", "going", "and", "but",
    "from", "with", "when", "what", "why", "how", "our", "their", "his", "her",
    "its", "not", "for", "you", "your", "dear", "best", "just", "only", "more",
    "most", "some", "than", "then", "into", "over", "after", "before", "about",
    "there", "here", "they", "we", "my", "if", "so", "or", "as", "at", "on",
    "in", "of", "an", "saas", "dear",
}
QUOTE_SHARE = 0.45


def _reject_speaker(speaker: str) -> bool:
    parts = speaker.split()
    if len(parts) < 2:
        return True
    if any(part.casefold() in _BAD_FIRST or part.casefold() in _NOT_A_NAME for part in parts):
        return True
    folded = speaker.casefold()
    if "lemkin" in folded:
        return True
    return False


def find_third_party_quotes(text: str) -> list[dict]:
    """Return quotations whose nearby name is not Jason Lemkin."""
    found = []
    seen = set()
    for match in _QUOTE.finditer(text or ""):
        span = re.sub(r"\s+", " ", match.group(1)).strip()
        tail = (text or "")[match.end() : match.end() + 80]
        newline = tail.find("\n")
        if newline >= 0:
            tail = tail[:newline]
        name = _NAME.search(tail)
        if not name:
            continue
        speaker = name.group(1)
        if _reject_speaker(speaker):
            continue
        role = tail[: name.start()].strip(" \t-–—,:")
        role = re.sub(r"\s+", " ", role)
        key = (speaker, span)
        if key in seen:
            continue
        seen.add(key)
        found.append(
            {
                "speaker": speaker,
                "role": role,
                "span": span,
                "word_count": len(span.split()),
            }
        )
    return found


def quote_attribution(text: str, base_status: str, base_position: str) -> dict:
    """Adjust attribution when a third-party quotation dominates the document."""
    quotes = find_third_party_quotes(text)
    words = len(re.sub(r"\s+", " ", text or "").split())
    quoted_words = sum(item["word_count"] for item in quotes)
    share = (quoted_words / words) if words else 0.0
    status = base_status
    position = base_position
    if quotes and share >= QUOTE_SHARE:
        status = "third_party_quotation"
        position = "quoted_third_party"
    return {
        "quotes": quotes,
        "quoted_word_share": round(share, 4),
        "attribution_status": status,
        "author_position": position,
        "quoted_speakers": [item["speaker"] for item in quotes],
    }


def index_text_with_quotes(text: str, quotes: list[dict]) -> str:
    """Prefix the index text. The original sentences stay verbatim below it."""
    if not quotes:
        return text
    lines = ["[Attribution note: quoted text below is not Jason Lemkin's own claim.]"]
    for quote in quotes:
        role = f", {quote['role']}" if quote.get("role") else ""
        lines.append(f"[Quoted speaker: {quote['speaker']}{role}.]")
    lines.append("[End attribution note.]")
    return "\n".join(lines) + "\n" + text
