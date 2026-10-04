"""
Strip SaaStr / blog recirculation tails and noisy social footers from corpus text.

Used by Spark (MinIO raw→processed), host_finetune cleaners, the scraped-JSONL
cleaner, and RAG context formatting so training + retrieval are not dominated
by "Related Posts" blocks, event ads, or LinkedIn scrape chrome.

Kept under ``spark_jobs/`` so ``clean_and_embed.py`` can import it when Airflow
SparkSubmit runs from ``/opt/airflow/spark_jobs`` (that directory is volume-mounted;
repo root is not).
"""
from __future__ import annotations

import html
import re

# Trailing blocks (WordPress / SaaStr style).
_FOOTER_HEADING = (
    r"Related Posts|Related articles|Related Article|"
    r"You May Also Like|READ MORE|Read more on SaaStr|Continue reading"
)
_FOOTER_START_PATTERNS = tuple(
    re.compile(p, re.IGNORECASE)
    for p in (
        rf"\n\s*(?:{_FOOTER_HEADING})\s*\n",
    )
)
# A chunk that is only the recirculation block starts with the heading on its own line.
_FOOTER_ONLY = re.compile(rf"^\s*(?:{_FOOTER_HEADING})\s*(?:\n|$)", re.IGNORECASE)

# Short posts that are event ads, not essays that mention an event.
PROMO_MAX_WORDS = 100
_DEAR_SAASTR = re.compile(r"^\s*dear saastr\b", re.IGNORECASE)
_NAMED_EVENT = re.compile(
    r"\b(?:"
    r"saastr\s+(?:ai\s+)?annual"
    r"|saastr\s+europa"
    r"|saastr\s+deploy"
    r"|saastrdeploy"
    r"|saastr\s+scale"
    r"|holiday\s+party"
    r"|saastr\s+workshop"
    r"|workshop\s+wednesday"
    r")\b",
    re.IGNORECASE,
)
_CTA = re.compile(
    r"\b(?:sponsors?|sponsorships?|sponsored|register|registration|registrations|"
    r"tickets?|join us|see you there|sign up|apply now)\b",
    re.IGNORECASE,
)
_SUPPORT_TICKET = re.compile(r"\bsupport tickets?\b", re.IGNORECASE)
# A pitch headline, not a stats or hiring title that merely contains "sponsor".
_TITLE_PITCH = re.compile(
    r"\b(?:"
    r"last chance to (?:sponsor|apply|submit)"
    r"|reasons to sponsor"
    r"|sponsor saastr"
    r"|become a saastr sponsor"
    r"|(?:why )?you should sponsor"
    r"|stuff to sponsor"
    r"|need leads"
    r"|want(?:\s+\d+)?\s+more leads"
    r"|want leads\?"
    r"|tickets will never"
    r"|no-cost vip"
    r"|vip tickets"
    r"|sponsor webinar"
    r"|sign up now"
    r"|apply now"
    r"|as (?:a |an )?(?:[\w’']+ ){0,3}sponsor"
    r")\b",
    re.IGNORECASE,
)

_FEED_POST = re.compile(r"^\s*Feed post(?:\s+number)?\b", re.IGNORECASE)
_HASHTAG_TOKEN = re.compile(r"\bhashtag\s+(?=#)", re.IGNORECASE)
# Gap immediately after the scheme. Spaces later in the path stay put.
_BROKEN_URL = re.compile(r"(https?://)[ \t\r\n]+", re.IGNORECASE)
# A newline that splits one URL path. The next line must continue that path.
_URL_PATH_BREAK = re.compile(r"(https?://[^\s<>\"']+?)\n([^\s<>\"'\n]+)")
_NEW_HOST = re.compile(r"^(?:https?://|www\.|[A-Za-z0-9-]+\.[A-Za-z]{2,})")
_URL_CONTINUATION = re.compile(r"^[a-z0-9\-._~:/?#\[\]@!$&'()*+,;=%]")

# Attribution card pasted from an embedded tweet. The essay after the card stays.
_TWITTER_CARD = re.compile(
    r"\n[ \t]*[—–-][ \t]*Jason\b[^\n]*\(@jasonlk\)[^\n]*"
    r"(?:\n[ \t]*[A-Z][a-z]+ \d{1,2}, \d{4})?",
    re.IGNORECASE,
)


def strip_footer_noise(text: str, *, footer_zone_frac: float = 0.62) -> str:
    """Remove recirculation footer and common social tails.

    ``footer_zone_frac``: only treat a match as a footer if it starts at or
    after this fraction of string length (avoids killing mid-article mentions).
    """
    if not text:
        return text
    s = text.strip()
    if _FOOTER_ONLY.match(s):
        return ""
    if len(s) < 120:
        return text

    cut = len(s)
    zone = int(len(s) * footer_zone_frac)

    for rx in _FOOTER_START_PATTERNS:
        for m in rx.finditer(s):
            if m.start() >= zone:
                cut = min(cut, m.start())

    if cut < len(s):
        s = s[:cut].rstrip()

    # Drop the attribution card only. A following paragraph is the rest of the essay.
    s = _TWITTER_CARD.sub("\n", s)
    return s.strip()


def _is_title_pitch(title: str) -> bool:
    """Sponsor or ticket-sale headline. Dear SaaStr essays and support-ticket titles stay."""
    t = (title or "").strip()
    if not t or _DEAR_SAASTR.match(t):
        return False
    if _SUPPORT_TICKET.search(t):
        return False
    return bool(_TITLE_PITCH.search(t))


def is_event_promo(text: str, title: str = "") -> bool:
    """True when a row is an event, ticket, or sponsor ad.

    A short post must name an event and include a call to action. A long essay
    that only mentions the event is kept. A sponsor or ticket-sale title is
    dropped even when the body is long. ``Dear SaaStr`` columns are kept.
    """
    if _DEAR_SAASTR.match((title or "").strip()):
        return False
    if _is_title_pitch(title):
        return True
    body = text or ""
    if len(body.split()) >= PROMO_MAX_WORDS:
        return False
    return bool(_NAMED_EVENT.search(body) and _CTA.search(body))


def is_linkedin_feed_chrome(text: str) -> bool:
    """True when a LinkedIn export captured the feed card, not one authored post."""
    return bool(_FEED_POST.match(text or ""))


def strip_linkedin_chrome(text: str) -> str:
    """Remove the literal ``hashtag`` token scraped in front of ``#`` tags."""
    if not text:
        return text
    return _HASHTAG_TOKEN.sub("", text)


def _join_url_path_break(match: re.Match[str]) -> str:
    """Rejoin a newline inside one path. A new host on the next line stays put."""
    fragment = match.group(2)
    if _NEW_HOST.match(fragment) or not _URL_CONTINUATION.match(fragment):
        return match.group(0)
    return match.group(1) + fragment


def repair_broken_urls(text: str) -> str:
    """Join a broken URL scheme, then a newline that splits the path.

    Spaces in ordinary prose, including a space inside a path, stay put.
    A following host such as ``pic.twitter.com`` stays on its own line.
    """
    if not text or "http" not in text.lower():
        return text
    text = _BROKEN_URL.sub(r"\1", text)
    previous = None
    while previous != text:
        previous = text
        text = _URL_PATH_BREAK.sub(_join_url_path_break, text)
    return text


def unescape_html(text: str) -> str:
    """Decode HTML entities such as ``&gt;`` left in auto-captions."""
    if not text or "&" not in text:
        return text
    return html.unescape(text)


def scrub_corpus_text(
    text: str,
    *,
    title: str = "",
    linkedin: bool = False,
    unescape: bool = False,
) -> tuple[str | None, str | None, dict]:
    """Apply the shared scrap rules to one document body.

    Returns ``(cleaned_text, None, flags)`` or ``(None, drop_reason, flags)``.
    Drop reasons are ``empty``, ``linkedin_feed_chrome``, ``event_promo``, and
    ``empty_after_scrub``. Callers still enforce source-specific length rules
    such as the X minimum of 25 characters.
    """
    flags = {"footer": False, "url": False, "hashtag": False, "html": False}
    current = (text or "").strip()
    if not current:
        return None, "empty", flags
    if linkedin and is_linkedin_feed_chrome(current):
        return None, "linkedin_feed_chrome", flags
    if linkedin:
        stripped = strip_linkedin_chrome(current)
        flags["hashtag"] = stripped != current
        current = stripped
    if unescape:
        unescaped = unescape_html(current)
        flags["html"] = unescaped != current
        current = unescaped
    foot = strip_footer_noise(current)
    flags["footer"] = foot != current
    current = foot.strip()
    repaired = repair_broken_urls(current)
    flags["url"] = repaired != current
    current = repaired.strip()
    if is_event_promo(current, title):
        return None, "event_promo", flags
    if not current:
        return None, "empty_after_scrub", flags
    return current, None, flags
