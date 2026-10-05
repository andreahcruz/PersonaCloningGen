"""Record-level L2 cleaner.

``scrub_corpus_text`` remains the only text scrub. This module owns source
gates, drop codes, and identity copying. It does not import ``clean_row``.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

from host_finetune.canonical_ids import (
    raw_record_id,
    source_identity_for_raw,
)

_SPARK_JOBS = Path(__file__).resolve().parents[1] / "spark_jobs"
if str(_SPARK_JOBS) not in sys.path:
    sys.path.insert(0, str(_SPARK_JOBS))
from corpus_footer_scrub import scrub_corpus_text

# Same floor the frozen row gate uses before the shared scrub.
MIN_X_CHARS = 25

# filename, clean kind, identity medium, text field, title field, scrub kwargs.
SOURCE_SPECS = (
    ("jasonlemkin_blog.jsonl", "blog", "blog", "content", "title", {}),
    ("jasonlemkinlinkedin.jsonl", "linkedin", "linkedin", "content", None, {"linkedin": True}),
    ("jasonlk_originals.jsonl", "x", "x", "text", None, {}),
    (
        "jasonmlemkinyoutubetranscripts.jsonl",
        "youtube",
        "youtube_jason",
        "transcript_text",
        "video_title",
        {"unescape": True},
    ),
    (
        "saastryoutubetranscripts.jsonl",
        "youtube",
        "youtube_saastr",
        "transcript_text",
        "video_title",
        {"unescape": True},
    ),
)

# First reason returned by the frozen gate order.
DROP_CODES = {
    "reply_or_repost": "REPLY_OR_REPOST",
    "too_short": "TOO_SHORT",
    "empty": "EMPTY_TEXT",
    "empty_transcript": "EMPTY_TRANSCRIPT",
    "linkedin_feed_chrome": "LINKEDIN_FEED_CHROME",
    "event_promo": "EVENT_PROMO",
    "empty_after_scrub": "EMPTY_AFTER_SCRUB",
}

SPEC_BY_MEDIUM = {spec[2]: spec for spec in SOURCE_SPECS}
_FLAG_ORDER = ("footer", "url", "hashtag", "html")


@dataclass(frozen=True)
class CleaningResult:
    """One raw record after the frozen cleaning rules.

    ``drop_reason`` is the stable code. ``frozen_reason`` is the historical
    string. Only the first gate is recorded.
    """

    kept: bool
    cleaned_text: str | None
    drop_reason: str | None
    frozen_reason: str | None
    source_id: str | None
    raw_record_id: str | None
    source_identity_kind: str | None
    external_id: str | None
    medium: str
    text_field: str
    cleaned_record: dict | None
    flags: tuple[tuple[str, bool], ...]


def _spec(medium: str) -> tuple:
    spec = SPEC_BY_MEDIUM.get(medium)
    if spec is None:
        raise ValueError(f"unknown medium {medium}")
    return spec


def _blank_flags() -> dict:
    return {key: False for key in _FLAG_ORDER}


def apply_source_policy(
    kind: str,
    record: dict,
    text_field: str,
    title_field: str | None,
    scrub_kwargs: dict,
) -> tuple[dict | None, str | None, dict]:
    """Frozen record gates, then one call to ``scrub_corpus_text``.

    The return is ``(cleaned_record, frozen_reason, flags)``. A drop leaves
    the record as ``None``. This is the single record-policy implementation.
    """
    flags = _blank_flags()
    if kind == "x" and (record.get("is_reply") or record.get("is_repost_or_quote")):
        return None, "reply_or_repost", flags
    raw = record.get(text_field)
    raw_text = raw if isinstance(raw, str) else ""
    if kind == "x":
        stripped = raw_text.strip()
        if not stripped:
            return None, "empty", flags
        if len(stripped) < MIN_X_CHARS:
            return None, "too_short", flags
    title = ""
    if title_field:
        title = record.get(title_field) or ""
        if not isinstance(title, str):
            title = str(title)
    cleaned, reason, flags = scrub_corpus_text(raw_text, title=title, **scrub_kwargs)
    if reason == "empty" and kind == "youtube":
        reason = "empty_transcript"
    if reason:
        return None, reason, flags
    out = dict(record)
    out[text_field] = cleaned
    return out, None, flags


def clean_record(medium: str, record: dict) -> CleaningResult:
    """Apply L2 policy to one raw record.

    Identity is taken from ``record`` before any text scrub. Cleaned text is
    not an input to ``source_id`` or ``raw_record_id``.
    """
    _filename, kind, medium_name, text_field, title_field, scrub_kwargs = _spec(medium)
    try:
        identity = source_identity_for_raw(medium_name, record)
        capture = raw_record_id(medium_name, record)
        source_id = identity.source_id
        kind_name = identity.source_identity_kind
        external_id = identity.external_id
    except (ValueError, TypeError):
        source_id = None
        capture = None
        kind_name = None
        external_id = None
    cleaned, reason, flags = apply_source_policy(
        kind, record, text_field, title_field, dict(scrub_kwargs)
    )
    flag_pairs = tuple(sorted((key, bool(value)) for key, value in flags.items()))
    if reason or cleaned is None:
        code = DROP_CODES.get(reason or "", reason)
        return CleaningResult(
            kept=False,
            cleaned_text=None,
            drop_reason=code,
            frozen_reason=reason,
            source_id=source_id,
            raw_record_id=capture,
            source_identity_kind=kind_name,
            external_id=external_id,
            medium=medium_name,
            text_field=text_field,
            cleaned_record=None,
            flags=flag_pairs,
        )
    text = cleaned.get(text_field)
    return CleaningResult(
        kept=True,
        cleaned_text=text if isinstance(text, str) else None,
        drop_reason=None,
        frozen_reason=None,
        source_id=source_id,
        raw_record_id=capture,
        source_identity_kind=kind_name,
        external_id=external_id,
        medium=medium_name,
        text_field=text_field,
        cleaned_record=cleaned,
        flags=flag_pairs,
    )
