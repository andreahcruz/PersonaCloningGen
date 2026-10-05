"""Shadow cleaner parity. The full corpus check writes only under artifacts/refactor/cleaning."""
from pathlib import Path

import pytest

from host_finetune.build_shadow_cleaned import publish
from host_finetune.canonical_cleaning import SOURCE_SPECS, clean_record
from host_finetune.canonical_ids import raw_record_id, source_identity_for_raw
from host_finetune.clean_scraped import _SOURCES

ROOT = Path(__file__).resolve().parents[1]


def test_shadow_modules_do_not_call_cleaning_writers():
    cleaner = (ROOT / "host_finetune" / "canonical_cleaning.py").read_text(encoding="utf-8")
    builder = (ROOT / "host_finetune" / "build_shadow_cleaned.py").read_text(encoding="utf-8")
    assert "clean_file(" not in cleaner
    assert "clean_file(" not in builder
    assert "clean_scraped(" not in cleaner
    assert "clean_scraped(" not in builder
    assert "AutoTokenizer" not in builder
    assert "fit_sft_to_context" not in builder
    assert "open(" not in cleaner


def test_source_specs_match_the_frozen_cleaner():
    projected = [(spec[0], spec[1], spec[3], spec[4], spec[5]) for spec in SOURCE_SPECS]
    assert projected == list(_SOURCES)


def test_reply_is_the_first_drop_and_identity_is_kept():
    record = {
        "id": "99",
        "text": "hi",
        "is_reply": True,
        "url": "https://x.com/jasonlk/status/99",
    }
    result = clean_record("x", record)
    assert result.kept is False
    assert result.drop_reason == "REPLY_OR_REPOST"
    assert result.frozen_reason == "reply_or_repost"
    assert result.source_id == source_identity_for_raw("x", record).source_id
    assert result.raw_record_id == raw_record_id("x", record)


def test_empty_youtube_transcript_keeps_the_video_identity():
    record = {"video_id": "abc", "transcript_text": None, "video_title": "Talk"}
    result = clean_record("youtube_jason", record)
    assert result.drop_reason == "EMPTY_TRANSCRIPT"
    assert result.source_id == source_identity_for_raw("youtube_jason", record).source_id
    assert result.raw_record_id == raw_record_id("youtube_jason", record)


def test_repeated_youtube_captures_share_source_id_only():
    first = {"video_id": "abc", "video_title": "Talk", "transcript_text": "alpha " * 30}
    second = {"video_id": "abc", "video_title": "Talk", "transcript_text": "beta " * 30}
    left = clean_record("youtube_jason", first)
    right = clean_record("youtube_jason", second)
    assert left.kept and right.kept
    assert left.source_id == right.source_id
    assert left.raw_record_id != right.raw_record_id


def test_linkedin_fallback_is_not_a_platform_id():
    record = {
        "content": "A founder note about pricing and retention that is not a feed card.",
        "urn": "",
        "url": "",
    }
    result = clean_record("linkedin", record)
    assert result.kept is True
    assert result.source_identity_kind == "content_fallback"
    assert result.external_id is None
    assert result.source_id == source_identity_for_raw("linkedin", record).source_id


def test_saastr_transcript_is_not_dropped_for_being_saastr():
    record = {"video_id": "saastr1", "video_title": "A talk", "transcript_text": "lesson " * 40}
    result = clean_record("youtube_saastr", record)
    assert result.kept is True
    assert result.drop_reason is None


def test_blog_footer_scrub_does_not_rename_the_source():
    record = {
        "url": "https://www.saastr.com/Post",
        "title": "Pricing",
        "date": "2020-01-01",
        "tags": ["saas"],
        "content": ("Sentence about SaaS. " * 40) + "\n\nRelated Posts\n\nAnother post\n",
    }
    result = clean_record("blog", record)
    repeat = clean_record("blog", record)
    assert result.kept is True
    assert "Related Posts" not in (result.cleaned_text or "")
    assert result.cleaned_text == repeat.cleaned_text
    assert result.source_id == source_identity_for_raw("blog", record).source_id
    assert result.raw_record_id == raw_record_id("blog", record)


def test_short_event_promo_drops_before_a_later_rule():
    record = {
        "url": "https://www.saastr.com/tickets",
        "title": "See you there",
        "date": "2020-01-01",
        "tags": [],
        "content": "Join us at SaaStr Annual. Register for tickets.",
    }
    result = clean_record("blog", record)
    assert result.kept is False
    assert result.drop_reason == "EVENT_PROMO"


@pytest.fixture(scope="module")
def shadow_report() -> dict:
    return publish()["stored"]


def test_shadow_cleaned_matches_frozen_files(shadow_report):
    assert shadow_report["raw_total"] == 32835
    assert shadow_report["kept_total"] == 30370
    assert shadow_report["dropped_total"] == 2465
    assert shadow_report["membership_parity"] is True
    assert shadow_report["order_parity"] is True
    assert shadow_report["content_parity"] is True
    assert shadow_report["byte_parity"] is True
    assert shadow_report["identity_breaks"] == 0
    assert shadow_report["drop_reasons"] == {
        "event_promo": 450,
        "linkedin_feed_chrome": 623,
        "too_short": 1203,
        "empty": 150,
        "empty_transcript": 39,
    }
    by_medium = {row["medium"]: row for row in shadow_report["per_source"]}
    assert by_medium["youtube_saastr"]["shadow_cleaned_count"] == by_medium["youtube_saastr"]["frozen_cleaned_count"]
    assert by_medium["youtube_saastr"]["shadow_cleaned_count"] > 0
    assert shadow_report["sft_nopromo"] == {
        "frozen_rows": 42771,
        "shadow_rows": 42771,
        "mismatches": 0,
        "membership_order_content": True,
    }
