"""Determinism tests for external source ids and raw capture ids. No dataset IO."""
import unicodedata

import pytest

from host_finetune.canonical_ids import (
    CONTENT_FALLBACK,
    CONTEXT_FIT_VERSION,
    PLATFORM_URN,
    PLATFORM_VIDEO_ID,
    RELABEL_VERSION,
    canonical_raw_json,
    document_id,
    experiment_row_id,
    model_row_id,
    rag_row_id,
    raw_payload_sha256,
    raw_record_id,
    relabel_row_id,
    source_identity_for_raw,
)


def _blog(**overrides):
    record = {
        "url": "https://www.saastr.com/Post",
        "title": "T",
        "date": "2020-01-01",
        "tags": ["a"],
        "content": "raw body",
        "scraped_at": "t1",
    }
    record.update(overrides)
    return record


def test_same_blog_url_keeps_source_id_when_content_changes():
    first = source_identity_for_raw("blog", _blog(content="raw A", scraped_at="t1"))
    second = source_identity_for_raw(
        "blog",
        _blog(url="HTTPS://www.SaaStr.com/Post/#share", content="raw B", scraped_at="t2"),
    )
    assert first.source_id == second.source_id
    assert first.source_identity_kind == "platform_url"
    assert first.external_id == "https://www.saastr.com/Post"
    assert raw_record_id("blog", _blog(content="raw A")) != raw_record_id("blog", _blog(content="raw B"))


def test_different_blog_urls_do_not_collide():
    left = source_identity_for_raw("blog", _blog(url="https://www.saastr.com/one"))
    right = source_identity_for_raw("blog", _blog(url="https://www.saastr.com/two"))
    assert left.source_id != right.source_id


def test_x_source_id_ignores_missing_or_changed_text():
    bare = source_identity_for_raw("x", {"id": "99", "text": None, "scraped_at": "later"})
    changed = source_identity_for_raw("x", {"text": "hello", "id": "99", "scraped_at": "earlier"})
    assert bare.source_id == changed.source_id
    assert bare.source_identity_kind == "platform_tweet_id"
    assert bare.external_id == "99"
    assert raw_record_id("x", {"id": "99", "text": None}) != raw_record_id("x", {"id": "99", "text": "hello"})


def test_linkedin_urn_wins_and_fallback_is_typed():
    by_urn = source_identity_for_raw(
        "linkedin",
        {"urn": "urn:li:activity:1", "content": "A", "published_text": "2 hours ago", "url": "https://x"},
    )
    same_urn = source_identity_for_raw(
        "linkedin",
        {"urn": "urn:li:activity:1", "content": "B", "published_text": "3 days ago"},
    )
    fallback = source_identity_for_raw("linkedin", {"content": "A", "published_text": "2 hours ago"})
    assert by_urn.source_id == same_urn.source_id
    assert by_urn.source_identity_kind == PLATFORM_URN
    assert by_urn.external_id == "urn:li:activity:1"
    assert fallback.source_identity_kind == CONTENT_FALLBACK
    assert fallback.external_id is None
    assert fallback.source_id != by_urn.source_id
    assert raw_record_id("linkedin", {"urn": "urn:li:activity:1", "content": "A"}) != raw_record_id(
        "linkedin",
        {"urn": "urn:li:activity:1", "content": "B"},
    )


def test_linkedin_fallback_follows_raw_body_and_not_relative_time():
    same = source_identity_for_raw("linkedin", {"content": "same post", "url": "", "urn": ""})
    again = source_identity_for_raw(
        "linkedin",
        {"content": "same post", "published_text": "2 hours ago"},
    )
    other = source_identity_for_raw("linkedin", {"content": "different post"})
    assert same.source_id == again.source_id
    assert same.source_identity_kind == CONTENT_FALLBACK
    assert same.source_id != other.source_id
    assert raw_record_id("linkedin", {"content": "same post", "published_text": "2 hours ago"}) == raw_record_id(
        "linkedin",
        {"content": "same post", "published_text": "3 days ago"},
    )


def test_youtube_source_id_survives_a_new_transcript_capture():
    shared = {"video_id": "abc", "fetched_at_utc": "t1", "video_title": "Talk"}
    first = source_identity_for_raw("youtube_jason", {**shared, "transcript_text": "take one"})
    second = source_identity_for_raw(
        "youtube_jason",
        {**shared, "transcript_text": "take two", "fetched_at_utc": "t2"},
    )
    other_channel = source_identity_for_raw(
        "youtube_saastr",
        {"video_id": "abc", "transcript_text": "take one"},
    )
    assert first.source_id == second.source_id
    assert first.source_identity_kind == PLATFORM_VIDEO_ID
    assert first.external_id == "abc"
    assert first.source_id != other_channel.source_id
    first_raw = raw_record_id("youtube_jason", {**shared, "transcript_text": "take one"})
    second_raw = raw_record_id("youtube_jason", {**shared, "transcript_text": "take two"})
    repeat_raw = raw_record_id(
        "youtube_jason",
        {**shared, "transcript_text": "take one", "fetched_at_utc": "later", "listing_source": "search"},
    )
    assert first_raw == repeat_raw
    assert first_raw != second_raw


def test_raw_record_hash_is_independent_of_dict_order_os_newline_and_cleaned_text():
    left = {
        "url": "https://www.saastr.com/Post",
        "content": "line\r\nline",
        "title": "café",
        "scraped_at": "t1",
        "cleaned_text": "scrubbed",
    }
    right = {
        "cleaned_text": "different scrub",
        "title": "café",
        "content": "line\r\nline",
        "url": "https://www.saastr.com/Post",
        "scraped_at": "t2",
    }
    decomposed = dict(left)
    decomposed["title"] = unicodedata.normalize("NFD", "café")
    assert raw_record_id("blog", left) == raw_record_id("blog", right)
    assert raw_record_id("blog", left) == raw_record_id("blog", decomposed)
    blob = canonical_raw_json("blog", left)
    assert "\n" not in blob
    assert "\r" not in blob
    assert "café" in blob
    assert "scraped_at" not in blob
    assert "cleaned_text" not in blob
    assert raw_record_id("x", {"id": "1"}) == raw_record_id("x", {"id": "1", "text": None})
    assert raw_record_id("x", {"id": "1", "text": None}) != raw_record_id("x", {"id": "1", "text": ""})


def test_document_id_follows_the_capture_not_the_external_source():
    record = _blog()
    other = _blog(content="other capture")
    source = source_identity_for_raw("blog", record).source_id
    first = document_id(raw_record_id("blog", record))
    second = document_id(raw_record_id("blog", other))
    assert first != source
    assert first != second
    assert document_id(raw_record_id("blog", record)) == first


def test_model_row_id_tracks_the_capture_and_ignores_balance_split_and_eligibility():
    document = document_id(raw_record_id("blog", _blog()))
    base = model_row_id(document, fit_version=CONTEXT_FIT_VERSION, chunk_ordinal=0, output="body")
    again = model_row_id(document, fit_version=CONTEXT_FIT_VERSION, chunk_ordinal=0, output="body")
    next_chunk = model_row_id(document, fit_version=CONTEXT_FIT_VERSION, chunk_ordinal=1, output="body")
    other_fit = model_row_id(document, fit_version="fit512-keepbreaks-v2", chunk_ordinal=0, output="body")
    other_output = model_row_id(document, fit_version=CONTEXT_FIT_VERSION, chunk_ordinal=0, output="edited")
    assert base == again
    assert len({base, next_chunk, other_fit, other_output}) == 4
    relabel = relabel_row_id(
        base,
        relabel_version=RELABEL_VERSION,
        sft_role="unchanged",
        instruction="Write as Jason.",
    )
    assert rag_row_id(relabel) == relabel
    assert relabel != relabel_row_id(
        base,
        relabel_version=RELABEL_VERSION,
        sft_role="continuation",
        instruction="Write as Jason.",
    )


def test_raw_payload_hash_is_not_the_canonical_capture_id():
    first = b'{"content":"body","scraped_at":"t1","url":"https://www.saastr.com/Post"}'
    second = b'{"content":"body","scraped_at":"t2","url":"https://www.saastr.com/Post"}'
    assert raw_payload_sha256(first) != raw_payload_sha256(second)
    record = {"content": "body", "url": "https://www.saastr.com/Post", "scraped_at": "t1"}
    other = dict(record)
    other["scraped_at"] = "t2"
    assert raw_record_id("blog", record) == raw_record_id("blog", other)
    assert raw_payload_sha256(first) != raw_record_id("blog", record)


def test_experiment_row_id_is_not_a_source_id():
    assert experiment_row_id(26148) == "train_26148"
    source = source_identity_for_raw("blog", _blog()).source_id
    assert experiment_row_id(26148) != source
    with pytest.raises(ValueError):
        experiment_row_id(-1)
