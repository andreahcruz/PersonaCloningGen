"""Before-state and post-migration contract for L2 cleaning ownership.

Expected reasons stay frozen. A mismatch is not repaired by editing them.
"""
import pytest

from host_finetune.build_shadow_cleaned import publish
from host_finetune.canonical_cleaning import SPEC_BY_MEDIUM, clean_record
from host_finetune.canonical_ids import raw_record_id, source_identity_for_raw
from host_finetune.clean_scraped import clean_row

# Frozen corpus acceptance counts. Semantic parity is required.
# Byte parity is the historical CRLF serialization, measured separately.
CORPUS = {
    "raw_total": 32835,
    "kept_total": 30370,
    "dropped_total": 2465,
    "cleaned_by_medium": {
        "blog": 3466,
        "linkedin": 2006,
        "x": 24277,
        "youtube_jason": 428,
        "youtube_saastr": 193,
    },
    "drop_reasons": {
        "event_promo": 450,
        "linkedin_feed_chrome": 623,
        "too_short": 1203,
        "empty": 150,
        "empty_transcript": 39,
    },
    "nopromo_rows": 42771,
}


def _agree(medium: str, record: dict):
    """Shadow result and the live ``clean_row`` tuple must name the same outcome."""
    result = clean_record(medium, record)
    _filename, kind, _medium, text_field, title_field, scrub_kwargs = SPEC_BY_MEDIUM[medium]
    cleaned, reason, _flags = clean_row(kind, record, text_field, title_field, dict(scrub_kwargs))
    assert result.frozen_reason == reason
    assert result.kept is (cleaned is not None)
    if cleaned is None:
        assert result.cleaned_text is None
        assert result.cleaned_record is None
    else:
        assert result.cleaned_text == cleaned[text_field]
        assert result.cleaned_record == cleaned
        assert list(result.cleaned_record) == list(record)
    repeat = clean_record(medium, record)
    assert repeat == result
    return result


def _blog(title: str, content: str) -> dict:
    return {
        "url": "https://www.saastr.com/pricing-note",
        "title": title,
        "date": "2020-01-01",
        "tags": ["saas"],
        "content": content,
    }


def _x(text: str, **flags) -> dict:
    record = {
        "id": "100",
        "text": text,
        "is_reply": False,
        "is_repost_or_quote": False,
        "url": "https://x.com/jasonlk/status/100",
    }
    record.update(flags)
    return record


def test_ownership_direction_and_import_cycle():
    import ast
    from pathlib import Path

    import host_finetune.canonical_cleaning as canonical
    import host_finetune.clean_scraped as legacy

    root = Path("host_finetune")
    cleaner_tree = ast.parse((root / "canonical_cleaning.py").read_text(encoding="utf-8"))
    shim_tree = ast.parse((root / "clean_scraped.py").read_text(encoding="utf-8"))

    def imported(tree):
        modules = []
        names = []
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                modules.append(node.module or "")
                names.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.Import):
                modules.extend(alias.name for alias in node.names)
        return modules, names

    cleaner_modules, cleaner_names = imported(cleaner_tree)
    shim_modules, shim_names = imported(shim_tree)
    assert "host_finetune.clean_scraped" not in cleaner_modules
    assert "clean_row" not in cleaner_names
    assert "corpus_footer_scrub" not in shim_modules
    assert "scrub_corpus_text" not in shim_names
    assert "clean_record" in shim_names
    assert "apply_source_policy" in canonical.__dict__
    assert legacy.clean_row.__module__ == "host_finetune.clean_scraped"
    assert canonical.clean_record.__module__ == "host_finetune.canonical_cleaning"


def test_canonical_module_stays_at_l2():
    from pathlib import Path

    text = Path("host_finetune/canonical_cleaning.py").read_text(encoding="utf-8")
    for banned in (
        "filter_event_promos",
        "split_long_document",
        "fit_sft_to_context",
        "select_balanced_rows",
        "split_groups",
        "relabel_continuations",
        "train_only_index",
        "AutoTokenizer",
    ):
        assert banned not in text


def test_blog_footer_and_url_repair_keep_identity():
    body = ("Sentence about SaaS pricing. " * 40) + "\n\nRelated Posts\n\nAnother post\n"
    body = body.replace("Sentence about SaaS pricing. ", "See https://\nsaastr.ai/ai-vc. ", 1)
    record = _blog("Pricing", body)
    result = _agree("blog", record)
    assert result.kept is True
    assert result.drop_reason is None
    assert "Related Posts" not in result.cleaned_text
    assert "https://saastr.ai/ai-vc" in result.cleaned_text
    assert "https://\n" not in result.cleaned_text
    assert "&gt;" not in (record["content"])
    assert result.source_id == source_identity_for_raw("blog", record).source_id
    assert result.raw_record_id == raw_record_id("blog", record)
    assert result.cleaned_text != record["content"]
    assert result.raw_record_id != raw_record_id("blog", {**record, "content": result.cleaned_text})


def test_blog_does_not_unescape_html():
    record = _blog("Margins", "Margins &gt; growth. " * 20)
    result = _agree("blog", record)
    assert result.kept is True
    assert "&gt;" in result.cleaned_text


def test_blog_event_promo_and_dear_saastr_exception():
    promo = _blog("See you there", "Join us at SaaStr Annual. Register for tickets.")
    assert _agree("blog", promo).drop_reason == "EVENT_PROMO"
    essay = _blog(
        "Dear SaaStr: Should I attend?",
        "Come to SaaStr Annual. Register now. The company still has to learn sales.",
    )
    kept = _agree("blog", essay)
    assert kept.kept is True
    assert kept.drop_reason is None


def test_blog_sponsor_title_drops_a_long_body():
    record = _blog("Last chance to sponsor", "Pricing lesson. " * 80)
    result = _agree("blog", record)
    assert result.drop_reason == "EVENT_PROMO"


@pytest.mark.parametrize(
    ("record", "reason"),
    [
        (_x("a" * 40), None),
        (_x("a" * 25), None),
        (_x("  " + ("a" * 25) + "  "), None),
        (_x("a" * 24), "TOO_SHORT"),
        (_x("  " + ("a" * 24)), "TOO_SHORT"),
        (_x(""), "EMPTY_TEXT"),
        (_x("   \n"), "EMPTY_TEXT"),
        (_x("a" * 40, is_reply=True), "REPLY_OR_REPOST"),
        (_x("a" * 40, is_repost_or_quote=True), "REPLY_OR_REPOST"),
        (_x("hi", is_reply=True, is_repost_or_quote=True), "REPLY_OR_REPOST"),
        (_x("Join us at SaaStr Annual. Register now."), "EVENT_PROMO"),
        (_x("Join SaaStr Annual."), "TOO_SHORT"),
    ],
)
def test_x_gates_in_frozen_order(record, reason):
    result = _agree("x", record)
    assert result.drop_reason == reason
    assert result.source_id == source_identity_for_raw("x", record).source_id
    assert result.raw_record_id == raw_record_id("x", record)


def test_linkedin_chrome_drop_and_hashtag_strip_keep_fallback_identity():
    normal = {
        "content": "A founder note about pricing and retention. hashtag #saas",
        "urn": "",
        "url": "",
    }
    kept = _agree("linkedin", normal)
    assert kept.kept is True
    assert "hashtag #" not in kept.cleaned_text
    assert "#saas" in kept.cleaned_text
    assert kept.source_identity_kind == "content_fallback"
    assert kept.external_id is None
    assert kept.source_id == source_identity_for_raw("linkedin", normal).source_id
    chrome = {
        "content": "Feed post number 4 Join us at SaaStr Annual. Register now.",
        "urn": "",
        "url": "",
    }
    dropped = _agree("linkedin", chrome)
    assert dropped.drop_reason == "LINKEDIN_FEED_CHROME"
    assert dropped.raw_record_id == raw_record_id("linkedin", chrome)


def test_youtube_unescape_empty_transcript_and_repeated_capture():
    spoken = {
        "video_id": "abc",
        "video_title": "Pricing",
        "transcript_text": "Sales &gt; marketing. " * 20,
    }
    kept = _agree("youtube_jason", spoken)
    assert kept.kept is True
    assert "&gt;" not in kept.cleaned_text
    assert "Sales > marketing." in kept.cleaned_text
    for blank in (None, "", "  \n"):
        empty = {"video_id": "abc", "video_title": "Last chance to sponsor", "transcript_text": blank}
        result = _agree("youtube_jason", empty)
        assert result.drop_reason == "EMPTY_TRANSCRIPT"
        assert result.source_id == source_identity_for_raw("youtube_jason", empty).source_id
    other = {
        "video_id": "abc",
        "video_title": "Pricing",
        "transcript_text": "A different capture of the same video. " * 10,
    }
    again = _agree("youtube_jason", other)
    assert again.kept is True
    assert again.source_id == kept.source_id
    assert again.raw_record_id != kept.raw_record_id
    saastr = {
        "video_id": "saastr1",
        "video_title": "A talk",
        "transcript_text": "Sales &gt; marketing. " * 20,
    }
    saastr_result = _agree("youtube_saastr", saastr)
    assert saastr_result.kept is True
    assert "&gt;" not in saastr_result.cleaned_text


@pytest.fixture(scope="module")
def shadow_report() -> dict:
    return publish()["stored"]


def test_corpus_contract_is_the_migration_gate(shadow_report):
    assert shadow_report["raw_total"] == CORPUS["raw_total"]
    assert shadow_report["kept_total"] == CORPUS["kept_total"]
    assert shadow_report["dropped_total"] == CORPUS["dropped_total"]
    assert shadow_report["drop_reasons"] == CORPUS["drop_reasons"]
    assert shadow_report["identity_breaks"] == 0
    assert shadow_report["membership_parity"] is True
    assert shadow_report["order_parity"] is True
    assert shadow_report["content_parity"] is True
    by_medium = {row["medium"]: row for row in shadow_report["per_source"]}
    for medium, count in CORPUS["cleaned_by_medium"].items():
        assert by_medium[medium]["shadow_cleaned_count"] == count
        assert by_medium[medium]["frozen_cleaned_count"] == count
        assert by_medium[medium]["byte_crlf"] is True
        assert by_medium[medium]["byte_lf"] is False
        assert by_medium[medium]["byte_newline"] == "CRLF"
    assert shadow_report["byte_parity"] is True
    assert shadow_report["sft_nopromo"]["frozen_rows"] == CORPUS["nopromo_rows"]
    assert shadow_report["sft_nopromo"]["shadow_rows"] == CORPUS["nopromo_rows"]
    assert shadow_report["sft_nopromo"]["mismatches"] == 0
