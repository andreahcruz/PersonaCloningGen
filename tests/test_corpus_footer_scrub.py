"""Shared scrap-cleaning rules used by Spark, the DAG, and clean_scraped."""
import corpus_footer_scrub as scrub


def test_footer_only_block_is_removed():
    text = "Related Posts\n5 Interesting Learnings from Klaviyo at $1B ARR\n"
    assert scrub.strip_footer_noise(text) == ""


def test_related_posts_tail_is_cut_and_an_early_mention_is_kept():
    body = ("A" * 200) + "\n\nRelated Posts\nSome other article\n"
    cleaned = scrub.strip_footer_noise(body)
    assert "Related Posts" not in cleaned
    assert cleaned.startswith("A" * 200)

    early = "Intro\n\nRelated Posts\nkept because early\n" + ("B" * 400)
    assert "Related Posts" in scrub.strip_footer_noise(early)


def test_short_sponsor_cta_is_a_promo_and_a_long_essay_is_not():
    promo = (
        "SaaStr AI Annual 2026 attendance is already 143% of last year. "
        "If you sponsor now, you're getting 43% more leads."
    )
    assert scrub.is_event_promo(promo)

    essay = (
        "Join 10,000 of us at SaaStr AI Annual 2026. " + ("Pricing still depends on retention. " * 40)
    )
    assert len(essay.split()) >= scrub.PROMO_MAX_WORDS
    assert not scrub.is_event_promo(essay)
    assert not scrub.is_event_promo(
        essay,
        title="Dear SaaStr: Should I Do a Dinner or a Booth at a Big Industry Event? Or Just Buy Tickets?",
    )


def test_footer_zone_tweet_card_keeps_the_following_essay():
    essay = "Intro paragraph about hiring. " * 40
    card = "\n— Jason ✨BeKind✨ Lemkin ⚫️ (@jasonlk)\nMay 19, 2021\n"
    rest = (
        "Finally, we can distill a lot of this down to one key criterion: "
        "Has Your VP of Sales sold at a startup?"
    )
    cleaned = scrub.strip_footer_noise(essay + card + rest)
    assert "one key criterion" in cleaned
    assert "Intro paragraph about hiring." in cleaned
    assert "(@jasonlk)" not in cleaned


def test_mid_article_tweet_embed_is_kept_and_a_trailing_signature_is_not():
    early = (
        ("Intro paragraph. " * 40)
        + "\n— Jason Lemkin (@jasonlk)\nFebruary 1, 2026\n"
        + ("The real essay continues. " * 80)
    )
    assert "The real essay continues" in scrub.strip_footer_noise(early)

    trailing = ("Intro paragraph. " * 80) + "\n— Jason Lemkin (@jasonlk)\n"
    cleaned = scrub.strip_footer_noise(trailing)
    assert "Intro paragraph" in cleaned
    assert "Jason Lemkin" not in cleaned


def test_sponsor_title_is_a_promo_and_support_tickets_are_not():
    assert scrub.is_event_promo(
        "A long article body. " * 80,
        title="Last Chance to Sponsor SaaStr Europa 2022: What It Will Be Like",
    )
    assert scrub.is_event_promo(
        "A long transcript. " * 80,
        title="SaaStr Insider Sponsor Webinar + AMA, July 28, 2022",
    )
    assert scrub.is_event_promo(
        "body " * 80,
        title="10 Reasons to Sponsor SaaStr Events and Media in 2023!",
    )
    assert not scrub.is_event_promo(
        "body " * 80,
        title="SaaStr AI App of the Week: HappyFox – Cut Support Tickets 50% With AI",
    )
    assert not scrub.is_event_promo(
        "word " * 120,
        title="44% of SaaStr AI 2025 Sponsors … Had CEOs and COOs Who Did Booth Duty",
    )


def test_linkedin_feed_chrome_is_detected_and_hashtag_token_is_removed():
    chrome = "Feed post number 132 Jason M. Lemkin • 3rd+ Visible to anyone"
    assert scrub.is_linkedin_feed_chrome(chrome)
    assert scrub.is_linkedin_feed_chrome("Feed post Jason M. Lemkin • 3rd+ Influencer")
    assert not scrub.is_linkedin_feed_chrome("SaaStr AI Annual is in May. Not a feed card.")
    assert scrub.strip_linkedin_chrome(
        "So proud to be partnered with Salesforce and hashtag #agentforce on the journey!"
    ) == "So proud to be partnered with Salesforce and #agentforce on the journey!"


def test_broken_scheme_is_joined_and_later_spaces_stay():
    assert scrub.repair_broken_urls("See https://\nsaastr.ai/ai-vc now") == "See https://saastr.ai/ai-vc now"
    assert scrub.repair_broken_urls("Try https:// saastr.ai/ai-vc") == "Try https://saastr.ai/ai-vc"
    spaced = "https://saastr.ai/valuation-calc ulator"
    assert scrub.repair_broken_urls(spaced) == spaced
    assert (
        scrub.repair_broken_urls("Try https://saastr.ai/valuation-calc\nulator now")
        == "Try https://saastr.ai/valuation-calculator now"
    )
    two_urls = "https://t.co/PfxUrjXlkS\npic.twitter.com/ghWQ8RmZru"
    assert scrub.repair_broken_urls(two_urls) == two_urls


def test_html_entities_in_captions_are_unescaped():
    assert scrub.unescape_html("Hello &gt;&gt; there &amp; you") == "Hello >> there & you"


def test_scrub_corpus_text_drops_each_dirty_case():
    cleaned, reason, flags = scrub.scrub_corpus_text(
        "Feed post number 1 " + ("x " * 30),
        linkedin=True,
    )
    assert cleaned is None and reason == "linkedin_feed_chrome"

    cleaned, reason, flags = scrub.scrub_corpus_text("AI &gt;&gt; sales", unescape=True)
    assert reason is None and cleaned == "AI >> sales" and flags["html"]

    body = ("Real essay about retention. " * 30) + "\n\nRelated Posts\nAnother headline\n"
    cleaned, reason, flags = scrub.scrub_corpus_text(body)
    assert reason is None and "Related Posts" not in cleaned and flags["footer"]
