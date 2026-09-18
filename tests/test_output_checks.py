from output_checks import looks_degenerate

PROSE = (
    "Most founders wait too long to hire their first head of sales. They think the product "
    "has to be perfect before anyone can sell it, but the opposite is true: a great seller "
    "finds the gaps in your pitch, your pricing and your onboarding faster than any survey. "
    "Hire early, give them real quota, and listen carefully to why deals are lost."
)


def test_repeated_token_is_degenerate():
    assert looks_degenerate("vibe " * 80)


def test_mostly_one_token_is_degenerate():
    assert looks_degenerate(("the " * 30) + PROSE)


def test_normal_prose_is_fine():
    assert not looks_degenerate(PROSE)


def test_length_alone_does_not_trigger_the_check():
    # Long drafts (blog posts, scripts) reuse common words far more overall than short ones.
    # Only the first window is inspected, so a long, varied text must not be penalised.
    varied = " ".join(f"topic{i % 900} point{i}" for i in range(3000))
    assert not looks_degenerate(varied)
    assert not looks_degenerate(PROSE + " " + varied)


def test_repeating_the_same_paragraph_is_flagged():
    assert looks_degenerate((PROSE + " ") * 40)


def test_short_or_empty_output_is_never_flagged_by_default():
    assert not looks_degenerate("")
    assert not looks_degenerate(None)  # type: ignore[arg-type]
    assert not looks_degenerate("vibe vibe vibe")


def test_min_words_can_be_lowered_for_one_sentence_smoke_answers():
    assert looks_degenerate("vibe " * 12, min_words=8)
    assert not looks_degenerate("Annual recurring revenue is the yearly value of your subscriptions.", min_words=8)
