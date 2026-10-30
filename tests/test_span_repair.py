"""Span repair deletes unsupported facts and keeps the rest of the draft."""

from host_finetune.span_repair import SPAN_REPAIR, repair_draft, substantive_eligible

QUOTE = (
    "AI means everything in SaaS needs to be a personalized experience for buyers today."
)


def _claim(text: str, speakers: list[str] | None = None) -> dict:
    item = {"claim": text}
    if speakers:
        item["quoted_speakers"] = speakers
    return item


def test_unsupported_amount_is_removed_and_the_supported_amount_stays():
    claim = "Capital One acquired Brex for $5.15 billion."
    draft = (
        "Capital One acquired Brex for $5.15 billion, and a later note added $60 billion. "
        "The rest of the post is about what founders should watch after a large exit."
    )
    repaired = repair_draft(draft, "Brex price", [_claim(claim)], medium="blog")
    assert repaired["applied"] is True
    assert repaired["contract"] == SPAN_REPAIR
    assert "$60 billion" not in repaired["answer"]
    assert "$5.15 billion" in repaired["answer"]
    assert "founders should watch" in repaired["answer"]
    assert repaired["warning"] == []
    assert repaired["word_retention"] >= 0.85


def test_unsupported_title_keeps_the_supported_speaker_name():
    draft = "Anneka Gupta, CEO of Asana, said " + QUOTE
    repaired = repair_draft(draft, "Who said it?", [_claim(QUOTE, ["Anneka Gupta"])], medium="blog")
    assert repaired["applied"] is True
    assert "CEO of Asana" not in repaired["answer"]
    assert "Anneka Gupta" in repaired["answer"]
    assert "personalized experience" in repaired["answer"]


def test_missing_speaker_name_is_inserted_before_the_quoted_span():
    repaired = repair_draft(QUOTE, "Who said it?", [_claim(QUOTE, ["Anneka Gupta"])], medium="x")
    assert repaired["applied"] is True
    assert repaired["answer"].startswith("Anneka Gupta ")
    assert QUOTE in repaired["answer"]


def test_a_dangling_preposition_is_not_left_behind():
    draft = (
        "Capital One announced that it will acquire Brex for $5 billion and that Brex has "
        "6,000+ customers who spend over $1 million annually on their cards."
    )
    repaired = repair_draft(draft, "Brex price", [_claim("No price was stated in the evidence.")], medium="linkedin")
    if repaired["applied"]:
        assert "for and" not in repaired["answer"]
        assert "has +" not in repaired["answer"]
        assert "over annually" not in repaired["answer"]
    else:
        assert repaired["answer"] == draft
        assert repaired["warning"]


def test_a_large_deletion_is_rejected_and_the_original_stays_with_a_warning():
    draft = "The price was $60 billion yesterday."
    repaired = repair_draft(draft, "price", [_claim("No price was stated.")], medium="x")
    assert repaired["applied"] is False
    assert repaired["answer"] == draft
    assert "unsupported_numeric" in repaired["warning"]
    assert "substantive_loss" in repaired["rejection"]


def test_substantive_eligibility_does_not_use_lexical_overlap():
    assert substantive_eligible({
        "completion": True,
        "format": True,
        "training_copy": True,
        "retrieval_copy": True,
        "topic_echo": False,
        "strict_pass": True,
        "word_retention": 0.9,
        "accepted_claims": 1,
        "claim_overlap": 1,
    })
    assert not substantive_eligible({
        "completion": True,
        "format": True,
        "training_copy": True,
        "retrieval_copy": True,
        "topic_echo": False,
        "strict_pass": False,
        "word_retention": 1,
        "accepted_claims": 1,
        "claim_overlap": 1,
        "lexical_overlap": 1,
    })
