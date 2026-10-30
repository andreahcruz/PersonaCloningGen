"""Strict factual checks stay separate from lexical overlap."""

from host_finetune.evidence_capsule import (
    PROMPT_CONTRACT,
    draft_retry_reason,
    final_evidence_instruction,
    media_jobs,
)
from host_finetune.factual_reliability import assess_factual, grounding_regression
from host_finetune.compare_adapters import MEDIUMS

QUOTE = (
    "AI means everything in SaaS needs to be a personalized experience for buyers today."
)


def _claim(text: str, speakers: list[str] | None = None) -> dict:
    item = {"claim": text, "source_id": "gd_parent", "support_span": text}
    if speakers:
        item["quoted_speakers"] = speakers
    return item


def test_extraction_failure_is_not_strictly_grounded():
    report = assess_factual(
        "Capital One is paying $5 billion for the company.",
        "What price did Capital One announce?",
        [],
        evidence_status="extraction_failed",
        hits=[{"id": "train_1", "text": "Capital One announced a price."}],
    )
    assert report["not_lexical_entailment"] is True
    assert report["pass"] is False
    assert "no_accepted_factual_evidence" in report["reasons"]
    assert "unsupported_numeric" in report["reasons"]


def test_role_requires_person_title_and_organization_in_the_claim():
    unsupported = assess_factual(
        "Anneka Gupta, CEO of Asana, made the point.",
        "Who said it?",
        [_claim(QUOTE, ["Anneka Gupta"])],
    )
    assert unsupported["role_assertions"][0]["verdict"] == "unsupported"
    supported_claim = "Anneka Gupta, Chief Product Officer at Rubrik, said " + QUOTE
    supported = assess_factual(
        "Anneka Gupta, Chief Product Officer at Rubrik, made the point.",
        "Who said it?",
        [_claim(supported_claim, ["Anneka Gupta"])],
    )
    assert supported["role_assertions"][0]["verdict"] == "supported"
    missing_org = assess_factual(
        "Anneka Gupta is the CEO according to the thread.",
        "Who said it?",
        [_claim(QUOTE, ["Anneka Gupta"])],
    )
    assert missing_org["role_assertions"][0]["verdict"] == "not_verifiable"


def test_quoted_span_without_the_speaker_is_unsupported():
    omitted = assess_factual(
        QUOTE,
        "Who said it?",
        [_claim(QUOTE, ["Anneka Gupta"])],
    )
    assert omitted["speaker_attribution"]["status"] == "omitted"
    assert omitted["pass"] is False
    named = assess_factual(
        "Anneka Gupta said " + QUOTE,
        "Who said it?",
        [_claim(QUOTE, ["Anneka Gupta"])],
    )
    assert named["speaker_attribution"]["status"] == "supported"


def test_retry_rejects_a_new_unsupported_number_and_requests_a_longer_x_post():
    before = assess_factual("The price was $5 billion in the announcement to buyers.", "price", [_claim("The price was $5 billion.")])
    after = assess_factual("The price was $5 billion and another $2 billion.", "price", [_claim("The price was $5 billion.")])
    assert grounding_regression(before, after) is True
    short = "Capital One paid. The rest is context for founders watching."
    assert len(short.split()) < 20
    assert draft_retry_reason(short, "pricing", "x", {"applicable": True, "pass": True}) == "x_below_operational_floor"
    leaked = "[E1] Capital One paid a price that founders should understand before they react today."
    assert draft_retry_reason(leaked, "pricing", "blog", {"applicable": True, "pass": True}) == "instruction_leak"


def test_prompt_contract_is_versioned_and_the_job_keeps_extraction_status():
    instruction = final_evidence_instruction(MEDIUMS[2], "Who said it?", [_claim(QUOTE)])
    assert PROMPT_CONTRACT in instruction
    assert "Do not give a person a title" in instruction
    jobs = media_jobs("Who said it?", [], [], "extraction_failed")
    assert "extraction failure" in jobs[0]["instruction"]
    assert PROMPT_CONTRACT in jobs[0]["instruction"]
    assert jobs[0]["prompt_contract"] == PROMPT_CONTRACT
    assert jobs[0]["evidence_status"] == "extraction_failed"
