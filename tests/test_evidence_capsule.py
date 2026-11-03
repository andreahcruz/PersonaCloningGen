"""Verified atomic evidence sits between frozen retrieval and Relabel."""
from contextlib import contextmanager
from pathlib import Path

import pytest

from host_finetune.compare_adapters import MEDIUMS, prompt_for
from host_finetune.evidence_capsule import (
    DISTILL_BACKEND,
    apply_verdicts,
    corrective_retry_improves,
    draft_retry_reason,
    evidence_provenance,
    extraction_user_prompt,
    fit_extraction_hits,
    final_evidence_instruction,
    interpret_verification,
    media_jobs,
    parse_atomic_evidence,
    partition_evidence,
    repeats_topic,
    require_adapter_toggle,
    verification_user_prompt,
)
from host_finetune.rag_source_copying import RAG_SOURCE_COPY_THRESHOLD, evaluate_rag_source_copying


RAW = (
    "Monday.com grew from $240m ARR to $400m ARR. Revenue was up 91 percent. "
    "Customer count was up 34 percent. The number of customers spending at least "
    "$50k grew from 264 to 793."
)
TITLE = (
    "What is the process of selling SaaS to a big corporation as a small startup?"
)
SPAN = "Revenue was up 91 percent."
CLAIM = "Monday.com reported roughly 91 percent year-over-year revenue growth."
ADVICE = "A sales team must provide a clear value proposition and unique selling points."


def _item(source_id, claim, span, claim_type="number"):
    return {
        "source_id": source_id,
        "claim": claim,
        "support_span": span,
        "claim_type": claim_type,
    }


class _AdapterModel:
    def __init__(self):
        self.adapter_enabled = True

    @contextmanager
    def disable_adapter(self):
        self.adapter_enabled = False
        try:
            yield
        finally:
            self.adapter_enabled = True


def test_unknown_source_id_is_rejected_and_a_real_span_is_kept():
    hits = [{"id": "train_10049", "text": RAW}]
    accepted, rejected = partition_evidence(
        [
            _item("train_99999", CLAIM, SPAN),
            _item("train_10049", CLAIM, SPAN),
        ],
        hits,
        topic="Monday.com ARR",
    )
    assert [item["reason"] for item in rejected] == ["unknown_source_id"]
    assert accepted[0]["source_id"] == "train_10049"
    assert accepted[0]["support_span"] == SPAN


def test_support_span_must_occur_in_the_named_source():
    hits = [{"id": "train_10049", "text": RAW}]
    accepted, rejected = partition_evidence(
        [
            _item("train_10049", CLAIM, ""),
            _item("train_10049", CLAIM, "invented span about pain points"),
            _item("train_10049", CLAIM, "  revenue   WAS up 91 percent. "),
        ],
        hits,
        topic="Monday.com ARR",
    )
    assert [item["reason"] for item in rejected] == ["missing_support", "support_not_in_source"]
    assert len(accepted) == 1
    assert accepted[0]["support_span"] == "  revenue   WAS up 91 percent. "


def test_unsupported_claim_is_removed_before_the_prompt():
    hits = [{"id": "train_10049", "text": RAW}]
    accepted, rejected = partition_evidence(
        [
            _item("train_10049", CLAIM, SPAN),
            _item("train_10049", ADVICE, "What is a value proposition?", "fact"),
        ],
        [{"id": "train_10049", "text": RAW + " What is a value proposition?"}],
        topic="Monday.com ARR",
    )
    assert [item["claim"] for item in accepted] == [CLAIM]
    assert rejected[0]["reason"] == "interrogative_assertion"
    jobs = media_jobs("Monday.com ARR", hits, accepted)
    assert ADVICE not in jobs[0]["instruction"]
    assert CLAIM in jobs[0]["instruction"]


def test_title_stub_cannot_carry_unsupported_background_advice():
    hits = [{"id": "train_20783", "text": TITLE}]
    prompt = extraction_user_prompt("Can an 8-person startup sell to a CIO?", hits)
    assert "title or stub" in prompt
    assert "do not answer it" in prompt
    assert "Do not expand it with background knowledge." in prompt
    assert "value proposition" not in prompt
    accepted, rejected = partition_evidence(
        [_item("train_20783", ADVICE, "startups need a clear value proposition", "fact")],
        hits,
    )
    assert accepted == []
    assert rejected[0]["reason"] == "support_not_in_source"
    titled, _ = partition_evidence(
        [_item("train_20783", ADVICE, TITLE, "fact")],
        hits,
    )
    assert titled == []
    jobs = media_jobs("Can an 8-person startup sell to a CIO?", hits, [])
    assert ADVICE not in jobs[0]["instruction"]
    assert "value proposition" not in jobs[0]["instruction"]


def test_support_spans_and_raw_passages_stay_out_of_the_relabel_prompt():
    hits = [{"id": "train_10049", "text": RAW, "distance": 0.21, "split": "train"}]
    claims = [_item("train_10049", CLAIM, SPAN)]
    jobs = media_jobs("Monday.com ARR", hits, claims)
    for job in jobs:
        assert SPAN not in job["instruction"]
        assert RAW not in job["instruction"]
        assert "train_10049" not in job["instruction"]
        assert CLAIM in job["instruction"]
        assert "ALLOWED FACTUAL EVIDENCE" in job["instruction"]
        assert "Do not mention the evidence" not in job["instruction"]
        assert job["instruction"].startswith(prompt_for(job["medium"], "Monday.com ARR"))
        assert job["instruction"].rstrip().endswith(f"about: Monday.com ARR")


def test_only_verified_claims_are_reused_across_media():
    hits = [{"id": "train_10049", "text": RAW}]
    claims = [_item("train_10049", CLAIM, SPAN)]
    jobs = media_jobs("Monday.com ARR", hits, claims)
    assert [job["medium"]["medium"] for job in jobs] == ["blog", "linkedin", "x", "talk"]
    assert len({job["evidence_block"] for job in jobs}) == 1
    assert len({id(job["claims"]) for job in jobs}) == 1
    assert jobs[0]["evidence_block"] == f"ALLOWED FACTUAL EVIDENCE\n\n[E1] {CLAIM}"


def test_retrieval_off_prompt_does_not_gain_evidence():
    assert "Factual evidence" not in prompt_for(MEDIUMS[0], "Hiring")
    prompted = final_evidence_instruction(MEDIUMS[0], "Hiring", [_item("train_1", CLAIM, SPAN)])
    assert prompted != prompt_for(MEDIUMS[0], "Hiring")


def test_final_copy_gate_uses_the_original_retrieved_passage():
    hits = [{"id": "train_10049", "text": RAW}]
    report = evaluate_rag_source_copying(RAW, hits, enabled=True)
    assert report["pass"] is False
    assert report["document_id"] == "train_10049"
    assert report["max_contiguous_words"] >= RAG_SOURCE_COPY_THRESHOLD
    long_claim = "Reported year-over-year growth was 91 percent and the rest of this sentence is commentary."
    accepted, rejected = partition_evidence(
        [_item("train_10049", long_claim, SPAN)],
        hits,
        topic="Monday.com ARR",
    )
    assert rejected == []
    assert accepted[0]["claim"] == long_claim
    assert accepted[0]["claim"] == long_claim
    jobs = media_jobs("Monday.com ARR", hits, accepted)
    assert RAW not in jobs[0]["instruction"]


def test_verifier_sees_only_the_claim_and_span():
    prompt = verification_user_prompt(CLAIM, SPAN)
    assert CLAIM in prompt
    assert SPAN in prompt
    lowered = prompt.casefold()
    assert "distance" not in lowered
    assert "linkedin" not in lowered
    assert "jason lemkin" not in lowered
    assert interpret_verification("SUPPORTED") == "SUPPORTED"
    assert interpret_verification("UNSUPPORTED") == "UNSUPPORTED"
    assert interpret_verification("maybe") == "UNSUPPORTED"


def test_provenance_keeps_raw_hits_when_claims_are_replaced():
    hits = [{"id": "train_10049", "text": RAW, "split": "train", "distance": 0.2}]
    saved = evidence_provenance(
        hits,
        {
            "proposed": [_item("train_10049", CLAIM, SPAN)],
            "verified": [_item("train_10049", CLAIM, SPAN)],
            "evidence_block": f"- {CLAIM}",
            "extraction_seconds": 1.0,
            "verification_seconds": 0.4,
        },
    )
    saved["verified_claims"] = []
    saved["evidence_block"] = ""
    assert saved["retrieval_hits"][0]["text"] == RAW
    assert saved["support_spans_in_generator_prompt"] is False
    assert "qwen" not in DISTILL_BACKEND


def test_unparseable_extraction_is_not_repaired():
    with pytest.raises(Exception, match="JSON"):
        parse_atomic_evidence("Monday.com grew very quickly and here are some notes")
    parsed = parse_atomic_evidence('```json\n[{"source_id": "train_10049"}]\n```')
    assert parsed == [{"source_id": "train_10049"}]
    truncated = '[{"source_id": "train_1", "claim": "still writing'
    with pytest.raises(Exception, match="JSON array"):
        parse_atomic_evidence(truncated)


def test_interior_quote_is_escaped_without_dropping_fields():
    raw = """[
  {
    "source_id": "train_12216",
    "claim": "Atlassian's users are split 50/50 between business and technical",
    "support_span": "50% of Users on "Business Side”, 50% Technical",
    "claim_type": "fact"
  }
]"""
    parsed = parse_atomic_evidence(raw)
    assert parsed[0]["source_id"] == "train_12216"
    assert parsed[0]["support_span"] == '50% of Users on "Business Side”, 50% Technical'
    assert parsed[0]["claim_type"] == "fact"


def test_missing_comma_is_inserted_without_adding_fields():
    raw = """[
  {
    "source_id": "train_5411",
    "claim": "Deals are getting smaller.",
    "support_span": "Deals are getting smaller"
    "claim_type": "fact"
  }
]"""
    parsed = parse_atomic_evidence(raw)
    assert list(parsed[0]) == ["source_id", "claim", "support_span", "claim_type"]
    assert parsed[0]["claim_type"] == "fact"


def test_truncated_tail_keeps_only_closed_objects():
    raw = """[
  {"source_id": "train_1", "claim": "kept", "support_span": "kept", "claim_type": "fact"},
  {"source_id": "train_2", "claim": "cut off"""
    parsed = parse_atomic_evidence(raw)
    assert parsed == [
        {
            "source_id": "train_1",
            "claim": "kept",
            "support_span": "kept",
            "claim_type": "fact",
        }
    ]


def test_generation_loads_one_model_and_does_not_name_qwen():
    root = Path(__file__).resolve().parents[1]
    source = (root / "host_finetune" / "compare_adapters.py").read_text(encoding="utf-8")
    capsule = (root / "host_finetune" / "evidence_capsule.py").read_text(encoding="utf-8")
    assert source.lower().count("qwen") == 0
    assert capsule.lower().count("qwen") == 0
    assert source.count("from_pretrained") == 1
    assert "disable_adapter" in source
    model = _AdapterModel()
    require_adapter_toggle(model)
    with model.disable_adapter():
        assert model.adapter_enabled is False
    assert model.adapter_enabled is True


def test_retrieval_configuration_stays_frozen_when_atomic_evidence_is_recorded(tmp_path):
    from host_finetune.compare_adapters import write_parameters

    write_parameters(
        tmp_path,
        retrieval=True,
        seed_base=42,
        embed_model="nomic-embed-text",
        retrieval_k=4,
        chroma_path="host_finetune/output/chroma_lemkin_train_only",
    )
    payload = __import__("json").loads((tmp_path / "config" / "run_parameters.json").read_text(encoding="utf-8"))
    assert payload["retrieval"]["collection"] == "lemkin_train_only"
    assert payload["retrieval"]["embedding_model"] == "nomic-embed-text"
    assert payload["retrieval"]["top_k"] == 4
    assert payload["retrieval"]["medium_filter"] is False
    assert payload["generation"]["temperature"] == 0.7
    assert payload["generation"]["rag_source_copy_threshold"] == RAG_SOURCE_COPY_THRESHOLD
    assert payload["generation"]["rag_source_copy_max_retries"] == 1
    distill = payload["evidence_distillation"]
    assert distill["representation"] == "atomic_claims"
    assert distill["adapter_enabled"] is False
    assert distill["do_sample"] is False
    assert distill["internal_copy_gate"] is False
    assert distill["support_spans_in_generator_prompt"] is False
    assert distill["second_model"] is None
    assert distill["llm_verifier_on_critical_path"] is False
    assert distill["numeric_grounding"] is True
    assert distill["numeric_role"] == "diagnostic_only"
    assert distill["numeric_blocks_output"] is False
    assert distill["repair_on_critical_path"] is False
    assert distill["style_rewrite_on_critical_path"] is False
    assert distill["base_generation_on_critical_path"] is False
    assert distill["lora_scale"] == 1.0
    assert distill["closer"] == "minimal"
    assert "qwen" not in distill["backend"]


def test_quoted_speaker_reaches_the_prompt_only_when_the_source_names_them():
    text = 'Anneka Gupta said "AI means everything in SaaS needs to be a personalized experience."'
    hits = [{
        "id": "gd_parent",
        "text": text,
        "graph_document_id": "gd_child",
        "chunk_id": "chunk-000",
        "attribution_status": "third_party_quotation",
        "quoted_speakers": ["Anneka Gupta", "Someone Else"],
    }]
    span = "AI means everything in SaaS needs to be a personalized experience."
    claim = "AI means everything in SaaS needs to be a personalized experience."
    accepted, rejected = partition_evidence(
        [_item("gd_child:chunk-000", claim, span)],
        hits,
    )
    assert rejected == []
    assert accepted[0]["source_id"] == "gd_parent"
    assert accepted[0]["quoted_speakers"] == ["Anneka Gupta"]
    instruction = final_evidence_instruction(MEDIUMS[1], "Who said it?", accepted)
    assert "Speaker: Anneka Gupta" in instruction
    assert "Someone Else" not in instruction
    assert "Attribute this statement" not in instruction
    assert "Do not copy speaker labels or evidence tags into the draft." in instruction
    assert "Do not present it as Jason Lemkin's own words" in instruction


def test_extraction_failure_is_not_described_as_irrelevant_sources():
    failed = final_evidence_instruction(MEDIUMS[0], "Brex", [], "extraction_failed")
    empty = final_evidence_instruction(MEDIUMS[0], "Brex", [], "no_accepted_claims")
    assert "extraction failure" in failed
    assert "not a finding that the sources were irrelevant" in failed
    assert "No claim passed source-span validation" in empty
    assert "irrelevant" not in empty
    assert "Do not repeat the user's question" in failed


def test_topic_echo_and_unsupported_number_request_one_retry():
    question = "What price did Capital One announce for acquiring Brex?"
    assert repeats_topic(question, question)
    assert repeats_topic(question + "\n\n$5.15 billion.", question)
    assert draft_retry_reason(question, question, "linkedin", {"applicable": True, "pass": True}) == "topic_echo"
    short = "Capital One announced a price."
    assert draft_retry_reason(short, "pricing", "blog", {"applicable": True, "pass": True}) == "below_length_band"
    numeric = {"applicable": True, "pass": False, "unsupported": [{"value": "$2 billion"}]}
    long = (
        "Capital One announced an acquisition and the rest of this paragraph explains why the "
        "number matters to founders who are watching late-stage exits, retention, and the "
        "hiring plan that follows a large outcome in enterprise software this year for buyers."
    )
    assert len(long.split()) >= 40
    assert draft_retry_reason(long, "pricing", "blog", numeric) == "unsupported_number"
    assert corrective_retry_improves("topic_echo", question, long, question, None, None)
    assert not corrective_retry_improves("topic_echo", question, question, question, None, None)


def test_style_exemplars_are_voice_and_do_not_change_the_evidence_block():
    hits = [{"id": "train_10049", "text": RAW}]
    claims = [_item("train_10049", CLAIM, SPAN)]
    plain = media_jobs("Monday.com ARR", hits, claims)
    styled = media_jobs(
        "Monday.com ARR",
        hits,
        claims,
        style_for=lambda medium: ["Short voice line."] if medium == "blog" else [],
    )
    assert plain[0]["instruction"] == final_evidence_instruction(MEDIUMS[0], "Monday.com ARR", claims)
    assert styled[0]["instruction"] != plain[0]["instruction"]
    assert "Style examples from train-owned source text." in styled[0]["instruction"]
    assert "They are not evidence for claims." in styled[0]["instruction"]
    assert "[Style 1]\nShort voice line." in styled[0]["instruction"]
    assert styled[0]["evidence_block"] == plain[0]["evidence_block"]
    assert styled[0]["style_applied"] is True
    assert styled[1]["style_applied"] is False
    assert styled[1]["instruction"] == plain[1]["instruction"]
    factual_side = styled[0]["instruction"].split("[Style 1]")[0] + styled[0]["instruction"].split("Short voice line.")[1]
    assert RAW not in factual_side
    assert SPAN not in factual_side
    voice_that_contains_passage = media_jobs(
        "Monday.com ARR",
        hits,
        claims,
        style_for=lambda _medium: [RAW],
    )
    assert RAW in voice_that_contains_passage[0]["instruction"]
    assert voice_that_contains_passage[0]["evidence_block"] == plain[0]["evidence_block"]
    assert "[E1]" in voice_that_contains_passage[0]["instruction"]


LATE_UPDATE = "April 12 is really, really late for an investor update in the post-Covid 19 world"


def test_a_verified_claim_may_repeat_its_short_source_passage():
    hits = [{"id": "gd_4eba33d02cb019ab6f7191ce", "text": LATE_UPDATE}]
    claims = [_item("gd_4eba33d02cb019ab6f7191ce", LATE_UPDATE, LATE_UPDATE, "fact")]
    accepted, rejected = partition_evidence(
        claims,
        hits,
        topic="What does a late investor update indicate about the month it describes?",
    )
    assert rejected == []
    jobs = media_jobs(
        "What does a late investor update indicate about the month it describes?",
        hits,
        accepted,
    )
    assert LATE_UPDATE in jobs[0]["instruction"]
    with pytest.raises(ValueError, match="outside_verified_claims"):
        media_jobs(
            "What does a late investor update indicate about the month it describes?",
            hits,
            accepted,
            relationships=[{
                "use_in_generation": True,
                "prompt_fact": LATE_UPDATE,
                "subject": "April",
                "predicate": "sent",
                "object": "update",
                "qualifiers": [],
            }],
        )


def test_extraction_fitting_drops_whole_trailing_hits_only():
    hits = [
        {"id": "gd_a", "text": "alpha WHOLE stays"},
        {"id": "gd_b", "text": "beta WHOLE stays"},
        {"id": "gd_c", "text": "gamma WHOLE stays"},
    ]

    def count_tokens(user: str) -> int:
        return 100 + 40 * user.count("WHOLE")

    kept, omitted = fit_extraction_hits("Topic", hits, count_tokens, limit=180)
    assert [hit["id"] for hit in kept] == ["gd_a", "gd_b"]
    assert [hit["id"] for hit in omitted] == ["gd_c"]
    assert kept[0]["text"] == hits[0]["text"]
    assert "alpha WHOLE" in extraction_user_prompt("Topic", kept)
    assert "gamma WHOLE" not in extraction_user_prompt("Topic", kept)
