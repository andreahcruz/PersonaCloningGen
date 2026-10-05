"""Deterministic numeric grounding is separate from the source-copy gate."""
from host_finetune.evidence_capsule import final_evidence_instruction, partition_evidence
from host_finetune.compare_adapters import MEDIUMS, prompt_for
from host_finetune.numeric_grounding import (
    evaluate_numeric_grounding,
    gates_failed,
    numbers_in,
)
from host_finetune.rag_source_copying import RAG_SOURCE_COPY_THRESHOLD, evaluate_rag_source_copying, retry_seed


SPAN_34 = "Customer Count Up “Just” 34%."
SOURCE_34 = "Revenue up 91%, Customer Count Up “Just” 34%. The rest of the note stays here."


def _item(source_id, claim, span, claim_type="number"):
    return {
        "source_id": source_id,
        "claim": claim,
        "support_span": span,
        "claim_type": claim_type,
    }


def test_supported_34_percent_claim_passes_without_an_llm_verdict():
    hits = [{"id": "train_10049", "text": SOURCE_34}]
    accepted, rejected = partition_evidence(
        [_item("train_10049", "Customer count is up 34%.", SPAN_34)],
        hits,
        topic="Monday.com",
    )
    assert rejected == []
    assert accepted[0]["claim"] == "Customer count is up 34%."


def test_question_titles_cannot_become_assertions():
    hits = [
        {"id": "train_a", "text": "Can an 8-person startup sell to a CIO?"},
        {
            "id": "train_b",
            "text": "Why do founders of startups refuse to sell them for hundreds of millions of dollars?",
        },
    ]
    _accepted, rejected = partition_evidence(
        [
            _item(
                "train_a",
                "An 8-person startup can sell to a CIO.",
                "Can an 8-person startup sell to a CIO?",
                "fact",
            ),
            _item(
                "train_b",
                "Founders of startups refuse to sell them for hundreds of millions of dollars.",
                "Why do founders of startups refuse to sell them for hundreds of millions of dollars?",
                "fact",
            ),
        ],
        hits,
    )
    assert [item["reason"] for item in rejected] == [
        "interrogative_assertion",
        "interrogative_assertion",
    ]


def test_claim_number_missing_from_the_span_is_rejected():
    hits = [{"id": "train_9156", "text": "Monday’s growth has slowed a little bit, but the growth is still incredible."}]
    _accepted, rejected = partition_evidence(
        [
            _item(
                "train_9156",
                "Monday.com is growing 68% at $550m in ARR.",
                "Monday’s growth has slowed a little bit, but the growth is still incredible.",
                "fact",
            )
        ],
        hits,
        topic="Monday.com at $550,000,000 in ARR",
    )
    assert rejected[0]["reason"] == "numeric_not_in_span"


def test_allowed_91_percent_passes_and_unallowed_110_fails():
    claims = [_item("train_10049", "Monday.com was growing 91% year-over-year.", "growing 91%")]
    grounded = evaluate_numeric_grounding(
        "Monday is still growing 91 percent.",
        "Monday.com",
        claims,
        enabled=True,
    )
    assert grounded["pass"] is True
    invented = evaluate_numeric_grounding(
        "And 110% NRR is the figure to remember.",
        "Monday.com",
        claims,
        enabled=True,
    )
    assert invented["pass"] is False
    assert invented["failure"] == "unsupported_numeric_grounding"
    assert invented["unsupported"][0]["canonical"] == "pct:110"
    assert "110%" in invented["unsupported"][0]["sentence"]


def test_dollar_magnitudes_normalize_together():
    assert numbers_in("$1B") == numbers_in("$1 billion") == numbers_in("$1,000,000,000")
    claims = [_item("train_14595", "Monday.com crossed $1,000,000,000 in ARR.", "$1,000,000,000 in ARR")]
    report = evaluate_numeric_grounding(
        "Monday.com later crossed $1B ARR, about $1 billion.",
        "Monday.com",
        claims,
        enabled=True,
    )
    assert report["pass"] is True


def test_stylistic_commentary_without_new_numbers_is_not_blocked():
    claims = [_item("train_10049", "10+ seat customers were 72% of revenue.", "72% of revenue")]
    report = evaluate_numeric_grounding(
        "The upmarket move is the part worth copying. It is a brand story as much as a sales story.",
        "Monday.com",
        claims,
        enabled=True,
    )
    assert report["pass"] is True
    assert report["unsupported"] == []


def test_source_copy_gate_still_rejects_a_long_original_passage():
    raw = (
        "We last looked in on Monday.com at two hundred forty million ARR but it grew "
        "so incredibly quickly and so effectively it is worth checking in again."
    )
    report = evaluate_rag_source_copying(raw, [{"id": "train_10049", "text": raw}], enabled=True)
    assert report["pass"] is False
    assert report["max_contiguous_words"] >= RAG_SOURCE_COPY_THRESHOLD


def test_retrieval_off_prompt_and_numeric_gate_stay_inactive():
    assert "ALLOWED FACTUAL EVIDENCE" not in prompt_for(MEDIUMS[0], "Hiring")
    assert "Do not add factual numbers" not in prompt_for(MEDIUMS[0], "Hiring")
    off = evaluate_numeric_grounding("Revenue grew 110%.", "Hiring", [], enabled=False)
    assert off["applicable"] is False
    assert off["pass"] is None
    assert gates_failed({"applicable": False, "pass": None}, off) is False


def test_final_closer_is_only_the_writing_request():
    claims = [_item("train_10049", "Revenue grew 91%.", "91%")]
    instruction = final_evidence_instruction(MEDIUMS[1], "Monday.com", claims)
    assert instruction.endswith("Now write the requested LinkedIn post about: Monday.com")
    assert "Do not mention the evidence, retrieval system, or these instructions." not in instruction
    assert "these instructions" not in instruction
    assert "Use only the factual specifics provided in ALLOWED FACTUAL EVIDENCE." in instruction
    assert "[E1] Revenue grew 91%." in instruction
    assert instruction.index("Do not add factual numbers") < instruction.index("\nALLOWED FACTUAL EVIDENCE\n")
    assert instruction.index("ALLOWED FACTUAL EVIDENCE") < instruction.index("Now write the requested")


def test_deployment_records_unsupported_numbers_without_retry_or_repair():
    from host_finetune.compare_adapters import final_response, kept_draft

    copy_pass = {"applicable": True, "pass": True, "max_contiguous_words": 3}
    copy_fail = {"applicable": True, "pass": False, "max_contiguous_words": 20}
    numeric_fail = {"applicable": True, "pass": False, "unsupported": [{"value": "$500M"}]}
    assert final_response(copy_pass) == "keep"
    assert final_response(copy_fail) == "copy_retry"
    first = {
        "answer": "Monday is worth $500M.",
        "report": copy_pass,
        "numeric": numeric_fail,
    }
    kept = kept_draft(first, None)
    assert kept["chosen"]["answer"] == "Monday is worth $500M."
    assert kept["source_copy_flag"] is False
    still_copying = {
        "answer": "A second pasted draft.",
        "report": copy_fail,
        "numeric": {"applicable": True, "pass": True, "unsupported": []},
    }
    flagged = kept_draft({**first, "report": copy_fail}, still_copying)
    assert flagged["chosen"]["answer"] == "Monday is worth $500M."
    assert flagged["source_copy_flag"] is True
    clean_retry = {
        "answer": "A clean draft.",
        "report": copy_pass,
        "numeric": {"applicable": True, "pass": True, "unsupported": []},
    }
    replaced = kept_draft({**first, "report": copy_fail}, clean_retry)
    assert replaced["chosen"]["answer"] == "A clean draft."
    assert replaced["attempt"] == 2
    assert replaced["source_copy_flag"] is False


def test_repair_helper_remains_available_off_the_deployment_path():
    from host_finetune.grounding_repair import response_to_gates

    copy_pass = {"applicable": True, "pass": True}
    numeric_fail = {"applicable": True, "pass": False, "failure": "unsupported_numeric_grounding"}
    assert response_to_gates(copy_pass, numeric_fail) == "repair"
    assert response_to_gates({"applicable": True, "pass": False}, {"applicable": True, "pass": True}) == "copy_retry"
    assert response_to_gates(copy_pass, {"applicable": True, "pass": True}) == "accept"
    assert retry_seed(42) == 10042
