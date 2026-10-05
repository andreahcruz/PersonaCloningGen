"""One adapter-off repair follows a numeric-grounding failure."""
from contextlib import contextmanager

import pytest

from host_finetune.compare_adapters import MEDIUMS, prompt_for
from host_finetune.grounding_repair import (
    repair_under_disabled_adapter,
    repair_user_prompt,
    response_to_gates,
    settle_cell,
)
from host_finetune.numeric_grounding import evaluate_numeric_grounding
from host_finetune.rag_source_copying import RAG_SOURCE_COPY_THRESHOLD, evaluate_rag_source_copying


RAW = (
    "We last looked in on Monday.com at two hundred forty million ARR but it grew "
    "so incredibly quickly and so effectively it is worth checking in again."
)
SPAN = "Customer Count Up Just 34%."
ACCEPTED = "Customer count is up 34%."
REJECTED = "Monday.com is growing 68% at $550m in ARR."


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


def _completion(answer, *, copy_pass=True, numeric_pass=True, seed=42):
    return {
        "answer": answer,
        "seed": seed,
        "report": {"applicable": True, "pass": copy_pass, "max_contiguous_words": 2 if copy_pass else 20},
        "numeric": {
            "applicable": True,
            "pass": numeric_pass,
            "failure": None if numeric_pass else "unsupported_numeric_grounding",
            "unsupported": [] if numeric_pass else [{"value": "$250M", "canonical": "usd:250000000"}],
        },
    }


def test_unsupported_number_triggers_one_repair_and_a_clean_draft_does_not():
    failing = {"applicable": True, "pass": False}
    passing = {"applicable": True, "pass": True}
    assert response_to_gates(passing, failing) == "repair"
    assert response_to_gates(passing, passing) == "accept"
    with pytest.raises(ValueError, match="no more than one repair"):
        settle_cell(_completion("draft", numeric_pass=False), repair={"answer": "x", "repair_index": 2, "report": {}, "numeric": {}})


def test_repair_prompt_lists_the_bad_values_and_only_accepted_evidence():
    prompt = repair_user_prompt(
        "Monday.com at $550,000,000 in ARR",
        "Monday is now worth $250M and grew 96%.",
        [{"claim": ACCEPTED, "support_span": SPAN, "source_id": "train_10049"}],
        ["$250M", "96%"],
    )
    assert "$250M" in prompt
    assert "96%" in prompt
    assert ACCEPTED in prompt
    assert REJECTED not in prompt
    assert RAW not in prompt
    assert SPAN not in prompt
    assert "distance" not in prompt.casefold()
    assert "Do NOT explain your edits." in prompt


def test_adapter_is_disabled_for_the_repair_call():
    model = _AdapterModel()
    seen = {}

    def _edit():
        seen["during"] = model.adapter_enabled
        return "at that stage"

    assert repair_under_disabled_adapter(model, _edit) == "at that stage"
    assert seen["during"] is False
    assert model.adapter_enabled is True


def test_generic_repair_can_pass_and_a_new_number_fails():
    claims = [{"claim": ACCEPTED}]
    topic = "Monday.com"
    generic = evaluate_numeric_grounding(
        "Monday was at that stage and the customer count is up 34%.",
        topic,
        claims,
        enabled=True,
    )
    assert generic["pass"] is True
    invented = evaluate_numeric_grounding(
        "Monday was at that stage and NRR is 110%.",
        topic,
        claims,
        enabled=True,
    )
    assert invented["pass"] is False
    assert invented["failure"] == "unsupported_numeric_grounding"


def test_source_copy_gate_runs_on_the_repaired_text():
    report = evaluate_rag_source_copying(RAW, [{"id": "train_10049", "text": RAW}], enabled=True)
    assert report["pass"] is False
    assert report["max_contiguous_words"] >= RAG_SOURCE_COPY_THRESHOLD


def test_raw_relabel_text_stays_separate_from_the_deployment_text():
    first = _completion("Monday is worth $250M.", numeric_pass=False)
    repair = _completion("Monday was at that stage.", numeric_pass=True, copy_pass=True)
    repair["repair_index"] = 1
    saved = settle_cell(first, repair=repair)
    assert saved["raw_model_output"] == "Monday is worth $250M."
    assert saved["answer"] == "Monday is worth $250M."
    assert saved["deployment_output"] == "Monday was at that stage."
    assert saved["deployment_accepted"] is True
    assert saved["repair_occurred"] is True
    saved["deployment_output"] = "changed later"
    assert saved["raw_model_output"] == "Monday is worth $250M."


def test_a_failed_repair_is_not_deployed_and_retrieval_off_is_unchanged():
    first = _completion("Monday is worth $250M.", numeric_pass=False)
    repair = _completion("Monday is worth $15B.", numeric_pass=False)
    repair["repair_index"] = 1
    saved = settle_cell(first, repair=repair)
    assert saved["deployment_output"] is None
    assert saved["deployment_accepted"] is False
    assert saved["raw_model_output"] == "Monday is worth $250M."
    assert "ALLOWED FACTUAL EVIDENCE" not in prompt_for(MEDIUMS[0], "Hiring")
    off = {"applicable": False, "pass": None}
    assert response_to_gates(off, off) == "accept"
