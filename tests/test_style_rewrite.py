"""Base draft then one style rewrite. Fallback is the base draft."""
from host_finetune.style_rewrite import (
    content_changes,
    grounded_base_prompt,
    select_deployment,
    style_rewrite_prompt,
)


CLAIMS = [{"claim": "Customer count is up 34%.", "support_span": "Customer Count Up Just 34%.", "source_id": "train_10049"}]
RAW = "We last looked in on Monday.com at two hundred forty million ARR and the prose continues from the retrieved source."


def _pass():
    return {"applicable": True, "pass": True}


def _fail():
    return {"applicable": True, "pass": False, "failure": "unsupported_numeric_grounding"}


def test_prompts_carry_accepted_claims_and_not_the_retrieved_passage():
    base = grounded_base_prompt("blog", "Monday.com ARR", CLAIMS)
    style = style_rewrite_prompt("blog", "Monday.com ARR", "The customer count is up 34%.", CLAIMS)
    assert "Customer count is up 34%." in base
    assert "Customer count is up 34%." in style
    assert "in the style of Jason Lemkin" not in base
    assert "Do not imitate Jason Lemkin specifically." in base
    assert "sounds like Jason Lemkin" in style
    for prompt in (base, style):
        assert RAW not in prompt
        assert "Customer Count Up Just 34%." not in prompt
        assert "train_10049" not in prompt
        assert "[E1]" not in prompt


def test_invalid_base_stops_before_a_rewrite():
    saved = select_deployment("The count is up 34%.", "unsupported_numeric_grounding", None, None)
    assert saved["deployment_output"] is None
    assert saved["style_rewrite_accepted"] is False
    assert saved["stopped_reason"] == "unsupported_numeric_grounding"
    try:
        select_deployment("bad", "rag_source_copying", "styled", None)
    except ValueError as exc:
        assert "cannot enter" in str(exc)
    else:
        raise AssertionError("invalid base was styled")


def test_failed_rewrite_falls_back_once_and_a_clean_rewrite_is_deployed():
    base = "The customer count is up 34%."
    fallback = select_deployment(base, None, "The count is up 125%.", "unsupported_numeric_grounding")
    assert fallback["deployment_output"] == base
    assert fallback["deployment_stage"] == "grounded_base_draft"
    assert fallback["style_rewrite_accepted"] is False
    assert fallback["fallback_reason"] == "unsupported_numeric_grounding"
    accepted = select_deployment(base, None, "Customer count is up 34%. Full stop.", None)
    assert accepted["deployment_output"] == "Customer count is up 34%. Full stop."
    assert accepted["deployment_stage"] == "relabel_style_rewrite"
    assert accepted["style_rewrite_accepted"] is True
    assert accepted["fallback_reason"] is None


def test_content_changes_list_added_quantities_and_names():
    changes = content_changes(
        "Customer count is up 34% at Monday.com.",
        "Customer count is up 34%, and Wix added a $250M round.",
    )
    assert "$250M" in changes["quantities_added"]
    assert "34%" not in changes["quantities_added"]
    assert "Wix" in changes["names_added"]
