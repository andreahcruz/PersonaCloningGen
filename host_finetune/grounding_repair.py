"""One factual edit of a Relabel draft. The adapter stays off.

This is not another sample. A numeric-grounding failure is repaired once,
then both deployment gates run again.
"""
from __future__ import annotations

from host_finetune.evidence_capsule import verified_claim_block

REPAIR_MAX_NEW_TOKENS = 1024
REPAIR_DO_SAMPLE = False
BASE_DIAGNOSTIC_CELLS = {
    ("gold_001", "blog"),
    ("gold_002", "blog"),
    ("gold_002", "linkedin"),
    ("gold_002", "talk"),
}


def numeric_failed(report: dict | None) -> bool:
    return bool(report and report.get("applicable") and not report.get("pass"))


def copy_failed(report: dict | None) -> bool:
    return bool(report and report.get("applicable") and not report.get("pass"))


def response_to_gates(copy_report: dict, numeric_report: dict) -> str:
    """Numeric failures are repaired. A copy-only failure keeps the one seed retry."""
    if numeric_failed(numeric_report):
        return "repair"
    if copy_failed(copy_report):
        return "copy_retry"
    return "accept"


def unsupported_surfaces(numeric_report: dict) -> list[str]:
    values = []
    seen = set()
    for item in numeric_report.get("unsupported") or []:
        surface = (item.get("value") or "").strip()
        if surface and surface not in seen:
            seen.add(surface)
            values.append(surface)
    return values


def repair_user_prompt(topic: str, draft: str, claims: list[dict], unsupported: list[str]) -> str:
    """Accepted claims and the bad quantities only. Raw passages are not included."""
    listed = "\n".join(f"- {value}" for value in unsupported) or "- (none)"
    return (
        "You are repairing a generated draft for factual grounding.\n"
        "Preserve the wording, organization, tone, and style as much as possible.\n\n"
        "The following numeric factual values are unsupported:\n"
        f"{listed}\n\n"
        "Allowed factual evidence:\n"
        f"{verified_claim_block(claims)}\n\n"
        f"Topic: {topic}\n\n"
        "Draft:\n"
        f"{draft.strip()}\n\n"
        "Edit only what is necessary to remove or correct unsupported factual specifics.\n"
        "You may:\n"
        "- remove an unsupported number\n"
        "- replace an unsupported number with a supported number when the intended claim is clearly supported\n"
        "- rewrite a sentence generically if no supported numeric substitute exists\n"
        "Examples:\n"
        "'$250M' may become 'at that stage' if no supported amount belongs there.\n"
        "'grew 96%' may become 'grew rapidly' if no supported growth value belongs there.\n"
        "Do NOT invent replacement facts.\n"
        "Do NOT introduce any numeric value that does not occur in the allowed evidence or topic.\n"
        "Do NOT add new examples.\n"
        "Do NOT add new factual claims.\n"
        "Do NOT imitate or intensify Jason Lemkin's style.\n"
        "Do NOT explain your edits.\n"
        "Return only the corrected draft."
    )


def repair_under_disabled_adapter(model, repair_fn):
    """Run the edit only while the Relabel adapter is off."""
    from host_finetune.evidence_capsule import require_adapter_toggle

    require_adapter_toggle(model)
    with model.disable_adapter():
        if getattr(model, "adapter_enabled", False):
            raise RuntimeError("adapter still enabled during repair")
        return repair_fn()


def settle_cell(first: dict, *, repair: dict | None = None, copy_retry: dict | None = None) -> dict:
    """Keep the raw Relabel draft. Deployment text is stored separately."""
    if repair is not None and copy_retry is not None:
        raise ValueError("numeric repair and a seed retry are not combined")
    if repair is not None and not numeric_failed(first["numeric"]):
        raise ValueError("a numerically grounded draft is not repaired")
    if repair is not None and repair.get("repair_index", 1) != 1:
        raise ValueError("no more than one repair")
    raw = first["answer"]
    if repair is not None:
        repair_ok = not numeric_failed(repair["numeric"]) and not copy_failed(repair["report"])
        deployment = repair["answer"] if repair_ok else None
        judged_copy = repair["report"]
        judged_numeric = repair["numeric"]
    elif copy_retry is not None:
        retry_ok = not numeric_failed(copy_retry["numeric"]) and not copy_failed(copy_retry["report"])
        deployment = copy_retry["answer"] if retry_ok else None
        judged_copy = copy_retry["report"]
        judged_numeric = copy_retry["numeric"]
    else:
        original_ok = not numeric_failed(first["numeric"]) and not copy_failed(first["report"])
        deployment = raw if original_ok else None
        judged_copy = first["report"]
        judged_numeric = first["numeric"]
    return {
        "raw_model_output": raw,
        "deployment_output": deployment,
        "deployment_accepted": deployment is not None,
        "answer": raw,
        "repair_occurred": repair is not None,
        "judged_copy": judged_copy,
        "judged_numeric": judged_numeric,
    }
