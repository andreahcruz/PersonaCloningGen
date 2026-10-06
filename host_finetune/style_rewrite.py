"""Base grounded draft, then one Relabel style rewrite.

An invalid base draft stops the cell. A failed rewrite falls back to the
base draft. There is no retry and no repair.
"""
from __future__ import annotations

from host_finetune.evidence_capsule import introduced_names
from host_finetune.grounding_repair import copy_failed, numeric_failed
from host_finetune.numeric_grounding import extract_quantities
from host_finetune.relabel_continuations import sentence_final

MEDIUM_NAME = {
    "blog": "blog post",
    "linkedin": "LinkedIn post",
    "x": "X post",
    "talk": "talk",
}
LEAK_MARKERS = (
    "do not mention the evidence",
    "do not imitate jason lemkin",
    "style-only",
    "do not explain the rewrite",
    "you are performing",
    "factual constraints",
    "retrieval process",
    "[e1]",
    "allowed factual evidence",
    "these instructions",
)


def claim_lines(claims: list[dict]) -> str:
    lines = [f"- {item['claim'].strip()}" for item in claims if (item.get("claim") or "").strip()]
    return "\n".join(lines) if lines else "- (none)"


def grounded_base_prompt(medium: str, topic: str, claims: list[dict]) -> str:
    """Factual draft. No persona request, raw passages, or support spans."""
    name = MEDIUM_NAME[medium]
    return (
        f"Write a {name} about: {topic}\n\n"
        "Use only the factual evidence below for factual specifics.\n\n"
        f"{claim_lines(claims)}\n\n"
        "Rules:\n"
        "- Do not add factual numbers, dates, dollar values, percentages, counts, "
        "company-specific metrics, or other concrete factual details that are not supplied above.\n"
        "- If evidence is sparse, stay general rather than inventing details.\n"
        "- Produce a complete useful response.\n"
        "- Do not imitate Jason Lemkin specifically.\n"
        "- Do not mention the evidence or retrieval process."
    )


def style_rewrite_prompt(medium: str, topic: str, draft: str, claims: list[dict]) -> str:
    """Full-strength Relabel sees the draft and the accepted claims only."""
    name = MEDIUM_NAME[medium]
    return (
        "You are performing a STYLE-ONLY rewrite.\n\n"
        f"Rewrite the draft so it sounds like Jason Lemkin and fits the requested {name}.\n\n"
        "Preserve the factual content of the draft.\n\n"
        "You may change:\n"
        "- wording\n"
        "- rhythm\n"
        "- sentence structure\n"
        "- emphasis\n"
        "- rhetorical style\n"
        "- formatting appropriate to the medium\n\n"
        "You must NOT introduce:\n"
        "- new facts\n"
        "- new examples\n"
        "- new companies or people\n"
        "- new dates\n"
        "- new percentages\n"
        "- new dollar values\n"
        "- new counts\n"
        "- new ARR/NRR/growth metrics\n"
        "- new quantitative claims\n\n"
        f"Topic: {topic}\n\n"
        "Draft:\n"
        f"{draft.strip()}\n\n"
        "Accepted factual constraints:\n"
        f"{claim_lines(claims)}\n\n"
        "Do not explain the rewrite.\n"
        "Return only the rewritten content."
    )


def instruction_leak(text: str) -> bool:
    folded = (text or "").casefold()
    return any(marker in folded for marker in LEAK_MARKERS)


def stage_failure(text: str, numeric: dict, copy: dict) -> str | None:
    """None means the stage can be deployed."""
    if not (text or "").strip():
        return "empty"
    if numeric_failed(numeric):
        return "unsupported_numeric_grounding"
    if copy_failed(copy):
        return "rag_source_copying"
    if instruction_leak(text):
        return "instruction_leak"
    if not sentence_final(text):
        return "mid_sentence"
    return None


def select_deployment(
    base_answer: str,
    base_failure: str | None,
    rewrite_answer: str | None,
    rewrite_failure: str | None,
) -> dict:
    """Invalid base stops the cell. A failed rewrite deploys the base draft."""
    if base_failure:
        if rewrite_answer is not None:
            raise ValueError("an invalid base draft cannot enter the style stage")
        return {
            "deployment_output": None,
            "deployment_stage": None,
            "style_rewrite_accepted": False,
            "fallback_reason": None,
            "stopped_reason": base_failure,
        }
    if rewrite_answer is None:
        raise ValueError("a valid base draft requires one style rewrite")
    if rewrite_failure:
        return {
            "deployment_output": base_answer,
            "deployment_stage": "grounded_base_draft",
            "style_rewrite_accepted": False,
            "fallback_reason": rewrite_failure,
            "stopped_reason": None,
        }
    return {
        "deployment_output": rewrite_answer,
        "deployment_stage": "relabel_style_rewrite",
        "style_rewrite_accepted": True,
        "fallback_reason": None,
        "stopped_reason": None,
    }


def content_changes(base: str, rewrite: str) -> dict:
    """Quantities and names that differ. This is a diagnostic, not a gate."""
    base_quantities = extract_quantities(base)
    rewrite_quantities = extract_quantities(rewrite)
    base_canonical = {item["canonical"] for item in base_quantities}
    rewrite_canonical = {item["canonical"] for item in rewrite_quantities}
    base_names = set(introduced_names(base))
    rewrite_names = set(introduced_names(rewrite))
    return {
        "quantities_added": _surfaces(rewrite_quantities, rewrite_canonical - base_canonical),
        "quantities_removed": _surfaces(base_quantities, base_canonical - rewrite_canonical),
        "names_added": sorted(rewrite_names - base_names),
        "names_removed": sorted(base_names - rewrite_names),
    }


def _surfaces(items: list[dict], canonicals: set[str]) -> list[str]:
    found = []
    seen = set()
    for item in items:
        canonical = item["canonical"]
        if canonical in canonicals and canonical not in seen:
            seen.add(canonical)
            found.append(item["value"])
    return found
