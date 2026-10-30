"""Delete or narrow unsupported spans in a finished draft.

The original prose is kept when the edit would add a new factual or copying
failure, or when it would drop more than 15 percent of the words. A rejected
edit is recorded as a warning beside the draft. The warning is not inserted
into the prose.
"""

from __future__ import annotations

import re

from host_finetune.evidence_capsule import repeats_topic
from host_finetune.factual_reliability import (
    _fold,
    _grams,
    assess_factual,
    embeds_topic,
    grounding_regression,
    role_assertions,
    speaker_attribution,
)
from host_finetune.numeric_grounding import extract_quantities
from host_finetune.rag_source_copying import evaluate_rag_source_copying
from host_finetune.relabel_continuations import sentence_final
from host_finetune.unified_eval import copying_gate, format_gate

SPAN_REPAIR = "span-repair-v1"
SUBSTANTIVE_ELIGIBILITY = "substantive-quality-v1"
WORD_RETENTION = 0.85


def _evidence_text(claims: list[dict] | None) -> str:
    return _fold("\n".join((item.get("claim") or "") for item in (claims or [])))


def _supported_people(claims: list[dict] | None) -> set[str]:
    names = set()
    for claim in claims or []:
        for name in claim.get("quoted_speakers") or []:
            if name:
                names.add(str(name))
    return names


def _person_supported(person: str, claims: list[dict] | None) -> bool:
    folded = _fold(person)
    if not folded:
        return False
    if folded in _evidence_text(claims):
        return True
    return any(_fold(name) == folded for name in _supported_people(claims))


def _expand_quantity(answer: str, start: int, end: int) -> tuple[int, int]:
    """Include a glued plus sign and a preposition that would otherwise dangle."""
    if end < len(answer) and answer[end] == "+":
        end += 1
    prefix = answer[:start]
    match = re.search(r"(?:^|\s)(for|over|under|about|around|at|than)\s+$", prefix, re.IGNORECASE)
    follower = answer[end:].lstrip()
    if match and re.match(r"(?:and|or|that|who|which|annually|today|,|\.)", follower, re.IGNORECASE):
        start = match.start(1)
    return start, end


def _broken_span(before: str, after: str) -> bool:
    """True when deletion leaves a preposition or plus sign with nothing attached."""
    pattern = re.compile(
        r"\b(?:for|over|under|about|around|at|than)\s+(?:and|or|that|who|which|annually|today|[,.])"
        r"|\s\+\s|\bhas\s+\+",
        re.IGNORECASE,
    )
    return bool(pattern.search(after)) and not bool(pattern.search(before))


def _edits(answer: str, claims: list[dict] | None, numeric: dict) -> list[tuple[int, int, str, str]]:
    """Spans to replace, as start, end, replacement, reason."""
    unsupported = {item.get("canonical") for item in (numeric.get("unsupported") or [])}
    edits = []
    for item in extract_quantities(answer):
        if item.get("canonical") in unsupported and item.get("start") is not None:
            start, end = _expand_quantity(answer, item["start"], item["end"])
            edits.append((start, end, "", "unsupported_quantity"))
    occupied = [(start, end) for start, end, _replacement, _reason in edits]
    for assertion in role_assertions(answer, claims):
        if assertion.get("verdict") == "supported":
            continue
        surface = assertion.get("text") or ""
        start = answer.find(surface)
        if start < 0:
            continue
        end = start + len(surface)
        if any(start < stop and end > begin for begin, stop in occupied):
            continue
        person = assertion.get("person") or ""
        replacement = person if _person_supported(person, claims) else ""
        reason = "unsupported_role" if assertion.get("verdict") == "unsupported" else "role_not_verifiable"
        edits.append((start, end, replacement, reason))
        occupied.append((start, end))
    return edits


def _apply(answer: str, edits: list[tuple[int, int, str, str]]) -> str:
    updated = answer
    for start, end, replacement, _reason in sorted(edits, key=lambda item: item[0], reverse=True):
        updated = updated[:start] + replacement + updated[end:]
    return updated


def _tidy(text: str) -> str:
    cleaned = re.sub(r"[ \t]{2,}", " ", text)
    cleaned = re.sub(r" +([,.;:!?])", r"\1", cleaned)
    cleaned = re.sub(r"\( +", "(", cleaned)
    cleaned = re.sub(r" +\)", ")", cleaned)
    return cleaned.strip()


def _insert_speaker(answer: str, claims: list[dict] | None) -> tuple[str, list[str]]:
    report = speaker_attribution(answer, claims)
    if report.get("status") != "omitted":
        return answer, []
    inserted = []
    updated = answer
    for claim in claims or []:
        speakers = [str(name) for name in (claim.get("quoted_speakers") or []) if name]
        if not speakers or any(_fold(name) in _fold(updated) for name in speakers):
            continue
        overlap = _grams(claim.get("claim") or "") & _grams(updated)
        placed = None
        for gram in overlap:
            index = _fold(updated).find(gram)
            if index >= 0 and (placed is None or index < placed):
                placed = index
        if placed is None:
            continue
        updated = updated[:placed] + speakers[0] + " " + updated[placed:]
        inserted.append(speakers[0])
    return updated, inserted


def _word_ratio(before: str, after: str) -> float:
    base = len((before or "").split())
    if base == 0:
        return 1.0
    return len((after or "").split()) / base


def claim_overlap_count(answer: str, claims: list[dict] | None) -> int:
    """Accepted claims that still share a 6-word span with the draft."""
    answer_grams = _grams(answer)
    count = 0
    for claim in claims or []:
        if _grams(claim.get("claim") or "") & answer_grams:
            count += 1
    return count


def substantive_eligible(checks: dict) -> bool:
    """Version substantive-quality-v1. Lexical overlap is not an input."""
    return bool(
        checks.get("completion")
        and checks.get("format")
        and checks.get("training_copy")
        and checks.get("retrieval_copy")
        and not checks.get("topic_echo")
        and checks.get("strict_pass")
        and checks.get("word_retention", 0) >= WORD_RETENTION
        and (checks.get("accepted_claims", 0) == 0 or checks.get("claim_overlap", 0) > 0)
    )


def repair_draft(
    answer: str,
    topic: str,
    claims: list[dict] | None,
    *,
    hits: list[dict] | None = None,
    prompt: str = "",
    copy_index=None,
    evidence_status: str = "verified",
    medium: str = "",
) -> dict:
    """Return the original draft unless a smaller span edit passes every guard."""
    claims = claims or []
    before = assess_factual(
        answer, topic, claims, evidence_status=evidence_status, hits=hits, retrieval_on=True
    )
    warning = list(before["reasons"])
    base = {
        "contract": SPAN_REPAIR,
        "applied": False,
        "answer": answer,
        "original_answer": answer,
        "edits": [],
        "warning": warning,
        "rejection": None,
        "word_retention": 1.0,
    }
    if before["pass"]:
        base["warning"] = []
        base["rejection"] = "already_supported"
        return base
    edits = _edits(answer, claims, before["numeric"])
    edited, speakers = _insert_speaker(_tidy(_apply(answer, edits)), claims)
    edited = _tidy(edited)
    edit_log = [{"start": start, "end": end, "replacement": replacement, "reason": reason} for start, end, replacement, reason in edits]
    edit_log.extend({"reason": "speaker_name", "speaker": name} for name in speakers)
    if edited == answer or not edit_log:
        base["rejection"] = "no_span_edit"
        return base
    after = assess_factual(
        edited, topic, claims, evidence_status=evidence_status, hits=hits, retrieval_on=True
    )
    retention = _word_ratio(answer, edited)
    training = copying_gate({"answer": edited, "prompt": prompt}, copy_index)
    formatting = format_gate({"answer": edited, "medium": medium, "prompt": prompt})
    original_format = format_gate({"answer": answer, "medium": medium, "prompt": prompt})
    reasons = []
    if grounding_regression(before, after):
        reasons.append("new_grounding_failure")
    if not after["retrieval_source_copying"].get("pass", True):
        reasons.append("retrieval_source_copy")
    if copy_index is not None and not training.get("pass", False):
        reasons.append("training_copy")
    if retention < WORD_RETENTION:
        reasons.append("substantive_loss")
    if (answer or "").strip() and sentence_final(answer) and not sentence_final(edited):
        reasons.append("incomplete_sentence")
    if original_format.get("pass") and not formatting.get("pass"):
        reasons.append("format_regression")
    if (embeds_topic(edited, topic) or repeats_topic(edited, topic)) and not (
        embeds_topic(answer, topic) or repeats_topic(answer, topic)
    ):
        reasons.append("new_topic_echo")
    if _broken_span(answer, edited):
        reasons.append("broken_span")
    improved = set(after["reasons"]) < set(before["reasons"])
    if reasons or not improved:
        if not improved:
            reasons.append("not_improved")
        base["rejection"] = reasons
        base["word_retention"] = round(retention, 4)
        return base
    return {
        "contract": SPAN_REPAIR,
        "applied": True,
        "answer": edited,
        "original_answer": answer,
        "edits": edit_log,
        "warning": list(after["reasons"]),
        "rejection": None,
        "word_retention": round(retention, 4),
        "reliable": after["pass"],
        "factual": after,
    }


def retrieval_copy_pass(answer: str, hits: list[dict] | None) -> bool:
    report = evaluate_rag_source_copying(answer, hits, enabled=True)
    return bool(report.get("pass"))
