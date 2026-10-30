"""Deterministic factual checks. These are not lexical entailment.

A quantity, role, or speaker result is supported only when the accepted claim
text contains that fact. Anything else is unsupported or not verifiable.
"""

from __future__ import annotations

import re

from host_finetune.numeric_grounding import evaluate_numeric_grounding
from host_finetune.rag_source_copying import evaluate_rag_source_copying, rag_copy_tokens

_PERSON = r"[A-Z][a-z]+(?:['’][A-Z][a-z]+)?(?:\s+[A-Z][a-z]+(?:['’][A-Z][a-z]+)?){1,2}"
_ROLE = (
    r"(?:Chief(?:\s+[A-Z][a-z]+){1,3}|CEO|CPO|CTO|CFO|COO|"
    r"Co-Founder|Founder|President)"
)
_ORG = r"[A-Z][A-Za-z0-9.&'+-]*(?:\s+[A-Z][A-Za-z0-9.&'+-]*){0,3}"
_ROLE_TRIPLE = re.compile(
    rf"(?P<person>{_PERSON}),?\s+(?P<role>{_ROLE})\s+(?:of|at)\s+(?P<org>{_ORG})"
)
_ROLE_NO_ORG = re.compile(rf"(?P<person>{_PERSON}),?\s+(?:is\s+)?(?:the\s+)?(?P<role>{_ROLE})\b")
_LEAK_MARKERS = ("[E1]", "[E2]", "Speaker:", "ALLOWED FACTUAL", "Prompt contract:")
_GRAM = 6


def _fold(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").casefold())


def embeds_topic(answer: str, topic: str) -> bool:
    """True when the full question appears anywhere in the draft."""
    draft = _fold(answer)
    asked = _fold(topic)
    return bool(draft and asked and asked in draft)


def instruction_leak(answer: str) -> bool:
    """True when evidence labels or the prompt contract appear in the draft."""
    text = answer or ""
    return any(marker in text for marker in _LEAK_MARKERS)


def _claim_text(claims: list[dict] | None) -> str:
    return "\n".join((item.get("claim") or "") for item in (claims or []))


def _grams(text: str, width: int = _GRAM) -> set[str]:
    tokens = rag_copy_tokens(text)
    if len(tokens) < width:
        return set()
    return {" ".join(tokens[index:index + width]) for index in range(len(tokens) - width + 1)}


def role_assertions(answer: str, claims: list[dict] | None) -> list[dict]:
    """Person-role-organization lines. A missing organization is not verifiable."""
    evidence = _fold(_claim_text(claims))
    found = []
    covered: list[tuple[int, int]] = []
    source = answer or ""
    for match in _ROLE_TRIPLE.finditer(source):
        covered.append((match.start(), match.end()))
        person = match.group("person")
        role = match.group("role")
        org = match.group("org")
        supported = bool(evidence) and all(part.casefold() in evidence for part in (person, role, org))
        found.append({
            "text": match.group(0),
            "person": person,
            "role": role,
            "organization": org,
            "verdict": "supported" if supported else "unsupported",
            "reason": "claim_contains_person_role_and_organization" if supported else "identity_not_in_accepted_claims",
        })
    for match in _ROLE_NO_ORG.finditer(source):
        if any(start <= match.start() and match.end() <= end for start, end in covered):
            continue
        found.append({
            "text": match.group(0),
            "person": match.group("person"),
            "role": match.group("role"),
            "organization": None,
            "verdict": "not_verifiable",
            "reason": "role_without_organization",
        })
    return found


def speaker_attribution(answer: str, claims: list[dict] | None) -> dict:
    """Quoted-claim overlap without the speaker's name is an omission."""
    quoted = []
    for claim in claims or []:
        speakers = [str(name) for name in (claim.get("quoted_speakers") or []) if name]
        if speakers:
            quoted.append((claim.get("claim") or "", speakers))
    if not quoted:
        return {
            "applicable": False,
            "status": "not_applicable",
            "pass": True,
            "conflicts": [],
        }
    answer_grams = _grams(answer)
    conflicts = []
    named_without_span = False
    draft = answer or ""
    for claim_text, speakers in quoted:
        overlap = bool(_grams(claim_text) & answer_grams)
        named = any(_fold(name) in _fold(draft) for name in speakers)
        if overlap and not named:
            conflicts.append({
                "verdict": "unsupported",
                "reason": "quoted_span_without_speaker",
                "speakers": speakers,
            })
        elif named and not overlap:
            named_without_span = True
    if conflicts:
        status = "omitted"
        passed = False
    elif named_without_span:
        status = "not_verifiable"
        passed = False
    elif any(_grams(text) & answer_grams for text, _speakers in quoted):
        status = "supported"
        passed = True
    else:
        status = "not_verifiable"
        passed = False
    return {
        "applicable": True,
        "status": status,
        "pass": passed,
        "conflicts": conflicts,
        "method": "quoted_span_and_speaker_name_not_entailment",
    }


def assess_factual(
    answer: str,
    topic: str,
    claims: list[dict] | None,
    *,
    evidence_status: str = "verified",
    hits: list[dict] | None = None,
    retrieval_on: bool = True,
) -> dict:
    """Score one draft. Lexical overlap is not an input and is not a result."""
    claims = claims or []
    numeric = evaluate_numeric_grounding(answer, topic, claims, enabled=True)
    roles = role_assertions(answer, claims)
    speaker = speaker_attribution(answer, claims)
    copying = evaluate_rag_source_copying(answer, hits, enabled=retrieval_on)
    accepted = evidence_status == "verified" and bool(claims)
    unsupported_roles = [item for item in roles if item["verdict"] == "unsupported"]
    unverifiable_roles = [item for item in roles if item["verdict"] == "not_verifiable"]
    reasons = []
    if not accepted:
        reasons.append("no_accepted_factual_evidence")
    if numeric.get("pass") is False:
        reasons.append("unsupported_numeric")
    if unsupported_roles:
        reasons.append("unsupported_role")
    if unverifiable_roles:
        reasons.append("role_not_verifiable")
    if speaker.get("applicable") and not speaker.get("pass"):
        reasons.append("speaker_" + speaker["status"])
    if copying.get("applicable") and not copying.get("pass"):
        reasons.append("retrieval_source_copy")
    return {
        "method": "deterministic_quantity_role_speaker_not_lexical_entailment",
        "not_lexical_entailment": True,
        "evidence_status": evidence_status,
        "accepted_claim_count": len(claims),
        "numeric": numeric,
        "role_assertions": roles,
        "speaker_attribution": speaker,
        "retrieval_source_copying": copying,
        "pass": not reasons,
        "reasons": reasons,
    }


def grounding_regression(before: dict, after: dict) -> bool:
    """True when the retry adds a copying, numeric, role, or speaker failure."""
    before_copy = (before.get("retrieval_source_copying") or {}).get("pass")
    after_copy = (after.get("retrieval_source_copying") or {}).get("pass")
    if before_copy and after_copy is False:
        return True

    def numbers(report: dict) -> set[str]:
        numeric = report.get("numeric") or {}
        return {item.get("canonical") for item in (numeric.get("unsupported") or [])}

    if numbers(after) - numbers(before):
        return True

    def roles(report: dict) -> set[tuple[str, str, str | None]]:
        return {
            (item.get("person"), item.get("role"), item.get("organization"))
            for item in (report.get("role_assertions") or [])
            if item.get("verdict") != "supported"
        }

    if roles(after) - roles(before):
        return True
    before_speaker = (before.get("speaker_attribution") or {}).get("status")
    after_speaker = (after.get("speaker_attribution") or {}).get("status")
    if after_speaker in {"omitted", "not_verifiable"} and before_speaker not in {"omitted", "not_verifiable"}:
        return True
    return False
