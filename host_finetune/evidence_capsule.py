"""Atomic evidence between frozen retrieval and Relabel generation.

The same loaded 4-bit base model extracts claims with the Relabel adapter off,
then checks each claim against its source span. Support spans stay in
provenance. Relabel sees only verified claim text.
"""
from __future__ import annotations

import json
import re

from host_finetune.numeric_grounding import numbers_in

EXTRACT_MAX_NEW_TOKENS = 2560
VERIFY_MAX_NEW_TOKENS = 24
DISTILL_DO_SAMPLE = False
DISTILL_SEED = 0
DISTILL_BACKEND = "same_loaded_4bit_base_adapter_off"
CLAIM_TYPES = ("fact", "number", "comparison", "position", "example")
_CLAIM_TYPE_SET = set(CLAIM_TYPES)
_QUESTION_START = re.compile(
    r"^(what|why|how|who|when|where|which|can|could|would|should|will|do|does|did|is|are|if)\b",
    re.IGNORECASE,
)
_METRIC_TOKENS = {
    "arr", "nrr", "acv", "smb", "ceo", "cio", "cios", "saas", "api", "ai", "us", "sf", "yoy", "fcf",
}
_COMMON_TOKENS = {
    "the", "and", "but", "for", "with", "from", "this", "that", "these", "those",
    "its", "their", "about", "after", "before", "into", "over", "under",
}


class EvidenceParseError(ValueError):
    """The extractor did not return a JSON array of claim objects."""


class EvidenceStageError(RuntimeError):
    """This topic has no usable verified evidence. Do not invent any."""

    def __init__(self, reason: str, record: dict):
        super().__init__(reason)
        self.reason = reason
        self.record = record


def require_adapter_toggle(model) -> None:
    """Refuse a second model when this loaded model cannot disable its adapter."""
    if not hasattr(model, "disable_adapter"):
        raise RuntimeError(
            "The loaded model has no PEFT disable_adapter context. "
            "Refusing to load a second model for evidence distillation."
        )


def normalize_support(text: str) -> str:
    """Whitespace and case only. The span is not rewritten."""
    folded = (text or "").casefold().replace("\u00a0", " ")
    folded = (
        folded.replace("\u2018", "'")
        .replace("\u2019", "'")
        .replace("\u201c", '"')
        .replace("\u201d", '"')
    )
    return re.sub(r"\s+", " ", folded).strip()


def extraction_user_prompt(topic: str, hits: list[dict]) -> str:
    """Information extraction. Raw passages are the input, not the final prompt."""
    blocks = []
    for hit in hits:
        blocks.append(f"[{hit.get('id')}]\n{(hit.get('text') or '').strip()}")
    supplied = "\n\n".join(blocks)
    kinds = ", ".join(CLAIM_TYPES)
    return (
        "You are extracting evidence from source passages.\n"
        "For each useful claim:\n"
        "1. state one short neutral claim\n"
        "2. identify the source document\n"
        "3. provide the exact source span that supports it\n"
        "Do not add information that is not explicitly supported.\n"
        "Do not infer generic advice.\n"
        "Do not combine unrelated facts.\n"
        "If a source is only a title or stub, extract only what that title or stub states.\n"
        "If the source is a question or a title, do not answer it and do not turn it into a factual assertion.\n"
        "Do not expand it with background knowledge.\n"
        "Do not imitate Jason Lemkin.\n"
        "Do not write the final blog, LinkedIn, X, or talk.\n"
        "Return structured JSON only.\n\n"
        "Return a JSON array. Each item uses these keys: "
        "source_id, claim, support_span, claim_type.\n"
        f"claim_type must be one of: {kinds}.\n"
        "support_span must be copied exactly from that source. "
        "Use the shortest exact phrase that supports the claim.\n\n"
        f"Topic: {topic}\n\n"
        "Source passages:\n"
        f"{supplied}"
    )


def verification_user_prompt(claim: str, support_span: str) -> str:
    """Claim and span only. No rank, distance, medium, or style instruction."""
    return (
        "You are checking whether a source span supports a claim.\n"
        "Use only the span. Do not add outside knowledge.\n"
        "Reply with exactly one word: SUPPORTED or UNSUPPORTED.\n\n"
        f"Claim: {claim}\n"
        f"Source span: {support_span}"
    )


def interpret_verification(text: str) -> str:
    """Only an explicit SUPPORTED label keeps a claim."""
    for word in re.findall(r"[A-Za-z]+", text or ""):
        folded = word.upper()
        if folded == "UNSUPPORTED":
            return "UNSUPPORTED"
        if folded == "SUPPORTED":
            return "SUPPORTED"
    return "UNSUPPORTED"


def _escape_interior_quotes(raw: str) -> str:
    """Escape a quote that sits inside a JSON string. A following quote ends the string."""
    chars = []
    in_string = False
    escaped = False
    for index, char in enumerate(raw):
        if in_string:
            if escaped:
                chars.append(char)
                escaped = False
                continue
            if char == "\\":
                chars.append(char)
                escaped = True
                continue
            if char == '"':
                nxt = ""
                for later in raw[index + 1 :]:
                    if later not in " \t\r\n":
                        nxt = later
                        break
                if nxt in ',}]:"' or nxt == "":
                    chars.append(char)
                    in_string = False
                else:
                    chars.append('\\"')
                continue
        elif char == '"':
            in_string = True
        chars.append(char)
    return "".join(chars)


def _loads_repaired(snippet: str):
    """Parse JSON after deterministic delimiter repairs. Do not add fields."""
    text = _escape_interior_quotes(snippet)
    for _ in range(32):
        try:
            return json.loads(text)
        except json.JSONDecodeError as exc:
            if "Expecting ',' delimiter" not in exc.msg:
                raise
            pos = exc.pos
            if pos is None or pos >= len(text) or text[pos] != '"':
                raise
            text = text[:pos] + "," + text[pos:]
    raise json.JSONDecodeError("too many missing commas", text, 0)


def _close_truncated_array(raw: str) -> str | None:
    """Keep objects that already closed. Drop the unfinished tail. Do not invent one."""
    if not raw.startswith("["):
        return None
    depth = 0
    in_string = False
    escaped = False
    last_complete = None
    for index, char in enumerate(raw):
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char in "{[":
            depth += 1
        elif char in "}]":
            depth -= 1
            if char == "}" and depth == 1:
                last_complete = index
    if last_complete is None:
        return None
    return raw[: last_complete + 1] + "]"


def parse_atomic_evidence(text: str) -> list[dict]:
    """Read a JSON array. Do not complete or correct truncated objects."""
    raw = (text or "").strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw).strip()
    start = raw.find("[")
    end = raw.rfind("]")
    if start < 0 or end <= start:
        closed = _close_truncated_array(raw[start:] if start >= 0 else "")
        if closed is None:
            raise EvidenceParseError("extractor did not return a JSON array")
        try:
            payload = _loads_repaired(closed)
        except json.JSONDecodeError as exc:
            raise EvidenceParseError("extractor JSON could not be parsed") from exc
        if not isinstance(payload, list) or not payload:
            raise EvidenceParseError("extractor did not return a JSON array")
        return payload
    snippet = raw[start : end + 1]
    try:
        payload = json.loads(snippet)
    except json.JSONDecodeError:
        try:
            payload = _loads_repaired(snippet)
        except json.JSONDecodeError as exc:
            raise EvidenceParseError("extractor JSON could not be parsed") from exc
    if not isinstance(payload, list):
        raise EvidenceParseError("extractor JSON was not an array")
    return payload


def span_is_interrogative(span: str) -> bool:
    """A question title, not a declarative sentence that happens to contain 'how'."""
    text = normalize_support(span)
    if "?" in (span or ""):
        return True
    return bool(_QUESTION_START.match(text))


def claim_asserts_answer(claim: str) -> bool:
    """True when the claim states a fact instead of reporting that a question was asked."""
    text = (claim or "").strip()
    if "?" in text:
        return False
    folded = text.casefold()
    framing = ("asks", "asked", "asking", "titled", "the title", "the question")
    return not any(phrase in folded for phrase in framing)


def introduced_names(claim: str) -> list[str]:
    """Proper names the claim adds. Sentence-initial words are not treated as names."""
    names = []
    for match in re.finditer(r"\b[A-Za-z][A-Za-z0-9]*(?:\.[A-Za-z0-9]+)+\b", claim or ""):
        names.append(match.group(0))
    for sentence in re.split(r"[.!?]\s+", (claim or "").strip()):
        tokens = sentence.split()
        for token in tokens[1:]:
            raw = token.strip(",;:()[]\"'")
            folded = raw.casefold()
            if not raw or folded in _METRIC_TOKENS or folded in _COMMON_TOKENS:
                continue
            if re.fullmatch(r"[A-Z]{2,6}", raw) or re.fullmatch(r"[A-Z][a-zA-Z]{2,}", raw):
                names.append(raw)
    return names


def validate_item(item: dict, hits_by_id: dict[str, dict], topic: str = "") -> str | None:
    """Deterministic acceptance. The optional LLM verifier does not run here."""
    if not isinstance(item, dict):
        return "not_an_object"
    source_id = item.get("source_id")
    if source_id not in hits_by_id:
        return "unknown_source_id"
    claim = item.get("claim")
    if not isinstance(claim, str) or not claim.strip():
        return "empty_claim"
    span = item.get("support_span")
    if not isinstance(span, str) or not span.strip():
        return "missing_support"
    claim_type = item.get("claim_type")
    if not isinstance(claim_type, str) or claim_type.strip().casefold() not in _CLAIM_TYPE_SET:
        return "invalid_claim_type"
    source = hits_by_id[source_id].get("text") or ""
    if normalize_support(span) not in normalize_support(source):
        return "support_not_in_source"
    if span_is_interrogative(span) and claim_asserts_answer(claim):
        return "interrogative_assertion"
    missing_numbers = numbers_in(claim) - numbers_in(span)
    if missing_numbers:
        return "numeric_not_in_span"
    backdrop = normalize_support(f"{span}\n{topic}")
    for name in introduced_names(claim):
        if normalize_support(name) not in backdrop:
            return "entity_not_in_source"
    return None


def partition_evidence(items: list, hits: list[dict], topic: str = "") -> tuple[list[dict], list[dict]]:
    """Split proposed items. Invalid spans are dropped, not edited."""
    hits_by_id = {hit.get("id"): hit for hit in hits}
    accepted = []
    rejected = []
    for item in items:
        reason = validate_item(item, hits_by_id, topic)
        if reason:
            rejected.append(_public_item(item, reason))
            continue
        accepted.append(
            {
                "source_id": item["source_id"],
                "claim": item["claim"].strip(),
                "support_span": item["support_span"],
                "claim_type": item["claim_type"].strip().casefold(),
            }
        )
    return accepted, rejected


def apply_verdicts(accepted: list[dict], verdicts: list[str]) -> tuple[list[dict], list[dict]]:
    """Keep SUPPORTED claims. Anything else is removed."""
    if len(verdicts) != len(accepted):
        raise ValueError("each validated claim needs one verifier result")
    verified = []
    rejected = []
    for item, verdict in zip(accepted, verdicts):
        label = interpret_verification(verdict)
        stamped = {**item, "verifier_output": verdict, "verdict": label}
        if label == "SUPPORTED":
            verified.append(stamped)
        else:
            rejected.append(stamped)
    return verified, rejected


def verified_claim_block(claims: list[dict]) -> str:
    """Accepted claims only. Spans and source ids stay out."""
    lines = [f"[E{index}] {item['claim'].strip()}" for index, item in enumerate(claims, start=1)]
    body = "\n".join(lines)
    if body:
        return f"ALLOWED FACTUAL EVIDENCE\n\n{body}"
    return "ALLOWED FACTUAL EVIDENCE"


def final_evidence_instruction(medium: dict, topic: str, claims: list[dict]) -> str:
    """Relabel prompt. The closer is only the writing request."""
    from host_finetune.compare_adapters import prompt_for

    name = {
        "blog": "blog post",
        "linkedin": "LinkedIn post",
        "x": "X post",
        "talk": "talk",
    }[medium["medium"]]
    block = verified_claim_block(claims)
    return (
        f"{prompt_for(medium, topic)}\n\n"
        "Use only the factual specifics provided in ALLOWED FACTUAL EVIDENCE.\n"
        "Do not add factual numbers, percentages, dollar amounts, dates, customer "
        "counts, ARR, NRR, growth rates, or other specific metrics from memory.\n"
        "If a factual detail is not in the evidence, omit it.\n"
        "You may provide analysis, opinion, transitions, and stylistic commentary, "
        "but specific factual claims must come from the evidence.\n\n"
        f"{block}\n\n"
        f"Now write the requested {name} about: {topic}"
    )


def freeze_hits(hits: list[dict]) -> list[dict]:
    """Copy retrieved rows so evidence text cannot replace them."""
    frozen = []
    for hit in hits:
        frozen.append(
            {
                "id": hit.get("id"),
                "text": hit.get("text"),
                "medium": hit.get("medium"),
                "group_id": hit.get("group_id"),
                "row_id": hit.get("row_id"),
                "split": hit.get("split"),
                "distance": hit.get("distance"),
            }
        )
    return frozen


def _public_item(item, reason: str) -> dict:
    if not isinstance(item, dict):
        return {"source_id": None, "claim": None, "support_span": None, "claim_type": None, "reason": reason}
    return {
        "source_id": item.get("source_id"),
        "claim": item.get("claim"),
        "support_span": item.get("support_span"),
        "claim_type": item.get("claim_type"),
        "reason": reason,
    }


def media_jobs(topic: str, hits: list[dict], claims: list[dict]) -> list[dict]:
    """One verified set, four media. The copy gate is not applied to claims."""
    from host_finetune.compare_adapters import MEDIUMS

    raw = freeze_hits(hits)
    jobs = []
    for medium in MEDIUMS:
        instruction = final_evidence_instruction(medium, topic, claims)
        for hit in raw:
            text = (hit.get("text") or "").strip()
            if text and text in instruction:
                raise ValueError("raw retrieved prose leaked into the final prompt")
        for claim in claims:
            if claim["claim"] not in instruction:
                raise ValueError("verified claim missing from the final prompt")
            span = claim.get("support_span") or ""
            if span and span not in claim["claim"] and span in instruction:
                raise ValueError("support span leaked into the final prompt")
            source_id = claim.get("source_id") or ""
            if source_id and source_id not in claim["claim"] and source_id in instruction:
                raise ValueError("source id leaked into the final prompt")
        jobs.append(
            {
                "medium": medium,
                "instruction": instruction,
                "hits": raw,
                "claims": claims,
                "evidence_block": verified_claim_block(claims),
            }
        )
    return jobs


def evidence_provenance(hits: list[dict], record: dict) -> dict:
    """Raw hits stay beside the claims. Replacing the claim list cannot erase them."""
    return {
        "retrieval_hits": freeze_hits(hits),
        "proposed_atomic_claims": record.get("proposed") or [],
        "validation_rejected": record.get("validation_rejected") or [],
        "verifier_rejected": record.get("verifier_rejected") or [],
        "verified_claims": record.get("verified") or [],
        "evidence_block": record.get("evidence_block") or "",
        "raw_extraction": record.get("raw_extraction"),
        "extraction_prompt": record.get("extraction_prompt"),
        "rendered_extraction_prompt": record.get("rendered_extraction_prompt"),
        "extraction_seconds": record.get("extraction_seconds"),
        "verification_seconds": record.get("verification_seconds"),
        "verification_records": record.get("verification_records") or [],
        "llm_verifier_on_critical_path": False,
        "allowed_numeric_values": record.get("allowed_numeric_values") or [],
        "distillation_adapter_enabled": False,
        "verification_adapter_enabled": False,
        "distillation_backend": DISTILL_BACKEND,
        "distillation_do_sample": DISTILL_DO_SAMPLE,
        "extraction_max_new_tokens": EXTRACT_MAX_NEW_TOKENS,
        "verification_max_new_tokens": VERIFY_MAX_NEW_TOKENS,
        "distillation_seed": DISTILL_SEED,
        "support_spans_in_generator_prompt": False,
        "evidence_stage_reason": record.get("reason"),
        "extraction_stop_reason": record.get("extraction_stop_reason"),
    }
