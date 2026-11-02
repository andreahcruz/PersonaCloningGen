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
        text = (hit.get("text") or "").strip()
        status = hit.get("attribution_status")
        prefix = ""
        if status:
            prefix = (
                f"Attribution status: {status}. "
                "Do not treat the passage as Jason Lemkin's personal position "
                "unless the passage itself says so.\n"
            )
        speakers = [str(name) for name in (hit.get("quoted_speakers") or []) if name]
        if speakers:
            prefix += (
                "Quoted speakers: "
                + ", ".join(speakers)
                + ". Text inside quotation marks is their statement, not "
                "Jason Lemkin's, unless the passage says he agrees.\n"
            )
        blocks.append(f"[{hit.get('id')}]\n{prefix}{text}")
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


def _resolve_hit(source_id: str | None, hits: list[dict]) -> dict | None:
    """Match a copied id to one hit. A parent id and a chunk id name the same source."""
    if not source_id:
        return None
    exact = [hit for hit in hits if hit.get("id") == source_id]
    if len(exact) == 1:
        return exact[0]
    matches = []
    for hit in hits:
        keys = {
            hit.get("id"),
            hit.get("graph_document_id"),
            hit.get("parent_graph_document_id"),
        }
        graph_id = hit.get("graph_document_id")
        chunk_id = hit.get("chunk_id")
        if graph_id and chunk_id:
            keys.add(f"{graph_id}:{chunk_id}")
        if source_id in keys:
            matches.append(hit)
    if len(matches) == 1:
        return matches[0]
    return None


def annotate_attribution(claims: list[dict], hits: list[dict]) -> list[dict]:
    """Keep a quoted speaker only when that name is already in the source text."""
    by_id = {hit.get("id"): hit for hit in hits}
    annotated = []
    for claim in claims:
        item = dict(claim)
        hit = by_id.get(item.get("source_id"))
        if hit is None:
            annotated.append(item)
            continue
        status = hit.get("attribution_status")
        if status:
            item["attribution_status"] = status
        source = normalize_support(hit.get("text") or "")
        names = []
        for name in hit.get("quoted_speakers") or []:
            text = str(name).strip()
            if text and normalize_support(text) in source:
                names.append(text)
        if names:
            item["quoted_speakers"] = names
        annotated.append(item)
    return annotated


def partition_evidence(items: list, hits: list[dict], topic: str = "") -> tuple[list[dict], list[dict]]:
    """Split proposed items. Invalid spans are dropped, not edited."""
    hits_by_id = {hit.get("id"): hit for hit in hits}
    accepted = []
    rejected = []
    for item in items:
        resolved = _resolve_hit(item.get("source_id") if isinstance(item, dict) else None, hits)
        checked = item
        if isinstance(item, dict) and resolved is not None and item.get("source_id") != resolved.get("id"):
            checked = {**item, "source_id": resolved.get("id")}
            hits_by_id = {**hits_by_id, resolved.get("id"): resolved}
        reason = validate_item(checked, hits_by_id, topic)
        if reason:
            rejected.append(_public_item(item, reason))
            continue
        accepted.append(
            {
                "source_id": checked["source_id"],
                "claim": checked["claim"].strip(),
                "support_span": checked["support_span"],
                "claim_type": checked["claim_type"].strip().casefold(),
            }
        )
    return annotate_attribution(accepted, hits), rejected


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
    lines = []
    for index, item in enumerate(claims, start=1):
        lines.append(f"[E{index}] {item['claim'].strip()}")
        speakers = [str(name) for name in (item.get("quoted_speakers") or []) if name]
        if speakers:
            lines.append("Speaker: " + ", ".join(speakers))
    body = "\n".join(lines)
    if body:
        return f"ALLOWED FACTUAL EVIDENCE\n\n{body}"
    return "ALLOWED FACTUAL EVIDENCE"


PROMPT_CONTRACT = "factual-reliability-v1"
OPERATIONAL_X_MIN_WORDS = 20


def final_evidence_instruction(
    medium: dict,
    topic: str,
    claims: list[dict],
    evidence_status: str = "verified",
    style_exemplars: list[str] | None = None,
    relationships: list[dict] | None = None,
    context_pad: str | None = None,
) -> str:
    """Relabel prompt. The closer is only the writing request.

    Style exemplars are optional voice text. An empty list leaves the prompt
    unchanged. They are not added to the factual evidence block.

    ``relationships`` adds a separate source-supported block. ``None`` or an
    empty list leaves the prompt unchanged. A context pad is a length control
    and cannot be combined with relationship evidence.
    """
    from host_finetune.compare_adapters import prompt_for
    from host_finetune.train_only_index import format_voice_exemplars

    name = {
        "blog": "blog post",
        "linkedin": "LinkedIn post",
        "x": "X post",
        "talk": "talk",
    }[medium["medium"]]
    from host_finetune.graph_relationships import context_pad_block, relationship_prompt_block

    block = verified_claim_block(claims)
    if relationships and context_pad:
        raise ValueError("relationship evidence and a context pad cannot be combined")
    supplement = ""
    relation_block = relationship_prompt_block(relationships)
    if relation_block:
        supplement = "\n\n" + relation_block
    elif context_pad:
        pad_block = context_pad_block(context_pad)
        if pad_block:
            supplement = "\n\n" + pad_block
    voice = format_voice_exemplars(style_exemplars)
    opening = prompt_for(medium, topic)
    if voice:
        opening = f"{opening}\n\n{voice}"
    status_note = ""
    if evidence_status == "extraction_failed":
        status_note = (
            "Evidence extraction failed to return parseable claims. "
            "That is an extraction failure, not a finding that the sources were irrelevant. "
            "Do not invent specific facts, numbers, names, or quotations.\n\n"
        )
    elif evidence_status == "no_accepted_claims":
        status_note = (
            "No claim passed source-span validation. "
            "Do not invent specific facts, numbers, names, or quotations.\n\n"
        )
    return (
        f"{opening}\n\n"
        "Use only the factual specifics provided in ALLOWED FACTUAL EVIDENCE.\n"
        "Do not add factual numbers, percentages, dollar amounts, dates, customer "
        "counts, ARR, NRR, growth rates, or other specific metrics from memory.\n"
        "If a factual detail is not in the evidence, omit it.\n"
        f"Prompt contract: {PROMPT_CONTRACT}.\n"
        "When a claim has a Speaker line, attribute that statement to that person. "
        "Do not present it as Jason Lemkin's own words unless the claim says he agrees.\n"
        "Do not give a person a title, employer, or role unless that title and organization "
        "are both written in the evidence.\n"
        "Do not copy speaker labels or evidence tags into the draft.\n"
        "Do not repeat the user's question as the draft or as its opening line.\n"
        "Finish the last sentence.\n"
        "A LinkedIn post must be at least 40 words. An X post must be at least 20 words.\n"
        "You may provide analysis, opinion, transitions, and stylistic commentary, "
        "but specific factual claims must come from the evidence.\n\n"
        f"{status_note}"
        f"{block}"
        f"{supplement}\n\n"
        f"Now write the requested {name} about: {topic}"
    )


_EXTRA_HIT_FIELDS = (
    "rank",
    "score",
    "score_type",
    "backend",
    "retrieval_mode",
    "attribution_status",
    "author_position",
    "graph_document_id",
    "parent_graph_document_id",
    "chunk_id",
    "provenance",
    "source_row_ids",
    "source_file",
    "source_line",
    "quoted_speakers",
)


def freeze_hits(hits: list[dict]) -> list[dict]:
    """Copy retrieved rows so evidence text cannot replace them."""
    frozen = []
    for hit in hits:
        item = {
            "id": hit.get("id"),
            "text": hit.get("text"),
            "medium": hit.get("medium"),
            "group_id": hit.get("group_id"),
            "row_id": hit.get("row_id"),
            "split": hit.get("split"),
            "distance": hit.get("distance"),
        }
        for key in _EXTRA_HIT_FIELDS:
            if key not in hit or hit.get(key) is None:
                continue
            value = hit.get(key)
            item[key] = list(value) if isinstance(value, list) else value
        frozen.append(item)
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


def repeats_topic(answer: str, topic: str) -> bool:
    """True when the draft is the question or opens by repeating it."""
    def folded(text: str) -> str:
        return re.sub(r"\s+", " ", (text or "").strip().casefold())

    draft = folded(answer)
    asked = folded(topic)
    if not draft or not asked:
        return False
    return draft == asked or draft.startswith(asked)


def draft_retry_reason(
    answer: str,
    topic: str,
    medium: str,
    numeric_report: dict | None,
    claims: list[dict] | None = None,
    evidence_status: str = "verified",
) -> str | None:
    """One reason for a single corrective retry. None means keep the draft."""
    from host_finetune.factual_reliability import embeds_topic, instruction_leak, role_assertions, speaker_attribution
    from host_finetune.relabel_continuations import sentence_final
    from host_finetune.unified_eval import LENGTH_BANDS

    checked_claims = claims if evidence_status == "verified" else []
    if repeats_topic(answer, topic) or embeds_topic(answer, topic):
        return "topic_echo"
    if instruction_leak(answer):
        return "instruction_leak"
    words = len((answer or "").split())
    floor, _ceiling = LENGTH_BANDS.get(medium, (1, 5000))
    if words < floor:
        return "below_length_band"
    if medium == "x" and words < OPERATIONAL_X_MIN_WORDS:
        return "x_below_operational_floor"
    if (answer or "").strip() and not sentence_final(answer):
        return "incomplete_sentence"
    if numeric_report and numeric_report.get("applicable") and numeric_report.get("pass") is False:
        return "unsupported_number"
    if any(item["verdict"] != "supported" for item in role_assertions(answer, checked_claims)):
        return "unsupported_role"
    speaker = speaker_attribution(answer, checked_claims)
    if speaker.get("applicable") and not speaker.get("pass"):
        return "speaker_attribution"
    return None


def corrective_addendum(reason: str, numeric_report: dict | None) -> str:
    """Tell the model what to change. Do not supply a replacement fact."""
    from host_finetune.grounding_repair import unsupported_surfaces

    if reason == "topic_echo":
        return (
            "The draft repeated the question. Write the requested piece itself. "
            "Do not repeat the question."
        )
    if reason == "below_length_band":
        return (
            "The draft is too short for the requested medium. "
            "Write a complete piece and do not repeat the question."
        )
    if reason == "incomplete_sentence":
        return "The draft stopped mid-sentence. Finish the last sentence. Do not add new numbers."
    if reason == "x_below_operational_floor":
        return (
            "The X post is under 20 words. Write at least 20 words of the post itself. "
            "Do not repeat the question and do not add numbers, titles, or employers."
        )
    if reason == "instruction_leak":
        return "Remove evidence labels, speaker labels, and prompt text from the draft. Do not add numbers."
    if reason == "unsupported_role":
        return (
            "Remove titles, employers, and roles that are not written in the evidence. "
            "Do not invent a replacement title or company."
        )
    if reason == "speaker_attribution":
        return (
            "Name the quoted speaker from the evidence when you use that speaker's statement. "
            "Do not present the quotation as Jason Lemkin's own words, and do not add a title or company."
        )
    surfaces = unsupported_surfaces(numeric_report or {})
    listed = ", ".join(surfaces[:8]) or "the quantities that are not in the evidence"
    return (
        f"Remove these quantities that are not in the evidence: {listed}. "
        "Do not replace them with other numbers."
    )


def _still_echoes(answer: str, topic: str) -> bool:
    from host_finetune.factual_reliability import embeds_topic

    return repeats_topic(answer, topic) or embeds_topic(answer, topic)


def corrective_retry_improves(
    reason: str,
    first_answer: str,
    second_answer: str,
    topic: str,
    first_numeric: dict | None,
    second_numeric: dict | None,
    claims: list[dict] | None = None,
) -> bool:
    """Keep a retry only when it repairs the recorded failure."""
    from host_finetune.factual_reliability import instruction_leak, role_assertions, speaker_attribution
    from host_finetune.relabel_continuations import sentence_final

    if reason == "topic_echo":
        return not _still_echoes(second_answer, topic)
    if reason == "below_length_band":
        return len((second_answer or "").split()) > len((first_answer or "").split()) and not _still_echoes(
            second_answer, topic
        )
    if reason == "incomplete_sentence":
        return sentence_final(second_answer or "")
    if reason == "x_below_operational_floor":
        return len((second_answer or "").split()) > len((first_answer or "").split()) and not _still_echoes(
            second_answer, topic
        )
    if reason == "instruction_leak":
        return not instruction_leak(second_answer)
    if reason == "unsupported_role":
        def bad(text: str) -> int:
            return sum(item["verdict"] != "supported" for item in role_assertions(text, claims))

        return bad(second_answer) < bad(first_answer)
    if reason == "speaker_attribution":
        return bool(speaker_attribution(second_answer, claims).get("pass")) and not speaker_attribution(
            first_answer, claims
        ).get("pass")
    if reason == "unsupported_number":
        if (second_numeric or {}).get("pass"):
            return True
        before = len((first_numeric or {}).get("unsupported") or [])
        after = len((second_numeric or {}).get("unsupported") or [])
        return after < before
    return False


def _without_voice_block(instruction: str, voice: str) -> str:
    """Drop the voice slot before leak checks. Style text is not factual evidence."""
    if not voice:
        return instruction
    if voice not in instruction:
        raise ValueError("style block missing from the final prompt")
    return instruction.replace(voice, "", 1)


def media_jobs(
    topic: str,
    hits: list[dict],
    claims: list[dict],
    evidence_status: str = "verified",
    style_for=None,
    relationships: list[dict] | None = None,
    context_pad: str | None = None,
) -> list[dict]:
    """One verified set, four media. The copy gate is not applied to claims.

    ``style_for`` is optional. When it is absent, prompts match the retrieval
    path that does not load voice exemplars. Retrieved passages are still
    refused outside the voice slot.
    """
    from host_finetune.compare_adapters import MEDIUMS
    from host_finetune.graph_relationships import relationship_prompt_block
    from host_finetune.train_only_index import format_voice_exemplars

    relation_block = relationship_prompt_block(relationships)

    raw = freeze_hits(hits)
    jobs = []
    for medium in MEDIUMS:
        exemplars = style_for(medium["medium"]) if style_for is not None else None
        voice = format_voice_exemplars(exemplars)
        instruction = final_evidence_instruction(
            medium,
            topic,
            claims,
            evidence_status,
            style_exemplars=exemplars,
            relationships=relationships,
            context_pad=context_pad,
        )
        checked = _without_voice_block(instruction, voice)
        allowed_relation_text = set()
        for relation in relationships or []:
            for key in ("subject", "predicate", "object", "prompt_fact"):
                value = relation.get(key)
                if value:
                    allowed_relation_text.add(str(value))
            for qualifier in relation.get("qualifiers") or []:
                allowed_relation_text.add(str(qualifier))
            description = str(relation.get("graph_description") or "")
            relation_zone = ""
            if "SOURCE-SUPPORTED RELATIONSHIPS" in checked:
                relation_zone = checked.split("SOURCE-SUPPORTED RELATIONSHIPS", 1)[1]
            if (
                description
                and description not in str(relation.get("predicate") or "")
                and description in relation_zone
            ):
                raise ValueError("graph summary leaked into the final prompt")
            span = str(relation.get("support_span") or "")
            if span and span not in allowed_relation_text and span in checked:
                raise ValueError("relationship support span leaked into the final prompt")
            for source_id in relation.get("source_document_ids") or []:
                if source_id and source_id not in allowed_relation_text and source_id in checked:
                    raise ValueError("relationship source id leaked into the final prompt")
        for hit in raw:
            text = (hit.get("text") or "").strip()
            if text and text in checked:
                raise ValueError("raw retrieved prose leaked into the final prompt")
        for claim in claims:
            if claim["claim"] not in instruction:
                raise ValueError("verified claim missing from the final prompt")
            span = claim.get("support_span") or ""
            if span and span not in claim["claim"] and span in checked:
                raise ValueError("support span leaked into the final prompt")
            source_id = claim.get("source_id") or ""
            if source_id and source_id not in claim["claim"] and source_id in checked:
                raise ValueError("source id leaked into the final prompt")
        cleaned = [text.strip() for text in (exemplars or []) if text and text.strip()]
        jobs.append(
            {
                "medium": medium,
                "instruction": instruction,
                "hits": raw,
                "claims": claims,
                "evidence_block": verified_claim_block(claims),
                "evidence_status": evidence_status,
                "prompt_contract": PROMPT_CONTRACT,
                "style_applied": bool(voice),
                "style_exemplar_count": len(cleaned),
                "relationships": list(relationships or []),
                "relationship_block": relation_block if relationships else "",
                "context_pad": bool(context_pad),
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
        "evidence_status": record.get("evidence_status"),
        "extraction_attempts": record.get("extraction_attempts"),
        "extraction_stop_reason": record.get("extraction_stop_reason"),
    }
