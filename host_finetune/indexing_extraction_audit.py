"""Offline diagnostics for LightRAG 1.5.7 extraction records.

These functions classify cached text. They do not open the expansion store,
call a model, or insert edges.
"""

from __future__ import annotations

import asyncio
import re
from collections import Counter

TUPLE_DELIMITER = "<|#|>"
COMPLETION_DELIMITER = "<|COMPLETE|>"
EXPECTED_RELATION_FIELDS = 5
POLICY_VERSION = "relation-normalization-v1"

_SCHEMA_KEYWORDS = re.compile(r"^relationship_keywords$", re.IGNORECASE)
_SCHEMA_DESCRIPTION = re.compile(r"^relationship_description$", re.IGNORECASE)
_NUMERIC_NAME = re.compile(
    r"(\$\s?\d|\d+(?:\.\d+)?\s*%|\b20\d{2}\b|\b\d+(?:\.\d+)?\s*(?:million|billion|m|b|k)\b)",
    re.IGNORECASE,
)
_YEAR = re.compile(r"\b20\d{2}\b")
_UNIT = re.compile(r"(\$|%|\bpercent\b|\bmillion\b|\bbillion\b|\bcustomers?\b|\brevenue\b|\bgrowth\b|\barr\b|\bnrr\b)", re.IGNORECASE)


def extraction_stage(prompt: str) -> str:
    """Separate the first extraction call from the one gleaning continuation."""
    text = prompt or ""
    if "Based on the last extraction task" in text[:400]:
        return "gleaning"
    if "Extract entities and relationships" in text[:400]:
        return "initial"
    return "other"


def split_output_lines(text: str) -> list[str]:
    """Split one cached model response into candidate records."""
    pieces = re.split(r"\n|<\|COMPLETE\|>|<\|complete\|>", text or "")
    return [piece.strip() for piece in pieces if piece and piece.strip()]


def record_fields(record: str) -> list[str]:
    return [part.strip() for part in (record or "").split(TUPLE_DELIMITER)]


def _keyword_like(value: str) -> bool:
    text = (value or "").strip()
    if not text or len(text) > 80 or any(mark in text for mark in ".!?;"):
        return False
    parts = [part.strip() for part in text.split(",")]
    if not 1 <= len(parts) <= 6:
        return False
    for part in parts:
        words = part.split()
        if not words or len(words) > 3:
            return False
    return True


def _comma_separated_claim(value: str) -> tuple[str, str] | None:
    """A keyword list and a sentence joined by a comma instead of the delimiter."""
    if "," not in (value or ""):
        return None
    left, right = value.split(",", 1)
    if _keyword_like(left) and _description_like(right):
        return left.strip(), right.strip()
    return None


def _schema_label_repair(fields: list[str]) -> dict | None:
    """The model echoed the prompt's field names as extra columns."""
    if len(fields) == 6 and _SCHEMA_KEYWORDS.match(fields[3]) and _description_like(fields[5]):
        return {"keywords": fields[4].strip(), "description": fields[5].strip(), "repair": "schema_labels"}
    if (
        len(fields) == 7
        and _SCHEMA_KEYWORDS.match(fields[3])
        and _SCHEMA_DESCRIPTION.match(fields[5])
        and _description_like(fields[6])
    ):
        return {"keywords": fields[4].strip(), "description": fields[6].strip(), "repair": "schema_labels"}
    return None


def _description_like(value: str) -> bool:
    text = (value or "").strip()
    return len(text.split()) >= 6 or len(text) > 80 or text.endswith(".")


def classify_relation_fields(fields: list[str]) -> dict:
    """Classify one split record. This does not decide that the claim is true."""
    prefix = (fields[0] if fields else "").casefold()
    if "relation" not in prefix:
        return {"kind": "not_relation", "field_count": len(fields), "recoverable": False}
    count = len(fields)
    source = fields[1] if count > 1 else ""
    target = fields[2] if count > 2 else ""
    same = bool(source) and source.casefold() == target.casefold()
    if count == EXPECTED_RELATION_FIELDS and same:
        return {
            "kind": "self_relation_dropped",
            "field_count": count,
            "recoverable": False,
            "reason": "the parser drops a relation whose endpoints are the same",
        }
    if count == EXPECTED_RELATION_FIELDS:
        return {"kind": "accepted_five", "field_count": count, "recoverable": False}
    if count == 4 and _comma_separated_claim(fields[3]):
        return {
            "kind": "comma_between_keywords_and_description",
            "field_count": count,
            "recoverable": True,
            "reason": "a comma was used where the field delimiter belongs",
        }
    if count == 4 and _description_like(fields[3]) and not _keyword_like(fields[3]):
        return {
            "kind": "omitted_keywords",
            "field_count": count,
            "recoverable": True,
            "reason": "the fourth field is a sentence, so the keyword field was left out",
        }
    if count == 4 and _keyword_like(fields[3]):
        return {
            "kind": "omitted_description",
            "field_count": count,
            "recoverable": False,
            "reason": "keywords without a description are not a relationship claim",
        }
    if count == 4:
        return {
            "kind": "four_fields_ambiguous",
            "field_count": count,
            "recoverable": False,
            "reason": "the last field is neither a clear keyword list nor a sentence",
        }
    labeled = _schema_label_repair(fields)
    if labeled is not None:
        return {
            "kind": "schema_labels",
            "field_count": count,
            "recoverable": True,
            "reason": "the row contains the prompt's field names as columns",
        }
    if count > EXPECTED_RELATION_FIELDS and _keyword_like(fields[3] if count > 3 else ""):
        return {
            "kind": "delimiter_inside_description",
            "field_count": count,
            "recoverable": False,
            "reason": "the extra fields are not a known schema, so joining them could invent a claim",
        }
    if count > EXPECTED_RELATION_FIELDS:
        return {
            "kind": "extra_fields_ambiguous",
            "field_count": count,
            "recoverable": False,
            "reason": "joining the tail would invent a boundary the record does not show",
        }
    return {"kind": "too_few_fields", "field_count": count, "recoverable": False}


def recover_relation(fields: list[str]) -> dict | None:
    """Return a syntactic repair, or None when the repair would invent a claim.

    The result is not source-validated. Callers must check the chunk before
    any edge is stored.
    """
    verdict = classify_relation_fields(fields)
    if not verdict.get("recoverable"):
        return None
    kind = verdict["kind"]
    labeled = _schema_label_repair(fields)
    if labeled is not None:
        source, target = fields[1], fields[2]
        keywords, description, repair_name = labeled["keywords"], labeled["description"], labeled["repair"]
    elif kind == "comma_between_keywords_and_description":
        source, target = fields[1], fields[2]
        keywords, description = _comma_separated_claim(fields[3])
        repair_name = kind
    elif kind == "omitted_keywords":
        source, target, description = fields[1], fields[2], fields[3]
        keywords = ""
        repair_name = kind
    else:
        return None
    if not source.strip() or not target.strip() or source.casefold() == target.casefold():
        return None
    if not description.strip():
        return None
    return {
        "src_id": source.strip(),
        "tgt_id": target.strip(),
        "keywords": keywords.strip(),
        "description": description.strip(),
        "repair": repair_name,
        "semantic_status": "unverified",
    }


def _rejection_label(kind: str) -> str:
    if kind in {"omitted_description"}:
        return "rejected_missing_description"
    if kind in {"four_fields_ambiguous", "omitted_keywords", "extra_fields_ambiguous", "too_few_fields"}:
        return "rejected_ambiguous"
    return "rejected_unsupported"


def normalize_relation_record(line: str) -> tuple[str, str]:
    """Return a five-field line when the shape is one of the two known repairs.

    Every other relation line is returned unchanged so the installed parser
    still rejects it. Entity lines are not passed here.
    """
    fields = record_fields(line)
    verdict = classify_relation_fields(fields)
    kind = verdict["kind"]
    if kind in {"accepted_five", "self_relation_dropped"}:
        return line, "valid_original"
    repair = recover_relation(fields) if kind in {"schema_labels", "comma_between_keywords_and_description"} else None
    if repair is None:
        return line, _rejection_label(kind)
    normalized = TUPLE_DELIMITER.join([
        "relation",
        repair["src_id"],
        repair["tgt_id"],
        repair["keywords"],
        repair["description"],
    ])
    label = "repaired_schema_label" if repair["repair"] == "schema_labels" else "repaired_comma_packed"
    return normalized, label


def normalize_extraction_result(text: str) -> tuple[str, dict[str, int]]:
    """Normalize relation rows in one model response. The input string is not mutated."""
    counts: dict[str, int] = {}
    pieces = re.split(r"(<\|COMPLETE\|>|<\|complete\|>)", text or "")
    rebuilt: list[str] = []
    for piece in pieces:
        if piece == "<|COMPLETE|>" or piece == "<|complete|>":
            rebuilt.append(piece)
            continue
        lines = []
        for line in piece.split("\n"):
            stripped = line.strip()
            if stripped.lower().startswith("relation"):
                replacement, label = normalize_relation_record(stripped)
                counts[label] = counts.get(label, 0) + 1
                prefix = line[: len(line) - len(line.lstrip())]
                lines.append(prefix + replacement)
            else:
                lines.append(line)
        rebuilt.append("\n".join(lines))
    return "".join(rebuilt), counts


_PARSER_ORIGINAL = None
_PARSER_INSTALLED = False


def install_parser_normalizer(counters: dict[str, int] | None = None) -> None:
    """Normalize text before LightRAG 1.5.7 parses it.

    This replaces the process-local function LightRAG calls. It does not edit
    the installed package and it does not write the LLM cache.
    """
    global _PARSER_ORIGINAL, _PARSER_INSTALLED
    if _PARSER_INSTALLED:
        return
    from importlib.metadata import version

    import lightrag.operate as operate

    installed = version("lightrag-hku")
    if installed != "1.5.7":
        raise RuntimeError(f"relation normalizer is pinned to LightRAG 1.5.7, found {installed}")
    original = operate._process_extraction_result

    async def _normalized(result, *args, **kwargs):
        rewritten, stats = normalize_extraction_result(result)
        if counters is not None:
            for key, value in stats.items():
                counters[key] = counters.get(key, 0) + value
        return await original(rewritten, *args, **kwargs)

    operate._process_extraction_result = _normalized
    _PARSER_ORIGINAL = original
    _PARSER_INSTALLED = True


def uninstall_parser_normalizer() -> None:
    """Restore the installed parser for this process."""
    global _PARSER_ORIGINAL, _PARSER_INSTALLED
    if not _PARSER_INSTALLED:
        return
    import lightrag.operate as operate

    operate._process_extraction_result = _PARSER_ORIGINAL
    _PARSER_ORIGINAL = None
    _PARSER_INSTALLED = False


def endpoint_cooccurrence(source: str, target: str, chunk_text: str) -> bool:
    """Both names occur in the chunk. Co-occurrence is not a relationship."""
    text = (chunk_text or "").casefold()
    return bool(source and target) and source.casefold() in text and target.casefold() in text


def description_overlap(description: str, chunk_text: str) -> bool:
    """At least one content word from the description also occurs in the chunk."""
    words = [
        word for word in re.findall(r"[A-Za-z]{5,}", description or "")
        if word.casefold() not in {"which", "their", "there", "about", "would", "could", "should"}
    ]
    text = (chunk_text or "").casefold()
    return any(word.casefold() in text for word in words[:12])


def numeric_fact_gaps(name: str, description: str) -> list[str]:
    """Missing anchors for a number that was stored as its own node."""
    if not _NUMERIC_NAME.search(name or ""):
        return []
    blob = f"{name} {description or ''}"
    gaps = []
    if not _UNIT.search(blob):
        gaps.append("unit_or_metric")
    if not _YEAR.search(description or "") and not _YEAR.search(name or ""):
        gaps.append("time_period")
    proper = re.findall(r"\b[A-Z][a-z]{2,}\b", description or "")
    if not proper:
        gaps.append("named_entity_in_description")
    return gaps


def alias_candidates(names: list[str]) -> list[dict]:
    """Group exact normalization candidates. This does not merge nodes."""
    groups: dict[str, list[str]] = {}
    for name in names:
        key = re.sub(r"[^a-z0-9]", "", (name or "").casefold())
        if len(key) < 3:
            continue
        groups.setdefault(key, [])
        if name not in groups[key]:
            groups[key].append(name)
    return [
        {"key": key, "names": values}
        for key, values in sorted(groups.items())
        if len(values) > 1
    ]


def relation_identity(fields: list[str]) -> tuple[str, str, int]:
    source = fields[1].casefold() if len(fields) > 1 else ""
    target = fields[2].casefold() if len(fields) > 2 else ""
    pair = tuple(sorted((source, target)))
    return (pair[0], pair[1], len(fields))


def summarize_responses(entries: list[dict], chunk_text: dict[str, str] | None = None) -> dict:
    """Count raw rows, unique dropped pairs, and syntactic repairs.

    ``entries`` are cache rows with ``return``, ``chunk_id``, and
    ``original_prompt``. Warning lines are not the unit. A row repeated in
    gleaning is counted again as a row and once as a repeat.
    """
    texts = chunk_text or {}
    field_counts: Counter[str] = Counter()
    kinds: Counter[str] = Counter()
    stages: Counter[str] = Counter()
    unique_malformed: set[tuple] = set()
    unique_recoverable: set[tuple] = set()
    accepted_pairs: set[tuple] = set()
    recoverable_with_both_names = 0
    recoverable_with_overlap = 0
    recoverable_checked = 0
    repeated_in_response = 0
    gleaning_repeats = 0
    accepted_rows = 0
    chunks_with_malformed: set[str] = set()
    attempted_by_chunk: Counter[str] = Counter()
    accepted_by_chunk: Counter[str] = Counter()
    initial_malformed: dict[str, set[tuple]] = {}
    examples: list[dict] = []

    ordered = sorted(entries, key=lambda row: (row.get("chunk_id") or "", int(row.get("create_time") or 0)))
    for entry in ordered:
        stage = extraction_stage(entry.get("original_prompt") or "")
        stages[stage] += 1
        chunk_id = entry.get("chunk_id") or ""
        seen_here: set[tuple] = set()
        for line in split_output_lines(entry.get("return") or ""):
            fields = record_fields(line)
            if not fields or "relation" not in fields[0].casefold():
                continue
            verdict = classify_relation_fields(fields)
            kinds[verdict["kind"]] += 1
            field_counts[str(verdict["field_count"])] += 1
            identity = (chunk_id, *relation_identity(fields))
            if identity in seen_here:
                repeated_in_response += 1
            seen_here.add(identity)
            attempted_by_chunk[chunk_id] += 1
            if verdict["kind"] == "accepted_five":
                accepted_rows += 1
                accepted_by_chunk[chunk_id] += 1
                accepted_pairs.add((chunk_id, *relation_identity(fields)[:2]))
                continue
            if verdict["kind"] == "not_relation":
                continue
            unique_malformed.add(identity)
            chunks_with_malformed.add(chunk_id)
            if stage == "initial":
                initial_malformed.setdefault(chunk_id, set()).add(identity)
            elif stage == "gleaning" and identity in initial_malformed.get(chunk_id, set()):
                gleaning_repeats += 1
            repair = recover_relation(fields)
            if repair is not None:
                unique_recoverable.add(identity)
                body = texts.get(chunk_id, "")
                if body:
                    recoverable_checked += 1
                    if endpoint_cooccurrence(repair["src_id"], repair["tgt_id"], body):
                        recoverable_with_both_names += 1
                        if description_overlap(repair["description"], body):
                            recoverable_with_overlap += 1
            if len(examples) < 24 and verdict["kind"] not in {"accepted_five", "not_relation"}:
                examples.append({
                    "chunk_id": chunk_id,
                    "stage": stage,
                    "kind": verdict["kind"],
                    "field_count": verdict["field_count"],
                    "fields": fields[:8],
                    "repair": repair["repair"] if repair else None,
                })

    return {
        "responses": len(entries),
        "stages": dict(stages),
        "relation_rows_by_field_count": dict(field_counts),
        "relation_rows_by_kind": dict(kinds),
        "accepted_five_rows": accepted_rows,
        "unique_malformed_records": len(unique_malformed),
        "unique_syntactic_repairs": len(unique_recoverable),
        "repeated_rows_inside_one_response": repeated_in_response,
        "gleaning_repeats_of_initial_malformed": gleaning_repeats,
        "chunks_with_a_malformed_relation": len(chunks_with_malformed),
        "chunks_whose_relation_rows_were_all_rejected": sum(
            1 for chunk_id, attempts in attempted_by_chunk.items() if accepted_by_chunk[chunk_id] == 0 and attempts
        ),
        "repairs_checked_against_chunk_text": recoverable_checked,
        "repairs_with_both_endpoints_in_chunk": recoverable_with_both_names,
        "repairs_with_endpoint_and_description_overlap": recoverable_with_overlap,
        "malformed_pairs_already_accepted_in_the_same_chunk": sum(
            1 for chunk_id, source, target, _count in unique_malformed
            if (chunk_id, source, target) in accepted_pairs
        ),
        "examples": examples,
    }


async def retrieve_task_exceptions(tasks: list[asyncio.Task]) -> list[BaseException]:
    """Cancel unfinished tasks and retrieve every exception.

    ``asyncio.wait`` alone leaves a finished task's exception unretrieved.
    LightRAG's chunk-extraction wait does that for tasks it cancels after the
    first failure. The merge path uses ``gather(..., return_exceptions=True)``.
    """
    for task in tasks:
        if not task.done():
            task.cancel()
    if not tasks:
        return []
    results = await asyncio.gather(*tasks, return_exceptions=True)
    found = []
    for result in results:
        if isinstance(result, BaseException) and not isinstance(result, asyncio.CancelledError):
            found.append(result)
    return found
