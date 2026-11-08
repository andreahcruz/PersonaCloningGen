"""Derived graph-ready documents from the frozen train-only retrieval rows.

This module reads the Relabel dataset, its split assignments, and cleaned
source lines. It writes a new document set. It does not edit raw files, the
cleaned corpus, the Relabel training file, split assignments, or Chroma.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

from host_finetune.filter_event_promos import DIRECT_ANNOUNCEMENT, EVENT
from host_finetune.numeric_grounding import extract_quantities
from host_finetune.train_only_index import (
    DEFAULT_ASSIGNMENTS,
    DEFAULT_DATASET,
    DEFAULT_GOLD_DISPOSITIONS,
    DEFAULT_GOLD_EVAL,
    assignment_rows,
    blocked_group_ids,
    headline_overlap_groups,
    load_jsonl,
    select_train_documents,
)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CLEANED = ROOT / "data" / "cleaned"
POLICY_VERSION = "graph-ready-v1"
_URL = re.compile(r"https?://\S+|www\.\S+", re.I)
_SENTENCE_END = re.compile(r"[.!?]")
_BOILERPLATE = re.compile(
    r"\b(cookie policy|accept all cookies|subscribe to our newsletter|"
    r"share this (?:post|article)|skip to content|all rights reserved)\b",
    re.I,
)
_GUEST = re.compile(r"\b(?:interview|guest)\b|\bwith\b.+\b(?:founder|ceo)\b", re.I)
_JASON_X = {"jasonlk", "jasonlemkin"}
_TRANSCRIPT_MEDIA = {"youtube_jason", "youtube_saastr"}


@dataclass(frozen=True)
class GraphReadyPolicy:
    """Inclusion rules for the derived graph corpus. Defaults fail closed."""

    version: str = POLICY_VERSION
    max_cleaned_residual_words: int = 12
    exclude_reposts: bool = True
    exclude_other_authors: bool = True
    exclude_unattributed_transcripts: bool = True
    exclude_event_promos: bool = True
    exclude_link_only: bool = True
    exclude_headline_stubs: bool = True
    dedupe_exact: bool = True
    quantity_cap: int = 40


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def _words(text: str) -> list[str]:
    return _norm(text).split()


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _graph_id(source_file: str, source_line: int, row_ids: list[str], method: str) -> str:
    material = "\n".join(
        [POLICY_VERSION, source_file, str(source_line), method, ",".join(row_ids)]
    )
    return "gd_" + _sha256_text(material)[:24]


def cleaned_body(record: dict | None) -> str:
    if not record:
        return ""
    for key in ("content", "text", "transcript_text", "cleaned_text"):
        value = record.get(key)
        if isinstance(value, str) and value.strip():
            return value
    return ""


def source_date_raw(record: dict | None) -> str | None:
    if not record:
        return None
    for key in ("date", "created_at", "upload_date_iso", "upload_date", "published_text"):
        value = record.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _suffix_prefix_words(left: str, right: str, minimum: int = 8) -> int:
    previous = left.split()
    current = right.split()
    best = 0
    limit = min(len(previous), len(current), 80)
    for size in range(minimum, limit + 1):
        if previous[-size:] == current[:size]:
            best = size
    return best


def join_ordered_chunks(texts: list[str]) -> str:
    """Join ordered slices and drop a repeated word suffix of at least eight words."""
    parts: list[str] = []
    for text in texts:
        piece = (text or "").strip()
        if not piece:
            continue
        if parts:
            shared = _suffix_prefix_words(parts[-1], piece)
            if shared:
                piece = " ".join(piece.split()[shared:]).strip()
        if piece:
            parts.append(piece)
    return "\n\n".join(parts)


def _find_chunk(cleaned_norm: str, chunk: str, start: int) -> int:
    piece = _norm(chunk)
    if not piece:
        return -1
    return cleaned_norm.find(piece, start)


def lead_and_interior_gaps(cleaned: str, chunks: list[str]) -> dict | None:
    """Word counts before the first chunk and between later chunks.

    None when a chunk cannot be found in order. A zero interior gap means the
    eligible chunks are adjacent in the cleaned source. It does not mean the
    cleaned page has no text outside those chunks.
    """
    residual = _norm(cleaned)
    if not residual:
        return None
    cursor = 0
    starts = []
    ends = []
    for chunk in chunks:
        found = _find_chunk(residual, chunk, cursor)
        if found < 0:
            return None
        piece = _norm(chunk)
        starts.append(found)
        ends.append(found + len(piece))
        cursor = found + len(piece)
    lead = len(_words(residual[: starts[0]])) if starts else 0
    interior = []
    for previous_end, nxt in zip(ends, starts[1:]):
        interior.append(len(_words(residual[previous_end:nxt])))
    return {"lead_words": lead, "interior_gap_words": interior}


def cleaned_residual_words(cleaned: str, chunks: list[str]) -> int | None:
    """Words left after removing each chunk in order. None when a chunk is absent."""
    residual = _norm(cleaned)
    if not residual:
        return None
    for chunk in chunks:
        piece = _norm(chunk)
        if not piece:
            continue
        found = residual.find(piece)
        if found < 0:
            return None
        residual = _norm(residual[:found] + " " + residual[found + len(piece) :])
    return len(_words(residual))


def attribution_for(medium: str, title: str, record: dict | None) -> str:
    """Separate account ownership from verified personal authorship."""
    if medium in _TRANSCRIPT_MEDIA:
        if medium == "youtube_saastr":
            return "channel_transcript_unattributed"
        return "unattributed_transcript"
    if record is None:
        return "cleaned_record_missing"
    if medium == "x":
        if record.get("is_repost_or_quote") is True:
            return "repost_or_quote"
        username = str(record.get("username") or "").strip().lower()
        if username and username not in _JASON_X:
            return "other_author"
        if username in _JASON_X:
            return "author_account"
        return "account_unknown"
    if medium == "linkedin":
        author = str(record.get("author") or "")
        if not author.strip():
            return "author_missing"
        if "lemkin" in author.casefold():
            return "named_author"
        return "other_named_author"
    if medium == "blog":
        if _GUEST.search(title or ""):
            return "possible_guest_or_interview"
        return "publication_page"
    return "unclassified_medium"


def author_position(status: str) -> str:
    if status in {"named_author", "author_account"}:
        return "account_or_name_attributed"
    return "not_verified"


def _quantities(text: str, cap: int) -> list[dict]:
    rows = []
    for item in extract_quantities(text):
        rows.append(
            {
                "value": item.get("value"),
                "canonical": item.get("canonical"),
                "sentence": (item.get("sentence") or "")[:240],
            }
        )
        if len(rows) >= cap:
            break
    return rows


def exclusion_reasons(
    text: str,
    medium: str,
    title: str,
    status: str,
    policy: GraphReadyPolicy,
) -> list[str]:
    reasons = []
    stripped = (text or "").strip()
    words = _words(stripped)
    without_urls = _words(_URL.sub(" ", stripped))
    has_url = bool(_URL.search(stripped))
    has_end = bool(_SENTENCE_END.search(stripped))
    has_number = bool(extract_quantities(stripped))
    if not stripped:
        reasons.append("empty")
    if policy.exclude_link_only and has_url and len(without_urls) < 4:
        reasons.append("link_only")
    if policy.exclude_headline_stubs and medium not in {"x", "linkedin"}:
        if len(words) < 8 and not has_end and not has_number:
            reasons.append("headline_or_stub")
    if medium in {"x", "linkedin"} and len(words) < 3 and not has_end and not has_number:
        reasons.append("extreme_short")
    if _BOILERPLATE.search(stripped) and len(words) < 40:
        reasons.append("navigation_boilerplate")
    if policy.exclude_event_promos:
        title_ok = not (title or "").lower().startswith("dear saastr:")
        if title_ok and EVENT.search(title or "") and DIRECT_ANNOUNCEMENT.search(title or ""):
            if len(words) < 120:
                reasons.append("event_promo_template")
    if policy.exclude_reposts and status == "repost_or_quote":
        reasons.append("repost_or_quote")
    if policy.exclude_other_authors and status in {"other_author", "other_named_author"}:
        reasons.append("other_author")
    if policy.exclude_unattributed_transcripts and status in {
        "unattributed_transcript",
        "channel_transcript_unattributed",
    }:
        reasons.append("unattributed_transcript")
    return reasons


def _flags(text: str, title: str, medium: str, extra: list[str]) -> list[str]:
    flags = list(extra)
    words = _words(text)
    if len(words) < 40:
        flags.append("short_under_40_words")
    if not _SENTENCE_END.search(text or "") and medium != "x":
        flags.append("no_terminal_punctuation")
    if _GUEST.search(title or ""):
        flags.append("possible_guest_or_interview")
    if EVENT.search(f"{title}\n{text}") and "event_promo_template" not in flags:
        flags.append("mentions_event")
    return flags


def _order_conflict(members: list[dict]) -> bool:
    priors = [member.get("prior_row_index") for member in members]
    if all(isinstance(value, int) for value in priors):
        if any(left >= right for left, right in zip(priors, priors[1:])):
            return True
    parts = [member.get("part_index") for member in members]
    if all(isinstance(value, int) for value in parts):
        if any(left >= right for left, right in zip(parts, parts[1:])):
            return True
    indexes = [member["row_index"] for member in members]
    return any(left >= right for left, right in zip(indexes, indexes[1:]))


def _member(
    index: int,
    row: dict,
    rec: dict,
    eligible_ids: set[str],
) -> dict:
    return {
        "row_id": f"train_{index}",
        "row_index": index,
        "text": (row.get("output") or "").strip(),
        "split": rec.get("split"),
        "group_id": rec.get("group_id"),
        "medium": rec.get("source_platform"),
        "source_file": rec.get("source_file"),
        "source_line": rec.get("source_line"),
        "prior_row_index": rec.get("prior_row_index"),
        "part_index": rec.get("part_index"),
        "base_title": rec.get("base_title") or "",
        "sft_role": rec.get("sft_role"),
        "eligible": f"train_{index}" in eligible_ids,
    }


def _document(
    members: list[dict],
    text: str,
    method: str,
    status: str,
    date_raw: str | None,
    flags: list[str],
    reasons: list[str],
    policy: GraphReadyPolicy,
    cleaned_used: bool,
) -> dict:
    ordered = sorted(members, key=lambda item: item["row_index"])
    source_file = str(ordered[0].get("source_file") or "")
    try:
        source_line = int(ordered[0].get("source_line"))
    except (TypeError, ValueError):
        source_line = -1
    row_ids = [item["row_id"] for item in ordered]
    group_ids = sorted({item["group_id"] for item in ordered if item.get("group_id")})
    quantities = _quantities(text, policy.quantity_cap)
    return {
        "graph_document_id": _graph_id(source_file, source_line, row_ids, method),
        "text": text,
        "text_sha256": _sha256_text(text),
        "medium": ordered[0].get("medium"),
        "split": "train",
        "group_id": group_ids[0] if group_ids else None,
        "group_ids": group_ids,
        "source_file": source_file,
        "source_line": source_line,
        "experiment_row_ids": row_ids,
        "primary_row_index": ordered[0]["row_index"],
        "reconstruction_method": method,
        "cleaned_document_used": cleaned_used,
        "attribution_status": status,
        "author_position": author_position(status),
        "source_date_raw": date_raw,
        "source_date_is_absolute": bool(date_raw and re.match(r"\d{4}", date_raw)),
        "numeric_surfaces": quantities,
        "numeric_surface_count": len(extract_quantities(text)),
        "flags": flags,
        "exclusion_reasons": reasons,
        "word_count": len(_words(text)),
        "member_count": len(ordered),
        "policy_version": policy.version,
        "included": not reasons,
    }


def _emit_singles(
    members: list[dict],
    cleaned: dict | None,
    policy: GraphReadyPolicy,
    extra_flags: list[str],
) -> list[dict]:
    documents = []
    for member in sorted(members, key=lambda item: item["row_index"]):
        title = member.get("base_title") or ""
        status = attribution_for(member.get("medium") or "", title, cleaned)
        reasons = exclusion_reasons(member["text"], member.get("medium") or "", title, status, policy)
        flags = _flags(member["text"], title, member.get("medium") or "", extra_flags)
        if member.get("sft_role") == "continuation":
            flags.append("continuation_fragment")
        documents.append(
            _document(
                [member],
                member["text"],
                "single_chunk",
                status,
                source_date_raw(cleaned),
                flags,
                reasons,
                policy,
                False,
            )
        )
    return documents


def assemble_source_documents(
    members: list[dict],
    cleaned_by_key: dict[tuple[str, int], dict | None],
    policy: GraphReadyPolicy | None = None,
) -> dict:
    """Group aligned rows by cleaned source line and apply the graph-ready rules.

    ``members`` must include ineligible siblings. Only rows with ``eligible``
    true can enter the graph. A source line that mixes splits is not stitched.
    A source line with an unknown split is dropped entirely.
    """
    policy = policy or GraphReadyPolicy()
    grouped: dict[tuple[str, int] | None, list[dict]] = defaultdict(list)
    for member in members:
        name = member.get("source_file")
        line = member.get("source_line")
        if not name or line is None or str(line).strip() == "":
            grouped[None].append(member)
            continue
        grouped[(str(name), int(line))].append(member)

    included: list[dict] = []
    quarantined: list[dict] = []
    pre_counts: Counter[str] = Counter()

    def _keep(document: dict) -> None:
        if document["included"]:
            included.append(document)
        else:
            quarantined.append(document)

    for key, group in grouped.items():
        eligible = [
            item for item in group if item.get("eligible") and item.get("split") == "train"
        ]
        if key is None:
            for item in eligible:
                pre_counts["missing_provenance"] += 1
                _keep(
                    _document(
                        [item],
                        item.get("text") or "",
                        "single_chunk",
                        "missing_provenance",
                        None,
                        ["missing_provenance"],
                        ["missing_provenance"],
                        policy,
                        False,
                    )
                )
            continue
        splits = {item.get("split") for item in group}
        if any(split not in {"train", "validation", "test"} for split in splits):
            pre_counts["unknown_split_rows"] += sum(
                1 for item in group if item.get("split") not in {"train", "validation", "test"}
            )
            for item in eligible:
                pre_counts["unknown_split"] += 1
                _keep(
                    _document(
                        [item],
                        item.get("text") or "",
                        "single_chunk",
                        "unknown_split",
                        None,
                        ["unknown_split"],
                        ["unknown_split"],
                        policy,
                        False,
                    )
                )
            continue
        if not eligible:
            continue
        cleaned = cleaned_by_key.get(key)
        if any(item.get("split") != "train" for item in group):
            pre_counts["cross_split_source"] += 1
            for document in _emit_singles(eligible, cleaned, policy, ["cross_split_sibling"]):
                _keep(document)
            continue
        ordered = sorted(eligible, key=lambda item: item["row_index"])
        ineligible = [item for item in group if not item.get("eligible")]
        if _order_conflict(ordered):
            pre_counts["order_conflict"] += 1
            for document in _emit_singles(ordered, cleaned, policy, ["order_conflict"]):
                _keep(document)
            continue
        if ineligible:
            pre_counts["ineligible_sibling"] += 1
            for document in _emit_singles(
                ordered, cleaned, policy, ["ineligible_sibling_not_joined"]
            ):
                _keep(document)
            continue
        title = ordered[0].get("base_title") or ""
        medium = ordered[0].get("medium") or ""
        status = attribution_for(medium, title, cleaned)
        chunks = [item["text"] for item in ordered]
        body = cleaned_body(cleaned)
        extra: list[str] = []
        if len({item.get("group_id") for item in ordered}) > 1:
            extra.append("multiple_group_ids")
        residual = cleaned_residual_words(body, chunks) if body else None
        if residual is not None and residual <= policy.max_cleaned_residual_words:
            text = body
            method = "cleaned_document"
            cleaned_used = True
        else:
            text = chunks[0] if len(chunks) == 1 else join_ordered_chunks(chunks)
            method = "single_chunk" if len(chunks) == 1 else "ordered_eligible_chunks"
            cleaned_used = False
            if not body:
                extra.append("cleaned_source_missing")
            elif residual is None:
                extra.append("chunks_not_contained_in_cleaned")
            else:
                extra.append("cleaned_residual_exceeds_limit")
                extra.append(f"cleaned_residual_words_{residual}")
        if any(item.get("sft_role") == "continuation" for item in ordered) and method == "single_chunk":
            extra.append("continuation_fragment")
        if date_raw := source_date_raw(cleaned):
            if not re.match(r"\d{4}", date_raw):
                extra.append("relative_date_unresolved")
        reasons = exclusion_reasons(text, medium, title, status, policy)
        flags = _flags(text, title, medium, extra)
        _keep(
            _document(
                ordered,
                text,
                method,
                status,
                source_date_raw(cleaned),
                flags,
                reasons,
                policy,
                cleaned_used,
            )
        )

    if policy.dedupe_exact:
        included, duplicates = _dedupe_exact(included)
        quarantined.extend(duplicates)
    else:
        duplicates = []

    return {
        "policy": policy,
        "included": included,
        "quarantined": quarantined,
        "pre_counts": dict(pre_counts),
        "exact_duplicates_removed": len(duplicates),
    }


def _dedupe_exact(documents: list[dict]) -> tuple[list[dict], list[dict]]:
    groups: dict[str, list[dict]] = defaultdict(list)
    for document in documents:
        groups[_norm(document["text"]).casefold()].append(document)
    kept = []
    removed = []
    for copies in groups.values():
        copies = sorted(
            copies,
            key=lambda item: (item["source_file"], item["source_line"], item["graph_document_id"]),
        )
        kept.append(copies[0])
        for duplicate in copies[1:]:
            removed.append(
                {
                    **duplicate,
                    "included": False,
                    "exclusion_reasons": [*duplicate["exclusion_reasons"], "exact_duplicate_source"],
                    "duplicate_of": copies[0]["graph_document_id"],
                }
            )
    return kept, removed


def eligible_train_ids(
    dataset: list[dict],
    assignments: list[dict],
    dispositions: list[dict],
    gold_rows: list[dict],
) -> tuple[set[str], dict]:
    """Same selector the v2 RAG stage uses: gold families, then headline overlap."""
    family = blocked_group_ids(assignments, dispositions)
    family_docs, family_stats = select_train_documents(dataset, assignments, family)
    headline, _details = headline_overlap_groups(dataset, assignments, gold_rows, family)
    final_docs, final_stats = select_train_documents(dataset, assignments, family | headline)
    ids = {doc["id"] for doc in final_docs}
    stats = {
        "family_excluded_gold": family_stats["excluded_gold"],
        "final_excluded_gold": final_stats["excluded_gold"],
        "headline_additional": final_stats["excluded_gold"] - family_stats["excluded_gold"],
        "eligible": len(ids),
        "held_out_train_26148_present": "train_26148" in ids,
        "family_indexed_before_headline": len(family_docs),
    }
    return ids, stats


def aligned_members(
    dataset: list[dict],
    assignments: list[dict],
    eligible_ids: set[str],
) -> list[dict]:
    records = assignment_rows(assignments)
    if len(records) != len(dataset):
        raise ValueError(
            f"assignment rows ({len(records)}) != dataset rows ({len(dataset)})"
        )
    members = []
    for index, (row, rec) in enumerate(zip(dataset, records)):
        if rec.get("row_index") != index:
            raise ValueError(f"assignment row_index {rec.get('row_index')} != {index}")
        members.append(_member(index, row, rec, eligible_ids))
    return members


def load_cleaned_keys(root: Path, keys: set[tuple[str, int]]) -> dict[tuple[str, int], dict | None]:
    by_file: dict[str, set[int]] = defaultdict(set)
    for name, line in keys:
        by_file[name].add(line)
    found: dict[tuple[str, int], dict | None] = {key: None for key in keys}
    for name, lines in by_file.items():
        path = root / name
        if not path.is_file():
            continue
        last = max(lines)
        with path.open(encoding="utf-8") as handle:
            for number, raw in enumerate(handle, start=1):
                if number in lines and raw.strip():
                    found[(name, number)] = json.loads(raw)
                if number >= last:
                    break
    return found


def summarize(result: dict, eligibility: dict | None = None) -> dict:
    included = result["included"]
    quarantined = result["quarantined"]
    by_medium = Counter(doc["medium"] for doc in included)
    quarantine_medium = Counter(doc["medium"] for doc in quarantined)
    reasons = Counter()
    for doc in quarantined:
        for reason in doc["exclusion_reasons"]:
            reasons[reason] += 1
    methods = Counter(doc["reconstruction_method"] for doc in included)
    attribution = Counter(doc["attribution_status"] for doc in included)
    positions = Counter(doc["author_position"] for doc in included)
    flags = Counter()
    for doc in included:
        for flag in doc["flags"]:
            if flag.startswith("cleaned_residual_words_"):
                flags["cleaned_residual_exceeds_limit"] += 0
                continue
            flags[flag] += 1
    multi_member = sum(1 for doc in included if doc["member_count"] > 1)
    included_rows = [row_id for doc in included for row_id in doc["experiment_row_ids"]]
    quarantined_rows = [row_id for doc in quarantined for row_id in doc["experiment_row_ids"]]
    included_set = set(included_rows)
    quarantined_set = set(quarantined_rows)
    pre_counts = dict(result["pre_counts"])
    for name in (
        "missing_provenance",
        "unknown_split",
        "unknown_split_rows",
        "cross_split_source",
        "order_conflict",
        "ineligible_sibling",
    ):
        pre_counts.setdefault(name, 0)
    length_by_medium = {}
    for medium in sorted({doc.get("medium") for doc in included}):
        lengths = sorted(doc["word_count"] for doc in included if doc.get("medium") == medium)
        if not lengths:
            continue
        def _at(q: float, values: list[int] = lengths) -> int:
            return values[min(len(values) - 1, max(0, int(round(q * (len(values) - 1)))))]
        length_by_medium[medium] = {
            "documents": len(lengths),
            "median_words": _at(0.5),
            "p10_words": _at(0.1),
            "p90_words": _at(0.9),
            "under_20_words": sum(1 for value in lengths if value < 20),
            "under_40_words": sum(1 for value in lengths if value < 40),
        }
    return {
        "policy_version": result["policy"].version,
        "eligibility": eligibility,
        "included_documents": len(included),
        "quarantined_documents": len(quarantined),
        "eligible_rows_represented": len(included_rows),
        "unique_included_rows": len(included_set),
        "unique_quarantined_rows": len(quarantined_set),
        "rows_in_both_included_and_quarantine": len(included_set & quarantined_set),
        "multi_member_documents": multi_member,
        "exact_duplicates_removed": result["exact_duplicates_removed"],
        "reconstruction_pre_counts": pre_counts,
        "length_by_medium": length_by_medium,
        "included_by_medium": dict(by_medium),
        "quarantine_by_medium": dict(quarantine_medium),
        "exclusion_reasons": dict(reasons),
        "reconstruction_methods": dict(methods),
        "attribution_status": dict(attribution),
        "author_position": dict(positions),
        "flags": dict(flags),
        "documents_with_numeric_surfaces": sum(
            1 for doc in included if doc["numeric_surface_count"]
        ),
    }


def select_index_sample(
    documents: list[dict],
    count: int,
    seed: int = 298,
    max_medium_fraction: float | None = None,
) -> list[dict]:
    """Deterministic medium-stratified sample. The same seed returns the same ids."""
    if count < 1:
        raise ValueError("sample count must be positive")
    import random

    by_medium: dict[str, list[dict]] = defaultdict(list)
    for document in documents:
        by_medium[str(document.get("medium") or "unknown")].append(document)
    for rows in by_medium.values():
        rows.sort(key=lambda item: item["graph_document_id"])
    media = sorted(by_medium)
    if count >= len(documents):
        chosen = [item["graph_document_id"] for rows in by_medium.values() for item in rows]
        return [doc for doc in documents if doc["graph_document_id"] in set(chosen)]
    quotas = {medium: 0 for medium in media}
    remaining = count
    # Give every medium one slot when the sample is large enough, then fill by size.
    if count >= len(media):
        for medium in media:
            quotas[medium] = 1
        remaining -= len(media)
    weights = {medium: len(by_medium[medium]) for medium in media}
    total = sum(weights.values()) or 1
    fractional = []
    for medium in media:
        share = remaining * weights[medium] / total
        whole = int(share)
        quotas[medium] += whole
        fractional.append((share - whole, medium))
    leftover = count - sum(quotas.values())
    for _fraction, medium in sorted(fractional, reverse=True):
        if leftover <= 0:
            break
        quotas[medium] += 1
        leftover -= 1
    if max_medium_fraction is not None:
        if not 0 < max_medium_fraction <= 1:
            raise ValueError("max_medium_fraction must be between 0 and 1")
        cap = max(1, int(count * max_medium_fraction))
        excess = 0
        for medium in media:
            if quotas[medium] > cap:
                excess += quotas[medium] - cap
                quotas[medium] = cap
        while excess:
            candidates = [
                medium
                for medium in media
                if quotas[medium] < cap and quotas[medium] < len(by_medium[medium])
            ]
            if not candidates:
                break
            medium = max(candidates, key=lambda item: len(by_medium[item]) - quotas[item])
            quotas[medium] += 1
            excess -= 1
    rng = random.Random(seed)
    selected = []
    for medium in media:
        rows = by_medium[medium]
        take = min(quotas[medium], len(rows))
        if take <= 0:
            continue
        if take == len(rows):
            selected.extend(rows)
            continue
        # Spread across the length-sorted list so short and long documents both appear.
        ordered = sorted(rows, key=lambda item: (item["word_count"], item["graph_document_id"]))
        step = len(ordered) / take
        indexes = []
        for offset in range(take):
            indexes.append(min(len(ordered) - 1, int(offset * step)))
        # Break ties with the seed without changing the length spread when indexes collide.
        chosen_indexes = []
        for index in indexes:
            if index not in chosen_indexes:
                chosen_indexes.append(index)
        cursor = 0
        while len(chosen_indexes) < take:
            candidate = (chosen_indexes[-1] + 1 + rng.randrange(len(ordered))) % len(ordered)
            if candidate not in chosen_indexes:
                chosen_indexes.append(candidate)
            cursor += 1
            if cursor > len(ordered) * 2:
                break
        selected.extend(ordered[index] for index in chosen_indexes[:take])
    if len(selected) < min(count, len(documents)):
        chosen = {item["graph_document_id"] for item in selected}
        rest = sorted(
            (doc for doc in documents if doc["graph_document_id"] not in chosen),
            key=lambda item: item["graph_document_id"],
        )
        selected.extend(rest[: count - len(selected)])
    selected.sort(key=lambda item: item["graph_document_id"])
    return selected


def review_sample(included: list[dict], quarantined: list[dict], limit: int = 8) -> list[dict]:
    """Fixed sample for a person to read. It is not a precision measurement."""
    picked = []
    by_medium: dict[str, list[dict]] = defaultdict(list)
    for document in sorted(included, key=lambda item: item["graph_document_id"]):
        by_medium[str(document.get("medium"))].append(document)
    for medium in sorted(by_medium):
        picked.extend(by_medium[medium][:2])
    reasons_seen = set()
    for document in sorted(quarantined, key=lambda item: item["graph_document_id"]):
        novel = [reason for reason in document["exclusion_reasons"] if reason not in reasons_seen]
        if not novel:
            continue
        reasons_seen.update(document["exclusion_reasons"])
        picked.append(document)
        if len(reasons_seen) >= 4:
            break
    return picked[: max(limit, 12)]


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def write_graph_ready(result: dict, audit: dict, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_jsonl(out_dir / "documents.jsonl", result["included"])
    _write_jsonl(out_dir / "quarantine.jsonl", result["quarantined"])
    sample = review_sample(result["included"], result["quarantined"])
    _write_jsonl(out_dir / "review_sample.jsonl", sample)
    index_sample = select_index_sample(
        result["included"],
        min(160, len(result["included"])),
        seed=298,
        max_medium_fraction=0.4,
    )
    (out_dir / "sample_160_ids.json").write_text(
        json.dumps(
            [document["graph_document_id"] for document in index_sample],
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    (out_dir / "audit.json").write_text(
        json.dumps(audit, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_project_corpus(
    dataset_path: Path = DEFAULT_DATASET,
    assignments_path: Path = DEFAULT_ASSIGNMENTS,
    dispositions_path: Path = DEFAULT_GOLD_DISPOSITIONS,
    gold_path: Path = DEFAULT_GOLD_EVAL,
    cleaned_dir: Path = DEFAULT_CLEANED,
    policy: GraphReadyPolicy | None = None,
) -> tuple[dict, dict]:
    dataset = load_jsonl(dataset_path)
    assignments = load_jsonl(assignments_path)
    dispositions = load_jsonl(dispositions_path)
    gold_rows = load_jsonl(gold_path)
    eligible_ids, eligibility = eligible_train_ids(dataset, assignments, dispositions, gold_rows)
    members = aligned_members(dataset, assignments, eligible_ids)
    keys = {
        (str(item["source_file"]), int(item["source_line"]))
        for item in members
        if item.get("source_file") and item.get("source_line") is not None
    }
    cleaned = load_cleaned_keys(cleaned_dir, keys)
    result = assemble_source_documents(members, cleaned, policy)
    audit = summarize(result, eligibility)
    audit["inputs"] = {
        "dataset_sha256": file_sha256(dataset_path),
        "assignments_sha256": file_sha256(assignments_path),
        "dispositions_sha256": file_sha256(dispositions_path),
        "gold_eval_sha256": file_sha256(gold_path),
        "dataset_rows": len(dataset),
        "assignment_rows": len(assignment_rows(assignments)),
    }
    return result, audit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "artifacts" / "lightrag" / "model_f" / "graph_ready",
    )
    args = parser.parse_args()
    result, audit = build_project_corpus()
    write_graph_ready(result, audit, args.out)
    print(json.dumps({key: audit[key] for key in (
        "included_documents",
        "quarantined_documents",
        "eligible_rows_represented",
        "reconstruction_methods",
        "exclusion_reasons",
        "eligibility",
    )}, indent=2))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
