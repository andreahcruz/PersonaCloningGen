"""Deterministic document priority for LightRAG indexing.

The score prefers attributed, knowledge-dense, moderately long documents and
spreads the queue across medium and a small keyword topic. It is a scheduling
heuristic, not a relevance grade. Token and chunk counts are character
estimates, not tokenizer output.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path

from host_finetune.lightrag_backend import OMISSION_MARKER

CHUNK_TOKEN_SIZE = 400
CHARS_PER_TOKEN_ESTIMATE = 4

# Pilot failures from EXP-20261029-004. They stay in the ledger and out of the
# default queue. A later session can include them with --include-deferred.
HISTORICAL_PILOT_IDS = frozenset(
    {
        "gd_1d361440c168ddd12ff26855",
        "gd_5ffdc3a083492246413be6f2",
        "gd_8276b7d28a0837fa1e71181a",
        "gd_8a579b23707dd4f77e169619",
        "gd_a4ad0243c9bcfec4c527a703",
        "gd_aa82ba1eab089122de370f68",
        "gd_f76cf95212bfc67508398f63",
        "gd_f9a237e6ccf11fe4de9924ad",
    }
)

ATTRIBUTION_WEIGHT = {
    "author_account": 40.0,
    "named_author": 38.0,
    "publication_page": 28.0,
    "third_party_quotation": 6.0,
}
MEDIUM_WEIGHT = {"blog": 18.0, "linkedin": 16.0, "x": 6.0}
TOPIC_PATTERNS = (
    ("hiring", re.compile(r"\b(hiring|hires?|vp of sales|salesperson|quota)\b", re.I)),
    ("pricing", re.compile(r"\b(pricing|price|arr|revenue|churn)\b", re.I)),
    ("fundraising", re.compile(r"\b(fund|series [a-d]|investor|venture|valuation)\b", re.I)),
    ("retention", re.compile(r"\b(retention|nrr|net retention|logo)\b", re.I)),
    ("ai", re.compile(r"\b(ai|gpt|llm|agent|model)\b", re.I)),
)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def estimate_tokens(text: str) -> int:
    """Approximate tokens as characters/4. This is not a model tokenizer count."""
    return max(1, (len(text) + CHARS_PER_TOKEN_ESTIMATE - 1) // CHARS_PER_TOKEN_ESTIMATE)


def estimate_chunks(text: str, chunk_token_size: int = CHUNK_TOKEN_SIZE) -> int:
    return max(1, (estimate_tokens(text) + chunk_token_size - 1) // chunk_token_size)


def topic_labels(text: str, medium: str) -> list[str]:
    labels = [str(medium or "unknown")]
    for name, pattern in TOPIC_PATTERNS:
        if pattern.search(text or ""):
            labels.append(name)
    if len(labels) == 1:
        labels.append("general")
    return labels


def _length_score(words: int, medium: str) -> float:
    if medium == "x":
        if words < 8:
            return 2.0
        if words < 40:
            return 8.0
        return 6.0
    if medium == "linkedin":
        if words < 40:
            return 4.0
        if words <= 400:
            return 14.0
        if words <= 900:
            return 10.0
        return 6.0
    if words < 80:
        return 4.0
    if words <= 1200:
        return 16.0
    if words <= 2500:
        return 10.0
    return 4.0


def _complexity_penalty(chunks: int) -> float:
    if chunks <= 3:
        return 0.0
    if chunks <= 8:
        return 4.0
    return 10.0 + chunks


def priority_score(document: dict) -> float:
    medium = str(document.get("medium") or "unknown")
    words = int(document.get("word_count") or 0)
    chunks = estimate_chunks(document.get("text") or "")
    score = ATTRIBUTION_WEIGHT.get(str(document.get("attribution_status") or ""), 10.0)
    score += MEDIUM_WEIGHT.get(medium, 4.0)
    score += _length_score(words, medium)
    score -= _complexity_penalty(chunks)
    if int(document.get("numeric_surface_count") or 0) > 0:
        score += 4.0
    method = document.get("reconstruction_method")
    if method == "cleaned_document":
        score += 3.0
    elif method == "ordered_eligible_chunks":
        score += 1.0
    if OMISSION_MARKER in (document.get("text") or ""):
        score -= 6.0
    flags = set(document.get("flags") or [])
    if "continuation_fragment" in flags:
        score -= 3.0
    if "contains_third_party_quotation" in flags:
        score -= 2.0
    return round(score, 4)


def _opening_key(text: str) -> str:
    words = re.findall(r"[A-Za-z0-9$%]+", (text or "").lower())
    return " ".join(words[:40])


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def build_schedule(
    documents: list[dict],
    *,
    corpus_hash: str,
    fingerprint: str,
    indexing_mode: str,
) -> tuple[list[dict], dict]:
    """Return ledger rows and a corpus summary.

    Exact ``text_sha256`` duplicates keep the higher-priority copy and mark
    the rest skipped. Historical pilot failures stay pending but deferred.
    """
    included = []
    quarantined = []
    for document in documents:
        reasons = list(document.get("exclusion_reasons") or [])
        text = document.get("text") or ""
        if (
            not document.get("included", True)
            or reasons
            or document.get("split") not in {None, "train"}
            or not text.strip()
        ):
            quarantined.append(document)
        else:
            included.append(document)

    scored = []
    for document in included:
        text = document.get("text") or ""
        scored.append(
            {
                "document": document,
                "score": priority_score(document),
                "chunks": estimate_chunks(text),
                "tokens": estimate_tokens(text),
                "topics": topic_labels(text, str(document.get("medium") or "unknown")),
                "opening": _opening_key(text),
            }
        )
    scored.sort(key=lambda item: (-item["score"], item["document"]["graph_document_id"]))

    seen_hash: dict[str, str] = {}
    seen_opening: dict[str, str] = {}
    eligible_rows = []
    skipped_dupes = 0
    for item in scored:
        document = item["document"]
        digest = document.get("text_sha256") or hashlib.sha256(document["text"].encode("utf-8")).hexdigest()
        doc_id = document["graph_document_id"]
        if digest in seen_hash:
            eligible_rows.append(
                _row(
                    document,
                    item,
                    corpus_hash,
                    fingerprint,
                    indexing_mode,
                    status="skipped",
                    selected=False,
                    rank=None,
                    defer_reason="duplicate_text_sha256",
                    near_duplicate_of=seen_hash[digest],
                    quality_notes="exact text hash already queued",
                )
            )
            skipped_dupes += 1
            continue
        seen_hash[digest] = doc_id
        near = None
        opening = item["opening"]
        if opening and opening in seen_opening and seen_opening[opening] != doc_id:
            near = seen_opening[opening]
            item["score"] = round(item["score"] - 25.0, 4)
        elif opening:
            seen_opening[opening] = doc_id
        item["near_duplicate_of"] = near
        eligible_rows.append(item)

    queue = [item for item in eligible_rows if isinstance(item, dict) and "document" in item]
    buckets: dict[str, list[dict]] = defaultdict(list)
    for item in sorted(queue, key=lambda entry: (-entry["score"], entry["document"]["graph_document_id"])):
        topic = item["topics"][1] if len(item["topics"]) > 1 else "general"
        bucket = f"{item['document'].get('medium') or 'unknown'}:{topic}"
        buckets[bucket].append(item)
    ordered = []
    names = sorted(buckets)
    while any(buckets[name] for name in names):
        for name in names:
            if buckets[name]:
                ordered.append(buckets[name].pop(0))

    rows = []
    rank = 0
    deferred = 0
    for item in ordered:
        document = item["document"]
        defer = None
        selected = True
        if document["graph_document_id"] in HISTORICAL_PILOT_IDS:
            defer = "pilot160_not_retried"
            selected = False
            deferred += 1
        else:
            rank += 1
        rows.append(
            _row(
                document,
                item,
                corpus_hash,
                fingerprint,
                indexing_mode,
                status="pending",
                selected=selected,
                rank=rank if selected else None,
                defer_reason=defer,
                near_duplicate_of=item.get("near_duplicate_of"),
                quality_notes=None,
            )
        )
    for item in eligible_rows:
        if isinstance(item, dict) and "document_id" in item:
            rows.append(item)
    for document in quarantined:
        text = document.get("text") or ""
        reasons = list(document.get("exclusion_reasons") or [])
        if not text.strip():
            reasons.append("empty_text")
        if document.get("split") not in {None, "train"}:
            reasons.append("non_train")
        placeholder = {
            "score": 0.0,
            "chunks": estimate_chunks(text) if text else 0,
            "tokens": estimate_tokens(text) if text else 0,
            "topics": topic_labels(text, str(document.get("medium") or "unknown")),
        }
        rows.append(
            _row(
                document,
                placeholder,
                corpus_hash,
                fingerprint,
                indexing_mode,
                status="quarantined",
                selected=False,
                rank=None,
                defer_reason=None,
                near_duplicate_of=None,
                quality_notes="; ".join(reasons) or "excluded by graph-ready policy",
            )
        )

    selected_rows = [row for row in rows if row["selected"]]
    summary = {
        "documents_seen": len(documents),
        "eligible": sum(1 for row in rows if row["status"] != "quarantined"),
        "selected": len(selected_rows),
        "deferred_pilot": deferred,
        "skipped_duplicate_text": skipped_dupes,
        "quarantined": sum(1 for row in rows if row["status"] == "quarantined"),
        "by_medium": _count(selected_rows, "source_medium"),
        "by_attribution": _count(selected_rows, "source_attribution_status"),
    }
    return rows, summary


def _count(rows: list[dict], key: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        name = str(row.get(key) or "unknown")
        counts[name] = counts.get(name, 0) + 1
    return counts


def _row(
    document: dict,
    item: dict,
    corpus_hash: str,
    fingerprint: str,
    indexing_mode: str,
    *,
    status: str,
    selected: bool,
    rank: int | None,
    defer_reason: str | None,
    near_duplicate_of: str | None,
    quality_notes: str | None,
) -> dict:
    text = document.get("text") or ""
    doc_id = document["graph_document_id"]
    return {
        "document_id": doc_id,
        "original_source_id": document.get("parent_graph_document_id") or doc_id,
        "source_medium": document.get("medium"),
        "source_attribution_status": document.get("attribution_status"),
        "topic_labels": item["topics"],
        "source_path": f"{document.get('source_file')}:{document.get('source_line')}",
        "source_hash": document.get("text_sha256")
        or hashlib.sha256(text.encode("utf-8")).hexdigest(),
        "corpus_version_hash": corpus_hash,
        "estimated_input_tokens": item["tokens"],
        "estimated_chunks": item["chunks"],
        "word_count": int(document.get("word_count") or len(text.split())),
        "has_omission": OMISSION_MARKER in text,
        "reconstruction_method": document.get("reconstruction_method"),
        "selection_priority": item["score"],
        "selection_rank": rank,
        "selected": selected,
        "indexing_mode": indexing_mode,
        "configuration_fingerprint": fingerprint,
        "status": status,
        "defer_reason": defer_reason,
        "near_duplicate_of": near_duplicate_of,
        "quality_notes": quality_notes,
    }


def validation_batch(rows: list[dict]) -> list[str]:
    """One short X post, one LinkedIn post, one moderate blog, one omission.

    Chosen from the already ranked selected rows so the sample is deterministic.
    """
    ranked = [
        row
        for row in rows
        if row.get("selected") and row.get("status") == "pending" and not row.get("defer_reason")
    ]
    ranked.sort(key=lambda row: (row.get("selection_rank") is None, row.get("selection_rank") or 0, row["document_id"]))

    def pick(predicate) -> str | None:
        for row in ranked:
            if predicate(row):
                return row["document_id"]
        return None

    chosen = []
    for doc_id in (
        pick(lambda row: row.get("source_medium") == "x" and int(row.get("word_count") or 0) >= 12),
        pick(lambda row: row.get("source_medium") == "linkedin" and int(row.get("word_count") or 0) >= 80),
        pick(
            lambda row: row.get("source_medium") == "blog"
            and not row.get("has_omission")
            and 1 <= int(row.get("estimated_chunks") or 0) <= 3
        ),
        pick(lambda row: bool(row.get("has_omission")) and int(row.get("estimated_chunks") or 99) <= 6),
    ):
        if doc_id and doc_id not in chosen:
            chosen.append(doc_id)
    return chosen
