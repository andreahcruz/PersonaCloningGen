"""Retrieval metrics that stay separate from generation quality.

Ranked metrics are emitted only when the qrels say how complete they are.
Unjudged retrieved chunks are never treated as ground-truth relevance.
"""
from __future__ import annotations

import math
import re
from collections import Counter

RELEVANT_GRADE = 2
_LIST_START = re.compile(r"^(?:#\d+|\d+\.)\b")
_SENT_END = re.compile(r"[.!?…][\"')\]]*$")


def chunk_flags(text: str) -> dict:
    """Surface chunk problems. These are diagnostics, not relevance labels."""
    stripped = (text or "").strip()
    words = stripped.split()
    has_url = "http://" in stripped or "https://" in stripped or "www." in stripped
    return {
        "words": len(words),
        "very_short": len(words) < 20,
        "short": len(words) < 40,
        "mid_list": bool(_LIST_START.match(stripped)),
        "headline_or_link": has_url and len(words) < 30,
        "no_sentence_end": bool(stripped) and _SENT_END.search(stripped) is None,
        "starts_lower": bool(stripped[:1].islower()),
    }


def _grades_for(query_id: str, qrels: list[dict]) -> dict[str, int]:
    grades: dict[str, int] = {}
    for row in qrels:
        if row.get("record_type") == "meta":
            continue
        if row.get("query_id") != query_id:
            continue
        doc_id = row.get("doc_id")
        if not doc_id:
            continue
        grades[str(doc_id)] = int(row.get("grade") or 0)
    return grades


def _lookup(grades: dict[str, int], doc_id: str, *, missing_as_zero: bool) -> int | None:
    if doc_id in grades:
        return grades[doc_id]
    if missing_as_zero:
        return 0
    return None


def _dcg(grades: list[int]) -> float:
    total = 0.0
    for index, grade in enumerate(grades):
        total += (2 ** grade - 1) / math.log2(index + 2)
    return total


def _precision(hits: list[dict], grades: dict[str, int], k: int, min_grade: int, *, missing_as_zero: bool) -> float | None:
    top = hits[:k]
    if len(top) < k:
        return None
    values = [_lookup(grades, hit.get("id"), missing_as_zero=missing_as_zero) for hit in top]
    if any(value is None for value in values):
        return None
    relevant = sum(value >= min_grade for value in values)
    return relevant / k


def _mrr(hits: list[dict], grades: dict[str, int], min_grade: int, *, missing_as_zero: bool) -> float | None:
    values = [_lookup(grades, hit.get("id"), missing_as_zero=missing_as_zero) for hit in hits]
    if any(value is None for value in values):
        return None
    for index, value in enumerate(values, 1):
        if value >= min_grade:
            return 1 / index
    return 0.0


def _ndcg(hits: list[dict], grades: dict[str, int], k: int, *, missing_as_zero: bool) -> float | None:
    top = hits[:k]
    if len(top) < k:
        return None
    values = [_lookup(grades, hit.get("id"), missing_as_zero=missing_as_zero) for hit in top]
    if any(value is None for value in values):
        return None
    ideal_source = list(grades.values())
    if missing_as_zero:
        ideal_source.extend(0 for _ in range(max(0, k - len(ideal_source))))
    ideal = sorted(ideal_source, reverse=True)[:k]
    idcg = _dcg(ideal)
    if idcg == 0:
        return None
    return _dcg(values) / idcg


def _recall(hits: list[dict], grades: dict[str, int], k: int, min_grade: int) -> float | None:
    relevant_ids = {doc_id for doc_id, grade in grades.items() if grade >= min_grade}
    if not relevant_ids:
        return None
    found = {hit.get("id") for hit in hits[:k]} & relevant_ids
    return len(found) / len(relevant_ids)


def evaluate_query(
    query_id: str,
    hits: list[dict],
    qrels: list[dict],
    *,
    k_values: tuple[int, ...] = (1, 4),
    min_grade: int = RELEVANT_GRADE,
    relevance_scope: str = "none",
) -> dict:
    """Score one retrieved list.

    ``relevance_scope`` is ``none``, ``judged_pool``, or ``corpus``.
    Corpus Recall@k is returned only for ``corpus``. Pool recall is named
    separately and is not corpus recall. Under ``corpus``, a document missing
    from the qrels is grade 0. Under ``judged_pool``, a missing grade leaves
    Precision, MRR, and nDCG unset.
    """
    grades = _grades_for(query_id, qrels)
    ids = [hit.get("id") for hit in hits]
    groups = [hit.get("group_id") for hit in hits if hit.get("group_id")]
    distances = [hit.get("distance") for hit in hits if isinstance(hit.get("distance"), (int, float))]
    platforms = [hit.get("medium") or hit.get("source_platform") for hit in hits]
    judged = [hit for hit in hits if hit.get("id") in grades]
    relevant_judged = [hit for hit in judged if grades[hit["id"]] >= min_grade]
    flags = [chunk_flags(hit.get("text") or "") for hit in hits]
    missing_as_zero = relevance_scope == "corpus"
    ranked_ready = relevance_scope in {"judged_pool", "corpus"} and (bool(grades) or missing_as_zero)
    row = {
        "query_id": query_id,
        "n_hits": len(hits),
        "empty": len(hits) == 0,
        "duplicate_document_ids": len(ids) != len(set(ids)),
        "duplicate_group_ids": len(groups) != len(set(groups)),
        "distances": distances,
        "platforms": platforms,
        "unique_platforms": len({item for item in platforms if item}),
        "unique_groups": len(set(groups)),
        "fragment_hits": sum(
            1 for flag in flags if flag["mid_list"] or flag["headline_or_link"] or flag["very_short"]
        ),
        "judged_hits": len(judged),
        "unjudged_hits": len(hits) - len(judged),
        "judged_relevant_hits": len(relevant_judged),
        "evidence_precision_among_judged": (
            len(relevant_judged) / len(judged) if judged else None
        ),
    }
    if ranked_ready:
        for k in k_values:
            row[f"precision_at_{k}"] = _precision(
                hits, grades, k, min_grade, missing_as_zero=missing_as_zero
            )
            row[f"ndcg_at_{k}"] = _ndcg(hits, grades, k, missing_as_zero=missing_as_zero)
        row["mrr"] = _mrr(hits, grades, min_grade, missing_as_zero=missing_as_zero)
        if relevance_scope == "corpus":
            for k in k_values:
                row[f"recall_at_{k}"] = _recall(hits, grades, k, min_grade)
        elif relevance_scope == "judged_pool":
            for k in k_values:
                row[f"pool_recall_at_{k}"] = _recall(hits, grades, k, min_grade)
    return row


def summarize_retrieval(
    queries: list[dict],
    qrels: list[dict] | None = None,
    *,
    k_values: tuple[int, ...] = (1, 4),
    min_grade: int = RELEVANT_GRADE,
    relevance_scope: str = "none",
) -> dict:
    """Aggregate per-query traces. Ranked IR metrics stay null without qrels."""
    qrels = qrels or []
    rows = [
        evaluate_query(
            item["query_id"],
            item.get("hits") or [],
            qrels,
            k_values=k_values,
            min_grade=min_grade,
            relevance_scope=relevance_scope,
        )
        for item in queries
    ]
    distances = [value for row in rows for value in row["distances"]]
    platforms = Counter(platform for row in rows for platform in row["platforms"] if platform)
    n = len(rows) or 1
    summary = {
        "n_queries": len(rows),
        "relevance_scope": relevance_scope,
        "relevant_grade_min": min_grade,
        "empty_retrieval_rate": sum(row["empty"] for row in rows) / n if rows else None,
        "duplicate_document_rate": sum(row["duplicate_document_ids"] for row in rows) / n if rows else None,
        "duplicate_group_rate": sum(row["duplicate_group_ids"] for row in rows) / n if rows else None,
        "mean_fragment_hits": (
            sum(row["fragment_hits"] for row in rows) / len(rows) if rows else None
        ),
        "mean_unique_platforms": (
            sum(row["unique_platforms"] for row in rows) / len(rows) if rows else None
        ),
        "mean_unique_groups": (
            sum(row["unique_groups"] for row in rows) / len(rows) if rows else None
        ),
        "platform_counts": dict(platforms),
        "mean_cosine_distance": sum(distances) / len(distances) if distances else None,
        "median_cosine_distance": _median(distances),
        "corpus_recall_reported": relevance_scope == "corpus",
        "reason_ranked_metrics_omitted": (
            None
            if relevance_scope in {"judged_pool", "corpus"}
            else "no complete relevance judgments; Precision, MRR, nDCG, and Recall are not reported"
        ),
    }
    if relevance_scope in {"judged_pool", "corpus"}:
        for k in k_values:
            summary[f"precision_at_{k}"] = _mean(row.get(f"precision_at_{k}") for row in rows)
            summary[f"ndcg_at_{k}"] = _mean(row.get(f"ndcg_at_{k}") for row in rows)
            if relevance_scope == "corpus":
                summary[f"recall_at_{k}"] = _mean(row.get(f"recall_at_{k}") for row in rows)
            else:
                summary[f"pool_recall_at_{k}"] = _mean(row.get(f"pool_recall_at_{k}") for row in rows)
        summary["mrr"] = _mean(row.get("mrr") for row in rows)
        judged = [
            row["evidence_precision_among_judged"]
            for row in rows
            if row["evidence_precision_among_judged"] is not None
        ]
        summary["evidence_precision_among_judged"] = sum(judged) / len(judged) if judged else None
    return {"summary": summary, "queries": rows}


def _mean(values) -> float | None:
    numbers = [value for value in values if isinstance(value, (int, float))]
    if not numbers:
        return None
    return sum(numbers) / len(numbers)


def _median(values: list[float]) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[mid]
    return (ordered[mid - 1] + ordered[mid]) / 2
