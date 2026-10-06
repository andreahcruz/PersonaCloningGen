"""Candidate pool and judgment schema for retrieval qrels.

Quality flags are not relevance grades. Missing judgments are not grade 0.
A later retriever's unseen candidate stays unlabeled until it is judged.
"""
from __future__ import annotations

import hashlib
from collections import Counter

from host_finetune.retrieval_eval import chunk_flags
from host_finetune.split_groups import _jaccard, matches_gold_field, tokens

NEAR_DUPLICATE_JACCARD = 0.80
GRADES = (0, 1, 2, 3)
POOL_ORIGIN = "dense_rag_v1"


def text_sha256(text: str) -> str:
    return hashlib.sha256((text or "").encode("utf-8")).hexdigest()


def quality_flags(
    text: str,
    *,
    sft_role: str | None,
    topic: str,
    earlier_hits: list[dict],
    group_id: str | None,
) -> dict:
    """Deterministic chunk notes. None of these fields is a relevance grade."""
    marked = chunk_flags(text)
    own_tokens = tokens((text or "").casefold())
    duplicate_of = None
    for earlier in earlier_hits:
        if group_id and earlier.get("group_id") == group_id:
            duplicate_of = earlier.get("document_id") or earlier.get("id")
            break
    near_duplicate = None
    for earlier in earlier_hits:
        if group_id and earlier.get("group_id") == group_id:
            continue
        score = _jaccard(own_tokens, tokens((earlier.get("text") or "").casefold()))
        if score >= NEAR_DUPLICATE_JACCARD:
            near_duplicate = earlier.get("document_id") or earlier.get("id")
            break
    overlap, _score = matches_gold_field(text, "", topic, "topic")
    has_url = "http://" in (text or "") or "https://" in (text or "") or "www." in (text or "")
    return {
        "fragment": bool(marked["mid_list"] or marked["starts_lower"] or marked["very_short"]),
        "continuation": sft_role == "continuation",
        "duplicate_of": duplicate_of,
        "near_duplicate": near_duplicate,
        "source_family_redundant": duplicate_of is not None,
        "headline_only": bool(marked["headline_or_link"]),
        "URL_heavy": bool(has_url and marked["short"]),
        "insufficient_context": bool(
            marked["very_short"] or marked["starts_lower"] or (sft_role == "continuation" and marked["mid_list"])
        ),
        "held_out_overlap_suspected": bool(overlap),
        "reviewer_notes": "",
    }


def annotate_query(query_id: str, topic: str, hits: list[dict], provenance: dict[int, dict]) -> list[dict]:
    """Attach provenance and flags to one raw top-N list. Order is rank."""
    annotated = []
    for rank, hit in enumerate(hits, 1):
        row_id = hit.get("row_id")
        source = provenance.get(int(row_id), {}) if row_id is not None else {}
        text = hit.get("text") or ""
        flags = quality_flags(
            text,
            sft_role=source.get("sft_role"),
            topic=topic,
            earlier_hits=annotated,
            group_id=hit.get("group_id"),
        )
        annotated.append(
            {
                "query_id": query_id,
                "topic": topic,
                "rank": rank,
                "document_id": hit.get("id"),
                "cosine_distance": hit.get("distance"),
                "group_id": hit.get("group_id"),
                "medium": hit.get("medium"),
                "row_id": row_id,
                "split": hit.get("split") or source.get("split"),
                "text": text,
                "candidate_text_hash": text_sha256(text),
                "sft_role": source.get("sft_role"),
                "part_index": source.get("part_index"),
                "part_count": source.get("part_count"),
                "source_file": source.get("source_file"),
                "source_line": source.get("source_line"),
                "source_url": source.get("source_url"),
                "source_id": source.get("source_id"),
                "source_config": "factual_dense_rag_v1",
                "pool_origin": POOL_ORIGIN,
                **flags,
            }
        )
    return annotated


def judgment_view(raw_hits: list[dict]) -> list[dict]:
    """One labeling unit per group, keeping the best-ranked chunk.

    Later raw hits from the same group stay in the raw trace. They are listed
    as siblings here and are not dropped.
    """
    units = []
    by_group: dict[str, dict] = {}
    for hit in raw_hits:
        group_id = hit.get("group_id") or f"doc:{hit.get('document_id')}"
        current = by_group.get(group_id)
        if current is None:
            unit = {
                "query_id": hit["query_id"],
                "document_id": hit["document_id"],
                "group_id": hit.get("group_id"),
                "rank": hit["rank"],
                "cosine_distance": hit.get("cosine_distance"),
                "medium": hit.get("medium"),
                "candidate_text_hash": hit["candidate_text_hash"],
                "text": hit["text"],
                "sibling_document_ids": [],
                "source_config": hit["source_config"],
                "pool_origin": hit["pool_origin"],
                "fragment": hit["fragment"],
                "continuation": hit["continuation"],
                "headline_only": hit["headline_only"],
                "URL_heavy": hit["URL_heavy"],
                "insufficient_context": hit["insufficient_context"],
                "held_out_overlap_suspected": hit["held_out_overlap_suspected"],
            }
            by_group[group_id] = unit
            units.append(unit)
        else:
            current["sibling_document_ids"].append(hit["document_id"])
    return units


def accepted_qrels(judgments: list[dict]) -> list[dict]:
    """Return evaluator qrels for accepted grades only.

    Unlabeled, proposed, and rejected rows are ignored. Grade 0 is a real
    judgment of irrelevant, not the default for a missing row.
    """
    rows = []
    for row in judgments:
        if row.get("record_type") == "meta":
            continue
        if row.get("review_status") != "accepted":
            continue
        grade = row.get("relevance_grade")
        if grade not in GRADES:
            continue
        if not row.get("query_id") or not row.get("document_id"):
            continue
        rows.append(
            {
                "query_id": row["query_id"],
                "doc_id": row["document_id"],
                "grade": int(grade),
            }
        )
    return rows


def inherit_group_grades(judgments: list[dict], raw_hits: list[dict]) -> list[dict]:
    """Copy an accepted representative grade onto same-group siblings.

    An explicit accepted judgment for a sibling wins over inheritance.
    Unjudged groups stay unlabeled.
    """
    accepted = {
        (row["query_id"], row["document_id"]): int(row["relevance_grade"])
        for row in judgments
        if row.get("review_status") == "accepted" and row.get("relevance_grade") in GRADES
    }
    representative_grade: dict[tuple[str, str, str], int] = {}
    for hit in raw_hits:
        origin = hit.get("pool_origin") or ""
        key = (hit["query_id"], origin, hit.get("group_id") or "")
        if key[2] and key not in representative_grade:
            grade = accepted.get((hit["query_id"], hit["document_id"]))
            if grade is not None:
                representative_grade[key] = grade
    expanded = []
    seen = set()
    for hit in raw_hits:
        query_id = hit["query_id"]
        document_id = hit["document_id"]
        explicit = accepted.get((query_id, document_id))
        inherited = representative_grade.get((query_id, hit.get("pool_origin") or "", hit.get("group_id") or ""))
        if explicit is None and inherited is None:
            continue
        marker = (query_id, document_id)
        if marker in seen:
            continue
        seen.add(marker)
        expanded.append(
            {
                "query_id": query_id,
                "doc_id": document_id,
                "grade": explicit if explicit is not None else inherited,
            }
        )
    return expanded


def pool_diagnostics(traces: list[dict], ks: tuple[int, ...] = (4, 10, 20)) -> dict:
    """Unjudged retrieval diagnostics. This function does not emit ranked IR metrics."""
    per_k = {k: [] for k in ks}
    distances = []
    platforms = Counter()
    empty = 0
    duplicate_docs = 0
    duplicate_groups = 0
    fragment = 0
    headline = 0
    short = 0
    n_hits = 0
    for trace in traces:
        hits = trace.get("hits") or []
        if not hits:
            empty += 1
        ids = [hit.get("document_id") for hit in hits]
        groups = [hit.get("group_id") for hit in hits if hit.get("group_id")]
        if len(ids) != len(set(ids)):
            duplicate_docs += 1
        if len(groups) != len(set(groups)):
            duplicate_groups += 1
        for hit in hits:
            n_hits += 1
            if isinstance(hit.get("cosine_distance"), (int, float)):
                distances.append(hit["cosine_distance"])
            if hit.get("medium"):
                platforms[hit["medium"]] += 1
            if hit.get("fragment"):
                fragment += 1
            if hit.get("headline_only"):
                headline += 1
            marked = chunk_flags(hit.get("text") or "")
            if marked["short"]:
                short += 1
        for k in ks:
            window = hits[:k]
            groups_k = [hit.get("group_id") for hit in window if hit.get("group_id")]
            unique = len(set(groups_k))
            per_k[k].append(
                {
                    "unique_groups": unique,
                    "extra_same_group_fraction": ((len(window) - unique) / len(window)) if window else None,
                    "unique_platforms": len({hit.get("medium") for hit in window if hit.get("medium")}),
                }
            )
    n_queries = len(traces) or 1
    summary = {
        "n_queries": len(traces),
        "raw_hits": n_hits,
        "empty_retrieval_rate": empty / n_queries if traces else None,
        "duplicate_document_rate": duplicate_docs / n_queries if traces else None,
        "duplicate_group_rate": duplicate_groups / n_queries if traces else None,
        "mean_cosine_distance": sum(distances) / len(distances) if distances else None,
        "median_cosine_distance": _median(distances),
        "fragment_rate": fragment / n_hits if n_hits else None,
        "headline_only_rate": headline / n_hits if n_hits else None,
        "short_chunk_rate": short / n_hits if n_hits else None,
        "platform_counts": dict(platforms),
        "ranked_metrics_omitted": True,
        "reason": "no accepted relevance judgments",
    }
    for k, rows in per_k.items():
        summary[f"mean_distinct_groups_at_{k}"] = _mean(row["unique_groups"] for row in rows)
        summary[f"mean_extra_same_group_fraction_at_{k}"] = _mean(
            row["extra_same_group_fraction"] for row in rows
        )
        summary[f"mean_source_diversity_at_{k}"] = _mean(row["unique_platforms"] for row in rows)
    return summary


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
