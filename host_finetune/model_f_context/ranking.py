"""Deterministic evidence ranking. Scores from different retrievers stay apart.

LightRAG weights and Chroma distances are never added together. The order is a
lexicographic key: support, query overlap, specificity, provenance, redundancy,
then the original retrieval order and a stable id.
"""

from __future__ import annotations

import re

from host_finetune.graph_relationships import is_weak_entity

_STOP = {
    "the", "and", "for", "with", "that", "this", "from", "into", "about", "what",
    "does", "jason", "lemkin", "say", "how", "why", "when", "where", "who", "are",
    "was", "were", "have", "has", "had", "not", "but", "you", "your", "their",
    "them", "they", "his", "her", "its", "after", "before", "only", "than", "then",
    "into", "over", "under", "a", "an", "of", "to", "in", "on", "or", "as", "at",
    "by", "be", "is", "it",
}
_YEAR = re.compile(r"\b(?:19|20)\d{2}\b")
_TOKEN = re.compile(r"[a-z0-9$%]+")


def query_terms(text: str) -> set[str]:
    terms = set()
    for token in _TOKEN.findall((text or "").casefold()):
        if len(token) < 3 or token in _STOP:
            continue
        terms.add(token)
    return terms


def _years(text: str) -> set[str]:
    return set(_YEAR.findall(text or ""))


def _overlap(terms: set[str], text: str) -> int:
    if not terms:
        return 0
    return len(terms & query_terms(text))


def _content_words(text: str) -> set[str]:
    return {token for token in _TOKEN.findall((text or "").casefold()) if len(token) >= 4 and token not in _STOP}


def rank_evidence(
    *,
    query: str,
    primary: list,
    relationships: list,
    descriptions: list,
    paths: list,
    profile: str,
) -> dict:
    """Return ordered units and an audit row for every candidate."""
    terms = query_terms(query)
    query_years = _years(query)
    passage_by_doc: dict[str, list[str]] = {}
    for item in primary:
        passage_by_doc.setdefault(item.document_id, []).append(item.passage)
        parent = (item.leakage_lineage or {}).get("parent_graph_document_id")
        if parent:
            passage_by_doc.setdefault(str(parent), []).append(item.passage)

    def temporal(text: str) -> int:
        found = _years(text)
        if not query_years or not found:
            return 0
        if query_years & found:
            return 1
        return -1

    def redundant_relationship(item) -> bool:
        words = _content_words(item.predicate or "")
        if len(words) < 2:
            return False
        documents = set(item.supporting_document_ids)
        for document_id in documents:
            for passage in passage_by_doc.get(document_id, []):
                folded = passage.casefold()
                if words and all(word in folded for word in words):
                    return True
        return False

    units = []

    def add(unit: dict) -> None:
        unit["exclude_reasons"] = [reason for reason in unit["exclude_reasons"] if reason]
        units.append(unit)

    for index, item in enumerate(primary):
        text = item.passage
        add({
            "evidence_id": item.evidence_id,
            "channel": "primary_source",
            "text": text,
            "priority": 4,
            "query_overlap": _overlap(terms, text),
            "temporal": temporal(text),
            "specificity": min(len(_content_words(text)), 12),
            "provenance": 5 if item.document_id and item.passage else 1,
            "redundant": False,
            "retrieval_order": item.rank if item.rank is not None else index,
            "weight_hint": 0,
            "score_type": item.score_type,
            "exclude_reasons": [],
            "record": item,
        })

    relationship_overlaps = []
    prepared_relationships = []
    for index, item in enumerate(relationships):
        searchable = " ".join([
            item.subject, item.object, item.predicate or "", " ".join(item.keywords),
        ])
        overlap = _overlap(terms, searchable)
        relationship_overlaps.append(overlap)
        reasons = []
        if item.support_status != "direct" or item.validation_decision != "accepted" or not item.predicate:
            reasons.append("unsupported_or_inferred")
        if is_weak_entity(item.subject) or is_weak_entity(item.object):
            named = item.subject.casefold() in (query or "").casefold() or item.object.casefold() in (query or "").casefold()
            if not named:
                reasons.append("weak_entity")
        if not item.predicate:
            reasons.append("missing_predicate")
        redundant = redundant_relationship(item)
        provenance = 0
        if item.verbatim_evidence:
            provenance += 2
        if item.supporting_document_ids:
            provenance += 2
        if item.supporting_chunk_ids or item.graph_source_ids:
            provenance += 1
        prepared_relationships.append({
            "evidence_id": item.relationship_id,
            "channel": "source_supported_relationship",
            "text": f"{item.subject} — {item.predicate} — {item.object}" if item.predicate else f"{item.subject} — {item.object}",
            "priority": 3 if item.support_status == "direct" and not reasons else 0,
            "query_overlap": overlap,
            "temporal": temporal(" ".join(item.temporal_qualifiers + item.numerical_qualifiers + [item.predicate or ""])),
            "specificity": len((item.predicate or "").split()),
            "provenance": provenance,
            "redundant": redundant,
            "retrieval_order": index,
            "weight_hint": (item.retrieval_score or 0) if item.score_type == "lightrag_weight" else 0,
            "score_type": item.score_type,
            "exclude_reasons": reasons,
            "record": item,
        })
    any_relationship_overlap = any(value > 0 for value in relationship_overlaps)
    if profile in {"relationships", "descriptions", "multihop"} and any_relationship_overlap:
        for unit in prepared_relationships:
            if unit["query_overlap"] == 0 and "unsupported_or_inferred" not in unit["exclude_reasons"]:
                unit["exclude_reasons"].append("unrelated_to_request")
    elif profile in {"relationships", "descriptions", "multihop"} and prepared_relationships:
        for unit in prepared_relationships:
            if unit["query_overlap"] == 0:
                unit["exclude_reasons"].append("no_query_overlap")
    if profile == "source":
        for unit in prepared_relationships:
            unit["exclude_reasons"].append("profile_excludes_channel")
    for unit in prepared_relationships:
        add(unit)

    description_overlaps = []
    prepared_descriptions = []
    unsupported_relationships = {
        item.relationship_id
        for item in relationships
        if item.support_status != "direct" or item.validation_decision != "accepted"
    }
    selected_relationship_text = " ".join(
        unit["text"] for unit in prepared_relationships if not unit["exclude_reasons"]
    ).casefold()
    primary_text = " ".join(item.passage for item in primary).casefold()
    for index, item in enumerate(descriptions):
        if item.validation_status == "rejected" or not item.description:
            prepared_descriptions.append({
                "evidence_id": item.description_id,
                "channel": "derived_description",
                "text": item.identity,
                "priority": 1,
                "query_overlap": 0,
                "temporal": 0,
                "specificity": 0,
                "provenance": 0,
                "redundant": False,
                "retrieval_order": index,
                "weight_hint": 0,
                "score_type": item.score_type,
                "exclude_reasons": ["rejected_description" if item.validation_status == "rejected" else "empty_description"],
                "record": item,
            })
            description_overlaps.append(0)
            continue
        searchable = f"{item.identity} {item.description}"
        overlap = _overlap(terms, searchable)
        description_overlaps.append(overlap)
        reasons = []
        if item.kind == "relationship" and item.identity in unsupported_relationships:
            reasons.append("unsupported_relationship_description")
        if item.kind == "entity" and is_weak_entity(item.identity) and item.identity.casefold() not in (query or "").casefold():
            reasons.append("weak_entity")
        folded = re.sub(r"\s+", " ", item.description).casefold()
        if folded and (folded in selected_relationship_text or folded in primary_text):
            reasons.append("repetitive_description")
        if item.classification != "derived_context":
            reasons.append("misclassified_description")
        prepared_descriptions.append({
            "evidence_id": item.description_id,
            "channel": "derived_description",
            "text": f"{item.identity}: {item.description}",
            "priority": 1,
            "query_overlap": overlap,
            "temporal": temporal(item.description),
            "specificity": min(len(_content_words(item.description)), 12),
            "provenance": 2 if item.source_ids or item.file_paths else 0,
            "redundant": "repetitive_description" in reasons,
            "retrieval_order": index,
            "weight_hint": (item.retrieval_score or 0) if item.score_type == "lightrag_weight" else 0,
            "score_type": item.score_type,
            "exclude_reasons": reasons,
            "record": item,
        })
    any_description_overlap = any(value > 0 for value in description_overlaps)
    if profile in {"descriptions", "multihop"} and any_description_overlap:
        for unit, overlap in zip(prepared_descriptions, description_overlaps):
            if overlap == 0 and "rejected_description" not in unit["exclude_reasons"]:
                unit["exclude_reasons"].append("unrelated_to_request")
    elif profile in {"descriptions", "multihop"} and prepared_descriptions:
        for unit in prepared_descriptions:
            unit["exclude_reasons"].append("no_query_overlap")
    if profile not in {"descriptions", "multihop"}:
        for unit in prepared_descriptions:
            unit["exclude_reasons"].append("profile_excludes_channel")
    for unit in prepared_descriptions:
        add(unit)

    path_overlaps = []
    prepared_paths = []
    for index, item in enumerate(paths):
        reasons = []
        if profile != "multihop":
            reasons.append("profile_excludes_channel")
        if item.hops > 2:
            reasons.append("path_too_long")
        if not item.edge_ids:
            reasons.append("empty_path")
        overlap = _overlap(terms, " ".join(item.edge_facts))
        path_overlaps.append(overlap)
        prepared_paths.append((index, item, reasons, overlap))
    any_path_overlap = any(value > 0 for value in path_overlaps)
    for index, item, reasons, overlap in prepared_paths:
        if profile == "multihop" and any_path_overlap and overlap == 0:
            reasons.append("unrelated_to_request")
        elif profile == "multihop" and path_overlaps and overlap == 0:
            reasons.append("no_query_overlap")
        add({
            "evidence_id": item.path_id,
            "channel": "multi_hop_path",
            "text": item.display,
            "priority": 2,
            "query_overlap": overlap,
            "temporal": temporal(item.display),
            "specificity": item.hops,
            "provenance": len(item.edge_provenance),
            "redundant": False,
            "retrieval_order": index,
            "weight_hint": 0,
            "score_type": "path_structure",
            "exclude_reasons": reasons,
            "record": item,
        })

    def sort_key(unit: dict) -> tuple:
        weight = unit["weight_hint"] if unit["score_type"] == "lightrag_weight" else 0
        return (
            0 if not unit["exclude_reasons"] else 1,
            -unit["priority"],
            -unit["query_overlap"],
            -unit["temporal"],
            -unit["specificity"],
            -unit["provenance"],
            1 if unit["redundant"] else 0,
            -weight,
            unit["retrieval_order"],
            unit["evidence_id"],
        )

    units.sort(key=sort_key)
    audit = []
    for unit in units:
        reasons = list(unit["exclude_reasons"])
        if not reasons:
            reasons = [
                f"priority:{unit['priority']}",
                f"query_overlap:{unit['query_overlap']}",
                f"provenance:{unit['provenance']}",
                f"score_type:{unit['score_type']}",
                f"tie_break:{unit['evidence_id']}",
            ]
        audit.append({
            "evidence_id": unit["evidence_id"],
            "channel": unit["channel"],
            "decision": "excluded" if unit["exclude_reasons"] else "selected_candidate",
            "reasons": reasons,
        })
    grouped = {
        "primary_source": [unit for unit in units if unit["channel"] == "primary_source"],
        "source_supported_relationship": [unit for unit in units if unit["channel"] == "source_supported_relationship"],
        "multi_hop_path": [unit for unit in units if unit["channel"] == "multi_hop_path"],
        "derived_description": [unit for unit in units if unit["channel"] == "derived_description"],
    }
    return {"units": units, "grouped": grouped, "audit": audit}
