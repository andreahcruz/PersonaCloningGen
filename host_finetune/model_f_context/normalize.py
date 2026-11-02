"""Turn a LightRAG retrieval payload into the Model F evidence bundle.

This module does not open graph storage and does not call a model. It accepts
either a raw ``aquery_data`` payload or the project payload already produced
by ``normalize_lightrag_data``. Generated descriptions stay on channel C.
"""

from __future__ import annotations

import re
from typing import Any

from host_finetune.model_f_context.schema import (
    SEP,
    AuditRecord,
    DerivedDescription,
    RelationshipEvidence,
    SourceEvidence,
    description_identity,
    relationship_identity,
    source_identity,
    split_stored,
)

OMISSION_MARKER = "[Omitted same-source text"
_INJECTION = re.compile(
    r"ignore (?:all |any |previous |prior )?instructions|"
    r"system prompt|you are now|<\|im_start\|>|\[INST\]|"
    r"disregard (?:the |all )?(?:previous |prior )?(?:instructions|rules)|"
    r"reveal (?:the |your )?(?:system )?prompt",
    re.IGNORECASE,
)
_NUMBER = re.compile(
    r"\$\s*\d[\d,]*(?:\.\d+)?\s*(?:billion|million|thousand|k|m|b)?|"
    r"\b\d+(?:\.\d+)?\s*(?:%|percent)\b|"
    r"\b(?:19|20)\d{2}\b",
    re.IGNORECASE,
)


def _mode(payload: dict) -> str | None:
    metadata = payload.get("metadata") if isinstance(payload.get("metadata"), dict) else {}
    raw = payload.get("mode") or metadata.get("query_mode")
    if raw is None:
        return None
    return str(raw)


def _raw_data(payload: dict) -> dict:
    data = payload.get("data")
    return data if isinstance(data, dict) else {}


def _validation_bundle(payload: dict) -> dict | None:
    bundle = payload.get("relationships")
    if isinstance(bundle, dict) and (
        bundle.get("contract") == "source-verified-relationship-v1"
        or "accepted" in bundle
        or "rejected" in bundle
        or "quarantined" in bundle
    ):
        return bundle
    return None


def _lineage(row: dict) -> dict[str, Any]:
    groups = [str(item) for item in (row.get("group_ids") or []) if item]
    group = row.get("group_id")
    if group and str(group) not in groups:
        groups.append(str(group))
    return {
        "source_group": None if group is None else str(group),
        "group_ids": groups,
        "split": row.get("split"),
        "parent_graph_document_id": row.get("parent_graph_document_id"),
        "graph_document_id": row.get("graph_document_id") or row.get("file_path"),
        "source_file": row.get("source_file"),
        "source_line": row.get("source_line"),
        "source_row_ids": [str(item) for item in (row.get("source_row_ids") or row.get("experiment_row_ids") or [])],
        "duplicate_family": row.get("duplicate_family") or row.get("family_id"),
    }


def _reference_status(tokens: list[str], catalog: dict[str, dict] | None) -> str:
    if catalog is None:
        return "not_checked"
    if not tokens:
        return "missing"
    found = False
    for token in tokens:
        if token in catalog:
            found = True
            break
        if "-chunk-" in token and token.split("-chunk-", 1)[0] in catalog:
            found = True
            break
    return "resolved" if found else "unresolved"


def _score(value: Any, default_type: str) -> tuple[float | None, str]:
    if isinstance(value, bool) or value is None:
        return None, default_type
    if isinstance(value, (int, float)):
        return float(value), default_type
    return None, default_type


def _keywords(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    found = []
    for part in re.split(r"[,;]", str(value or "")):
        word = re.sub(r"\s+", " ", part).strip()
        if word and word.casefold() not in {"related_to", "related", "relationship"}:
            found.append(word)
    return found


def _numbers(text: str | None) -> list[str]:
    found = []
    for match in _NUMBER.finditer(text or ""):
        surface = re.sub(r"\s+", " ", match.group(0)).strip()
        if surface not in found:
            found.append(surface)
    return found


def _primary_from_hit(hit: dict, index: int) -> SourceEvidence | None:
    if not isinstance(hit, dict):
        return None
    passage = str(hit.get("text") or hit.get("content") or "").strip()
    if not passage:
        return None
    document_id = str(
        hit.get("graph_document_id")
        or hit.get("parent_graph_document_id")
        or hit.get("id")
        or ""
    )
    chunk_id = str(hit.get("chunk_id") or "")
    lineage = _lineage(hit)
    score, score_type = _score(hit.get("score"), str(hit.get("score_type") or "unscored_rank"))
    if hit.get("distance") is not None and score is None:
        score, score_type = _score(hit.get("distance"), "chroma_distance")
    rank = hit.get("rank")
    return SourceEvidence(
        evidence_id=source_identity(document_id, chunk_id, passage),
        document_id=document_id,
        chunk_id=chunk_id,
        passage=passage,
        medium=None if hit.get("medium") is None else str(hit.get("medium")),
        source_url=None if hit.get("source_url") is None else str(hit.get("source_url")),
        publication_date=None if hit.get("publication_date") is None else str(hit.get("publication_date")),
        attribution_status=None if hit.get("attribution_status") is None else str(hit.get("attribution_status")),
        author_position=None if hit.get("author_position") is None else str(hit.get("author_position")),
        quoted_speakers=[str(item) for item in (hit.get("quoted_speakers") or []) if item],
        retrieval_score=score,
        score_type=score_type,
        rank=int(rank) if isinstance(rank, int) else index + 1,
        source_group=lineage["source_group"],
        leakage_lineage=lineage,
    )


def _verbatim(chunk: str, source: str) -> str | None:
    if chunk and chunk in source:
        return chunk
    return None


def _primary_from_chunk(
    chunk: dict,
    catalog: dict[str, dict] | None,
    failed_ids: set[str],
    index: int,
) -> tuple[SourceEvidence | None, str | None]:
    if not isinstance(chunk, dict):
        return None, "malformed_chunk"
    content = str(chunk.get("content") or chunk.get("text") or "")
    file_path = str(chunk.get("file_path") or "")
    chunk_id = str(chunk.get("chunk_id") or "")
    if not content.strip():
        return None, "missing_passage"
    document = (catalog or {}).get(file_path)
    if catalog is not None and document is None:
        return None, "unresolved_source_reference"
    if document is not None:
        parent = str(document.get("parent_graph_document_id") or file_path)
        if file_path in failed_ids or parent in failed_ids:
            return None, "failed_document"
        if document.get("split") not in {None, "train"}:
            return None, "non_train_document"
        verbatim = _verbatim(content, str(document.get("text") or ""))
        if verbatim is None:
            return None, "chunk_not_in_source"
        if OMISSION_MARKER in verbatim:
            return None, "omission_marker_span"
        content = verbatim
        row = {
            **document,
            "chunk_id": chunk_id,
            "text": content,
            "graph_document_id": file_path,
        }
    else:
        row = {
            "text": content,
            "chunk_id": chunk_id,
            "graph_document_id": file_path,
            "file_path": file_path,
        }
    evidence = _primary_from_hit(row, index)
    return evidence, None


def _support_status(record: dict) -> str:
    decision = str(record.get("validation_status") or "")
    kind = str(record.get("support_kind") or "")
    if decision == "accepted" and kind == "directly_stated":
        return "direct"
    if kind == "inferred_multi_source" or decision == "quarantined":
        return "inferred"
    if decision == "rejected":
        return "unsupported"
    return "unverified"


def _relationship_from_validated(record: dict, weights: dict[str, float]) -> RelationshipEvidence | None:
    if not isinstance(record, dict):
        return None
    subject = str(record.get("subject") or record.get("src_id") or "").strip()
    object_ = str(record.get("object") or record.get("tgt_id") or "").strip()
    if not subject or not object_:
        return RelationshipEvidence(
            relationship_id=str(record.get("relationship_id") or relationship_identity(subject or "missing", object_ or "missing")),
            subject=subject,
            object=object_,
            predicate=None,
            keywords=_keywords(record.get("keywords")),
            support_status="unsupported",
            validation_decision="rejected",
            validation_reason=str(record.get("rejection_reason") or "malformed_relationship"),
            score_type="unscored_rank",
        )
    predicate = record.get("predicate")
    predicate_text = None if predicate in {None, ""} else str(predicate)
    span = record.get("support_span")
    span_text = None if not span else str(span)
    identity = str(record.get("relationship_id") or relationship_identity(subject, object_))
    chunk_ids = [str(item) for item in (record.get("graph_chunk_ids") or record.get("supporting_chunk_ids") or [])]
    documents = [str(item) for item in (record.get("source_document_ids") or record.get("graph_document_ids") or [])]
    weight = weights.get(identity)
    if weight is None:
        weight = weights.get(relationship_identity(subject, object_))
    return RelationshipEvidence(
        relationship_id=identity,
        subject=subject,
        object=object_,
        predicate=predicate_text,
        keywords=_keywords(record.get("keywords")),
        graph_source_ids=chunk_ids or split_stored(str(record.get("source_id") or "")),
        supporting_document_ids=documents,
        supporting_chunk_ids=chunk_ids,
        verbatim_evidence=span_text,
        support_status=_support_status(record),
        validation_decision=str(record.get("validation_status") or "not_validated"),
        validation_reason=None if record.get("rejection_reason") in {None, ""} else str(record.get("rejection_reason")),
        temporal_qualifiers=[str(item) for item in (record.get("qualifiers") or []) if item],
        numerical_qualifiers=_numbers(span_text),
        quoted_speakers=[str(item) for item in (record.get("quoted_speakers") or []) if item],
        retrieval_score=weight,
        score_type="lightrag_weight" if weight is not None else "unscored_rank",
        source_reference_status="resolved" if documents or chunk_ids else "missing",
    )


def _relationship_unvalidated(row: dict, *, reason: str, decision: str, status: str) -> RelationshipEvidence | None:
    if not isinstance(row, dict):
        return None
    subject = str(row.get("src_id") or row.get("subject") or "").strip()
    object_ = str(row.get("tgt_id") or row.get("object") or "").strip()
    identity = relationship_identity(subject, object_) if subject or object_ else "rel:malformed"
    weight, score_type = _score(row.get("weight"), "lightrag_weight")
    chunks = split_stored(str(row.get("source_id") or ""))
    documents = split_stored(str(row.get("file_path") or ""))
    return RelationshipEvidence(
        relationship_id=identity,
        subject=subject,
        object=object_,
        predicate=None,
        keywords=_keywords(row.get("keywords")),
        graph_source_ids=chunks,
        supporting_document_ids=documents,
        supporting_chunk_ids=chunks,
        verbatim_evidence=None,
        support_status=status,
        validation_decision=decision,
        validation_reason=reason,
        retrieval_score=weight,
        score_type=score_type if weight is not None else "unscored_rank",
    )


def _description_from_entity(row: dict, catalog: dict[str, dict] | None) -> DerivedDescription | None:
    if not isinstance(row, dict):
        return None
    name = str(row.get("entity_name") or row.get("name") or "").strip()
    text = str(row.get("description") or "").strip()
    if not name or not text:
        return None
    if _INJECTION.search(name) or _INJECTION.search(text):
        return DerivedDescription(
            description_id=description_identity("entity", name),
            kind="entity",
            identity=name,
            description="",
            clauses=[],
            entity_type=None if not row.get("entity_type") else str(row.get("entity_type")),
            validation_status="rejected",
            source_reference_status="not_checked",
        )
    paths = split_stored(str(row.get("file_path") or ""))
    sources = split_stored(str(row.get("source_id") or ""))
    weight, score_type = _score(row.get("weight") or row.get("rank"), "lightrag_weight")
    return DerivedDescription(
        description_id=description_identity("entity", name),
        kind="entity",
        identity=name,
        description=text,
        clauses=split_stored(text) or [text],
        entity_type=None if not row.get("entity_type") else str(row.get("entity_type")),
        source_ids=sources,
        file_paths=paths,
        retrieval_score=weight,
        score_type=score_type if weight is not None else "unscored_rank",
        confidence=None,
        validation_status=None,
        source_reference_status=_reference_status(paths + sources, catalog),
    )


def _description_from_relationship(row: dict, catalog: dict[str, dict] | None) -> DerivedDescription | None:
    if not isinstance(row, dict):
        return None
    subject = str(row.get("src_id") or row.get("subject") or "").strip()
    object_ = str(row.get("tgt_id") or row.get("object") or "").strip()
    text = str(row.get("description") or row.get("graph_description") or "").strip()
    if not text or not (subject or object_):
        return None
    identity = relationship_identity(subject, object_) if subject and object_ else (subject or object_)
    if _INJECTION.search(text) or _INJECTION.search(subject) or _INJECTION.search(object_):
        return DerivedDescription(
            description_id=description_identity("relationship", identity),
            kind="relationship",
            identity=identity,
            description="",
            validation_status="rejected",
        )
    paths = split_stored(str(row.get("file_path") or ""))
    sources = split_stored(str(row.get("source_id") or ""))
    if not sources:
        sources = [str(item) for item in (row.get("graph_chunk_ids") or [])]
    weight, score_type = _score(row.get("weight"), "lightrag_weight")
    return DerivedDescription(
        description_id=description_identity("relationship", identity),
        kind="relationship",
        identity=identity,
        description=text,
        clauses=split_stored(text) or [text],
        source_ids=sources,
        file_paths=paths or [str(item) for item in (row.get("graph_document_ids") or [])],
        retrieval_score=weight,
        score_type=score_type if weight is not None else "unscored_rank",
        confidence=None,
        validation_status=None,
        source_reference_status=_reference_status(paths + sources, catalog),
    )


def _dedupe(items: list, key) -> tuple[list, list[AuditRecord]]:
    kept = []
    audit = []
    seen = set()
    for item in items:
        identity = key(item)
        if identity in seen:
            audit.append(AuditRecord(identity, item.channel, "excluded", ["duplicate_record"]))
            continue
        seen.add(identity)
        kept.append(item)
    return kept, audit


def _candidate_rows(payload: dict, raw: dict) -> list[dict]:
    rows = []
    for item in raw.get("relationships") or []:
        if isinstance(item, dict):
            rows.append(item)
    for item in payload.get("relationship_candidates") or []:
        if isinstance(item, dict):
            rows.append(item)
    return rows


def _weights(rows: list[dict]) -> dict[str, float]:
    found = {}
    for row in rows:
        subject = str(row.get("src_id") or row.get("subject") or "").strip()
        object_ = str(row.get("tgt_id") or row.get("object") or "").strip()
        if not subject or not object_:
            continue
        weight, _score_type = _score(row.get("weight"), "lightrag_weight")
        if weight is None:
            continue
        found[relationship_identity(subject, object_)] = weight
    return found


def _validated_rows(bundle: dict) -> list[dict]:
    rows = []
    for key in ("accepted", "quarantined", "rejected"):
        for item in bundle.get(key) or []:
            if isinstance(item, dict):
                rows.append(item)
    return rows


def _entity_source_rows(payload: dict, raw: dict) -> tuple[list[dict], str]:
    """Prefer raw entity rows. Otherwise use a description pass-through list.

    ``query_documents`` keeps descriptions on ``entity_descriptions`` and does
    not copy ``data.entities``. The experiment payload may carry the same rows
    as ``raw_entities``. Neither list is invented when both are absent.
    """
    raw_entities = [row for row in (raw.get("entities") or []) if isinstance(row, dict)]
    if raw_entities:
        return raw_entities, "raw_entities"
    for key in ("entity_descriptions", "raw_entities"):
        rows = [row for row in (payload.get(key) or []) if isinstance(row, dict)]
        if rows:
            return rows, key
    return [], "absent"


def normalize_graph_context(
    payload: dict | None,
    catalog: dict[str, dict] | None = None,
    *,
    failed_document_ids: set[str] | None = None,
) -> dict[str, Any]:
    """Normalize one saved or live retrieval payload. Does not query an index."""
    payload = payload or {}
    if not isinstance(payload, dict):
        raise TypeError("retrieval payload must be a dict")
    raw = _raw_data(payload)
    failed_ids = set(failed_document_ids or ())
    diagnostics: dict[str, Any] = {
        "payload_shape": "aquery_data" if raw else "project_normalized" if payload.get("hits") is not None else "unknown",
        "mode": _mode(payload),
        "entity_descriptions_not_retained": 0,
        "malformed_records": 0,
        "notes": [],
    }
    audit: list[AuditRecord] = []
    primary: list[SourceEvidence] = []
    if isinstance(payload.get("hits"), list):
        for index, hit in enumerate(payload["hits"]):
            item = _primary_from_hit(hit, index)
            if item is None:
                diagnostics["malformed_records"] += 1
                audit.append(AuditRecord(f"src:malformed:{index}", "primary_source", "excluded", ["malformed_chunk"]))
                continue
            if OMISSION_MARKER in item.passage:
                audit.append(AuditRecord(item.evidence_id, "primary_source", "excluded", ["omission_marker_span"]))
                continue
            primary.append(item)
        if raw.get("chunks"):
            diagnostics["notes"].append("hits_already_budgeted_raw_chunks_not_added")
    elif raw.get("chunks"):
        for index, chunk in enumerate(raw.get("chunks") or []):
            item, reason = _primary_from_chunk(chunk, catalog, failed_ids, index)
            if item is None:
                diagnostics["malformed_records"] += 1
                audit.append(AuditRecord(f"src:chunk:{index}", "primary_source", "excluded", [reason or "malformed_chunk"]))
                continue
            primary.append(item)
    primary, duplicate_audit = _dedupe(primary, lambda item: (item.document_id, item.chunk_id, item.passage))
    audit.extend(duplicate_audit)

    candidate_rows = _candidate_rows(payload, raw)
    weights = _weights(candidate_rows)
    relationships: list[RelationshipEvidence] = []
    bundle = _validation_bundle(payload)
    if bundle is not None:
        for record in _validated_rows(bundle):
            item = _relationship_from_validated(record, weights)
            if item is None:
                diagnostics["malformed_records"] += 1
                audit.append(AuditRecord("rel:malformed", "source_supported_relationship", "excluded", ["malformed_relationship"]))
                continue
            if not item.subject or not item.object:
                item.validation_decision = "rejected"
                item.validation_reason = "malformed_relationship"
                item.support_status = "unsupported"
                item.predicate = None
            relationships.append(item)
        diagnostics["notes"].append("used_source_verified_relationship_bundle")
    elif candidate_rows and catalog is not None:
        from host_finetune.graph_relationships import validate_relationships

        checked = validate_relationships(candidate_rows, catalog, failed_document_ids=failed_ids)
        for record in _validated_rows(checked):
            item = _relationship_from_validated(record, weights)
            if item is None:
                continue
            item.source_reference_status = _reference_status(
                item.supporting_document_ids + item.supporting_chunk_ids,
                catalog,
            )
            if item.source_reference_status == "unresolved":
                item.validation_reason = item.validation_reason or "stale_or_unresolved_source"
            relationships.append(item)
        diagnostics["notes"].append("validated_against_supplied_catalog")
        diagnostics["validation_contract"] = checked.get("contract")
    else:
        for row in candidate_rows:
            subject = str(row.get("src_id") or "").strip()
            object_ = str(row.get("tgt_id") or "").strip()
            if not subject or not object_:
                item = _relationship_unvalidated(
                    row, reason="malformed_relationship", decision="rejected", status="unsupported",
                )
                diagnostics["malformed_records"] += 1
            elif not split_stored(str(row.get("file_path") or "")) and not split_stored(str(row.get("source_id") or "")):
                item = _relationship_unvalidated(
                    row, reason="missing_source_document", decision="rejected", status="unsupported",
                )
            else:
                item = _relationship_unvalidated(
                    row, reason="catalog_not_supplied", decision="not_validated", status="unverified",
                )
            if item is not None:
                item.predicate = None
                relationships.append(item)
        if candidate_rows:
            diagnostics["notes"].append("relationships_not_validated_without_catalog_or_bundle")

    relationships, relationship_dupes = _dedupe(relationships, lambda item: item.relationship_id)
    audit.extend(relationship_dupes)

    descriptions: list[DerivedDescription] = []
    entity_rows, entity_origin = _entity_source_rows(payload, raw)
    if entity_rows:
        if entity_origin != "raw_entities":
            diagnostics["notes"].append("entity_descriptions_read_from_query_payload")
        for row in entity_rows:
            item = _description_from_entity(row, catalog)
            if item is None:
                diagnostics["malformed_records"] += 1
                name = str((row or {}).get("entity_name") or (row or {}).get("name") or "missing")
                audit.append(AuditRecord(description_identity("entity", name), "derived_description", "excluded", ["malformed_entity"]))
                continue
            descriptions.append(item)
    else:
        stripped = [
            row for row in (payload.get("excluded") or [])
            if isinstance(row, dict)
            and row.get("reason") == "generated_summary_not_a_source_span"
            and row.get("kind") == "entities"
        ]
        diagnostics["entity_descriptions_not_retained"] = len(stripped)
        if stripped:
            diagnostics["notes"].append(
                "entity_descriptions_were_removed_before_this_payload_and_were_not_invented"
            )
    for row in candidate_rows:
        item = _description_from_relationship(row, catalog)
        if item is not None:
            descriptions.append(item)
    descriptions, description_dupes = _dedupe(
        descriptions,
        lambda item: (item.kind, item.identity, re.sub(r"\s+", " ", item.description).casefold()),
    )
    audit.extend(description_dupes)
    for item in descriptions:
        if item.classification != "derived_context":
            item.classification = "derived_context"
        if item.confidence is not None:
            item.confidence = None

    mode = _mode(payload)
    if mode == "naive" and not relationships and not descriptions:
        diagnostics["notes"].append("naive_mode_has_no_graph_rows")
    if mode == "local" and not entity_rows and diagnostics["entity_descriptions_not_retained"] == 0 and not descriptions:
        diagnostics["notes"].append("local_mode_without_entity_rows")
    diagnostics["separator"] = SEP
    return {
        "query": str(payload.get("query") or ""),
        "mode": mode,
        "primary": primary,
        "relationships": relationships,
        "descriptions": descriptions,
        "audit": audit,
        "diagnostics": diagnostics,
        "raw_payload": payload,
    }
