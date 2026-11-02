"""Versioned evidence records for one Model F graph-context pipeline.

Channel A is original author text. Channel B is a relationship only when a
source passage supports that edge. Channel C is a generated graph description
and is never a verified fact. Confidence stays unset unless a caller measured it.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass, field
from typing import Any

SCHEMA_VERSION = "model-f-evidence-v1"
CHANNEL_PRIMARY = "primary_source"
CHANNEL_RELATIONSHIP = "source_supported_relationship"
CHANNEL_DESCRIPTION = "derived_description"
CHANNEL_PATH = "multi_hop_path"
SEP = "<SEP>"


def _stable_id(prefix: str, *parts: str) -> str:
    raw = SEP.join(part or "" for part in parts)
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:12]
    return f"{prefix}:{digest}"


def split_stored(value: str | None) -> list[str]:
    """Split LightRAG's ``<SEP>`` lists. HTML-escaped separators count too."""
    text = str(value or "").replace("&lt;SEP&gt;", SEP)
    parts = []
    for part in text.split(SEP):
        cleaned = part.strip()
        if cleaned and cleaned != "unknown_source":
            parts.append(cleaned)
    return parts


@dataclass
class SourceEvidence:
    """Original author text. Highest factual authority in the bundle."""

    evidence_id: str
    document_id: str
    chunk_id: str
    passage: str
    medium: str | None = None
    source_url: str | None = None
    publication_date: str | None = None
    attribution_status: str | None = None
    author_position: str | None = None
    quoted_speakers: list[str] = field(default_factory=list)
    retrieval_score: float | None = None
    score_type: str = "unscored_rank"
    rank: int | None = None
    source_group: str | None = None
    leakage_lineage: dict[str, Any] = field(default_factory=dict)
    channel: str = CHANNEL_PRIMARY

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class RelationshipEvidence:
    """One graph edge. A predicate is never invented from the endpoint names."""

    relationship_id: str
    subject: str
    object: str
    predicate: str | None = None
    keywords: list[str] = field(default_factory=list)
    graph_source_ids: list[str] = field(default_factory=list)
    supporting_document_ids: list[str] = field(default_factory=list)
    supporting_chunk_ids: list[str] = field(default_factory=list)
    verbatim_evidence: str | None = None
    support_status: str = "unverified"
    validation_decision: str = "not_validated"
    validation_reason: str | None = None
    temporal_qualifiers: list[str] = field(default_factory=list)
    numerical_qualifiers: list[str] = field(default_factory=list)
    quoted_speakers: list[str] = field(default_factory=list)
    retrieval_score: float | None = None
    score_type: str = "unscored_rank"
    source_reference_status: str = "not_checked"
    channel: str = CHANNEL_RELATIONSHIP

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class DerivedDescription:
    """Generated graph text. Supplementary context, not an independent fact."""

    description_id: str
    kind: str
    identity: str
    description: str
    clauses: list[str] = field(default_factory=list)
    entity_type: str | None = None
    source_ids: list[str] = field(default_factory=list)
    file_paths: list[str] = field(default_factory=list)
    retrieval_score: float | None = None
    score_type: str = "unscored_rank"
    classification: str = "derived_context"
    confidence: float | None = None
    validation_status: str | None = None
    source_reference_status: str = "not_checked"
    channel: str = CHANNEL_DESCRIPTION

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        if payload["confidence"] is not None and not isinstance(payload["confidence"], (int, float)):
            raise ValueError("description confidence must be measured, not invented")
        return payload


@dataclass
class EvidencePath:
    """A chain of source-supported edges. The chain is not one stated claim."""

    path_id: str
    hops: int
    edge_ids: list[str]
    nodes: list[str]
    edge_facts: list[str]
    edge_provenance: list[dict[str, Any]]
    explicit_endpoint_claim: bool
    endpoint_claim_note: str
    display: str
    channel: str = CHANNEL_PATH

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class AuditRecord:
    evidence_id: str
    channel: str
    decision: str
    reasons: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def source_identity(document_id: str, chunk_id: str, passage: str) -> str:
    if document_id and chunk_id:
        return f"src:{document_id}:{chunk_id}"
    return _stable_id("src", document_id, chunk_id, passage)


def relationship_identity(subject: str, object_: str) -> str:
    return f"{subject}{SEP}{object_}"


def description_identity(kind: str, identity: str) -> str:
    return _stable_id("desc", kind, identity)


def path_identity(edge_ids: list[str]) -> str:
    return _stable_id("path", *edge_ids)


def compact_predicate(keywords: list[str] | None, predicate: str | None) -> str:
    """A short supported phrase. This is not a copy of the source passage."""
    phrases = [str(word).strip() for word in (keywords or []) if len(str(word).split()) >= 2]
    if phrases:
        return ", ".join(phrases[:4])
    words = re.sub(r"\s+", " ", predicate or "").split()
    if len(words) > 8:
        return " ".join(words[:6])
    return " ".join(words)


def bundle_dict(
    *,
    query: str,
    mode: str | None,
    primary: list[SourceEvidence],
    relationships: list[RelationshipEvidence],
    descriptions: list[DerivedDescription],
    paths: list[EvidencePath],
    audit: list[AuditRecord],
    diagnostics: dict[str, Any],
    raw_payload: dict | None = None,
    include_raw: bool = False,
) -> dict[str, Any]:
    """Serializable bundle. The raw payload stays attached unless omitted."""
    encoded = json.dumps(raw_payload or {}, ensure_ascii=False, sort_keys=True, default=str)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "query": query,
        "mode": mode,
        "primary": [item.to_dict() for item in primary],
        "relationships": [item.to_dict() for item in relationships],
        "descriptions": [item.to_dict() for item in descriptions],
        "paths": [item.to_dict() for item in paths],
        "audit": [item.to_dict() for item in audit],
        "diagnostics": diagnostics,
        "raw_payload_sha256": hashlib.sha256(encoded.encode("utf-8")).hexdigest(),
        "raw_payload_keys": sorted((raw_payload or {}).keys()),
    }
    if include_raw:
        payload["raw_payload"] = raw_payload or {}
    return payload
