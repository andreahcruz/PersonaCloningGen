"""Read-only graph quality report. Nothing is written back to a store."""

from __future__ import annotations

from pathlib import Path

from host_finetune.graph_relationships import is_weak_entity
from host_finetune.indexing_extraction_audit import (
    alias_candidates,
    classify_relation_fields,
    record_fields,
)
from host_finetune.model_f_experiment.snapshot import LIVE_STORE_NAME, SnapshotError

# Malformed extraction affects generation only if a repaired edge was stored.
# A cache-only malformed row is a quality finding, not an automatic edit.
_AFFECTS_GENERATION = {
    "weak_entity_in_graph",
    "edge_missing_provenance",
    "description_missing_provenance",
    "duplicate_edge",
}
_AFFECTS_RETRIEVAL = {
    "weak_entity_in_graph",
    "alias_group",
    "orphan_vector",
    "duplicate_edge",
}
_PRONOUNS = {"i", "you", "we", "they", "he", "she", "it"}


def _impact(kind: str) -> str:
    retrieval = kind in _AFFECTS_RETRIEVAL
    generation = kind in _AFFECTS_GENERATION
    if retrieval and generation:
        return "affects_retrieval_and_generation"
    if retrieval:
        return "affects_retrieval"
    if generation:
        return "affects_generation"
    return "cosmetic_or_cache_only"


def audit_records(records: dict) -> dict:
    """Classify supplied names, edges, descriptions, vectors, and cache lines.

    ``records`` is an in-memory export. This function does not open the live
    store and does not apply relationship recovery.
    """
    findings = []
    names = [str(item) for item in (records.get("entity_names") or [])]
    for name in names:
        if is_weak_entity(name) or name.casefold() in _PRONOUNS:
            findings.append({
                "kind": "weak_entity_in_graph",
                "name": name,
                "impact": _impact("weak_entity_in_graph"),
                "note": "A pronoun, generic label, or numeric fragment can enter retrieval neighborhoods.",
            })
    for group in alias_candidates(names):
        findings.append({
            "kind": "alias_group",
            "names": group["names"],
            "impact": _impact("alias_group"),
            "note": "Spelling variants split evidence across nodes. This audit does not merge them.",
        })
    seen_edges = set()
    for edge in records.get("edges") or []:
        source = str(edge.get("source") or "")
        target = str(edge.get("target") or "")
        key = tuple(sorted((source.casefold(), target.casefold())))
        if key in seen_edges and source and target:
            findings.append({
                "kind": "duplicate_edge",
                "source": source,
                "target": target,
                "impact": _impact("duplicate_edge"),
            })
        seen_edges.add(key)
        if source and target and not (edge.get("source_ids") or edge.get("chunk_ids")):
            findings.append({
                "kind": "edge_missing_provenance",
                "source": source,
                "target": target,
                "impact": _impact("edge_missing_provenance"),
            })
    for item in records.get("descriptions") or []:
        if item.get("text") and not (item.get("source_ids") or item.get("file_paths")):
            findings.append({
                "kind": "description_missing_provenance",
                "name": item.get("name"),
                "impact": _impact("description_missing_provenance"),
                "note": "A description without a source must not be shown as a verified fact.",
            })
    vector_ids = {str(item) for item in (records.get("vector_ids") or [])}
    graph_ids = {str(item) for item in (records.get("graph_chunk_ids") or [])}
    for vector_id in sorted(vector_ids - graph_ids):
        findings.append({
            "kind": "orphan_vector",
            "id": vector_id,
            "impact": _impact("orphan_vector"),
        })
    for chunk_id in sorted(graph_ids - vector_ids):
        if vector_ids:
            findings.append({
                "kind": "missing_vector",
                "id": chunk_id,
                "impact": "affects_retrieval",
            })
    cache_counts: dict[str, int] = {}
    for line in records.get("extraction_lines") or []:
        verdict = classify_relation_fields(record_fields(str(line)))
        if verdict["kind"] in {"not_relation", "accepted_five"}:
            continue
        cache_counts[verdict["kind"]] = cache_counts.get(verdict["kind"], 0) + 1
        findings.append({
            "kind": verdict["kind"],
            "field_count": verdict["field_count"],
            "recoverable": bool(verdict.get("recoverable")),
            "impact": "cosmetic_or_cache_only",
            "note": "Historical recovery is not applied to the index.",
        })
    return {
        "findings": findings,
        "cache_kind_counts": cache_counts,
        "recovery_applied": False,
        "graph_modified": False,
    }


def _ids(payload: dict) -> set[str]:
    found = set()
    rows = payload.get("data") if isinstance(payload, dict) else None
    if isinstance(payload, dict) and rows is None:
        rows = list(payload.values())
    for row in rows or []:
        if isinstance(row, dict):
            for key in ("__id__", "id", "chunk_id"):
                if row.get(key):
                    found.add(str(row[key]))
        elif row:
            found.add(str(row))
    return found


def export_audit_records(snapshot: Path) -> dict:
    """Read a snapshot copy into the audit schema. The snapshot is not modified."""
    from host_finetune.model_f_experiment.snapshot import assert_snapshot_path

    assert_snapshot_path(snapshot)
    import json

    names: list[str] = []
    entities_path = snapshot / "kv_store_full_entities.json"
    descriptions = []
    if entities_path.is_file():
        entities = json.loads(entities_path.read_text(encoding="utf-8"))
        for row in entities.values() if isinstance(entities, dict) else []:
            if not isinstance(row, dict):
                continue
            for name in row.get("entity_names") or []:
                names.append(str(name))
    edges = []
    graph_path = snapshot / "graph_chunk_entity_relation.graphml"
    if graph_path.is_file():
        import xml.etree.ElementTree as ET

        from host_finetune.graph_relationships import _GRAPHML_NS, read_graphml_relationships

        tree = ET.parse(graph_path)
        node_keys = {
            key.get("id"): key.get("attr.name")
            for key in tree.findall("g:key", _GRAPHML_NS)
            if key.get("for") == "node"
        }
        for node in tree.findall(".//g:node", _GRAPHML_NS):
            fields = {}
            for item in list(node):
                attr = node_keys.get(item.get("key"))
                if attr:
                    fields[attr] = item.text or ""
            if node.get("id"):
                names.append(str(node.get("id")))
            if fields.get("description"):
                descriptions.append({
                    "name": node.get("id"),
                    "text": fields.get("description"),
                    "source_ids": [fields["source_id"]] if fields.get("source_id") else [],
                    "file_paths": [fields["file_path"]] if fields.get("file_path") else [],
                })
        for edge in read_graphml_relationships(graph_path):
            edges.append({
                "source": edge.get("src_id"),
                "target": edge.get("tgt_id"),
                "source_ids": [edge["source_id"]] if edge.get("source_id") else [],
                "chunk_ids": [edge["chunk_id"]] if edge.get("chunk_id") else [],
            })
    vector_ids: set[str] = set()
    chunks_path = snapshot / "vdb_chunks.json"
    if chunks_path.is_file():
        vector_ids = _ids(json.loads(chunks_path.read_text(encoding="utf-8")))
    graph_chunk_ids: set[str] = set()
    text_path = snapshot / "kv_store_text_chunks.json"
    if text_path.is_file():
        graph_chunk_ids = set(json.loads(text_path.read_text(encoding="utf-8")))
    lines = []
    cache_path = snapshot / "kv_store_llm_response_cache.json"
    if cache_path.is_file():
        cache = json.loads(cache_path.read_text(encoding="utf-8"))
        for row in cache.values() if isinstance(cache, dict) else []:
            if isinstance(row, dict) and row.get("cache_type") == "extract":
                lines.extend(str(row.get("return") or "").splitlines())
    return {
        "entity_names": names,
        "edges": edges,
        "descriptions": descriptions,
        "vector_ids": sorted(vector_ids),
        "graph_chunk_ids": sorted(graph_chunk_ids),
        "extraction_lines": lines,
        "recovery_applied": False,
    }


def audit_directory(path: Path) -> dict:
    if LIVE_STORE_NAME in path.resolve().parts:
        raise SnapshotError("refusing to audit the live expansion store; audit a snapshot copy")
    export_path = path / "graph_audit_export.json"
    if not export_path.is_file():
        raise SnapshotError(
            "snapshot audit expects graph_audit_export.json written by a later export step; the live graph was not opened"
        )
    import json

    return audit_records(json.loads(export_path.read_text(encoding="utf-8")))
