"""Read-only comparison of the indexing ledger and LightRAG stores.

The functions below do not delete vectors, graph rows, or ledger rows.
"""

from __future__ import annotations

import json
import sqlite3
import xml.etree.ElementTree as ET
from pathlib import Path

from host_finetune.lightrag_backend import OMISSION_MARKER

GRAPHML_NS = "http://graphml.graphdrawing.org/xmlns"


def chunk_owner(row: dict) -> str:
    full = str(row.get("full_doc_id") or "").strip()
    if full:
        return full
    chunk_id = str(row.get("__id__") or row.get("_id") or row.get("id") or "")
    marker = "-chunk-"
    if marker in chunk_id:
        return chunk_id.split(marker, 1)[0]
    return ""


def split_graph_path(value: str | None) -> list[str]:
    text = (value or "").replace("&lt;SEP&gt;", "<SEP>")
    return [part.strip() for part in text.split("<SEP>") if part.strip()]


def reconcile_records(
    *,
    ledger_rows: list[dict],
    doc_status: dict[str, dict],
    chunk_vectors: list[dict],
    text_chunks: dict[str, dict],
    full_entities: dict[str, dict],
    full_relations: dict[str, dict],
    graph_file_paths: list[str],
    graph_node_ids: list[str] | None = None,
) -> dict:
    """Classify stored documents and vectors. Inputs are already loaded."""
    ledger = {row["document_id"]: row for row in ledger_rows}
    duplicate_ledger = _duplicates(row["document_id"] for row in ledger_rows)
    duplicate_nodes = _duplicates(graph_node_ids or [])
    duplicate_vectors = _duplicates(
        str(row.get("__id__") or row.get("id") or "") for row in chunk_vectors
    )

    docs_on_graph: dict[str, int] = {}
    for path in graph_file_paths:
        for token in split_graph_path(path):
            docs_on_graph[token] = docs_on_graph.get(token, 0) + 1

    text_owners = {chunk_owner(row) for row in text_chunks.values()}
    vector_ids = {str(row.get("__id__") or "") for row in chunk_vectors}
    text_ids = set(text_chunks)

    completed_with_relations = []
    completed_entities_only = []
    completed_empty = []
    failed_residue = []
    interrupted_partial = []
    anchor_conflicts = []

    for doc_id, row in sorted(ledger.items()):
        status = doc_status.get(doc_id) or {}
        light_status = str(status.get("status") or "absent").lower()
        entity_count = int((full_entities.get(doc_id) or {}).get("count") or 0)
        relation_count = int((full_relations.get(doc_id) or {}).get("count") or 0)
        on_graph = docs_on_graph.get(doc_id, 0)
        vectors = [item for item in chunk_vectors if chunk_owner(item) == doc_id]
        ledger_status = row.get("status")
        record = {
            "document_id": doc_id,
            "ledger_status": ledger_status,
            "lightrag_status": light_status,
            "entity_count": entity_count,
            "relation_count": relation_count,
            "graph_mentions": on_graph,
            "chunk_vectors": len(vectors),
            "text_chunks": sum(1 for owner in (chunk_owner(item) for item in text_chunks.values()) if owner == doc_id),
            "kg_write_state": (status.get("metadata") or {}).get("kg_write_state"),
        }
        if ledger_status == "completed" and light_status == "processed" and entity_count > 0 and relation_count > 0 and on_graph > 0:
            completed_with_relations.append(record)
        elif ledger_status == "completed" and entity_count > 0 and relation_count == 0:
            completed_entities_only.append(record)
        elif ledger_status == "completed" and entity_count == 0 and relation_count == 0:
            completed_empty.append(record)
        if ledger_status in {"failed_permanent", "failed_retryable", "interrupted"} and vectors:
            failed_residue.append(record)
        if ledger_status in {"interrupted", "processing"} or light_status in {"processing", "parsing", "analyzing", "pending"}:
            if ledger_status != "completed":
                interrupted_partial.append(record)
        conflict = _anchor_conflict(record, ledger_status, light_status, on_graph, entity_count)
        if conflict:
            record = dict(record)
            record["conflict"] = conflict
            anchor_conflicts.append(record)

    unknown_vectors = []
    for item in chunk_vectors:
        owner = chunk_owner(item)
        chunk_id = str(item.get("__id__") or "")
        if owner not in ledger and owner not in doc_status:
            unknown_vectors.append({"chunk_id": chunk_id, "owner": owner, "reason": "unknown_owner"})
        elif chunk_id and chunk_id not in text_ids and owner in text_owners:
            unknown_vectors.append({"chunk_id": chunk_id, "owner": owner, "reason": "vector_without_text_chunk"})
        elif chunk_id and chunk_id not in text_ids:
            unknown_vectors.append({"chunk_id": chunk_id, "owner": owner, "reason": "vector_without_text_chunk"})

    failed_ids = {
        doc_id
        for doc_id, row in doc_status.items()
        if str(row.get("status") or "").lower() == "failed"
    }
    uncovered = []
    covered = []
    for item in chunk_vectors:
        owner = chunk_owner(item)
        if owner not in failed_ids:
            continue
        file_path = str(item.get("file_path") or "")
        full_doc = str(item.get("full_doc_id") or "")
        entry = {"chunk_id": str(item.get("__id__") or ""), "file_path": file_path, "full_doc_id": full_doc}
        if file_path in failed_ids or full_doc in failed_ids:
            covered.append(entry)
        else:
            uncovered.append(entry)

    for doc_id, status in doc_status.items():
        if doc_id in ledger:
            continue
        anchor_conflicts.append({
            "document_id": doc_id,
            "ledger_status": "absent",
            "lightrag_status": str(status.get("status") or ""),
            "conflict": "LightRAG document is not in the ledger",
        })

    return {
        "completed_with_relations": completed_with_relations,
        "completed_entities_only": completed_entities_only,
        "completed_empty": completed_empty,
        "failed_with_chunk_residue": failed_residue,
        "interrupted_or_partial": interrupted_partial,
        "anchor_conflicts": anchor_conflicts,
        "unknown_or_unowned_vectors": unknown_vectors,
        "duplicate_ledger_ids": duplicate_ledger,
        "duplicate_graph_node_ids": duplicate_nodes,
        "duplicate_chunk_vector_ids": [item for item in duplicate_vectors if item],
        "query_filter": {
            "failed_document_ids": sorted(failed_ids),
            "failed_vectors_covered": covered,
            "failed_vectors_not_covered": uncovered,
        },
        "counts": {
            "ledger_documents": len(ledger_rows),
            "lightrag_documents": len(doc_status),
            "chunk_vectors": len(chunk_vectors),
            "text_chunks": len(text_chunks),
            "graph_path_mentions": len(graph_file_paths),
        },
    }


def _anchor_conflict(record, ledger_status, light_status, on_graph, entity_count) -> str | None:
    del record
    if ledger_status == "completed" and light_status != "processed":
        return "ledger completed but LightRAG status is not processed"
    if ledger_status in {"failed_permanent", "failed_retryable"} and on_graph > 0:
        return "failed ledger row still appears on a graph node or edge"
    if ledger_status == "completed" and entity_count == 0 and on_graph > 0:
        return "completed row has graph mentions and no entity anchor"
    if ledger_status == "completed" and entity_count > 0 and on_graph == 0:
        return "completed row has entity anchors and no graph path"
    if light_status == "processed" and ledger_status != "completed":
        return "LightRAG processed a document the ledger does not mark completed"
    return None


def _duplicates(values) -> list[str]:
    seen = set()
    dupes = []
    for value in values:
        if not value or value in seen:
            if value and value not in dupes:
                dupes.append(value)
            continue
        seen.add(value)
    return dupes


def load_graph_paths(path: Path) -> tuple[list[str], list[str]]:
    tree = ET.parse(path)
    file_path_keys = set()
    for key in tree.iter(f"{{{GRAPHML_NS}}}key"):
        if key.attrib.get("attr.name") == "file_path":
            file_path_keys.add(key.attrib.get("id"))
    paths = []
    node_ids = []
    for node in tree.iter(f"{{{GRAPHML_NS}}}node"):
        if node.attrib.get("id"):
            node_ids.append(node.attrib["id"])
    for data in tree.iter(f"{{{GRAPHML_NS}}}data"):
        if data.attrib.get("key") in file_path_keys and data.text:
            paths.append(data.text)
    return paths, node_ids


def load_reconciliation(workdir: Path) -> dict:
    """Read one workdir. Opens the ledger read-only and does not write stores."""
    ledger_path = workdir / "indexing_ledger.sqlite"
    uri = ledger_path.resolve().as_uri() + "?mode=ro"
    connection = sqlite3.connect(uri, uri=True)
    connection.row_factory = sqlite3.Row
    try:
        ledger_rows = [dict(row) for row in connection.execute("SELECT * FROM documents")]
    finally:
        connection.close()
    doc_status = _read_json(workdir / "kv_store_doc_status.json")
    vectors = _read_json(workdir / "vdb_chunks.json")
    text_chunks = _read_json(workdir / "kv_store_text_chunks.json")
    entities = _read_json(workdir / "kv_store_full_entities.json")
    relations = _read_json(workdir / "kv_store_full_relations.json")
    paths, node_ids = load_graph_paths(workdir / "graph_chunk_entity_relation.graphml")
    report = reconcile_records(
        ledger_rows=ledger_rows,
        doc_status=doc_status,
        chunk_vectors=list(vectors.get("data") or []),
        text_chunks=text_chunks,
        full_entities=entities,
        full_relations=relations,
        graph_file_paths=paths,
        graph_node_ids=node_ids,
    )
    report["embedding_dim"] = vectors.get("embedding_dim")
    return report


def write_audit_bundle(workdir: Path, destination: Path) -> dict:
    """Write reconciliation, quality, orphan, and runtime reports. Does not edit stores."""
    from host_finetune.indexing_quality import classify_graph_quality, runtime_profile

    destination.mkdir(parents=True, exist_ok=True)
    report = load_reconciliation(workdir)
    ledger_path = workdir / "indexing_ledger.sqlite"
    uri = ledger_path.resolve().as_uri() + "?mode=ro"
    connection = sqlite3.connect(uri, uri=True)
    connection.row_factory = sqlite3.Row
    try:
        documents = [dict(row) for row in connection.execute("SELECT * FROM documents")]
        attempts = [dict(row) for row in connection.execute("SELECT * FROM attempts")]
    finally:
        connection.close()
    entities = _read_json(workdir / "kv_store_full_entities.json")
    relations = _read_json(workdir / "kv_store_full_relations.json")
    by_id = {row["document_id"]: row for row in documents}
    quality_rows = []
    seen = set()
    for record in (
        report["completed_with_relations"]
        + report["completed_entities_only"]
        + report["completed_empty"]
        + report["failed_with_chunk_residue"]
        + report["interrupted_or_partial"]
    ):
        doc_id = record["document_id"]
        if doc_id in seen:
            continue
        seen.add(doc_id)
        source = by_id.get(doc_id) or {}
        names = list((entities.get(doc_id) or {}).get("entity_names") or [])
        metrics = _json_or_none(source.get("extraction_metrics"))
        malformed = None
        if isinstance(metrics, dict) and "malformed_relation_warnings_measured" in metrics:
            malformed = int(metrics["malformed_relation_warnings_measured"])
        quality_rows.append(classify_graph_quality(
            indexing_status=source.get("status") or record["ledger_status"],
            entity_count=int((entities.get(doc_id) or {}).get("count") or 0),
            relation_count=int((relations.get(doc_id) or {}).get("count") or 0),
            entity_names=names,
            malformed_relations=malformed,
            extraction_incomplete=source.get("status") in {"failed_permanent", "failed_retryable", "interrupted", "processing"},
            source_text=OMISSION_MARKER if source.get("has_omission") else "",
        ) | {"document_id": doc_id})
    profile = runtime_profile(documents, attempts)
    orphans = report["unknown_or_unowned_vectors"]
    (destination / "reconciliation.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    (destination / "graph_quality.json").write_text(json.dumps(quality_rows, indent=2) + "\n", encoding="utf-8")
    (destination / "orphan_vectors.json").write_text(json.dumps(orphans, indent=2) + "\n", encoding="utf-8")
    (destination / "runtime_statistics.json").write_text(json.dumps(profile, indent=2) + "\n", encoding="utf-8")
    command = (
        "host_finetune\\.venv\\Scripts\\python.exe -X utf8 -m host_finetune.indexing_session "
        "--resume --max-documents 20 --max-runtime-minutes 45 --max-document-seconds 420"
    )
    (destination / "next_resume.txt").write_text(command + "\n", encoding="utf-8")
    return {
        "anchor_conflicts": len(report["anchor_conflicts"]),
        "orphan_vectors": len(orphans),
        "failed_vectors_not_covered": len(report["query_filter"]["failed_vectors_not_covered"]),
        "completed_with_relations": len(report["completed_with_relations"]),
        "completed_entities_only": len(report["completed_entities_only"]),
        "next_resume_command": command,
    }


def _json_or_none(value):
    if isinstance(value, dict):
        return value
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return None


def _read_json(path: Path):
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))
