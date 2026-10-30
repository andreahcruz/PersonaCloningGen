"""Reconcile the finished 160-document pilot index. Read-only."""

from __future__ import annotations

import json
import re
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

from host_finetune.entity_audit import entity_name_flags

ROOT = Path(__file__).resolve().parents[3]
WORK = ROOT / "artifacts" / "lightrag" / "model_f" / "storage_pilot160"
SMOKE = ROOT / "artifacts" / "lightrag" / "model_f" / "storage"
IDS = (
    ROOT
    / "experiments"
    / "EXP-20261027-001-graph-ready-audit"
    / "config"
    / "sample_160_ids.json"
)
V2 = ROOT / "artifacts" / "lightrag" / "model_f" / "graph_ready_v2" / "documents.jsonl"
LOG = Path(
    r"C:\Users\okevi\.cursor\projects\c-Users-okevi-SJSU-DATA298-298P\terminals\185770.txt"
)
OUT = Path(__file__).resolve().parents[1] / "metrics" / "index_audit.json"

NS = {"g": "http://graphml.graphdrawing.org/xmlns"}


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def source_docs(raw: str) -> list[str]:
    docs = []
    for piece in (raw or "").split("<SEP>"):
        piece = piece.strip()
        if not piece:
            continue
        docs.append(piece.split("-chunk-")[0].strip())
    return [doc for doc in docs if doc]


def status_text(value) -> str:
    if isinstance(value, dict):
        return str(value.get("status") or value.get("value") or "")
    return str(value or "")


def main() -> None:
    manifest_ids = json.loads(IDS.read_text(encoding="utf-8"))
    catalog = {
        row["graph_document_id"]: row
        for row in load_jsonl(WORK / "catalog.jsonl")
    }
    doc_status = json.loads((WORK / "kv_store_doc_status.json").read_text(encoding="utf-8"))
    skip = json.loads((WORK / "extraction_skip.json").read_text(encoding="utf-8"))
    index_manifest = json.loads((WORK / "index_manifest.json").read_text(encoding="utf-8"))
    parents = {}
    for row in catalog.values():
        parent = row.get("parent_graph_document_id") or row["graph_document_id"]
        parents.setdefault(parent, []).append(row["graph_document_id"])

    by_status = Counter()
    status_ids = {}
    for doc_id, row in doc_status.items():
        label = status_text(row.get("status") if isinstance(row, dict) else row)
        by_status[label] += 1
        status_ids.setdefault(label, []).append(doc_id)

    processed_ids = set(status_ids.get("processed", []))
    failed_ids = set(status_ids.get("failed", []))
    other_status = {
        key: sorted(value)
        for key, value in status_ids.items()
        if key not in {"processed", "failed"}
    }
    catalog_parents = set(parents)
    missing_from_catalog = sorted(set(manifest_ids) - catalog_parents)
    extra_catalog = sorted(catalog_parents - set(manifest_ids))
    processed_parents = set()
    for doc_id in processed_ids:
        row = catalog.get(doc_id, {})
        processed_parents.add(row.get("parent_graph_document_id") or doc_id)
    failed_parents = []
    for doc_id in sorted(failed_ids | set(skip)):
        row = catalog.get(doc_id, {})
        failed_parents.append(
            {
                "graph_document_id": doc_id,
                "parent_graph_document_id": row.get("parent_graph_document_id") or doc_id,
                "status": status_text((doc_status.get(doc_id) or {}).get("status")),
                "error_msg": (doc_status.get(doc_id) or {}).get("error_msg")
                or (doc_status.get(doc_id) or {}).get("error"),
                "chunks_count": (doc_status.get(doc_id) or {}).get("chunks_count"),
                "in_skip_list": doc_id in skip,
                "split": row.get("split"),
                "medium": row.get("medium"),
                "words": len((row.get("text") or "").split()),
            }
        )
    not_extracted_parents = sorted(
        (set(manifest_ids) - processed_parents)
    )
    splits = Counter(row.get("split") for row in catalog.values())
    non_train = [
        row["graph_document_id"]
        for row in catalog.values()
        if row.get("split") != "train"
    ]

    tree = ET.parse(WORK / "graph_chunk_entity_relation.graphml")
    keys = {}
    for key in tree.findall("g:key", NS):
        keys[key.attrib.get("id")] = key.attrib.get("attr.name")
    nodes = []
    for node in tree.findall("g:graph/g:node", NS):
        attrs = {}
        for data in node.findall("g:data", NS):
            name = keys.get(data.attrib.get("key"), data.attrib.get("key"))
            attrs[name] = data.text or ""
        nodes.append({"id": node.attrib.get("id"), **attrs})
    edges = []
    for edge in tree.findall("g:graph/g:edge", NS):
        attrs = {}
        for data in edge.findall("g:data", NS):
            name = keys.get(data.attrib.get("key"), data.attrib.get("key"))
            attrs[name] = data.text or ""
        edges.append(
            {
                "source": edge.attrib.get("source"),
                "target": edge.attrib.get("target"),
                **attrs,
            }
        )
    node_ids = {node["id"] for node in nodes}
    weak = []
    malformed = []
    for node in nodes:
        name = node.get("entity_name") or node["id"]
        flags = entity_name_flags(name)
        if flags:
            weak.append({"name": name, "flags": flags, "source_id": node.get("source_id", "")[:180]})
        if "<" in name or ">" in name or name.startswith("\"") or "source_entity" in name.casefold():
            malformed.append(name)
    node_sources = Counter()
    unknown_node_sources = []
    for node in nodes:
        parts = source_docs(node.get("source_id") or "")
        if not parts:
            unknown_node_sources.append(node.get("entity_name") or node["id"])
        for doc_id in parts:
            node_sources[doc_id] += 1
    edge_sources = Counter()
    dangling = []
    empty_source_edges = 0
    unknown_edge_sources = []
    for edge in edges:
        if edge["source"] not in node_ids or edge["target"] not in node_ids:
            dangling.append({"source": edge["source"], "target": edge["target"]})
        parts = source_docs(edge.get("source_id") or "")
        if not parts:
            empty_source_edges += 1
        for doc_id in parts:
            edge_sources[doc_id] += 1
            if doc_id not in catalog:
                unknown_edge_sources.append(doc_id)
    flag_counts = Counter(flag for item in weak for flag in item["flags"])
    weekday_names = sorted({item["name"] for item in weak if "weekday" in item["flags"]})
    failed_with_graph = sorted(
        (failed_ids | set(skip)) & (set(node_sources) | set(edge_sources))
    )
    non_train_sources = sorted(
        doc_id
        for doc_id in set(node_sources) | set(edge_sources)
        if catalog.get(doc_id, {}).get("split") not in {None, "train"}
        or (doc_id not in catalog and doc_id not in {"UNKNOWN"})
    )

    v2 = {}
    for row in load_jsonl(V2):
        parent = row.get("parent_graph_document_id") or row.get("graph_document_id")
        if parent in set(manifest_ids):
            v2[parent] = row
    marker_re = re.compile(r"\[Omitted same-source text between these sections: (\d+) words\.\]")
    marked = []
    for parent, row in v2.items():
        text = row.get("text") or ""
        gaps = [int(item) for item in marker_re.findall(text)]
        if gaps:
            marked.append(
                {
                    "parent": parent,
                    "graph_document_id": row.get("graph_document_id"),
                    "words": len(text.split()),
                    "gaps": gaps,
                    "extracted": (row.get("graph_document_id") in processed_ids),
                    "attribution_status": row.get("attribution_status"),
                }
            )
    chunks = json.loads((WORK / "kv_store_text_chunks.json").read_text(encoding="utf-8"))
    marker_chunks = []
    for chunk_id, chunk in chunks.items():
        content = chunk.get("content") or chunk.get("text") or ""
        if "Omitted same-source text" in content:
            marker_chunks.append(
                {
                    "chunk_id": chunk_id,
                    "full_doc_id": chunk.get("full_doc_id") or chunk.get("doc_id"),
                    "chars": len(content),
                    "gaps": [int(item) for item in marker_re.findall(content)],
                }
            )
    marker_chunk_ids = {item["chunk_id"] for item in marker_chunks}
    marker_doc_ids = {item["full_doc_id"] for item in marker_chunks}
    edges_from_marker_chunks = 0
    for edge in edges:
        pieces = [
            piece.strip()
            for piece in (edge.get("source_id") or "").split("<SEP>")
            if piece.strip()
        ]
        if any(piece in marker_chunk_ids for piece in pieces):
            edges_from_marker_chunks += 1

    log = LOG.read_text(encoding="utf-8", errors="replace")
    chunk_lines = re.findall(
        r"Chunk (\d+) of (\d+) extracted (\d+) Ent \+ (\d+) Rel (\S+)",
        log,
    )
    timeouts = re.findall(
        r"Worker execution timeout after (\d+)s",
        log,
    )
    failed_docs = re.findall(r"Failed to extract document \d+/\d+: (\S+)", log)
    ent = [int(item[2]) for item in chunk_lines]
    rel = [int(item[3]) for item in chunk_lines]
    by_doc_chunks = Counter(item[4].split("-chunk-")[0] for item in chunk_lines)
    header = {}
    for line in log.splitlines()[:8]:
        if line.startswith("elapsed_ms") or line.startswith("started_at") or line.startswith("ended_at"):
            key, _, value = line.partition(":")
            header[key] = value.strip()
    # elapsed is in the footer
    footer = {}
    for line in log.splitlines()[-6:]:
        if ":" in line and not line.startswith("-"):
            key, _, value = line.partition(":")
            footer[key.strip()] = value.strip()

    smoke_nodes = 0
    smoke_edges = 0
    smoke_graph = SMOKE / "graph_chunk_entity_relation.graphml"
    if smoke_graph.is_file():
        smoke_tree = ET.parse(smoke_graph)
        smoke_nodes = len(smoke_tree.findall("g:graph/g:node", NS))
        smoke_edges = len(smoke_tree.findall("g:graph/g:edge", NS))

    quotes = []
    for row in catalog.values():
        if row.get("quoted_speakers") or row.get("attribution_status") not in {None, "lemkin", "author"}:
            if row.get("quoted_speakers") or "quot" in str(row.get("attribution_status")):
                quotes.append(
                    {
                        "graph_document_id": row["graph_document_id"],
                        "parent": row.get("parent_graph_document_id"),
                        "attribution_status": row.get("attribution_status"),
                        "author_position": row.get("author_position"),
                        "quoted_speakers": row.get("quoted_speakers"),
                        "extracted": row["graph_document_id"] in processed_ids,
                    }
                )

    report = {
        "manifest_ids": len(manifest_ids),
        "catalog_documents": len(catalog),
        "catalog_parents": len(catalog_parents),
        "missing_from_catalog": missing_from_catalog,
        "extra_catalog_parents": extra_catalog,
        "doc_status_rows": len(doc_status),
        "doc_status_counts": dict(by_status),
        "other_status_ids": other_status,
        "successfully_extracted_documents": len(processed_ids),
        "successfully_extracted_parents": len(processed_parents),
        "manifest_parents_without_processed_status": not_extracted_parents,
        "failed_or_skipped_detail": failed_parents,
        "index_manifest_processed": sum(
            1 for row in index_manifest.get("documents", {}).values() if row.get("status") == "processed"
        ),
        "index_manifest_documents": len(index_manifest.get("documents", {})),
        "splits": dict(splits),
        "non_train_catalog_ids": non_train,
        "graph_nodes": len(nodes),
        "graph_edges": len(edges),
        "graph_key_names": sorted(set(keys.values())),
        "weak_entity_count": len(weak),
        "weak_flag_counts": dict(flag_counts),
        "weekday_entity_names": weekday_names,
        "weak_entities": weak[:80],
        "malformed_entity_names": malformed,
        "nodes_missing_source_id": unknown_node_sources[:40],
        "nodes_missing_source_id_count": len(unknown_node_sources),
        "dangling_edges": dangling[:20],
        "dangling_edge_count": len(dangling),
        "edges_missing_source_id": empty_source_edges,
        "unknown_edge_source_ids": sorted(set(unknown_edge_sources))[:40],
        "failed_or_skipped_ids_still_cited_by_graph": failed_with_graph,
        "non_catalog_or_non_train_graph_sources": non_train_sources[:40],
        "marked_pilot_documents": marked,
        "chunks_containing_omission_marker": marker_chunks,
        "omission_marker_chunk_count": len(marker_chunks),
        "omission_marker_document_count": len(marker_doc_ids),
        "edges_citing_an_omission_marker_chunk": edges_from_marker_chunks,
        "quoted_catalog_rows": quotes,
        "this_resume_log": {
            "chunk_extractions_logged": len(chunk_lines),
            "entity_mentions_sum": sum(ent),
            "relation_mentions_sum": sum(rel),
            "zero_relation_chunks": sum(1 for item in rel if item == 0),
            "documents_with_a_logged_chunk": len(by_doc_chunks),
            "timeout_events": len(timeouts),
            "timeout_seconds_seen": sorted(set(timeouts)),
            "failed_extract_documents_in_log": failed_docs,
            "token_fields_in_log": bool(re.search(r"prompt_eval|eval_count|input_tokens|output_tokens", log)),
            "retry_lines": len(re.findall(r"(?i)retry", log)),
            "chunk_size_overlap_logged": "size=400" in log and "overlap=100" in log,
            "footer": footer,
        },
        "smoke_graph_nodes": smoke_nodes,
        "smoke_graph_edges": smoke_edges,
        "sample_node_attr_keys": sorted({key for node in nodes[:3] for key in node}),
        "sample_edge_attr_keys": sorted({key for edge in edges[:3] for key in edge}),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({
        "catalog": report["catalog_documents"],
        "extracted": report["successfully_extracted_documents"],
        "status": report["doc_status_counts"],
        "not_extracted": report["manifest_parents_without_processed_status"],
        "nodes": report["graph_nodes"],
        "edges": report["graph_edges"],
        "weak": report["weak_entity_count"],
        "weak_flags": report["weak_flag_counts"],
        "weekdays": report["weekday_entity_names"],
        "unknown_sources": report["unknown_edge_source_ids"][:8],
        "marker_docs": report["omission_marker_document_count"],
        "marker_edges": report["edges_citing_an_omission_marker_chunk"],
        "malformed": report["malformed_entity_names"],
        "dangling": report["dangling_edge_count"],
        "marker_chunks": len(report["chunks_containing_omission_marker"]),
        "failed_still_cited": report["failed_or_skipped_ids_still_cited_by_graph"],
        "chunks_logged": report["this_resume_log"]["chunk_extractions_logged"],
        "tokens_logged": report["this_resume_log"]["token_fields_in_log"],
        "smoke": [smoke_nodes, smoke_edges],
    }, indent=2))


if __name__ == "__main__":
    main()
