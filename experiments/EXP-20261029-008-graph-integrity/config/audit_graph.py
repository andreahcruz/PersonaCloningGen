"""Read-only integrity audit of the finished pilot graph.

Does not write into the LightRAG workdir and does not call Ollama.
"""

from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

from host_finetune.entity_audit import entity_name_flags
from host_finetune.lightrag_backend import OMISSION_MARKER, read_failed_document_ids

ROOT = Path(__file__).resolve().parents[3]
WORK = ROOT / "artifacts" / "lightrag" / "model_f" / "storage_pilot160"
OUT = Path(__file__).resolve().parents[1] / "metrics" / "integrity.json"
NS = {"g": "http://graphml.graphdrawing.org/xmlns"}


def _ids(value: str | None) -> list[str]:
    text = (value or "").replace("&lt;SEP&gt;", "<SEP>")
    return [part for part in text.split("<SEP>") if part]


def main() -> None:
    failed = read_failed_document_ids(WORK)
    chunks = json.loads((WORK / "kv_store_text_chunks.json").read_text(encoding="utf-8"))
    chunk_rows = list(chunks.values())
    failed_chunks = [row for row in chunk_rows if row.get("file_path") in failed]
    omission_chunks = [row for row in chunk_rows if OMISSION_MARKER in (row.get("content") or "")]
    catalog = {}
    for line in (WORK / "catalog.jsonl").read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            catalog[row["graph_document_id"]] = row
    unknown_chunks = [row.get("file_path") for row in chunk_rows if row.get("file_path") not in catalog]
    quoted = [
        row["graph_document_id"]
        for row in catalog.values()
        if row.get("attribution_status") == "third_party_quotation"
    ]

    tree = ET.parse(WORK / "graph_chunk_entity_relation.graphml")
    nodes = tree.findall(".//g:node", NS)
    edges = tree.findall(".//g:edge", NS)
    weak = Counter()
    weak_examples = []
    failed_node_refs = 0
    failed_edge_refs = 0
    omission_edge_refs = 0
    omission_chunk_ids = {row.get("_id") for row in omission_chunks}
    for node in nodes:
        name = node.get("id") or ""
        flags = entity_name_flags(name)
        for flag in flags:
            weak[flag] += 1
        if flags and len(weak_examples) < 12:
            weak_examples.append({"name": name, "flags": flags})
        paths = _ids(_data(node, "d4"))
        if any(path in failed for path in paths):
            failed_node_refs += 1
    for edge in edges:
        paths = _ids(_data(edge, "d11"))
        sources = _ids(_data(edge, "d10"))
        if any(path in failed for path in paths):
            failed_edge_refs += 1
        if any(source in omission_chunk_ids for source in sources):
            omission_edge_refs += 1

    report = {
        "workdir": str(WORK.relative_to(ROOT)),
        "mutated_graph": False,
        "failed_document_ids": sorted(failed),
        "failed_document_count": len(failed),
        "stored_chunks": len(chunk_rows),
        "chunks_from_failed_documents": len(failed_chunks),
        "failed_chunk_file_paths": sorted({row.get("file_path") for row in failed_chunks}),
        "omission_marker_chunks": len(omission_chunks),
        "chunks_missing_from_catalog": unknown_chunks,
        "third_party_quotation_documents": quoted,
        "graph_nodes": len(nodes),
        "graph_edges": len(edges),
        "nodes_citing_failed_documents": failed_node_refs,
        "edges_citing_failed_documents": failed_edge_refs,
        "edges_citing_omission_chunks": omission_edge_refs,
        "weak_entity_flags": dict(weak),
        "weak_entity_examples": weak_examples,
        "query_time_exclusions": [
            "failed_extraction_chunk",
            "omission_marker_span",
            "generated_summary_not_a_source_span",
            "unknown_graph_document",
        ],
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({key: report[key] for key in (
        "failed_document_count",
        "chunks_from_failed_documents",
        "omission_marker_chunks",
        "edges_citing_omission_chunks",
        "nodes_citing_failed_documents",
        "weak_entity_flags",
        "third_party_quotation_documents",
    )}, indent=2))


def _data(element, key: str) -> str:
    for child in list(element):
        if child.get("key") == key:
            return child.text or ""
    return ""


if __name__ == "__main__":
    main()
