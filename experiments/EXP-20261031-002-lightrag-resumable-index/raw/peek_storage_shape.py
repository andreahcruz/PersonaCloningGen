"""Read-only peek at expansion storage shapes. Prints keys and counts only."""
import json
from pathlib import Path
import xml.etree.ElementTree as ET

root = Path(r"artifacts/lightrag/model_f/storage_expand_v1")
vdb = json.loads((root / "vdb_chunks.json").read_text(encoding="utf-8"))
row = (vdb.get("data") or [None])[0]
print("chunk_row_keys", sorted(row) if isinstance(row, dict) else type(row))
print("chunk_count", len(vdb.get("data") or []))
ids = []
for item in vdb.get("data") or []:
    ids.append(item.get("__id__") or item.get("id"))
print("chunk_ids", ids)
ents = json.loads((root / "kv_store_full_entities.json").read_text(encoding="utf-8"))
print("entity_docs", sorted(ents))
sample = next(iter(ents.values()))
print("entity_row_keys", sorted(sample) if isinstance(sample, dict) else type(sample))
rels = json.loads((root / "kv_store_full_relations.json").read_text(encoding="utf-8"))
print("relation_docs", sorted(rels))
if rels:
    sample_r = next(iter(rels.values()))
    print("relation_row_keys", sorted(sample_r) if isinstance(sample_r, dict) else type(sample_r))
status = json.loads((root / "kv_store_doc_status.json").read_text(encoding="utf-8"))
for doc_id, row in status.items():
    print("status", doc_id, row.get("status"), "chunks", row.get("chunks_count"), "kg", (row.get("metadata") or {}).get("kg_write_state"))
chunks = json.loads((root / "kv_store_text_chunks.json").read_text(encoding="utf-8"))
print("text_chunk_count", len(chunks))
one = next(iter(chunks.values()))
print("text_chunk_keys", sorted(one) if isinstance(one, dict) else type(one))
# graph node file_path sample
tree = ET.parse(root / "graph_chunk_entity_relation.graphml")
ns = {"g": "http://graphml.graphdrawing.org/xmlns"}
nodes = list(tree.iter("{http://graphml.graphdrawing.org/xmlns}node"))
edges = list(tree.iter("{http://graphml.graphdrawing.org/xmlns}edge"))
print("graph_nodes", len(nodes), "graph_edges", len(edges))
keys = {}
for key in tree.iter("{http://graphml.graphdrawing.org/xmlns}key"):
    keys[key.attrib.get("id")] = key.attrib.get("attr.name")
print("graph_keys", keys)
