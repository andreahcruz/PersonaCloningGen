import hashlib
import json
import sqlite3
from pathlib import Path

root = Path(r"artifacts/lightrag/model_f/storage_expand_v1")

def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

conn = sqlite3.connect(root / "indexing_ledger.sqlite")
conn.row_factory = sqlite3.Row
ids = (
    "gd_09e160ccf7663729c581d1e2",
    "gd_0270bf4a3abebc35861d117f",
    "gd_014b45cb99cfee97ee337556",
    "gd_756878d4d219336489c49168",
    "gd_0e1afddaca819c17829b209e",
    "gd_002be28d89a6ca9924e73be2",
    "gd_087a4675a11cf57ae08ea7cf",
)
print("fingerprint", conn.execute("select value from meta where key='configuration_fingerprint'").fetchone()[0])
for doc_id in ids:
    row = conn.execute(
        "select document_id, source_medium, word_count, estimated_chunks, status, selection_rank from documents where document_id=?",
        (doc_id,),
    ).fetchone()
    print(dict(row) if row else doc_id)
print("selection_manifest", sha256(root / "selection_manifest.jsonl"))
print("summary", sha256(root / "reports" / "indexing_summary.json"))
print("graphml", sha256(root / "graph_chunk_entity_relation.graphml"))
