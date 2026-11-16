"""Build the Elon Musk RAG index from the two selected public sources.

Selected sources:
* Zenodo's Elon Musk Twitter/X evolution dataset page (voice and phrasing)
* Hugging Face's Lex Fridman transcript dataset page (long-form reasoning)

This follows the Jason Lemkin Chroma pattern, but reads normalized records
from the Elon DuckDB database created by ``ingest_elon_data.py`` and writes an isolated ``elon_persona``
collection so the two personas cannot mix.

Run::

    python3 persona_pipeline/jobs/build_elon_rag.py
"""

from __future__ import annotations

import os
from pathlib import Path

import chromadb
import duckdb


ROOT = Path(__file__).resolve().parents[2]
DB_PATH = ROOT / "persona_pipeline" / "data" / "elon_musk.duckdb"
CHROMA_PATH = ROOT / "persona_pipeline" / "data" / "chroma_elon"
COLLECTION_NAME = "elon_persona"
CHUNK_SIZE = 450
CHUNK_OVERLAP = 60
FETCH_MULTIPLIER = 2

def chunk_text(text: str) -> list[str]:
    words = text.split()
    if len(words) <= CHUNK_SIZE:
        return [text]
    chunks = []
    start = 0
    while start < len(words):
        end = start + CHUNK_SIZE
        chunks.append(" ".join(words[start:end]))
        if end >= len(words):
            break
        start = end - CHUNK_OVERLAP
    return chunks


def main() -> None:
    if not DB_PATH.exists():
        raise FileNotFoundError(f"DuckDB not found: {DB_PATH}. Run elon_etl.py first.")

    con = duckdb.connect(str(DB_PATH), read_only=True)
    try:
        rows = con.execute("""
            SELECT source_type, source_type, source_url, title, text
            FROM elon_rag_documents
            WHERE text IS NOT NULL AND length(trim(text)) > 0
            ORDER BY source_type, title
        """).fetchall()
    finally:
        con.close()

    if not rows:
        raise RuntimeError("No successful documents found for the two selected sources.")

    CHROMA_PATH.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(CHROMA_PATH))
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass
    collection = client.get_or_create_collection(
        COLLECTION_NAME,
        metadata={"hnsw:space": "cosine", "persona": "Elon Musk"},
    )

    ids, documents, metadatas = [], [], []
    for source_name, source_type, url, title, text in rows:
        # Tweets are already short semantic units; interviews need overlap.
        chunks = [text] if source_type == "tweet" else chunk_text(text)
        for chunk_index, chunk in enumerate(chunks):
            ids.append(f"elon_{len(ids)}")
            documents.append(chunk)
            metadatas.append({
                "source_name": "Elon Musk tweets" if source_type == "tweet" else title,
                "source_type": source_type,
                "url": url,
                "title": title or source_name,
                "chunk_index": chunk_index,
                "persona": "elon_musk",
            })

    batch_size = 100
    for start in range(0, len(ids), batch_size):
        end = start + batch_size
        collection.add(
            ids=ids[start:end],
            documents=documents[start:end],
            metadatas=metadatas[start:end],
        )

    print(f"Added {len(ids)} chunks to '{COLLECTION_NAME}'")
    print(f"Chroma path: {CHROMA_PATH}")
    print(f"Sources indexed: {len(rows)}")


if __name__ == "__main__":
    main()
