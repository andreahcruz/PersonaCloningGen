"""
Phase 5 — Build ChromaDB vector store from DuckDB posts.

Chunks posts into ~500-word segments with 50-word overlap, embeds them
with all-MiniLM-L6-v2, and stores them in a local ChromaDB collection.

Run: python persona_pipeline/jobs/build_chromadb.py
"""
from __future__ import annotations

import os
import sys
import warnings
from pathlib import Path

REPO_ROOT = str(Path(__file__).resolve().parents[2])
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import chromadb
from chromadb.config import Settings
import duckdb

from persona_pipeline.config.config import CHROMA_PATH, DB_PATH

COLLECTION_NAME = "lemkin_persona"
CHUNK_SIZE = 500       # words
CHUNK_OVERLAP = 50     # words
BATCH_SIZE = 100


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    words = text.split()
    if len(words) <= chunk_size:
        return [text]
    chunks: list[str] = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunks.append(" ".join(words[start:end]))
        start = end - overlap
    return chunks


def main() -> None:
    if not os.path.exists(DB_PATH):
        warnings.warn(f"DuckDB not found at {DB_PATH}; run load_duckdb.py first.")
        return

    con = duckdb.connect(DB_PATH, read_only=True)
    try:
        rows = con.execute(
            "SELECT id, source, cleaned_text, is_b2b FROM all_posts WHERE word_count > 100"
        ).fetchall()
    finally:
        con.close()

    if not rows:
        warnings.warn("No posts with word_count > 100 found in DuckDB.")
        return

    os.makedirs(CHROMA_PATH, exist_ok=True)
    client = chromadb.PersistentClient(path=CHROMA_PATH)

    # Delete existing collection if present so we rebuild cleanly.
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass

    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    documents: list[str] = []
    metadatas: list[dict] = []
    ids: list[str] = []

    for post_id, source, text, is_b2b in rows:
        if not text:
            continue
        chunks = chunk_text(text)
        for ci, chunk in enumerate(chunks):
            chunk_id = f"{post_id}_c{ci}"
            documents.append(chunk)
            metadatas.append({
                "source": source or "",
                "is_b2b": bool(is_b2b),
                "post_id": str(post_id),
                "source_type": "persona_writing",
            })
            ids.append(chunk_id)

    # Add in batches
    total = len(documents)
    for i in range(0, total, BATCH_SIZE):
        end = min(i + BATCH_SIZE, total)
        collection.add(
            documents=documents[i:end],
            metadatas=metadatas[i:end],
            ids=ids[i:end],
        )

    print(f"[build_chromadb] Added {total} chunks to collection '{COLLECTION_NAME}'")
    print(f"[build_chromadb] ChromaDB path: {CHROMA_PATH}")


if __name__ == "__main__":
    main()
