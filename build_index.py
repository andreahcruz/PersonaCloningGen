#!/usr/bin/env python3
"""
Build a Chroma vector index from pg_chunks.parquet using Ollama /api/embed.
Run once, then use generate.py to retrieve and generate.

  python build_index.py --corpus data/pg_chunks.parquet --out data/index/
"""
import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import requests
import chromadb
from chromadb.config import Settings


OLLAMA_EMBED_URL = "http://localhost:11434/api/embed"
DEFAULT_EMBED_MODEL = "nomic-embed-text"
BATCH_SIZE = 32
COLLECTION_NAME = "pg_essays"


def ollama_embed(texts: list[str], base_url: str, model: str) -> list[list[float]]:
    """Call Ollama /api/embed. Batches if needed."""
    url = f"{base_url.rstrip('/')}/api/embed"
    all_embeddings = []
    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i : i + BATCH_SIZE]
        payload = {"model": model, "input": batch}
        resp = requests.post(url, json=payload, timeout=120)
        resp.raise_for_status()
        data = resp.json()
        all_embeddings.extend(data["embeddings"])
    return all_embeddings


def main():
    p = argparse.ArgumentParser(description="Build Chroma index from chunk parquet using Ollama embeddings.")
    p.add_argument("--corpus", required=True, help="Path to pg_chunks.parquet")
    p.add_argument("--out", required=True, help="Output directory for Chroma DB (e.g. data/index/)")
    p.add_argument("--ollama-base", default="http://localhost:11434", help="Ollama base URL")
    p.add_argument("--embed-model", default=DEFAULT_EMBED_MODEL, help="Ollama embed model name")
    args = p.parse_args()

    corpus_path = Path(args.corpus)
    if not corpus_path.exists():
        print(f"Error: corpus not found: {corpus_path}", file=sys.stderr)
        sys.exit(1)

    out_path = Path(args.out)
    out_path.mkdir(parents=True, exist_ok=True)

    print("Loading corpus...")
    df = pd.read_parquet(corpus_path)
    df = df.astype(str, errors="ignore")
    texts = df["chunk_text"].tolist()
    ids = [str(x) for x in df["chunk_id"].tolist()]
    metadatas = df[["doc_id", "chunk_index", "source", "format"]].copy()
    if "split" in df.columns:
        metadatas["split"] = df["split"]
    metadatas = metadatas.to_dict("records")
    for m in metadatas:
        for k, v in m.items():
            m[k] = str(v)[:100]

    print(f"Embedding {len(texts)} chunks with Ollama ({args.embed_model})...")
    embeddings = ollama_embed(texts, args.ollama_base, args.embed_model)
    if len(embeddings) != len(ids):
        print(f"Error: got {len(embeddings)} embeddings for {len(ids)} chunks", file=sys.stderr)
        sys.exit(1)

    print("Writing Chroma index...")
    client = chromadb.PersistentClient(path=str(out_path), settings=Settings(anonymized_telemetry=False))
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass
    collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata={"embed_model": args.embed_model, "description": "PG essay chunks for RAG"},
    )
    for i in range(0, len(ids), BATCH_SIZE):
        end = min(i + BATCH_SIZE, len(ids))
        collection.add(
            ids=ids[i:end],
            documents=texts[i:end],
            metadatas=metadatas[i:end],
            embeddings=embeddings[i:end],
        )
    print(f"Indexed {len(ids)} chunks at {out_path} (collection: {COLLECTION_NAME})")


if __name__ == "__main__":
    main()
