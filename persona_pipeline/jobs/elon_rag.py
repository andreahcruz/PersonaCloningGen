"""Query the Elon Musk RAG collection built by build_elon_rag.py."""

from __future__ import annotations

import argparse
from pathlib import Path
from collections import defaultdict

import chromadb


ROOT = Path(__file__).resolve().parents[2]
CHROMA_PATH = ROOT / "persona_pipeline" / "data" / "chroma_elon"
COLLECTION_NAME = "elon_persona"


def retrieve(query: str, n_results: int = 5) -> list[dict]:
    client = chromadb.PersistentClient(path=str(CHROMA_PATH))
    collection = client.get_collection(COLLECTION_NAME)
    result = collection.query(query_texts=[query], n_results=max(n_results * 3, 10))
    docs = result.get("documents", [[]])[0]
    metas = result.get("metadatas", [[]])[0]
    distances = result.get("distances", [[]])[0]
    selected = []
    seen = set()
    per_source = defaultdict(int)
    for doc, meta, distance in zip(docs, metas, distances):
        fingerprint = " ".join(doc.lower().split())[:180]
        source = meta.get("source_type", "unknown")
        if fingerprint in seen or per_source[source] >= max(2, n_results // 2 + 1):
            continue
        seen.add(fingerprint)
        per_source[source] += 1
        selected.append({"text": doc, "metadata": meta, "distance": distance})
        if len(selected) >= n_results:
            break
    return selected


def main() -> None:
    parser = argparse.ArgumentParser(description="Retrieve context from the Elon Musk RAG index")
    parser.add_argument("query")
    parser.add_argument("-k", type=int, default=5)
    args = parser.parse_args()
    for index, item in enumerate(retrieve(args.query, args.k), start=1):
        print(f"\n[{index}] {item['metadata']['source_name']} distance={item['distance']:.4f}")
        print(item["text"][:1200])


if __name__ == "__main__":
    main()
