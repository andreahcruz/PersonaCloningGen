"""
Phase 6 — PersonaRAG retrieval logic.

Key difference from naive RAG: the user query is *rewritten* to include
the persona's framing and vocabulary before retrieval, and results are
*reranked* by persona vocabulary alignment — so we retrieve chunks that
sound like Jason Lemkin talking about the topic, not just
topically-relevant chunks.

Reference: PersonaRAG (Zerhoudi & Granitzer, 2024, arXiv 2407.09394)

Run: python persona_pipeline/jobs/persona_rag.py
"""
from __future__ import annotations

import json
import os
import sys
import warnings
from pathlib import Path

REPO_ROOT = str(Path(__file__).resolve().parents[2])
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import chromadb

from persona_pipeline.config.config import CHROMA_PATH, PERSONA_PROFILE_PATH

COLLECTION_NAME = "lemkin_persona"


def load_persona_profile(path: str | None = None) -> dict:
    path = path or PERSONA_PROFILE_PATH
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def rewrite_query(user_query: str, persona_profile: dict) -> str:
    """
    Rewrite user query to include the persona's framing so that
    embedding-based retrieval surfaces content matching BOTH the topic
    AND the author's voice.
    """
    core_topics = persona_profile.get("core_topics", [])
    char_words = persona_profile.get("style", {}).get("characteristic_words", [])

    topic_str = ", ".join(core_topics[:4])
    vocab_str = " ".join(char_words[:15])

    rewritten = (
        f"{user_query} — in the context of {topic_str}. "
        f"Key vocabulary: {vocab_str}"
    )
    return rewritten


def persona_alignment_score(chunk_text: str, characteristic_words: list[str]) -> float:
    """
    Score a chunk by how much its vocabulary overlaps with the persona's
    characteristic words (higher = more persona-aligned).
    """
    if not chunk_text or not characteristic_words:
        return 0.0
    chunk_words = set(chunk_text.lower().split())
    char_set = set(characteristic_words)
    overlap = chunk_words & char_set
    return len(overlap) / len(char_set)


def persona_conditioned_retrieve(
    user_query: str,
    persona_profile: dict,
    n_results: int = 5,
) -> list[dict]:
    """
    PersonaRAG retrieval pipeline:
    1. Rewrite query through persona lens
    2. Retrieve 2x candidates from ChromaDB
    3. Rerank by persona alignment
    4. Return top n_results
    """
    if not os.path.exists(CHROMA_PATH):
        warnings.warn(f"ChromaDB not found at {CHROMA_PATH}; run build_chromadb.py first.")
        return []

    client = chromadb.PersistentClient(path=CHROMA_PATH)

    try:
        collection = client.get_collection(COLLECTION_NAME)
    except Exception:
        warnings.warn(f"Collection '{COLLECTION_NAME}' not found in ChromaDB.")
        return []

    # Step 1: rewrite query
    rewritten = rewrite_query(user_query, persona_profile)

    # Step 2: retrieve 2x candidates
    fetch_n = n_results * 2
    results = collection.query(
        query_texts=[rewritten],
        n_results=fetch_n,
    )

    if not results or not results.get("documents"):
        return []

    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0] if results.get("distances") else [0.0] * len(documents)

    char_words = persona_profile.get("style", {}).get("characteristic_words", [])

    # Step 3: score and rerank
    candidates = []
    for doc, meta, dist in zip(documents, metadatas, distances):
        alignment = persona_alignment_score(doc, char_words)
        candidates.append({
            "text": doc,
            "metadata": meta,
            "distance": dist,
            "persona_alignment": alignment,
        })

    candidates.sort(key=lambda c: c["persona_alignment"], reverse=True)

    # Step 4: return top n
    return candidates[:n_results]


def main() -> None:
    if not os.path.exists(PERSONA_PROFILE_PATH):
        warnings.warn("Persona profile not found; run build_persona_profile.py first.")
        return

    profile = load_persona_profile()

    test_queries = [
        "How should a SaaS founder price their product?",
        "When should a startup hire their first VP of Sales?",
        "What does net negative churn look like in practice?",
    ]

    for q in test_queries:
        print(f"\n{'='*60}")
        print(f"Query: {q}")
        rewritten = rewrite_query(q, profile)
        print(f"Rewritten: {rewritten[:120]}...")
        results = persona_conditioned_retrieve(q, profile, n_results=3)
        for i, r in enumerate(results, 1):
            snippet = r["text"][:100].replace("\n", " ")
            print(f"  [{i}] alignment={r['persona_alignment']:.3f}  {snippet}...")


if __name__ == "__main__":
    main()
