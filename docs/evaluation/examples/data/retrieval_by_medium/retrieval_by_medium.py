"""What would retrieval return if it were filtered by the requested medium? (PUK-34, "type-specific RAG")

Kevin's v1 retriever searches the whole train-only index and ignores the requested medium on purpose.
Kevin asked us to try a type-specific version. Before spending GPU time on a Colab run, this script answers the
cheap part: for each of the 30 gold topics, what are the top 4 passages when the search is limited to blog,
LinkedIn, X or talk (transcript) passages, compared with what the unfiltered search returns now?

It generates nothing and calls no judge. It needs the index, Ollama with the nomic-embed-text model, and two
packages (the versions Kevin used):

    pip install chromadb==1.5.5 requests
    python docs/evaluation/examples/data/retrieval_by_medium/retrieval_by_medium.py \
        --chroma-path <handoff>/REQUIRED/chroma_lemkin_train_only

The query is exactly Kevin's: the topic string, embedded with ``nomic-embed-text`` through Ollama's /api/embed on the
CPU, searched in the cosine space of collection ``lemkin_train_only`` with k = 4. The filter is the one his
unused helper ``medium_where`` defines: talk searches ``youtube_jason`` and the written mediums search their own name.

First it checks that the unfiltered search reproduces the passages saved in our Colab run for the same topics. If that
check fails, the rest is not to be trusted and the script says so.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import Counter
from pathlib import Path


HERE = Path(__file__).resolve().parent
COLLECTION = "lemkin_train_only"
EMBED_MODEL = "nomic-embed-text"
K = 4
# Kevin's COMPARISON_MEDIUM_TO_INDEX: what each requested medium is called in the index.
INDEX_MEDIUM = {"blog": "blog", "linkedin": "linkedin", "x": "x", "talk": "youtube_jason"}
VERY_SHORT_WORDS = 20  # Kevin's own "very short" flag


def embed(topic: str, base_url: str) -> list[float]:
    import requests

    body = {"model": EMBED_MODEL, "input": [topic], "options": {"num_gpu": 0}}
    response = requests.post(base_url.rstrip("/") + "/api/embed", json=body, timeout=300)
    response.raise_for_status()
    return response.json()["embeddings"][0]


def search(collection, vector: list[float], where: dict | None) -> list[dict]:
    kwargs = {"query_embeddings": [vector], "n_results": K, "include": ["documents", "metadatas", "distances"]}
    if where is not None:
        kwargs["where"] = where
    result = collection.query(**kwargs)
    hits = []
    for doc_id, text, meta, distance in zip(result["ids"][0], result["documents"][0], result["metadatas"][0], result["distances"][0]):
        hits.append({"id": doc_id, "medium": (meta or {}).get("medium"), "group_id": (meta or {}).get("group_id"),
                     "distance": distance, "words": len((text or "").split()), "text": text})
    return hits


def mean(values):
    values = list(values)
    return round(statistics.fmean(values), 4) if values else None


def summarize(topics: list[dict], unfiltered: dict, filtered: dict, pool: Counter) -> dict:
    """Per requested medium: how the filtered passages compare with the unfiltered ones."""
    out = {}
    for medium, index_name in INDEX_MEDIUM.items():
        un = [unfiltered[t["id"]] for t in topics]
        fl = [filtered[medium][t["id"]] for t in topics]
        gap = [mean(h["distance"] for h in f) - mean(h["distance"] for h in u) for u, f in zip(un, fl)]
        overlap = [len({h["id"] for h in u} & {h["id"] for h in f}) / K for u, f in zip(un, fl)]
        out[medium] = {
            "index_medium": index_name,
            "pool_documents": pool[index_name],
            "unfiltered_share_of_medium": mean(sum(h["medium"] == index_name for h in u) / K for u in un),
            "filtered_share_of_medium": mean(sum(h["medium"] == index_name for h in f) / K for f in fl),
            "unfiltered_mean_distance": mean(h["distance"] for u in un for h in u),
            "filtered_mean_distance": mean(h["distance"] for f in fl for h in f),
            "distance_gap_mean": mean(gap),
            "distance_gap_max": round(max(gap), 4),
            "filtered_best_hit_distance_mean": mean(f[0]["distance"] for f in fl),
            "unfiltered_mean_words": mean(h["words"] for u in un for h in u),
            "filtered_mean_words": mean(h["words"] for f in fl for h in f),
            "unfiltered_very_short_share": mean(h["words"] < VERY_SHORT_WORDS for u in un for h in u),
            "filtered_very_short_share": mean(h["words"] < VERY_SHORT_WORDS for f in fl for h in f),
            "overlap_with_unfiltered_share": mean(overlap),
            "filtered_unique_groups_mean": mean(len({h["group_id"] for h in f}) for f in fl),
        }
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--chroma-path", required=True, help="Folder chroma_lemkin_train_only from the handoff.")
    p.add_argument("--topics", default=str(HERE / "topics.json"))
    p.add_argument("--ollama", default="http://127.0.0.1:11434")
    p.add_argument("--out-json", default=str(HERE / "results.json"))
    p.add_argument("--examples", type=int, default=4, help="How many topics to print passages for, per medium.")
    args = p.parse_args()

    import chromadb
    from chromadb.config import Settings

    topics = json.loads(Path(args.topics).read_text(encoding="utf-8"))
    client = chromadb.PersistentClient(path=args.chroma_path, settings=Settings(anonymized_telemetry=False))
    collection = client.get_collection(COLLECTION)
    metas = collection.get(include=["metadatas"])["metadatas"]
    pool = Counter((m or {}).get("medium") for m in metas)
    print("documents:", collection.count(), dict(pool))

    unfiltered, filtered = {}, {m: {} for m in INDEX_MEDIUM}
    for topic in topics:
        vector = embed(topic["query"], args.ollama)
        unfiltered[topic["id"]] = search(collection, vector, None)
        for medium, index_name in INDEX_MEDIUM.items():
            filtered[medium][topic["id"]] = search(collection, vector, {"medium": index_name})

    same = sum([h["id"] for h in unfiltered[t["id"]]] == t["saved_top4_colab_seed43"] for t in topics)
    print(f"check: unfiltered top-4 equals the passages saved in the Colab run for {same} of {len(topics)} topics")
    if same < len(topics) - 2:
        print("WARNING: this does not reproduce the saved retrieval. Do not trust the numbers below.", file=sys.stderr)

    summary = summarize(topics, unfiltered, filtered, pool)
    for medium, s in summary.items():
        print(f"{medium:9s} pool {s['pool_documents']:6d} | distance unfiltered {s['unfiltered_mean_distance']} -> filtered {s['filtered_mean_distance']} "
              f"(gap {s['distance_gap_mean']:+.3f}, worst {s['distance_gap_max']:+.3f}) | words {s['unfiltered_mean_words']} -> {s['filtered_mean_words']} "
              f"| very short {s['unfiltered_very_short_share']} -> {s['filtered_very_short_share']} | kept from unfiltered {s['overlap_with_unfiltered_share']}")

    examples = {}
    for medium in INDEX_MEDIUM:
        examples[medium] = []
        for topic in topics[: args.examples]:
            best = filtered[medium][topic["id"]][0]
            examples[medium].append({"id": topic["id"], "topic": topic["query"], "distance": round(best["distance"], 4),
                                     "words": best["words"], "passage": best["text"][:300]})
    result = {
        "reproduction_check": {"topics_matching_saved_run": same, "topics": len(topics)},
        "index_documents": collection.count(), "index_by_medium": dict(pool), "k": K,
        "per_medium": summary, "examples_best_filtered_hit": examples,
        "unfiltered_top4_ids": {t["id"]: [h["id"] for h in unfiltered[t["id"]]] for t in topics},
        "filtered_top4_ids": {m: {t["id"]: [h["id"] for h in filtered[m][t["id"]]] for t in topics} for m in INDEX_MEDIUM},
        "filtered_top4_distances": {m: {t["id"]: [round(h["distance"], 4) for h in filtered[m][t["id"]]] for t in topics} for m in INDEX_MEDIUM},
        "unfiltered_top4_distances": {t["id"]: [round(h["distance"], 4) for h in unfiltered[t["id"]]] for t in topics},
    }
    Path(args.out_json).write_bytes(json.dumps(result, indent=1, ensure_ascii=False).encode("utf-8"))
    print("wrote", args.out_json)


if __name__ == "__main__":
    main()
