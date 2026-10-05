"""Read-only audit of the persisted Factual Dense RAG v1 index.

Does not rebuild the collection, does not judge relevance, and does not load
the generator. Ranked IR metrics stay out of this run because no qrels exist.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import chromadb

from host_finetune.rebuild_chroma import _embed_batch
from host_finetune.retrieval_eval import chunk_flags, summarize_retrieval
from host_finetune.train_only_index import (
    DEFAULT_ASSIGNMENTS,
    DEFAULT_DATASET,
    DEFAULT_GOLD_EVAL,
    DEFAULT_PERSIST_DIR,
    TRAIN_ONLY_COLLECTION,
    query_excerpts,
)

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = ROOT / "experiments" / "EXP-20261004-009-dense-rag-v1"
PAGE = 500
EXAMPLE_LIMIT = 8


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git() -> dict:
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip())
    return {"commit": commit, "dirty": dirty}


def _load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _roles() -> dict[int, str]:
    roles = {}
    with DEFAULT_ASSIGNMENTS.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if "row_index" in row:
                roles[int(row["row_index"])] = row.get("sft_role") or ""
    return roles


def _percentile(ordered: list[int], q: float) -> int | None:
    if not ordered:
        return None
    index = min(len(ordered) - 1, int(q * (len(ordered) - 1)))
    return ordered[index]


def _scan(collection) -> dict:
    count = collection.count()
    platforms: Counter[str] = Counter()
    splits: Counter[str] = Counter()
    flags: Counter[str] = Counter()
    roles: Counter[str] = Counter()
    role_map = _roles()
    groups: Counter[str] = Counter()
    word_counts: list[int] = []
    examples: dict[str, list[dict]] = {
        "mid_list": [],
        "very_short": [],
        "headline_or_link": [],
        "starts_lower": [],
    }
    missing_split = 0
    offset = 0
    while offset < count:
        page = collection.get(include=["documents", "metadatas"], limit=PAGE, offset=offset)
        ids = page.get("ids") or []
        if not ids:
            break
        documents = page.get("documents") or []
        metadatas = page.get("metadatas") or []
        for doc_id, text, meta in zip(ids, documents, metadatas):
            meta = meta or {}
            medium = meta.get("medium") or "missing"
            platforms[medium] += 1
            split = meta.get("split")
            if split:
                splits[str(split)] += 1
            else:
                missing_split += 1
            group = meta.get("group_id")
            if group:
                groups[str(group)] += 1
            row_id = meta.get("row_id")
            if row_id is not None and int(row_id) in role_map:
                roles[role_map[int(row_id)] or "missing"] += 1
            marked = chunk_flags(text or "")
            word_counts.append(marked["words"])
            for name in (
                "mid_list",
                "very_short",
                "short",
                "headline_or_link",
                "no_sentence_end",
                "starts_lower",
            ):
                if marked[name]:
                    flags[name] += 1
                    if name in examples and len(examples[name]) < EXAMPLE_LIMIT:
                        examples[name].append(
                            {
                                "id": doc_id,
                                "medium": medium,
                                "row_id": row_id,
                                "words": marked["words"],
                                "preview": (text or "")[:180],
                            }
                        )
        offset += len(ids)
    multi = [size for size in groups.values() if size > 1]
    ordered = sorted(word_counts)
    held = collection.get(ids=["train_26148"])
    return {
        "document_count": count,
        "platforms": dict(platforms),
        "splits": dict(splits),
        "missing_split": missing_split,
        "train_26148_present": bool(held.get("ids")),
        "flag_counts": dict(flags),
        "sft_role_counts": dict(roles),
        "n_groups": len(groups),
        "groups_with_multiple_chunks": len(multi),
        "documents_in_multi_chunk_groups": sum(multi),
        "max_group_size": max(groups.values()) if groups else 0,
        "word_count_p10": _percentile(ordered, 0.10),
        "word_count_p50": _percentile(ordered, 0.50),
        "word_count_p90": _percentile(ordered, 0.90),
        "examples": examples,
    }


def _retrieve(collection, topics: list[dict]) -> list[dict]:
    vectors = _embed_batch(
        [row["topic"] for row in topics],
        "http://127.0.0.1:11434",
        "nomic-embed-text",
        cpu=True,
    )
    traces = []
    for row, vector in zip(topics, vectors):
        hits = query_excerpts(collection, vector, 4, where=None)
        traces.append({"query_id": row["id"], "topic": row["topic"], "hits": hits})
    return traces


def main() -> int:
    EXPERIMENT.mkdir(parents=True)
    for name in ("config", "metrics", "raw"):
        (EXPERIMENT / name).mkdir(exist_ok=True)
    client = chromadb.PersistentClient(path=str(DEFAULT_PERSIST_DIR))
    collection = client.get_collection(TRAIN_ONLY_COLLECTION)
    corpus = _scan(collection)
    topics = _load_jsonl(DEFAULT_GOLD_EVAL)
    traces = _retrieve(collection, topics)
    diagnostics = summarize_retrieval(traces, relevance_scope="none")
    all_x = sum(
        1
        for trace in traces
        if trace["hits"] and {hit.get("medium") for hit in trace["hits"]} == {"x"}
    )
    diagnostics["summary"]["queries_with_all_x_hits"] = all_x
    git = _git()
    timestamp = datetime.now(timezone.utc).isoformat()
    retriever = {
        "name": "Factual Dense RAG v1",
        "collection": TRAIN_ONLY_COLLECTION,
        "persist_dir": str(DEFAULT_PERSIST_DIR),
        "document_count": corpus["document_count"],
        "embedding_model": "nomic-embed-text",
        "embed_device": "cpu",
        "distance_metric": "cosine",
        "query": "gold topic text only",
        "retrieval_k": 4,
        "medium_filter": False,
        "reranker": False,
        "metadata_fields": ["group_id", "medium", "row_id", "split"],
        "chunk_source": str(DEFAULT_DATASET),
        "chunk_unit": "relabel SFT train row output, not a Spark 500-word chunk",
        "dataset_sha256": _sha256(DEFAULT_DATASET),
        "assignments_sha256": _sha256(DEFAULT_ASSIGNMENTS),
        "gold_eval_sha256": _sha256(DEFAULT_GOLD_EVAL),
        "git": git,
        "timestamp": timestamp,
    }
    (EXPERIMENT / "config" / "retriever.json").write_text(
        json.dumps(retriever, indent=2) + "\n", encoding="utf-8"
    )
    (EXPERIMENT / "metrics" / "corpus.json").write_text(
        json.dumps(corpus, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (EXPERIMENT / "metrics" / "retrieval_diagnostics.json").write_text(
        json.dumps(diagnostics["summary"], indent=2) + "\n", encoding="utf-8"
    )
    with (EXPERIMENT / "raw" / "query_traces.jsonl").open("w", encoding="utf-8") as handle:
        for trace in traces:
            handle.write(json.dumps(trace, ensure_ascii=False) + "\n")
    (EXPERIMENT / "config" / "commands.txt").write_text(
        "host_finetune\\.venv\\Scripts\\python.exe -m host_finetune.audit_retrieval_corpus\n",
        encoding="utf-8",
    )
    manifest = (
        "experiment_id: EXP-20261004-009-dense-rag-v1\n"
        f"timestamp: {timestamp}\n"
        "execution_commands:\n"
        "  - host_finetune\\.venv\\Scripts\\python.exe -m host_finetune.audit_retrieval_corpus\n"
        "git:\n"
        f"  commit: {git['commit']}\n"
        f"  dirty: {str(git['dirty']).lower()}\n"
        "data:\n"
        "  source_manifest: n/a\n"
        "  source_hashes: {}\n"
        f"  processed_dataset_hash: {retriever['dataset_sha256']}\n"
        f"  split_manifest_hash: {retriever['assignments_sha256']}\n"
        f"  gold_eval_hash: {retriever['gold_eval_sha256']}\n"
        "rag:\n"
        "  persona_profile_hash: n/a\n"
        f"  chroma_collection: {TRAIN_ONLY_COLLECTION}\n"
        f"  index_version_or_snapshot: document_count={corpus['document_count']}\n"
        "  embedding_model: nomic-embed-text\n"
        "model:\n"
        "  model_name: n/a\n"
        "  model_or_adapter_hash: n/a\n"
        "  config: retrieval-only; generator not loaded\n"
        "  random_seed: n/a\n"
        "evaluation:\n"
        f"  evaluator_commit: {git['commit']}\n"
        "  judge_model: n/a\n"
        "  judge_config: no relevance judge; qrels do not exist\n"
    )
    (EXPERIMENT / "manifest.yaml").write_text(manifest, encoding="utf-8")
    print(json.dumps({
        "documents": corpus["document_count"],
        "platforms": corpus["platforms"],
        "flags": corpus["flag_counts"],
        "train_26148_present": corpus["train_26148_present"],
        "queries": diagnostics["summary"]["n_queries"],
        "mean_distance": diagnostics["summary"]["mean_cosine_distance"],
        "all_x_queries": all_x,
    }))
    return 0


if __name__ == "__main__":
    sys.exit(main())
