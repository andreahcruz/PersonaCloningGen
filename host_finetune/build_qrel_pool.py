"""Build the initial Dense RAG v1 candidate pool. Does not assign grades.

CPU embeddings only. Does not load a generator and does not write into
EXP-20261004-009-dense-rag-v1.
"""
from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import chromadb

from host_finetune.build_qrel_pool_docs import SCHEMA, JUDGE_PROMPT, README, notes_for
from host_finetune.qrel_pool import annotate_query, judgment_view, pool_diagnostics, text_sha256
from host_finetune.rebuild_chroma import _embed_batch
from host_finetune.train_only_index import (
    DEFAULT_ASSIGNMENTS,
    DEFAULT_DATASET,
    DEFAULT_GOLD_EVAL,
    DEFAULT_PERSIST_DIR,
    TRAIN_ONLY_COLLECTION,
    query_excerpts,
)

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = ROOT / "experiments" / "EXP-20261004-010-retrieval-qrels-v1"
FROZEN = "EXP-20261004-009-dense-rag-v1"
POOL_K = 20
FROZEN_DATASET = "1231835ecc5dcb608fefa0357d325affb2afbdc3aeade2a63b1dc42b2920897b"
FROZEN_SPLIT = "6e31aedf33bbc81488641e3a2d0d284199bd7c3e724d97d27e801a37010ea7fb"
FROZEN_GOLD = "d7c4f997899e28f28eb3ea9aa3417857da6f0666acbf72fd8ac31422034e520e"
FROZEN_SHA = "4006d4cee7768789c460c4c2ae1b1fcb1b124b03"


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
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _assignments() -> dict[int, dict]:
    rows = {}
    with DEFAULT_ASSIGNMENTS.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("record_type") == "meta" or "row_index" not in row:
                continue
            rows[int(row["row_index"])] = row
    return rows


def _sources(needed: dict[str, set[int]]) -> dict[tuple[str, int], dict]:
    found = {}
    cleaned = ROOT / "data" / "cleaned"
    for name, lines in needed.items():
        path = cleaned / name
        if not path.exists() or not lines:
            continue
        pending = set(lines)
        with path.open(encoding="utf-8") as handle:
            for number, line in enumerate(handle, 1):
                if number not in pending:
                    continue
                found[(name, number)] = json.loads(line)
                pending.discard(number)
                if not pending:
                    break
    return found


def _existing_labels() -> dict:
    files = list((ROOT / "experiments").glob("**/evidence_review.jsonl"))
    filled_on_topic = 0
    filled_claim = 0
    rows = 0
    for path in files:
        for line in path.open(encoding="utf-8"):
            if not line.strip():
                continue
            row = json.loads(line)
            rows += 1
            if str(row.get("on_topic") or "").strip():
                filled_on_topic += 1
            if str(row.get("claim_support") or "").strip():
                filled_claim += 1
    return {
        "evidence_review_files": [str(path.relative_to(ROOT)) for path in files],
        "evidence_review_rows": rows,
        "nonempty_on_topic": filled_on_topic,
        "nonempty_claim_support": filled_claim,
        "retrieval_qrels_found": False,
    }


def _provenance(hits: list[dict], assignments: dict[int, dict]) -> dict[int, dict]:
    needed: dict[str, set[int]] = defaultdict(set)
    for hit in hits:
        row_id = hit.get("row_id")
        if row_id is None:
            continue
        assignment = assignments.get(int(row_id))
        if not assignment:
            continue
        source_file = assignment.get("source_file")
        source_line = assignment.get("source_line")
        if source_file and source_line:
            needed[source_file].add(int(source_line))
    sources = _sources(needed)
    provenance = {}
    for hit in hits:
        row_id = hit.get("row_id")
        if row_id is None:
            continue
        assignment = assignments.get(int(row_id), {})
        source_file = assignment.get("source_file")
        source_line = assignment.get("source_line")
        record = sources.get((source_file, int(source_line))) if source_file and source_line else None
        record = record or {}
        provenance[int(row_id)] = {
            "sft_role": assignment.get("sft_role"),
            "split": assignment.get("split"),
            "part_index": assignment.get("part_index"),
            "part_count": assignment.get("part_count"),
            "source_file": source_file,
            "source_line": source_line,
            "source_url": record.get("url") or record.get("video_url"),
            "source_id": record.get("id"),
        }
    return provenance


def main() -> int:
    judgments_path = EXPERIMENT / "qrels_judgments.jsonl"
    if judgments_path.exists() and judgments_path.stat().st_size > 0:
        print("refusing to overwrite existing judgments", file=sys.stderr)
        return 2
    dataset_hash = _sha256(DEFAULT_DATASET)
    split_hash = _sha256(DEFAULT_ASSIGNMENTS)
    gold_hash = _sha256(DEFAULT_GOLD_EVAL)
    if (dataset_hash, split_hash, gold_hash) != (FROZEN_DATASET, FROZEN_SPLIT, FROZEN_GOLD):
        print("frozen dataset hashes do not match; pool not written", file=sys.stderr)
        return 2
    client = chromadb.PersistentClient(path=str(DEFAULT_PERSIST_DIR))
    collection = client.get_collection(TRAIN_ONLY_COLLECTION)
    count = collection.count()
    held = collection.get(ids=["train_26148"])
    if count != 21193 or held.get("ids"):
        print(f"index check failed count={count} held={held.get('ids')}", file=sys.stderr)
        return 2
    topics = _load_jsonl(DEFAULT_GOLD_EVAL)
    vectors = _embed_batch(
        [row["topic"] for row in topics],
        "http://127.0.0.1:11434",
        "nomic-embed-text",
        cpu=True,
    )
    retrieved = []
    for topic, vector in zip(topics, vectors):
        hits = query_excerpts(collection, vector, POOL_K, where=None)
        retrieved.append((topic, hits))
    flat_hits = [hit for _topic, hits in retrieved for hit in hits]
    provenance = _provenance(flat_hits, _assignments())
    raw_rows = []
    traces = []
    candidates = []
    for topic, hits in retrieved:
        annotated = annotate_query(topic["id"], topic["topic"], hits, provenance)
        raw_rows.extend(annotated)
        traces.append({"query_id": topic["id"], "topic": topic["topic"], "hits": annotated})
        candidates.extend(judgment_view(annotated))
    diagnostics = pool_diagnostics(traces)
    labels = _existing_labels()
    git = _git()
    timestamp = datetime.now(timezone.utc).isoformat()
    EXPERIMENT.mkdir(parents=True)
    for name in ("config", "metrics", "raw"):
        (EXPERIMENT / name).mkdir(exist_ok=True)
    manifest_body = {
        "pool_origin": "dense_rag_v1",
        "frozen_baseline_experiment": FROZEN,
        "frozen_git_sha": FROZEN_SHA,
        "collection": TRAIN_ONLY_COLLECTION,
        "document_count": count,
        "train_26148_present": False,
        "embedding_model": "nomic-embed-text",
        "embed_device": "cpu",
        "distance_metric": (collection.metadata or {}).get("hnsw:space"),
        "query": "gold topic text only",
        "pool_k": POOL_K,
        "production_retrieval_k": 4,
        "medium_filter": False,
        "reranker": False,
        "n_queries": len(traces),
        "raw_candidate_rows": len(raw_rows),
        "judgment_units_after_group_dedup": len(candidates),
        "unique_document_ids": len({row["document_id"] for row in raw_rows}),
        "unique_group_ids": len({row["group_id"] for row in raw_rows if row.get("group_id")}),
        "dataset_sha256": dataset_hash,
        "assignments_sha256": split_hash,
        "gold_eval_sha256": gold_hash,
        "candidate_file_sha256": None,
        "accepted_judgments": 0,
        "ranked_metrics": "omitted; no accepted qrels",
        "relevance_scope": "none",
        "git": git,
        "timestamp": timestamp,
        "existing_labels": labels,
    }
    _write_jsonl(EXPERIMENT / "raw" / "raw_top20_traces.jsonl", raw_rows)
    _write_jsonl(EXPERIMENT / "qrels_candidates.jsonl", candidates)
    judgments_path.write_text("", encoding="utf-8")
    manifest_body["candidate_file_sha256"] = text_sha256(
        (EXPERIMENT / "qrels_candidates.jsonl").read_text(encoding="utf-8")
    )
    (EXPERIMENT / "retrieval_pool_manifest.json").write_text(
        json.dumps(manifest_body, indent=2) + "\n", encoding="utf-8"
    )
    (EXPERIMENT / "qrels_schema.json").write_text(SCHEMA, encoding="utf-8")
    (EXPERIMENT / "config" / "relevance_judge_prompt.md").write_text(JUDGE_PROMPT, encoding="utf-8")
    (EXPERIMENT / "README.md").write_text(README, encoding="utf-8")
    (EXPERIMENT / "notes.md").write_text(notes_for(manifest_body, diagnostics), encoding="utf-8")
    (EXPERIMENT / "metrics" / "diagnostics.json").write_text(
        json.dumps(diagnostics, indent=2) + "\n", encoding="utf-8"
    )
    (EXPERIMENT / "metrics" / "existing_labels.json").write_text(
        json.dumps(labels, indent=2) + "\n", encoding="utf-8"
    )
    (EXPERIMENT / "metrics" / "ranked_metrics.json").write_text(
        json.dumps(
            {
                "relevance_scope": "none",
                "precision_at_1": None,
                "precision_at_4": None,
                "precision_at_10": None,
                "precision_at_20": None,
                "mrr": None,
                "ndcg_at_4": None,
                "ndcg_at_10": None,
                "ndcg_at_20": None,
                "pool_recall": None,
                "corpus_recall": None,
                "reason": "no accepted relevance judgments; unlabeled candidates are not grade 0",
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    (EXPERIMENT / "config" / "commands.txt").write_text(
        "host_finetune\\.venv\\Scripts\\python.exe -m host_finetune.build_qrel_pool\n",
        encoding="utf-8",
    )
    (EXPERIMENT / "manifest.yaml").write_text(_manifest_yaml(manifest_body, git, timestamp), encoding="utf-8")
    print(json.dumps({
        "raw_candidate_rows": len(raw_rows),
        "judgment_units_after_group_dedup": len(candidates),
        "unique_document_ids": manifest_body["unique_document_ids"],
        "unique_group_ids": manifest_body["unique_group_ids"],
        "distance_metric": manifest_body["distance_metric"],
        "nonempty_on_topic": labels["nonempty_on_topic"],
        "mean_distance": diagnostics["mean_cosine_distance"],
        "duplicate_group_rate": diagnostics["duplicate_group_rate"],
    }))
    return 0


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _manifest_yaml(body: dict, git: dict, timestamp: str) -> str:
    return (
        "experiment_id: EXP-20261004-010-retrieval-qrels-v1\n"
        f"timestamp: {timestamp}\n"
        "execution_commands:\n"
        "  - host_finetune\\.venv\\Scripts\\python.exe -m host_finetune.build_qrel_pool\n"
        "git:\n"
        f"  commit: {git['commit']}\n"
        f"  dirty: {str(git['dirty']).lower()}\n"
        "data:\n"
        "  source_manifest: n/a\n"
        "  source_hashes: {}\n"
        f"  processed_dataset_hash: {body['dataset_sha256']}\n"
        f"  split_manifest_hash: {body['assignments_sha256']}\n"
        f"  gold_eval_hash: {body['gold_eval_sha256']}\n"
        "rag:\n"
        "  persona_profile_hash: n/a\n"
        f"  chroma_collection: {TRAIN_ONLY_COLLECTION}\n"
        "  index_version_or_snapshot: document_count=21193\n"
        "  embedding_model: nomic-embed-text\n"
        "model:\n"
        "  model_name: n/a\n"
        "  model_or_adapter_hash: n/a\n"
        "  config: retrieval pool only; generator not loaded\n"
        "  random_seed: n/a\n"
        "evaluation:\n"
        f"  evaluator_commit: {git['commit']}\n"
        "  judge_model: n/a\n"
        "  judge_config: prompt stored; judge not run; no accepted qrels\n"
        "environment:\n"
        f"  python_version: {platform.python_version()}\n"
        "  package_snapshot: n/a\n"
        "  hardware: CPU retrieval; CUDA not initialized\n"
        "outputs:\n"
        "  raw: raw/raw_top20_traces.jsonl\n"
        "  metrics: metrics/diagnostics.json\n"
        "  warnings_and_failures: ranked IR metrics omitted because no judgments exist\n"
    )


if __name__ == "__main__":
    sys.exit(main())
