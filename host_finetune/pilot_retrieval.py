"""Retrieval-only comparison on the 160-document pilot.

The dense arm embeds the same graph-ready source texts with nomic-embed-text
and no query or document prefix, matching the Model E Chroma embedder. It does
not open or write the frozen Chroma directories. The optional chroma arm reads
the active frozen index and does not upsert. LightRAG modes read an existing
pilot index. Review source ids in the question file are not sent to any retriever.

Precision and recall are not computed. The output is a review sheet.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import requests

from host_finetune.graph_ready import load_jsonl
from host_finetune.lightrag_backend import LIGHTRAG_MODES, query_documents_sync
from host_finetune.lightrag_eval import retrieval_observation
from host_finetune.train_only_index import ollama_embed_body

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_QUESTIONS = ROOT / "host_finetune" / "pilot_questions.json"
DEFAULT_DOCS = ROOT / "artifacts" / "lightrag" / "model_f" / "graph_ready_v2" / "documents.jsonl"
DEFAULT_IDS = (
    ROOT
    / "experiments"
    / "EXP-20261027-001-graph-ready-audit"
    / "config"
    / "sample_160_ids.json"
)


def load_questions(path: Path) -> list[dict]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    for row in rows:
        if not row.get("query") or not row.get("id"):
            raise ValueError("each question needs an id and a query")
    return rows


def select_pilot_documents(documents: list[dict], id_file: Path) -> list[dict]:
    wanted = set(json.loads(id_file.read_text(encoding="utf-8")))
    chosen = [
        row
        for row in documents
        if row.get("split") == "train"
        and (
            row.get("graph_document_id") in wanted
            or row.get("parent_graph_document_id") in wanted
        )
    ]
    found = {
        row.get("parent_graph_document_id") or row["graph_document_id"] for row in chosen
    }
    missing = wanted - found
    if missing:
        raise SystemExit(f"pilot id file is missing {len(missing)} documents")
    return chosen


def _embed(texts: list[str], base_url: str, model: str) -> np.ndarray:
    vectors = []
    for start in range(0, len(texts), 32):
        batch = texts[start : start + 32]
        response = requests.post(
            f"{base_url.rstrip('/')}/api/embed",
            json=ollama_embed_body(batch, model, cpu=False),
            timeout=300,
        )
        response.raise_for_status()
        rows = response.json().get("embeddings") or []
        if len(rows) != len(batch):
            raise RuntimeError(f"embed returned {len(rows)} vectors for {len(batch)} texts")
        vectors.extend(rows)
    array = np.asarray(vectors, dtype=np.float32)
    norms = np.linalg.norm(array, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return array / norms


def row_source_index(documents: list[dict]) -> dict[str, dict]:
    """Map a Chroma train-row id onto the pilot graph document that contains it."""
    index = {}
    for document in documents:
        parent = document.get("parent_graph_document_id") or document.get("graph_document_id")
        info = {
            "graph_document_id": document.get("graph_document_id"),
            "parent_graph_document_id": parent,
            "attribution_status": document.get("attribution_status"),
            "quoted_speakers": list(document.get("quoted_speakers") or []),
            "split": document.get("split"),
            "medium": document.get("medium"),
        }
        for row_id in document.get("experiment_row_ids") or []:
            index[str(row_id)] = info
    return index


def annotate_chroma_hits(
    question: str,
    raw_hits: list[dict],
    sources: dict[str, dict],
    seconds: float,
) -> dict:
    """Label frozen-Chroma hits with a pilot source only when the row is in the pilot.

    A hit whose train-row id is outside the pilot keeps that train id. It is not
    rewritten into a graph document.
    """
    hits = []
    for rank, hit in enumerate(raw_hits, start=1):
        source = sources.get(str(hit.get("id") or ""))
        item = dict(hit)
        if source:
            item.update({key: value for key, value in source.items() if value is not None})
        item["rank"] = item.get("rank") or rank
        item["score_type"] = item.get("score_type") or "cosine_distance"
        hits.append(item)
    return {
        "backend": "chroma_v2",
        "mode": "dense",
        "query": question,
        "hits": hits,
        "excluded": [],
        "seconds": round(seconds, 3),
        "score_note": (
            "Cosine distance on the frozen full train-only Chroma index. "
            "Not limited to the pilot corpus."
        ),
    }


def dense_hits(
    question: str,
    documents: list[dict],
    matrix: np.ndarray,
    base_url: str,
    model: str,
    k: int,
) -> dict:
    started = time.perf_counter()
    query = _embed([question], base_url, model)[0]
    scores = matrix @ query
    order = np.argsort(-scores)[:k]
    hits = []
    for rank, index in enumerate(order, start=1):
        document = documents[int(index)]
        source_id = document.get("parent_graph_document_id") or document["graph_document_id"]
        hits.append(
            {
                "id": source_id,
                "text": (document.get("source_text") or document.get("text") or "")[:400],
                "medium": document.get("medium"),
                "split": document.get("split"),
                "graph_document_id": source_id,
                "attribution_status": document.get("attribution_status"),
                "author_position": document.get("author_position"),
                "quoted_speakers": document.get("quoted_speakers") or [],
                "distance": float(1.0 - scores[int(index)]),
                "rank": rank,
                "score_type": "cosine_distance",
            }
        )
    return {
        "backend": "pilot_dense",
        "mode": "dense",
        "query": question,
        "hits": hits,
        "excluded": [],
        "seconds": round(time.perf_counter() - started, 3),
        "score_note": "Cosine distance on the pilot source texts. Not a frozen Chroma collection.",
    }


def sheet_row(question: dict, payload: dict) -> dict:
    observed = retrieval_observation(payload)
    hits = []
    for hit in payload.get("hits") or []:
        hits.append(
            {
                "source_id": (
                    hit.get("parent_graph_document_id")
                    or hit.get("graph_document_id")
                    or hit.get("id")
                ),
                "graph_document_id": hit.get("graph_document_id"),
                "attribution_status": hit.get("attribution_status"),
                "quoted_speakers": hit.get("quoted_speakers") or [],
                "text_preview": (hit.get("text") or "")[:400],
                "rank": hit.get("rank"),
            }
        )
    review_ids = question.get("review_source_ids") or []
    hit_source_ids = [hit["source_id"] for hit in hits]
    designated = [source_id for source_id in hit_source_ids if source_id in set(review_ids)]
    return {
        "question_id": question["id"],
        "kind": question["kind"],
        "query": question["query"],
        "abstain_expected": question.get("abstain", False),
        "review_source_ids": review_ids,
        "designated_sources_retrieved": designated,
        "designated_source_covered": bool(designated) if review_ids else None,
        "designated_source_note": (
            "Overlap with the preselected review ids. Not precision or recall."
        ),
        "backend": observed["backend"],
        "mode": observed["mode"],
        "hit_source_ids": hit_source_ids,
        "unique_sources": observed["unique_sources"],
        "duplicate_source": observed["duplicate_source"],
        "empty": observed["empty"],
        "train_attributed": observed["train_attributed"],
        "excluded_reasons": observed["excluded_reasons"],
        "seconds": payload.get("seconds"),
        "token_usage": None,
        "hits": hits,
        "relevance": None,
        "precision_recall": "not measured; no human relevance judgments",
    }


def _open_chroma(version: str | None):
    """Open the active frozen collection. Callers must not upsert into it."""
    import chromadb
    from chromadb.config import Settings

    from host_finetune.rag_runtime import resolve_rag
    from host_finetune.train_only_index import assert_collection_allowed

    selected = resolve_rag(version)
    assert_collection_allowed(selected["collection"])
    if not Path(selected["path"]).exists():
        raise SystemExit(f"train-only index not found at {selected['path']}")
    client = chromadb.PersistentClient(
        path=str(selected["path"]),
        settings=Settings(anonymized_telemetry=False),
    )
    collection = client.get_collection(selected["collection"])
    print(
        f"chroma version={selected['version']} path={selected['path']} "
        f"collection={selected['collection']} documents={collection.count()}",
        flush=True,
    )
    return collection


def _chroma_query(question, collection, sources, base_url, model, k):
    from host_finetune.rebuild_chroma import _embed_batch
    from host_finetune.train_only_index import query_excerpts

    started = time.perf_counter()
    vector = _embed_batch([question], base_url, model, cpu=True)[0]
    raw = query_excerpts(collection, vector, k, where=None)
    return annotate_chroma_hits(question, raw, sources, time.perf_counter() - started)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--questions", type=Path, default=DEFAULT_QUESTIONS)
    parser.add_argument("--documents", type=Path, default=DEFAULT_DOCS)
    parser.add_argument("--id-file", type=Path, default=DEFAULT_IDS)
    parser.add_argument("--workdir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--modes", default="dense," + ",".join(LIGHTRAG_MODES))
    parser.add_argument("--evidence-budget", type=int, default=4)
    parser.add_argument("--base-url", default="http://localhost:11434")
    parser.add_argument("--embed-model", default="nomic-embed-text")
    parser.add_argument(
        "--num-ctx",
        type=int,
        default=None,
        help="Ollama context for LightRAG keyword calls. Omit to keep the backend default.",
    )
    parser.add_argument(
        "--rag-version",
        default=None,
        help="Frozen Chroma version for the chroma mode. Default is rag_active.json.",
    )
    args = parser.parse_args()
    questions = load_questions(args.questions)
    documents = select_pilot_documents(load_jsonl(args.documents), args.id_file)
    modes = [part.strip() for part in args.modes.split(",") if part.strip()]
    chroma_collection = None
    if "chroma" in modes:
        chroma_collection = _open_chroma(args.rag_version)
    matrix = None
    if "dense" in modes:
        texts = [row.get("source_text") or row.get("text") or "" for row in documents]
        matrix = _embed(texts, args.base_url, args.embed_model)
    rows = []
    for question in questions:
        for mode in modes:
            if mode == "dense":
                payload = dense_hits(
                    question["query"],
                    documents,
                    matrix,
                    args.base_url,
                    args.embed_model,
                    args.evidence_budget,
                )
            elif mode == "chroma":
                payload = _chroma_query(
                    question["query"],
                    chroma_collection,
                    row_source_index(documents),
                    args.base_url,
                    args.embed_model,
                    args.evidence_budget,
                )
            else:
                started = time.perf_counter()
                query_kwargs = {}
                if args.num_ctx is not None:
                    query_kwargs["num_ctx"] = args.num_ctx
                payload = query_documents_sync(
                    question["query"],
                    args.workdir,
                    mode,
                    base_url=args.base_url,
                    evidence_budget=args.evidence_budget,
                    max_per_source=1,
                    **query_kwargs,
                )
                payload["seconds"] = round(time.perf_counter() - started, 3)
            rows.append(sheet_row(question, payload))
            print(
                f"{question['id']} {mode} hits={rows[-1]['hit_source_ids']} seconds={rows[-1]['seconds']}",
                flush=True,
            )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
