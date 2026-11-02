"""Retrieval comparison for source-verified graph relationships.

Matched pilot dense search, LightRAG naive, hybrid, and mix. Review source ids
are not sent to any retriever. Overlap with those ids is not precision or recall.
This script does not modify the pilot graph or the frozen Chroma indexes.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parents[1]
RAW = OUT / "raw"
METRICS = OUT / "metrics"
QUESTIONS = ROOT / "host_finetune" / "graph_relationship_questions.json"
WORK = ROOT / "artifacts" / "lightrag" / "model_f" / "storage_pilot160"
DOCS = ROOT / "artifacts" / "lightrag" / "model_f" / "graph_ready_v2" / "documents.jsonl"
IDS = ROOT / "experiments" / "EXP-20261027-001-graph-ready-audit" / "config" / "sample_160_ids.json"
BASE = "http://localhost:11434"
MODES = ("naive", "hybrid", "mix")

sys.path.insert(0, str(ROOT))

from host_finetune.graph_ready import load_jsonl
from host_finetune.lightrag_backend import query_documents_sync
from host_finetune.pilot_retrieval import _embed, dense_hits, load_questions, select_pilot_documents


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git() -> dict:
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip())
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "dirty": None}
    return {"commit": commit, "dirty": dirty}


def _relation_public(row: dict) -> dict:
    return {
        "relationship_id": row.get("relationship_id"),
        "subject": row.get("subject"),
        "predicate": row.get("predicate"),
        "predicate_kind": row.get("predicate_kind"),
        "object": row.get("object"),
        "source_document_ids": row.get("source_document_ids") or [],
        "graph_chunk_ids": row.get("graph_chunk_ids") or [],
        "support_span": row.get("support_span"),
        "support_kind": row.get("support_kind"),
        "validation_status": row.get("validation_status"),
        "use_in_generation": row.get("use_in_generation"),
        "quoted_speakers": row.get("quoted_speakers") or [],
        "qualifiers": row.get("qualifiers") or [],
        "attribution_status": row.get("attribution_status"),
        "rejection_reason": row.get("rejection_reason"),
        "prompt_fact": row.get("prompt_fact"),
    }


def _sheet_row(question: dict, payload: dict) -> dict:
    hits = []
    for hit in payload.get("hits") or []:
        hits.append({
            "source_id": hit.get("parent_graph_document_id") or hit.get("graph_document_id") or hit.get("id"),
            "medium": hit.get("medium"),
            "attribution_status": hit.get("attribution_status"),
            "quoted_speakers": hit.get("quoted_speakers") or [],
            "text_preview": (hit.get("text") or "")[:500],
            "rank": hit.get("rank"),
        })
    bundle = payload.get("relationships") or {}
    accepted = [_relation_public(row) for row in (bundle.get("accepted") or [])]
    review = set(question.get("review_source_ids") or [])
    hit_ids = [hit["source_id"] for hit in hits]
    designated = [source_id for source_id in hit_ids if source_id in review]
    reasons = Counter(
        str((row or {}).get("rejection_reason") or row.get("validation_status"))
        for row in (bundle.get("rejected") or []) + (bundle.get("quarantined") or [])
    )
    return {
        "question_id": question["id"],
        "kind": question["kind"],
        "query": question["query"],
        "review_source_ids": question.get("review_source_ids") or [],
        "required_relationships_for_review": question.get("required_relationships") or [],
        "review_note": question.get("review_note"),
        "backend": payload.get("backend"),
        "mode": payload.get("mode"),
        "seconds": payload.get("seconds"),
        "hit_source_ids": hit_ids,
        "duplicate_source": len(hit_ids) != len(set(hit_ids)),
        "designated_sources_retrieved": designated,
        "designated_source_note": "Overlap with preselected review ids. Not precision or recall.",
        "hits": hits,
        "accepted_relationships": accepted,
        "accepted_for_generation": [row for row in accepted if row.get("use_in_generation")],
        "relationship_rejection_counts": dict(reasons),
        "relationship_contract": bundle.get("contract"),
    }


def _dense(questions, documents, matrix) -> list[dict]:
    rows = []
    for question in questions:
        payload = dense_hits(question["query"], documents, matrix, BASE, "nomic-embed-text", 4)
        payload["query"] = question["query"]
        payload["relationships"] = {
            "contract": "source-verified-relationship-v1",
            "accepted": [],
            "quarantined": [],
            "rejected": [],
            "prompt_block": "",
        }
        rows.append(payload)
        print(f"dense {question['id']} hits={[hit.get('id') for hit in payload['hits']]}", flush=True)
    return rows


def _lightrag(questions, mode: str) -> list[dict]:
    import time

    rows = []
    for question in questions:
        started = time.perf_counter()
        payload = query_documents_sync(
            question["query"],
            WORK,
            mode,
            base_url=BASE,
            evidence_budget=4,
            max_per_source=1,
            num_ctx=16384,
        )
        payload["seconds"] = round(time.perf_counter() - started, 3)
        payload["query"] = question["query"]
        accepted = (payload.get("relationships") or {}).get("accepted") or []
        print(
            f"{mode} {question['id']} hits={[hit.get('id') for hit in payload.get('hits') or []]} "
            f"accepted={len(accepted)} s={payload['seconds']}",
            flush=True,
        )
        rows.append(payload)
    return rows


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n", encoding="utf-8")


def main() -> None:
    RAW.mkdir(parents=True, exist_ok=True)
    METRICS.mkdir(parents=True, exist_ok=True)
    questions = load_questions(QUESTIONS)
    documents = select_pilot_documents(load_jsonl(DOCS), IDS)
    matrix_path = RAW / "pilot_dense_matrix.npy"
    if matrix_path.is_file():
        matrix = np.load(matrix_path)
        print(f"loaded dense matrix {matrix.shape}", flush=True)
    else:
        texts = [row.get("source_text") or row.get("text") or "" for row in documents]
        matrix = _embed(texts, BASE, "nomic-embed-text")
        np.save(matrix_path, matrix)
        print(f"embedded dense matrix {matrix.shape}", flush=True)
    dense_path = RAW / "frozen_pilot_dense.jsonl"
    if dense_path.is_file():
        print("keep existing dense freeze", flush=True)
    else:
        _write_jsonl(dense_path, _dense(questions, documents, matrix))
    for mode in MODES:
        path = RAW / f"frozen_{mode}.jsonl"
        if path.is_file():
            print(f"keep existing {mode}", flush=True)
            continue
        _write_jsonl(path, _lightrag(questions, mode))
    sheet = []
    for name, filename in (
        ("pilot_dense", "frozen_pilot_dense.jsonl"),
        ("naive", "frozen_naive.jsonl"),
        ("hybrid", "frozen_hybrid.jsonl"),
        ("mix", "frozen_mix.jsonl"),
    ):
        payloads = [json.loads(line) for line in (RAW / filename).read_text(encoding="utf-8").splitlines() if line.strip()]
        by_query = {row.get("query"): row for row in payloads}
        for question in questions:
            sheet.append(_sheet_row(question, by_query[question["query"]]))
    (RAW / "retrieval_sheet.json").write_text(json.dumps(sheet, indent=2, ensure_ascii=False), encoding="utf-8")
    summary = {
        "questions": len(questions),
        "note": "Designated-source overlap and relationship counts are review aids, not precision or recall.",
        "git": _git(),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "questions_sha256": _sha256(QUESTIONS),
        "catalog_sha256": _sha256(WORK / "catalog.jsonl"),
        "graphml_sha256": _sha256(WORK / "graph_chunk_entity_relation.graphml"),
        "by_question": {},
    }
    for question in questions:
        rows = [row for row in sheet if row["question_id"] == question["id"]]
        summary["by_question"][question["id"]] = {
            arm["mode"] or arm["backend"]: {
                "seconds": arm["seconds"],
                "hit_source_ids": arm["hit_source_ids"],
                "designated_sources_retrieved": arm["designated_sources_retrieved"],
                "accepted_for_generation": len(arm["accepted_for_generation"]),
                "accepted_total": len(arm["accepted_relationships"]),
                "rejection_counts": arm["relationship_rejection_counts"],
            }
            for arm in rows
        }
    (METRICS / "retrieval_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("wrote retrieval sheet", flush=True)


if __name__ == "__main__":
    main()
