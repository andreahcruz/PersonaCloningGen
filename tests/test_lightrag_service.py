from __future__ import annotations

import json

from lightrag_service.retriever import EMBED_DIMENSION, snapshot_status, verify_snapshot


def _snapshot(tmp_path):
    (tmp_path / "catalog.jsonl").write_text(
        json.dumps({"graph_document_id": "doc-1", "text": "source", "split": "train"}) + "\n",
        encoding="utf-8",
    )
    (tmp_path / "vdb_chunks.json").write_text(
        json.dumps({"embedding_dim": EMBED_DIMENSION, "data": [], "matrix": ""}), encoding="utf-8"
    )
    for name in {
        "graph_chunk_entity_relation.graphml",
        "kv_store_full_docs.json",
        "kv_store_text_chunks.json",
        "vdb_entities.json",
        "vdb_relationships.json",
    }:
        (tmp_path / name).write_text("{}", encoding="utf-8")
    return tmp_path


def test_snapshot_status_requires_the_model_f_shape(tmp_path):
    status = snapshot_status(tmp_path)
    assert status["ready"] is False
    assert "missing" in status["detail"]

    snapshot = _snapshot(tmp_path)
    (snapshot / "snapshot_hashes.json").write_text(json.dumps({"files": {}}), encoding="utf-8")
    status = snapshot_status(snapshot)
    assert status["ready"] is True
    assert status["documents"] == 1
    assert status["embedding_dimension"] == 768


def test_verify_snapshot_checks_the_recorded_hashes(tmp_path):
    snapshot = _snapshot(tmp_path)
    target = snapshot / "catalog.jsonl"
    import hashlib

    digest = hashlib.sha256(target.read_bytes()).hexdigest()
    (snapshot / "snapshot_hashes.json").write_text(
        json.dumps({"files": {"catalog.jsonl": digest}}), encoding="utf-8"
    )
    assert verify_snapshot(snapshot)["verified_files"] == 1

    target.write_text("changed\n", encoding="utf-8")
    try:
        verify_snapshot(snapshot)
    except RuntimeError as error:
        assert "verification failed" in str(error)
    else:
        raise AssertionError("expected a failed hash verification")
