"""Copy a finished LightRAG workdir. This does not query the graph.

The live expansion directory is refused unless the caller sets
``allow_expansion_source`` after the indexer has exited and the lock is gone.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

LIVE_STORE_NAME = "storage_expand_v1"
IN_FLIGHT = {"processing", "parsing", "analyzing", "pending"}


class SnapshotError(RuntimeError):
    """The workdir is not safe to copy."""


def assert_snapshot_path(path: Path) -> None:
    """Experiments read a copy. The live expansion directory is never a snapshot."""
    if LIVE_STORE_NAME in Path(path).resolve().parts:
        raise SnapshotError("experiments must use a snapshot copy, not the live expansion store")


def assert_snapshot_source(source: Path, *, allow_expansion_source: bool) -> None:
    if LIVE_STORE_NAME in source.resolve().parts and not allow_expansion_source:
        raise SnapshotError(
            "refusing to read the live expansion store; pass allow_expansion_source only after the indexer has exited"
        )
    if LIVE_STORE_NAME in source.resolve().parts and allow_expansion_source:
        return


def _lock_held(source: Path) -> bool:
    return (source / "indexing.lock").is_file()


def _in_flight_status(source: Path) -> list[str]:
    path = source / "kv_store_doc_status.json"
    if not path.is_file():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.values() if isinstance(payload, dict) else payload
    found = []
    for row in rows or []:
        if isinstance(row, dict) and str(row.get("status") or "").lower() in IN_FLIGHT:
            found.append(str(row.get("id") or row.get("file_path") or row.get("status")))
    return found


def _ledger_processing(source: Path) -> list[str]:
    path = source / "indexing_ledger.sqlite"
    if not path.is_file():
        return []
    import sqlite3

    connection = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    try:
        rows = connection.execute(
            "SELECT document_id FROM documents WHERE status = 'processing'"
        ).fetchall()
    except sqlite3.Error as exc:
        raise SnapshotError(f"ledger could not be read: {exc}") from exc
    finally:
        connection.close()
    return [str(row[0]) for row in rows]


def validate_source(source: Path, *, allow_expansion_source: bool = False) -> dict:
    assert_snapshot_source(source, allow_expansion_source=allow_expansion_source)
    if not source.is_dir():
        raise SnapshotError(f"snapshot source is not a directory: {source}")
    if _lock_held(source):
        raise SnapshotError(f"indexing lock is present: {source / 'indexing.lock'}")
    in_flight = _in_flight_status(source)
    processing = _ledger_processing(source)
    if in_flight or processing:
        raise SnapshotError(
            "documents are still processing: " + ", ".join((in_flight + processing)[:8])
        )
    return {
        "source": str(source),
        "lock_present": False,
        "in_flight": 0,
        "ledger_processing": 0,
    }


def _hash_tree(root: Path) -> dict[str, str]:
    hashes = {}
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        hashes[path.relative_to(root).as_posix()] = digest.hexdigest()
    return hashes


def copy_snapshot(source: Path, destination: Path, *, allow_expansion_source: bool = False) -> dict:
    """Copy a validated workdir and write SHA-256 hashes. The source is not modified."""
    report = validate_source(source, allow_expansion_source=allow_expansion_source)
    if destination.exists():
        raise SnapshotError(f"snapshot destination already exists: {destination}")
    if LIVE_STORE_NAME in destination.resolve().parts:
        raise SnapshotError("refusing to write the snapshot into the live expansion store")
    shutil.copytree(source, destination, ignore=shutil.ignore_patterns("STOP", "indexing.lock"))
    hashes = _hash_tree(destination)
    (destination / "snapshot_hashes.json").write_text(
        json.dumps({"algorithm": "sha256", "files": hashes}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    report["destination"] = str(destination)
    report["files"] = len(hashes)
    report["hash_of_hashes"] = hashlib.sha256(
        json.dumps(hashes, sort_keys=True).encode("utf-8")
    ).hexdigest()
    return report


REQUIRED_SNAPSHOT_FILES = (
    "catalog.jsonl",
    "graph_chunk_entity_relation.graphml",
    "vdb_chunks.json",
    "vdb_entities.json",
    "vdb_relationships.json",
)
EXPECTED_EMBEDDING_DIM = 768


def verify_snapshot(path: Path) -> dict:
    """Check a copied store. Does not open the live expansion directory."""
    assert_snapshot_path(path)
    if not path.is_dir():
        raise SnapshotError(f"snapshot is not a directory: {path}")
    if _lock_held(path):
        raise SnapshotError(f"indexing lock is present: {path / 'indexing.lock'}")
    missing = [name for name in REQUIRED_SNAPSHOT_FILES if not (path / name).is_file()]
    if missing:
        raise SnapshotError("snapshot is missing " + ", ".join(missing))
    vectors = json.loads((path / "vdb_chunks.json").read_text(encoding="utf-8"))
    dimension = vectors.get("embedding_dim")
    rows = vectors.get("data") or []
    if rows and isinstance(rows[0], dict) and isinstance(rows[0].get("vector"), list):
        dimension = len(rows[0]["vector"])
    if dimension != EXPECTED_EMBEDDING_DIM:
        raise SnapshotError(f"chunk embedding dimension is {dimension}, expected {EXPECTED_EMBEDDING_DIM}")
    recorded_path = path / "snapshot_hashes.json"
    if not recorded_path.is_file():
        raise SnapshotError("snapshot_hashes.json is missing")
    recorded = json.loads(recorded_path.read_text(encoding="utf-8"))["files"]
    actual = {
        name: digest
        for name, digest in _hash_tree(path).items()
        if name != "snapshot_hashes.json"
    }
    if actual != recorded:
        raise SnapshotError("snapshot hashes do not match the copied files")
    catalog = (path / "catalog.jsonl").read_bytes()
    return {
        "embedding_dim": dimension,
        "hash_match": True,
        "catalog_sha256": hashlib.sha256(catalog).hexdigest(),
        "files": len(actual),
    }
