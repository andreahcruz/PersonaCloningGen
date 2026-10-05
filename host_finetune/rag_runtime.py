"""Choose the active train-only Chroma index.

One file, ``host_finetune/rag_active.json``, selects v1 or v2. v1 is the
original ``lemkin_train_only`` directory. v2 is a directory copy of the
validated shadow index and keeps that index's collection name. ``lemkin_content``
is not a selectable version.
"""

from __future__ import annotations

import hashlib
import json
import logging
import shutil
from pathlib import Path

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[1]
ACTIVE_FILE = ROOT / "host_finetune" / "rag_active.json"
V1_DIR = ROOT / "host_finetune" / "output" / "chroma_lemkin_train_only"
V2_DIR = ROOT / "host_finetune" / "output" / "chroma_lemkin_train_only_v2"
V1_COLLECTION = "lemkin_train_only"
# Renaming the validated collection would rewrite its database. The name stays.
V2_COLLECTION = "lemkin_train_only_v2_shadow"
LEGACY_COLLECTION = "lemkin_content"
HELD_OUT_ID = "train_26148"


class RagConfigError(RuntimeError):
    """The active RAG selection is missing or points at a forbidden index."""


def catalog() -> dict[str, dict]:
    return {
        "v1": {"version": "v1", "path": V1_DIR, "collection": V1_COLLECTION},
        "v2": {"version": "v2", "path": V2_DIR, "collection": V2_COLLECTION},
    }


def read_active_version(path: Path | None = None) -> str:
    file = path or ACTIVE_FILE
    payload = json.loads(file.read_text(encoding="utf-8"))
    version = str(payload.get("version") or "").strip().lower()
    if version not in catalog():
        raise RagConfigError(f"invalid RAG version: {version!r}")
    return version


def resolve_rag(version: str | None = None, *, config_path: Path | None = None) -> dict:
    """Return the selected index. The default is ``rag_active.json``."""
    chosen = (version or read_active_version(config_path)).strip().lower()
    entry = catalog().get(chosen)
    if entry is None:
        raise RagConfigError(f"invalid RAG version: {chosen!r}")
    if entry["collection"] == LEGACY_COLLECTION or chosen == LEGACY_COLLECTION:
        raise RagConfigError("lemkin_content cannot be selected as a RAG version")
    if entry["path"].resolve() == V1_DIR.resolve() and chosen != "v1":
        raise RagConfigError("v2 resolves to the v1 directory")
    if V1_DIR.resolve() == V2_DIR.resolve():
        raise RagConfigError("v1 and v2 paths are not distinct")
    selected = {
        "version": entry["version"],
        "path": entry["path"],
        "collection": entry["collection"],
    }
    logger.info(
        "RAG version=%s path=%s collection=%s",
        selected["version"],
        selected["path"],
        selected["collection"],
    )
    return selected


def _file_hashes(directory: Path) -> dict[str, str]:
    hashes = {}
    for path in sorted(item for item in directory.rglob("*") if item.is_file()):
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        hashes[path.relative_to(directory).as_posix()] = digest.hexdigest()
    return hashes


def promote_shadow_directory(source: Path, destination: Path) -> dict:
    """Copy a complete Chroma directory. This does not open or rebuild it."""
    source = source.resolve()
    destination = destination.resolve()
    if source == V1_DIR.resolve() or V1_DIR.resolve() in source.parents:
        raise RagConfigError("refusing to use the v1 directory as a promotion source")
    if destination == V1_DIR.resolve() or V1_DIR.resolve() in destination.parents or destination in V1_DIR.resolve().parents:
        raise RagConfigError("refusing to write inside the v1 directory")
    if LEGACY_COLLECTION in destination.as_posix():
        raise RagConfigError("refusing to promote into a lemkin_content path")
    if not (source / "chroma.sqlite3").is_file():
        raise RagConfigError(f"source is not a Chroma directory: {source}")
    if destination.exists():
        raise RagConfigError(f"promotion destination already exists: {destination}")
    v1_db = V1_DIR / "chroma.sqlite3"
    before = hashlib.sha256(v1_db.read_bytes()).hexdigest()
    source_hashes = _file_hashes(source)
    shutil.copytree(source, destination)
    copied = _file_hashes(destination)
    after = hashlib.sha256(v1_db.read_bytes()).hexdigest()
    if copied != source_hashes:
        raise RagConfigError("promoted directory bytes do not match the validated shadow")
    if after != before:
        raise RagConfigError("v1 chroma.sqlite3 changed during promotion")
    return {
        "files": len(copied),
        "byte_identical": True,
        "v1_sha256_before": before,
        "v1_sha256_after": after,
    }
