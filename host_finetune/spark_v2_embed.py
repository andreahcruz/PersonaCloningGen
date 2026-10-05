"""Shadow embedding and Chroma stages. Production ``lemkin_train_only`` is never opened for write.

Vectors are read from a byte copy of the production directory. The live
database path is only hashed, so a Chroma client cannot touch it.
"""

from __future__ import annotations

import json
import math
import shutil
import struct
import urllib.request
from collections import Counter
from pathlib import Path

from host_finetune.canonical_embeddings import (
    DEFAULT_BASE,
    DOCUMENT_BATCH,
    MODEL,
    RETRIEVAL_K,
    embed_documents,
    embed_query,
    request_body,
)
from host_finetune.spark_v2 import (
    EXPECTED_RAG_ROWS,
    HELD_OUT_RAG_ID,
    ORDERED_EXECUTOR,
    ParityError,
    ROOT,
    STAGING_ROOT,
    _read_jsonl,
    assert_staging_dir,
    file_sha256,
    gate,
    write_manifest,
)
from host_finetune.train_only_index import DEFAULT_GOLD_EVAL, TRAIN_ONLY_COLLECTION, query_excerpts

PRODUCTION_DIR = ROOT / "host_finetune" / "output" / "chroma_lemkin_train_only"
PRODUCTION_DB = PRODUCTION_DIR / "chroma.sqlite3"
SHADOW_COLLECTION = "lemkin_train_only_v2_shadow"
FORBIDDEN_COLLECTIONS = {TRAIN_ONLY_COLLECTION, "lemkin_content"}
COPY_NAME = "_production_readonly_copy"


def production_db_sha256() -> str:
    return file_sha256(PRODUCTION_DB)


def assert_shadow_target(persist: Path, collection: str) -> Path:
    resolved = assert_staging_dir(persist)
    if collection in FORBIDDEN_COLLECTIONS:
        raise ParityError(f"refusing production collection name: {collection}")
    production = PRODUCTION_DIR.resolve()
    if resolved == production or production in resolved.parents or resolved in production.parents:
        raise ParityError(f"refusing to write inside production Chroma: {resolved}")
    return resolved


def ollama_identity(base_url: str | None = None, model: str = MODEL) -> dict:
    """Read the installed model. This does not pull or update it."""
    base = (base_url or DEFAULT_BASE).rstrip("/")
    try:
        with urllib.request.urlopen(base + "/api/version", timeout=10) as response:
            version = json.loads(response.read().decode("utf-8"))
    except Exception as exc:  # noqa: BLE001
        raise ParityError(f"Ollama is not reachable at {base}: {exc}") from exc
    try:
        request = urllib.request.Request(
            base + "/api/show",
            data=json.dumps({"model": model}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(request, timeout=30) as response:
            shown = json.loads(response.read().decode("utf-8"))
    except Exception as exc:  # noqa: BLE001
        raise ParityError(f"{model} is not installed locally; refusing to pull it: {exc}") from exc
    details = shown.get("details") or {}
    digest = shown.get("digest") or details.get("sha256")
    if not digest:
        with urllib.request.urlopen(base + "/api/tags", timeout=10) as response:
            tags = json.loads(response.read().decode("utf-8"))
        for item in tags.get("models") or []:
            if str(item.get("name", "")).split(":")[0] == model:
                digest = item.get("digest")
                break
    return {
        "model": model,
        "ollama_version": version.get("version"),
        "digest": digest,
        "parameter_size": details.get("parameter_size"),
        "family": details.get("family"),
        "modified_at": shown.get("modified_at"),
        "parameter_count_note": "num_ctx is a model parameter; this call does not set it",
        "pulled": False,
    }


def verify_rag_rows(rows: list[dict]) -> dict:
    """Final corpus gate. A failure here must stop embedding."""
    ids = [row["row_id"] for row in rows]
    mismatches = {
        "rows": int(len(rows) != EXPECTED_RAG_ROWS or len(set(ids)) != len(ids)),
        "held_out": int(HELD_OUT_RAG_ID in ids),
        "split": sum(row.get("split") != "train" for row in rows),
        "empty": sum(not str(row.get("document") or "").strip() for row in rows),
        "instruction_embedded": sum("instruction" in row and row.get("document") == row.get("instruction") for row in rows),
    }
    return mismatches


def select_sample(rows: list[dict]) -> list[dict]:
    """Deterministic mix of positions, media, and roles. Excludes train_26148."""
    by_id = {row["row_id"]: row for row in rows}
    chosen: list[dict] = []
    seen: set[str] = set()

    def add(row: dict | None) -> None:
        if row is None or row["row_id"] in seen or row["row_id"] == HELD_OUT_RAG_ID:
            return
        seen.add(row["row_id"])
        chosen.append(row)

    for row in rows[:8]:
        add(row)
    for neighbor in ("train_26147", "train_26149", "train_26146", "train_26150"):
        add(by_id.get(neighbor))
    for medium in ("blog", "x", "linkedin", "youtube_jason"):
        add(next((row for row in rows if row.get("medium") == medium), None))
    for role in ("opening", "continuation", "unchanged"):
        add(next((row for row in rows if row.get("sft_role") == role), None))
    # Spread a few later rows so the sample is not only the head of the file.
    for index in (100, 1000, 5000, 10000, 15000, 20000):
        if index < len(rows):
            add(rows[index])
    return chosen


def _f32(values: list[float]) -> list[float]:
    return [struct.unpack("<f", struct.pack("<f", float(value)))[0] for value in values]


def vector_delta(left: list[float], right: list[float]) -> dict:
    if len(left) != len(right) or not left:
        return {
            "dimension_mismatch": 1,
            "exact": False,
            "max_abs": None,
            "mean_abs": None,
            "cosine": None,
            "dimension": len(left),
        }
    max_abs = 0.0
    sum_abs = 0.0
    dot = 0.0
    left_norm = 0.0
    right_norm = 0.0
    exact = True
    for a_value, b_value in zip(left, right):
        difference = abs(a_value - b_value)
        if difference != 0.0:
            exact = False
        if difference > max_abs:
            max_abs = difference
        sum_abs += difference
        dot += a_value * b_value
        left_norm += a_value * a_value
        right_norm += b_value * b_value
    cosine = dot / math.sqrt(left_norm * right_norm) if left_norm and right_norm else 0.0
    return {
        "dimension_mismatch": 0,
        "exact": exact,
        "max_abs": max_abs,
        "mean_abs": sum_abs / len(left),
        "cosine": cosine,
        "dimension": len(left),
    }


def _percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round(fraction * (len(ordered) - 1))))
    return ordered[index]


def contract_from_sample(rows: list[dict]) -> dict:
    """Choose the narrow rule the sample actually supports."""
    if any(row["raw"]["dimension_mismatch"] for row in rows):
        raise ParityError("sample embedding dimension does not match production")
    exact = sum(1 for row in rows if row["raw"]["exact"])
    stored = sum(1 for row in rows if row["float32"]["exact"])
    max_abs = max(row["raw"]["max_abs"] for row in rows)
    min_cosine = min(row["raw"]["cosine"] for row in rows)
    if exact == len(rows):
        return {"mode": "exact", "max_ulps": 0, "evidence": "sample vectors matched element for element"}
    if stored == len(rows):
        return {
            "mode": "float32_storage",
            "max_ulps": 0,
            "evidence": (
                "fresh JSON floats differ from the stored vectors, and casting the fresh "
                "vector to float32 reproduces the stored vector on every sample row"
            ),
        }
    max_ulps = max(row["float32_ulps"] for row in rows)
    if max_ulps is not None and max_ulps <= 1:
        return {
            "mode": "float32_one_ulp",
            "max_ulps": 1,
            "evidence": (
                "casting the fresh Ollama JSON vector to float32 matches the stored Chroma "
                "vector within one float32 unit in the last place on every sample component; "
                f"bit-identical sample rows={stored} of {len(rows)}; raw max_abs={max_abs}; "
                f"raw min_cosine={min_cosine}"
            ),
        }
    raise ParityError(
        f"sample vectors are not explained by exact, float32, or one-ULP storage: "
        f"exact={exact} float32_exact={stored} max_ulps={max_ulps} max_abs={max_abs} min_cosine={min_cosine}"
    )


def row_within_contract(raw: dict, stored: dict, ulps: int | None, contract: dict) -> bool:
    if raw["dimension_mismatch"] or stored["dimension_mismatch"] or ulps is None:
        return False
    if contract["mode"] == "exact":
        return bool(raw["exact"])
    if contract["mode"] == "float32_storage":
        return bool(stored["exact"])
    if contract["mode"] == "float32_one_ulp":
        return ulps <= int(contract["max_ulps"])
    return False


def _copy_production(staging: Path) -> Path:
    """Byte-copy the production directory. The live path is not opened by Chroma."""
    destination = assert_shadow_target(staging / COPY_NAME, SHADOW_COLLECTION)
    source_hash = production_db_sha256()
    marker = destination / "source_sha256.txt"
    if destination.exists() and marker.is_file() and marker.read_text(encoding="utf-8").strip() == source_hash:
        return destination
    if destination.exists():
        shutil.rmtree(destination)
    shutil.copytree(PRODUCTION_DIR, destination)
    marker.write_text(source_hash + "\n", encoding="utf-8")
    return destination


def _production_client(copy_dir: Path):
    import chromadb
    from chromadb.config import Settings

    return chromadb.PersistentClient(path=str(copy_dir), settings=Settings(anonymized_telemetry=False))


def read_production_rows(copy_dir: Path, ids: list[str], collection=None) -> dict[str, dict]:
    if collection is None:
        collection = _production_client(copy_dir).get_collection(TRAIN_ONLY_COLLECTION)
    found: dict[str, dict] = {}
    for start in range(0, len(ids), 256):
        chunk = ids[start : start + 256]
        got = collection.get(ids=chunk, include=["embeddings", "documents", "metadatas"])
        embeddings = got.get("embeddings")
        documents = got.get("documents")
        metadatas = got.get("metadatas")
        embeddings = [] if embeddings is None else embeddings
        documents = [] if documents is None else documents
        metadatas = [] if metadatas is None else metadatas
        for row_id, embedding, document, metadata in zip(got.get("ids") or [], embeddings, documents, metadatas):
            found[row_id] = {
                "embedding": [float(value) for value in list(embedding)],
                "document": document,
                "metadata": metadata or {},
            }
    return found


def _summarize(deltas: list[dict]) -> dict:
    maxes = [row["max_abs"] for row in deltas if row["max_abs"] is not None]
    means = [row["mean_abs"] for row in deltas if row["mean_abs"] is not None]
    cosines = [row["cosine"] for row in deltas if row["cosine"] is not None]
    return {
        "exact": sum(1 for row in deltas if row["exact"]),
        "nonexact": sum(1 for row in deltas if not row["exact"] and not row["dimension_mismatch"]),
        "dimension_mismatches": sum(row["dimension_mismatch"] for row in deltas),
        "max_abs": max(maxes) if maxes else None,
        "mean_abs": sum(means) / len(means) if means else None,
        "min_cosine": min(cosines) if cosines else None,
        "p50_cosine": _percentile(cosines, 0.50),
        "p01_cosine": _percentile(cosines, 0.01),
        "p50_max_abs": _percentile(maxes, 0.50),
        "p95_max_abs": _percentile(maxes, 0.95),
        "p99_max_abs": _percentile(maxes, 0.99),
    }


def _ordered_float32(value: float) -> int:
    bits = struct.unpack("<I", struct.pack("<f", float(value)))[0]
    if bits & 0x80000000:
        return (~bits) & 0xFFFFFFFF
    return bits | 0x80000000


def float32_ulps(left: list[float], right: list[float]) -> int | None:
    """Largest float32 step between two vectors. None means the lengths differ."""
    if len(left) != len(right) or not left:
        return None
    return max(abs(_ordered_float32(a_value) - _ordered_float32(b_value)) for a_value, b_value in zip(left, right))


def compare_pair(fresh: list[float], stored: list[float]) -> dict:
    fresh_f32 = _f32(fresh)
    return {
        "raw": vector_delta(fresh, stored),
        "float32": vector_delta(fresh_f32, stored),
        "float32_ulps": float32_ulps(fresh_f32, stored),
    }


def stage_embed(
    staging: Path,
    run_id: str,
    rows: list[dict],
    *,
    base_url: str | None = None,
) -> dict:
    """Embed every RAG document and compare it with the stored production vector."""
    before = production_db_sha256()
    mismatches = verify_rag_rows(rows)
    if any(mismatches.values()):
        raise ParityError(f"RAG corpus is not ready to embed: {mismatches}")
    identity = ollama_identity(base_url)
    copy_dir = _copy_production(staging)
    production = _production_client(copy_dir).get_collection(TRAIN_ONLY_COLLECTION)
    sample_rows = select_sample(rows)
    sample_ids = [row["row_id"] for row in sample_rows]
    stored_sample = read_production_rows(copy_dir, sample_ids, production)
    fresh_sample = embed_documents([row["document"] for row in sample_rows], base_url=base_url)
    sample_report = []
    for row, fresh in zip(sample_rows, fresh_sample):
        stored = stored_sample.get(row["row_id"])
        if stored is None:
            raise ParityError(f"production vector missing for sample {row['row_id']}")
        pair = compare_pair(fresh, stored["embedding"])
        sample_report.append({"row_id": row["row_id"], "medium": row.get("medium"), "sft_role": row.get("sft_role"), **pair})
    contract = contract_from_sample(sample_report)
    dimension = len(fresh_sample[0]) if fresh_sample else 0
    fresh_by_id: dict[str, list[float]] = {}
    outside = []
    raw_deltas = []
    stored_deltas = []
    max_ulps_seen = 0
    for start in range(0, len(rows), DOCUMENT_BATCH):
        if start and start % 2048 == 0:
            print(f"embedded {start}/{len(rows)}", flush=True)
        batch = rows[start : start + DOCUMENT_BATCH]
        vectors = embed_documents([row["document"] for row in batch], base_url=base_url)
        if len(vectors) != len(batch):
            raise ParityError(f"embedder returned {len(vectors)} vectors for {len(batch)} documents")
        stored_batch = read_production_rows(copy_dir, [row["row_id"] for row in batch], production)
        for row, fresh in zip(batch, vectors):
            if len(fresh) != dimension:
                raise ParityError(f"dimension changed at {row['row_id']}")
            stored = stored_batch.get(row["row_id"])
            if stored is None:
                outside.append(row["row_id"])
                continue
            pair = compare_pair(fresh, stored["embedding"])
            raw_deltas.append(pair["raw"])
            stored_deltas.append(pair["float32"])
            if pair["float32_ulps"] is not None:
                max_ulps_seen = max(max_ulps_seen, pair["float32_ulps"])
            if not row_within_contract(pair["raw"], pair["float32"], pair["float32_ulps"], contract):
                outside.append(row["row_id"])
            fresh_by_id[row["row_id"]] = fresh
    summary = _summarize(raw_deltas)
    stored_summary = _summarize(stored_deltas)
    vector_mismatches = {
        "missing": len(set(row["row_id"] for row in rows) - set(fresh_by_id)),
        "unexpected": 0,
        "dimension": summary["dimension_mismatches"],
        "outside_contract": len(outside),
    }
    status = "FAIL" if any(vector_mismatches.values()) else "PASS"
    _write_vectors(staging / "embeddings", [row["row_id"] for row in rows], [fresh_by_id[row["row_id"]] for row in rows])
    sample_stats = _summarize([row["raw"] for row in sample_report])
    write_manifest(
        staging,
        "embed_rag",
        {
            "run_id": run_id,
            "executor": ORDERED_EXECUTOR,
            "input_artifact": "rag/rows.jsonl",
            "output_artifact": "embeddings/ids.json",
            "input_rows": len(rows),
            "output_rows": len(fresh_by_id),
            "input_sha256": file_sha256(staging / "rag" / "rows.jsonl"),
            "output_sha256": file_sha256(staging / "embeddings" / "ids.json"),
            "status": status,
            "mismatches": vector_mismatches,
            "details": {
                "claim": "CURRENT EMBEDDING BEHAVIOR REPRODUCED" if status == "PASS" else "VECTORS DIFFER",
                "historical_embed_command": "UNRECORDED",
                "model": identity,
                "batch_size": DOCUMENT_BATCH,
                "cpu": True,
                "normalization": False,
                "input_field": "document (stripped Relabel output)",
                "request_keys": sorted(request_body(["example"]).keys()),
                "dimension": dimension,
                "contract": contract,
                "sample_rows": len(sample_report),
                "sample": sample_stats,
                "full_raw": summary,
                "full_float32": stored_summary,
                "max_float32_ulps": max_ulps_seen,
                "outside_ids": outside[:20],
                "production_db_sha256_before": before,
                "production_db_sha256_after": production_db_sha256(),
            },
        },
    )
    if production_db_sha256() != before:
        raise ParityError("production chroma.sqlite3 changed during embedding")
    gate("embed_rag", vector_mismatches)
    return {"contract": contract, "dimension": dimension, "identity": identity, "vectors": len(fresh_by_id)}


def _write_vectors(directory: Path, ids: list[str], vectors: list[list[float]]) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    assert_staging_dir(directory)
    (directory / "ids.json").write_text(json.dumps(ids), encoding="utf-8")
    dimension = len(vectors[0]) if vectors else 0
    pieces = [struct.pack("<II", len(vectors), dimension)]
    for vector in vectors:
        pieces.append(struct.pack(f"<{dimension}f", *vector))
    (directory / "vectors.f32").write_bytes(b"".join(pieces))


def load_vectors(directory: Path) -> tuple[list[str], list[list[float]]]:
    ids = json.loads((directory / "ids.json").read_text(encoding="utf-8"))
    blob = (directory / "vectors.f32").read_bytes()
    count, dimension = struct.unpack_from("<II", blob, 0)
    values = struct.unpack_from(f"<{count * dimension}f", blob, 8)
    vectors = [list(values[index * dimension : (index + 1) * dimension]) for index in range(count)]
    if len(ids) != len(vectors):
        raise ParityError("embedding id file does not match the vector file")
    return ids, vectors


def stage_build_chroma(staging: Path, run_id: str, rows: list[dict]) -> dict:
    before = production_db_sha256()
    persist = assert_shadow_target(staging / "chroma", SHADOW_COLLECTION)
    ids, vectors = load_vectors(staging / "embeddings")
    by_id = {row["row_id"]: row for row in rows}
    if ids != [row["row_id"] for row in rows]:
        raise ParityError("embedding ids are not in RAG row order")
    if persist.exists():
        shutil.rmtree(persist)
    import chromadb
    from chromadb.config import Settings

    client = chromadb.PersistentClient(path=str(persist), settings=Settings(anonymized_telemetry=False))
    collection = client.get_or_create_collection(SHADOW_COLLECTION, metadata={"hnsw:space": "cosine"})
    for start in range(0, len(rows), 256):
        batch_ids = ids[start : start + 256]
        batch_vectors = vectors[start : start + 256]
        batch_rows = [by_id[row_id] for row_id in batch_ids]
        collection.upsert(
            ids=batch_ids,
            documents=[row["document"] for row in batch_rows],
            embeddings=batch_vectors,
            metadatas=[
                {
                    "group_id": row["group_id"],
                    "medium": row["medium"],
                    "row_id": int(str(row["row_id"]).split("_", 1)[1]),
                    "split": row["split"],
                }
                for row in batch_rows
            ],
        )
    count = collection.count()
    held = collection.get(ids=[HELD_OUT_RAG_ID])
    mismatches = {
        "count": int(count != EXPECTED_RAG_ROWS),
        "held_out": int(bool(held.get("ids"))),
    }
    status = "FAIL" if any(mismatches.values()) else "PASS"
    write_manifest(
        staging,
        "build_chroma_shadow",
        {
            "run_id": run_id,
            "executor": ORDERED_EXECUTOR,
            "input_artifact": "embeddings/vectors.f32",
            "output_artifact": "chroma",
            "input_rows": len(rows),
            "output_rows": count,
            "input_sha256": file_sha256(staging / "embeddings" / "vectors.f32"),
            "output_sha256": None,
            "status": status,
            "mismatches": mismatches,
            "details": {
                "collection": SHADOW_COLLECTION,
                "distance": "cosine",
                "production_collection_untouched": production_db_sha256() == before,
            },
        },
    )
    if production_db_sha256() != before:
        raise ParityError("production chroma.sqlite3 changed while building the shadow collection")
    gate("build_chroma_shadow", mismatches)
    return {"count": count, "collection": SHADOW_COLLECTION}


def stage_logical_chroma(staging: Path, run_id: str, rows: list[dict]) -> dict:
    before = production_db_sha256()
    embed_manifest = json.loads((staging / "manifests" / "embed_rag.json").read_text(encoding="utf-8"))
    contract = embed_manifest["details"]["contract"]
    ulp_limit = 0 if contract["mode"] in {"exact", "float32_storage"} else int(contract["max_ulps"])
    copy_dir = _copy_production(staging)
    production = _production_client(copy_dir).get_collection(TRAIN_ONLY_COLLECTION)
    ids = [row["row_id"] for row in rows]
    import chromadb
    from chromadb.config import Settings

    client = chromadb.PersistentClient(
        path=str(staging / "chroma"), settings=Settings(anonymized_telemetry=False)
    )
    collection = client.get_collection(SHADOW_COLLECTION)
    mismatches = Counter()
    exact_vectors = 0
    within_tolerance = 0
    worst_ulps = 0
    seen_shadow = 0
    for start in range(0, len(ids), 256):
        chunk = ids[start : start + 256]
        left_rows = read_production_rows(copy_dir, chunk, production)
        got = collection.get(ids=chunk, include=["embeddings", "documents", "metadatas"])
        right_rows = {}
        for row_id, embedding, document, metadata in zip(
            list(got.get("ids") or []),
            [] if got.get("embeddings") is None else got.get("embeddings"),
            [] if got.get("documents") is None else got.get("documents"),
            [] if got.get("metadatas") is None else got.get("metadatas"),
        ):
            right_rows[row_id] = {
                "embedding": [float(value) for value in list(embedding)],
                "document": document,
                "metadata": metadata or {},
            }
        seen_shadow += len(right_rows)
        for row_id in chunk:
            left = left_rows.get(row_id)
            right = right_rows.get(row_id)
            if left is None or right is None:
                mismatches["ids"] += 1
                continue
            if left["document"] != right["document"]:
                mismatches["documents"] += 1
            for field in ("group_id", "medium", "split"):
                if left["metadata"].get(field) != right["metadata"].get(field):
                    mismatches[field] += 1
            if left["metadata"].get("row_id") != right["metadata"].get("row_id"):
                mismatches["row_id"] += 1
            if left["embedding"] == right["embedding"]:
                exact_vectors += 1
                continue
            ulps = float32_ulps(left["embedding"], right["embedding"])
            worst_ulps = max(worst_ulps, ulps or 0)
            if ulps is not None and ulps <= ulp_limit:
                within_tolerance += 1
            else:
                mismatches["vectors"] += 1
    if collection.count() != len(ids):
        mismatches["ids"] += 1
    numeric = {key: int(value) for key, value in mismatches.items()}
    for key in ("ids", "documents", "group_id", "medium", "split", "row_id", "vectors"):
        numeric.setdefault(key, 0)
    status = "FAIL" if any(numeric.values()) else "PASS"
    write_manifest(
        staging,
        "logical_chroma_parity",
        {
            "run_id": run_id,
            "executor": ORDERED_EXECUTOR,
            "input_artifact": COPY_NAME,
            "output_artifact": "chroma",
            "input_rows": len(ids),
            "output_rows": seen_shadow,
            "input_sha256": before,
            "output_sha256": None,
            "status": status,
            "mismatches": numeric,
            "details": {
                "distance": "cosine",
                "contract_mode": contract["mode"],
                "ulp_limit": ulp_limit,
                "exact_vectors": exact_vectors,
                "vectors_within_tolerance": within_tolerance,
                "worst_ulps": worst_ulps,
                "physical_db_hash_required": False,
                "production_db_sha256_after": production_db_sha256(),
            },
        },
    )
    gate("logical_chroma_parity", numeric)
    return numeric


def compare_rankings(left: list[dict], right: list[dict]) -> dict:
    """Classify one query. A cutoff tie keeps the same closer hits and an equal distance."""
    left_ids = [hit["id"] for hit in left]
    right_ids = [hit["id"] for hit in right]
    score_gap = 0.0
    for one, two in zip(left, right):
        if one.get("distance") is not None and two.get("distance") is not None:
            score_gap = max(score_gap, abs(float(one["distance"]) - float(two["distance"])))
    cutoff_tie = False
    if left_ids != right_ids and left_ids[:-1] == right_ids[:-1] and left and right:
        cutoff_tie = left[-1].get("distance") == right[-1].get("distance")
    return {
        "exact_rank": left_ids == right_ids,
        "cutoff_tie": cutoff_tie,
        "same_set": set(left_ids) == set(right_ids),
        "score_gap": score_gap,
    }


def stage_retrieval(staging: Path, run_id: str, *, base_url: str | None = None) -> dict:
    """Embed each gold topic once and query both collections with that vector."""
    before = production_db_sha256()
    topics = _read_jsonl(DEFAULT_GOLD_EVAL)
    copy_dir = _copy_production(staging)
    production = _production_client(copy_dir).get_collection(TRAIN_ONLY_COLLECTION)
    import chromadb
    from chromadb.config import Settings

    shadow = chromadb.PersistentClient(
        path=str(staging / "chroma"), settings=Settings(anonymized_telemetry=False)
    ).get_collection(SHADOW_COLLECTION)
    exact = 0
    ties = []
    same_set = 0
    different = 0
    largest_gap = 0.0
    regenerated_gap = 0.0
    for topic in topics:
        vector = embed_query(str(topic["topic"]), base_url=base_url)
        again = embed_query(str(topic["topic"]), base_url=base_url)
        regenerated_gap = max(regenerated_gap, vector_delta(vector, again)["max_abs"] or 0.0)
        left = query_excerpts(production, vector, RETRIEVAL_K, where=None)
        right = query_excerpts(shadow, vector, RETRIEVAL_K, where=None)
        compared = compare_rankings(left, right)
        largest_gap = max(largest_gap, compared["score_gap"])
        if compared["exact_rank"]:
            exact += 1
        elif compared["cutoff_tie"]:
            ties.append(
                {
                    "query_id": topic.get("id"),
                    "production_id": left[-1]["id"],
                    "shadow_id": right[-1]["id"],
                    "distance": left[-1]["distance"],
                }
            )
        elif compared["same_set"]:
            same_set += 1
        else:
            different += 1
    mismatches = {
        "different_set": different,
        "different_order": same_set,
    }
    status = "FAIL" if any(mismatches.values()) else "PASS"
    write_manifest(
        staging,
        "retrieval_parity",
        {
            "run_id": run_id,
            "executor": ORDERED_EXECUTOR,
            "input_artifact": str(DEFAULT_GOLD_EVAL.relative_to(ROOT)),
            "output_artifact": "chroma",
            "input_rows": len(topics),
            "output_rows": exact,
            "input_sha256": file_sha256(DEFAULT_GOLD_EVAL),
            "output_sha256": None,
            "status": status,
            "mismatches": mismatches,
            "details": {
                "queries": len(topics),
                "k": RETRIEVAL_K,
                "reranker": False,
                "medium_filter": False,
                "query_field": "gold topic text",
                "exact_ranked_matches": exact,
                "cutoff_ties": ties,
                "same_set_different_order": same_set,
                "different_set": different,
                "largest_score_difference": largest_gap,
                "regenerated_query_max_abs": regenerated_gap,
                "shared_query_vector": True,
                "production_db_sha256_after": production_db_sha256(),
                "production_db_unchanged": production_db_sha256() == before,
            },
        },
    )
    if production_db_sha256() != before:
        raise ParityError("production chroma.sqlite3 changed during retrieval comparison")
    gate("retrieval_parity", mismatches)
    return {"queries": len(topics), "exact": exact, "different": different}


def embedding_input_is_document(row: dict) -> str:
    """The string the embedder receives. Instruction text is not the document."""
    return str(row.get("document") or "")
