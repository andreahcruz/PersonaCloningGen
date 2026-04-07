"""Peek at lemkin_content in Chroma (same tenant/database as the Airflow DAG).

Interpreting counts:
- "Rows per source" = Chroma vectors = one row per text *chunk* (after ~500-word splitting).
- "Distinct essay_id per source" = original *documents* that have at least one embedded
  chunk in Chroma (Spark assigns one essay_id per row after dedup). Compare these to your
  pipeline's *document* counts (MinIO raw after dedup), not to raw JSONL line counts.
"""

import json
import os
from collections import Counter, defaultdict

import chromadb
from chromadb.config import DEFAULT_DATABASE, DEFAULT_TENANT


def _count_by_source(col, batch_size: int) -> Counter:
    """Paginate through metadata only to build per-source row counts."""
    counts: Counter = Counter()
    offset = 0
    while True:
        batch = col.get(
            include=["metadatas"],
            limit=batch_size,
            offset=offset,
        )
        metas = batch.get("metadatas") or []
        if not metas:
            break
        for m in metas:
            src = (m or {}).get("source") or "unknown"
            counts[src] += 1
        if len(metas) < batch_size:
            break
        offset += batch_size
    return counts


def _distinct_essay_ids_by_source(col, batch_size: int) -> dict[str, set]:
    """How many original Spark documents (essay_id) appear per source — chunk grain excluded."""
    by_src: dict[str, set] = defaultdict(set)
    offset = 0
    while True:
        batch = col.get(
            include=["metadatas"],
            limit=batch_size,
            offset=offset,
        )
        metas = batch.get("metadatas") or []
        if not metas:
            break
        for m in metas:
            src = (m or {}).get("source") or "unknown"
            eid = (m or {}).get("essay_id")
            if eid is not None:
                by_src[src].add(eid)
        if len(metas) < batch_size:
            break
        offset += batch_size
    return dict(by_src)


def _print_sample_block(
    label: str,
    doc_id: str,
    text: str,
    meta: dict | None,
    emb,
) -> None:
    preview = (text[:400] + "…") if len(text) > 400 else text
    emb_info = ""
    if emb is not None and hasattr(emb, "__len__"):
        head = [float(x) for x in emb[:3]]
        emb_info = f"dim={len(emb)} head={head!r}"
    print(f"--- {label} id={doc_id!r}")
    print(f"    metadata: {json.dumps(meta, default=str)}")
    print(f"    embedding: {emb_info}")
    print(f"    document:\n{preview}\n")


def main() -> None:
    host = os.environ.get("CHROMA_HOST", "localhost")
    port = int(os.environ.get("CHROMA_PORT", "8000"))
    name = os.environ.get("CHROMA_COLLECTION_NAME", "lemkin_content")
    tenant = os.environ.get("CHROMA_TENANT", DEFAULT_TENANT)
    database = os.environ.get("CHROMA_DATABASE", DEFAULT_DATABASE)
    batch_size = int(os.environ.get("CHROMA_SCAN_BATCH", "5000"))
    samples_per_source = int(os.environ.get("CHROMA_SAMPLES_PER_SOURCE", "3"))

    client = chromadb.HttpClient(
        host=host, port=port, tenant=tenant, database=database
    )
    col = client.get_collection(name)

    print(f"host={host}:{port} tenant={tenant} database={database}")
    total = col.count()
    print(f"collection={name!r} total_rows={total}\n")

    if total == 0:
        print("Nothing to show.")
        return

    print("=== Rows per source (metadata.source) ===")
    counts = _count_by_source(col, batch_size=batch_size)
    counted = sum(counts.values())
    for src in sorted(counts.keys(), key=lambda s: (-counts[s], s)):
        print(f"  {src!r}: {counts[src]}")
    if counted != total:
        print(
            f"  (note: sum of buckets={counted} vs col.count()={total} — "
            "usually identical; mismatch can mean concurrent writes.)"
        )
    print()

    print("=== Distinct essay_id per source (original documents with ≥1 embedded chunk) ===")
    by_essay = _distinct_essay_ids_by_source(col, batch_size=batch_size)
    for src in sorted(by_essay.keys(), key=lambda s: (-len(by_essay[s]), s)):
        print(f"  {src!r}: {len(by_essay[src])} documents")
    print(
        "  (Each essay_id is one raw row after Spark dropDuplicates on "
        "source/title/url/text; long posts become many Chroma rows but one essay_id.)\n"
    )

    print(f"=== Up to {samples_per_source} sample(s) per source ===\n")
    for src in sorted(counts.keys()):
        got = col.get(
            where={"source": {"$eq": src}},
            limit=samples_per_source,
            include=["documents", "metadatas", "embeddings"],
        )
        ids = got.get("ids") or []
        docs = got.get("documents") or []
        metas = got.get("metadatas") or []
        embs = got.get("embeddings")
        if not ids:
            print(f"(no rows returned for source={src!r} — check metadata filter)\n")
            continue
        for i, doc_id in enumerate(ids):
            text = docs[i] if i < len(docs) else ""
            meta = metas[i] if i < len(metas) else None
            emb = embs[i] if embs is not None and i < len(embs) else None
            _print_sample_block(
                f"[{src}] sample {i + 1}/{len(ids)}",
                doc_id,
                text,
                meta,
                emb,
            )


if __name__ == "__main__":
    main()
