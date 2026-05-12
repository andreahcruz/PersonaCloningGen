"""Rebuild the ``lemkin_content`` Chroma collection from cleaned ``dataset.jsonl``.

This bypasses the Spark → MinIO → Airflow ``load_to_chroma`` path because the
data we want in RAG is now the **cleaned** corpus, not the raw MinIO chunks
that have mid-word seams and unpunctuated transcripts.

What it does:

    1. Read every row of ``host_finetune/data/dataset.jsonl``.
    2. Drop + recreate the ``lemkin_content`` collection on the Docker Chroma.
    3. Embed each row's ``output`` via Ollama ``/api/embed`` (default
       ``nomic-embed-text``) — same model the Spark job used, so the existing
       ``generate.py`` retrieval still works without changes.
    4. Upsert into Chroma in batches of ``BATCH_SIZE`` with metadata
       (doc_id, title, chunk_index, n_chunks, source_type).

Run::

    host_finetune\\.venv\\Scripts\\python.exe -m host_finetune.rebuild_chroma

Knobs (env vars / CLI):

    CHROMA_HOST          (default localhost)
    CHROMA_PORT          (default 8000)
    CHROMA_COLLECTION_NAME (default lemkin_content)
    OLLAMA_BASE          (default http://localhost:11434)
    OLLAMA_EMBED_MODEL   (default nomic-embed-text)
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

import requests

from host_finetune.config import DATA_DIR

logger = logging.getLogger(__name__)

DEFAULT_DATASET = DATA_DIR / "dataset.jsonl"
DEFAULT_COLLECTION = os.environ.get("CHROMA_COLLECTION_NAME", "lemkin_content")
DEFAULT_CHROMA_HOST = os.environ.get("CHROMA_HOST", "localhost")
DEFAULT_CHROMA_PORT = int(os.environ.get("CHROMA_PORT", "8000"))
DEFAULT_OLLAMA_BASE = os.environ.get("OLLAMA_BASE", "http://localhost:11434")
DEFAULT_EMBED_MODEL = os.environ.get("OLLAMA_EMBED_MODEL", "nomic-embed-text")
BATCH_SIZE = int(os.environ.get("CHROMA_REBUILD_BATCH", "64"))
EMBED_RETRIES = int(os.environ.get("OLLAMA_EMBED_RETRIES", "4"))

_TITLE_RE = re.compile(
    r"^Write in the style of Jason Lemkin about:\s*", re.IGNORECASE
)
_PART_RE = re.compile(r"\s*\(part\s+(\d+)\s+of\s+(\d+)\)\s*$", re.IGNORECASE)


def _parse_title(instr: str) -> tuple[str, int | None, int | None]:
    """Return (base_title, part_idx, total_parts)."""
    t = _TITLE_RE.sub("", (instr or "").strip())
    m = _PART_RE.search(t)
    if not m:
        return t.strip(), None, None
    return _PART_RE.sub("", t).strip(), int(m.group(1)), int(m.group(2))


def _embed_batch(
    texts: list[str], base_url: str, model: str
) -> list[list[float]]:
    """Embed a batch of texts in a single Ollama call.

    Ollama ``/api/embed`` accepts ``input`` as either ``str`` or ``list[str]``;
    batching is dramatically faster (20 ms/embed at batch=128 vs 2 s/embed at
    batch=1 on this hardware).
    """
    if not texts:
        return []
    url = f"{base_url.rstrip('/')}/api/embed"
    last_err: str | None = None
    for attempt in range(EMBED_RETRIES):
        try:
            resp = requests.post(
                url, json={"model": model, "input": texts}, timeout=300
            )
            resp.raise_for_status()
            data = resp.json()
            vecs = data.get("embeddings") or []
            if isinstance(vecs, list) and len(vecs) == len(texts):
                return vecs
            last_err = f"got {len(vecs)} vectors for {len(texts)} inputs"
        except Exception as e:  # noqa: BLE001
            last_err = str(e)
        time.sleep(1.5 * (2**attempt))
    raise RuntimeError(f"embed batch failed after {EMBED_RETRIES} tries: {last_err}")


def _load_rows(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", default=str(DEFAULT_DATASET))
    p.add_argument("--collection", default=DEFAULT_COLLECTION)
    p.add_argument("--chroma-host", default=DEFAULT_CHROMA_HOST)
    p.add_argument("--chroma-port", type=int, default=DEFAULT_CHROMA_PORT)
    p.add_argument("--ollama-base", default=DEFAULT_OLLAMA_BASE)
    p.add_argument("--embed-model", default=DEFAULT_EMBED_MODEL)
    p.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    p.add_argument("--sample", type=int, default=None,
                   help="Only embed the first N rows (smoke test).")
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    src = Path(args.dataset)
    if not src.is_file():
        print(f"error: missing dataset {src}", file=sys.stderr)
        sys.exit(2)

    rows = _load_rows(src)
    if args.sample is not None:
        rows = rows[: args.sample]
    logger.info("loaded %d rows from %s", len(rows), src)

    # ── Chroma client ──────────────────────────────────────────────────────
    import chromadb
    from chromadb.config import DEFAULT_DATABASE, DEFAULT_TENANT

    client = chromadb.HttpClient(
        host=args.chroma_host,
        port=args.chroma_port,
        tenant=os.environ.get("CHROMA_TENANT", DEFAULT_TENANT),
        database=os.environ.get("CHROMA_DATABASE", DEFAULT_DATABASE),
    )
    try:
        client.delete_collection(args.collection)
        logger.info("dropped existing collection %r", args.collection)
    except Exception as e:  # noqa: BLE001
        logger.info("no existing collection %r to drop (%s)", args.collection, e)

    collection = client.get_or_create_collection(
        args.collection, metadata={"hnsw:space": "cosine"}
    )

    # ── Group by title to assign chunk_index / n_chunks for metadata ──────
    titles = [_parse_title(r.get("instruction", "")) for r in rows]
    title_total: dict[str, int] = defaultdict(int)
    for base, _, _ in titles:
        title_total[base] += 1
    title_idx: dict[str, int] = defaultdict(int)

    # ── Embed in batches (Ollama supports multi-input /api/embed) ─────────
    # Build (doc_id, doc_text, metadata) triples first so we can batch.
    todo: list[tuple[str, str, dict]] = []
    for i, row in enumerate(rows):
        out = (row.get("output") or "").strip()
        if not out:
            continue
        base, _, _ = titles[i]
        title_idx[base] += 1
        chunk_index = title_idx[base] - 1
        n_chunks = title_total[base]
        doc_id = f"clean_v2_{i}"
        meta = {
            "doc_id": doc_id,
            "title": base,
            "chunk_index": chunk_index,
            "n_chunks": n_chunks,
            "source_type": "clean_v2",
        }
        todo.append((doc_id, out, meta))

    logger.info("prepared %d (doc_id, text, meta) triples for embedding", len(todo))

    upserted = 0
    failures = 0
    t_start = time.monotonic()
    for batch_start in range(0, len(todo), args.batch_size):
        batch = todo[batch_start : batch_start + args.batch_size]
        ids = [b[0] for b in batch]
        docs = [b[1] for b in batch]
        metas = [b[2] for b in batch]
        try:
            embs = _embed_batch(docs, args.ollama_base, args.embed_model)
        except Exception as e:  # noqa: BLE001
            failures += len(batch)
            logger.warning(
                "embed batch failure (rows %d..%d): %r",
                batch_start,
                batch_start + len(batch) - 1,
                e,
            )
            continue
        collection.upsert(
            ids=ids, documents=docs, embeddings=embs, metadatas=metas
        )
        upserted += len(batch)
        elapsed = time.monotonic() - t_start
        rate = upserted / max(1e-6, elapsed)
        eta = (len(todo) - upserted) / max(rate, 1e-6)
        logger.info(
            "upserted %d / %d (%.1f/s, eta %.0fs)",
            upserted,
            len(todo),
            rate,
            eta,
        )

    elapsed = time.monotonic() - t_start
    try:
        count = collection.count()
    except Exception:  # noqa: BLE001
        count = -1
    logger.info(
        "rebuild done: upserted=%d failures=%d collection_count=%d elapsed=%.1fs",
        upserted,
        failures,
        count,
        elapsed,
    )


if __name__ == "__main__":
    main()
