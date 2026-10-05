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
    CHROMA_COLLECTION_NAME (default lemkin_content; not the train-only index)
    CHROMA_PERSIST_DIR   (train-only persist dir; default host_finetune/output/chroma_lemkin_train_only)
    OLLAMA_BASE          (default http://localhost:11434)
    OLLAMA_EMBED_MODEL   (default nomic-embed-text)

``--train-only`` is a separate build. It reads the EXP-004 relabel file, keeps
the train split, drops gold-overlap families, and writes ``lemkin_train_only``.
It refuses to delete or upsert ``lemkin_content``. Embeddings send
``options.num_gpu = 0``.
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
from host_finetune.train_only_index import (
    DEFAULT_ASSIGNMENTS,
    DEFAULT_DATASET as TRAIN_ONLY_DATASET,
    DEFAULT_GOLD_DISPOSITIONS,
    DEFAULT_GOLD_EVAL,
    DEFAULT_PERSIST_DIR,
    TRAIN_ONLY_COLLECTION,
    assert_collection_allowed,
    blocked_group_ids,
    headline_overlap_groups,
    index_documents,
    load_jsonl,
    ollama_embed_body,
    gpu_used_conflict,
    select_train_documents,
    vram_conflict,
)

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
    texts: list[str], base_url: str, model: str, *, cpu: bool = False
) -> list[list[float]]:
    """Embed a batch of texts in a single Ollama call.

    Ollama ``/api/embed`` accepts ``input`` as either ``str`` or ``list[str]``;
    batching is dramatically faster (20 ms/embed at batch=128 vs 2 s/embed at
    batch=1 on this hardware). ``cpu=True`` sends ``num_gpu: 0``.
    """
    if not texts:
        return []
    url = f"{base_url.rstrip('/')}/api/embed"
    last_err: str | None = None
    for attempt in range(EMBED_RETRIES):
        try:
            resp = requests.post(
                url, json=ollama_embed_body(texts, model, cpu=cpu), timeout=300
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


def _nvidia_csv(query: str) -> str:
    import subprocess

    return subprocess.check_output(
        ["nvidia-smi", f"--query-{query}", "--format=csv,noheader,nounits"],
        text=True,
        encoding="utf-8",
    )


def query_compute_apps() -> dict[int, int]:
    """Map GPU compute-app pid to used MiB. Skip Windows ``[N/A]`` cells."""
    apps: dict[int, int] = {}
    for line in _nvidia_csv("compute-apps=pid,used_gpu_memory").splitlines():
        line = line.strip()
        if not line or "," not in line:
            continue
        pid_text, mem_text = [part.strip() for part in line.split(",", 1)]
        if not mem_text.isdigit():
            continue
        apps[int(pid_text)] = int(mem_text)
    return apps


def query_gpu_used_mib() -> list[int]:
    """Total memory.used for each GPU, in MiB."""
    used = []
    for line in _nvidia_csv("gpu=memory.used").splitlines():
        text = line.strip()
        if text.isdigit():
            used.append(int(text))
    if not used:
        raise RuntimeError("nvidia-smi did not report GPU memory.used")
    return used


def _probe_cpu_embed(embed_fn) -> None:
    """Embed one short string and stop if GPU memory rises."""
    before_apps = query_compute_apps()
    before_gpu = query_gpu_used_mib()
    embed_fn(["train-only index probe"])
    after_apps = query_compute_apps()
    after_gpu = query_gpu_used_mib()
    conflict = vram_conflict(before_apps, after_apps) or gpu_used_conflict(before_gpu, after_gpu)
    if conflict:
        raise SystemExit(f"stopping train-only embed: {conflict}")
    logger.info(
        "CPU embed probe left GPU memory within tolerance (%s -> %s MiB)",
        before_gpu,
        after_gpu,
    )


def run_train_only(args: argparse.Namespace) -> None:
    """Index EXP-004 train rows into a new persistent collection."""
    collection_name = args.collection or TRAIN_ONLY_COLLECTION
    assert_collection_allowed(collection_name)
    dataset = Path(args.dataset)
    assignments_path = Path(args.assignments)
    dispositions_path = Path(args.gold_dispositions)
    for path in (dataset, assignments_path, dispositions_path):
        if not path.is_file():
            raise SystemExit(f"error: missing {path}")
    dataset_rows = load_jsonl(dataset)
    assignment_rows = load_jsonl(assignments_path)
    prior_blocked = blocked_group_ids(assignment_rows, load_jsonl(dispositions_path))
    gold_path = Path(args.gold)
    if not gold_path.is_file():
        raise SystemExit(f"error: missing {gold_path}")
    extra_groups, extra_rows = headline_overlap_groups(
        dataset_rows, assignment_rows, load_jsonl(gold_path), prior_blocked
    )
    logger.info(
        "headline-overlap groups not already blocked=%d direct_matches=%d",
        len(extra_groups - prior_blocked),
        len(extra_rows),
    )
    for item in extra_rows:
        logger.info(
            "newly excluded row=%s group=%s split=%s platform=%s hits=%s",
            item["row_id"],
            item["group_id"],
            item["split"],
            item["source_platform"],
            item["hits"],
        )
    docs, stats = select_train_documents(
        dataset_rows,
        assignment_rows,
        prior_blocked | extra_groups,
    )
    stats["headline_overlap_new_groups"] = len(extra_groups - prior_blocked)
    stats["headline_overlap_direct_matches"] = len(extra_rows)
    logger.info("train-only selection %s", stats)
    if args.sample is not None:
        docs = docs[: args.sample]
    if not docs:
        raise SystemExit("error: train-only selection is empty")

    def embed_fn(texts: list[str]) -> list[list[float]]:
        return _embed_batch(texts, args.ollama_base, args.embed_model, cpu=True)

    if not args.skip_vram_probe:
        _probe_cpu_embed(embed_fn)
    if args.probe_only:
        logger.info("probe only; no documents upserted")
        return

    import chromadb
    from chromadb.config import Settings

    persist = Path(args.persist_dir)
    persist.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(
        path=str(persist), settings=Settings(anonymized_telemetry=False)
    )
    try:
        client.delete_collection(collection_name)
        logger.info("dropped existing collection %r in %s", collection_name, persist)
    except Exception as exc:  # noqa: BLE001
        logger.info("no existing collection %r to drop (%s)", collection_name, exc)
    collection = client.get_or_create_collection(
        collection_name, metadata={"hnsw:space": "cosine"}
    )
    # Batches of 64 long posts stalled Ollama. 16 stayed on CPU and returned.
    batch_size = min(args.batch_size, 16)
    count = index_documents(docs, collection, embed_fn, batch_size=batch_size)
    logger.info("train-only index count=%d persist=%s", count, persist)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", default=None)
    p.add_argument("--collection", default=None)
    p.add_argument("--train-only", action="store_true",
                   help="Index the EXP-004 train split into lemkin_train_only.")
    p.add_argument("--assignments", default=str(DEFAULT_ASSIGNMENTS))
    p.add_argument("--gold-dispositions", default=str(DEFAULT_GOLD_DISPOSITIONS))
    p.add_argument("--gold", default=str(DEFAULT_GOLD_EVAL))
    p.add_argument(
        "--persist-dir",
        default=os.environ.get("CHROMA_PERSIST_DIR", str(DEFAULT_PERSIST_DIR)),
    )
    p.add_argument("--probe-only", action="store_true",
                   help="CPU-embed one string, check VRAM, and do not upsert.")
    p.add_argument("--skip-vram-probe", action="store_true",
                   help="Skip the nvidia-smi check. Tests and offline stubs only.")
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

    if args.train_only:
        if args.dataset is None:
            args.dataset = str(TRAIN_ONLY_DATASET)
        run_train_only(args)
        return
    if args.dataset is None:
        args.dataset = str(DEFAULT_DATASET)
    if args.collection is None:
        args.collection = DEFAULT_COLLECTION

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
