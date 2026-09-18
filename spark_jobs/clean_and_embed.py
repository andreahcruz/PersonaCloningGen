"""
PySpark job: reads normalized Lemkin JSON from MinIO (raw/), applies transforms,
writes cleaned + embedded chunks to MinIO (chunks/).

Transforms:
  1. Strip HTML tags
  2. Parse date → timestamp
  3. Deduplicate rows
  4. Chunk text into ~500-word segments
  5. Add chunk_index + preserve source
  6. Generate nomic-embed-text vectors via Ollama

Embedding reliability: each Spark partition calls Ollama concurrently. Too many
partitions → timeouts / empty embeddings. Before embedding, the DataFrame is
repartitioned to SPARK_EMBED_PARTITIONS (default 2). Use 1 for a single-threaded
embed stream (slowest, safest for laptop Ollama). Increase if your Ollama host
handles parallel /api/embed load well.
"""
import json
import logging
import os
import random
import re
import time
from datetime import datetime
from pathlib import Path

import requests as req
from bs4 import BeautifulSoup
from pyspark import StorageLevel
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import (
    ArrayType,
    FloatType,
    StringType,
    StructField,
    StructType,
    TimestampType,
)

from corpus_footer_scrub import strip_footer_noise

MINIO_ENDPOINT = os.environ.get("MINIO_ENDPOINT", "http://minio:9000")
MINIO_ACCESS_KEY = os.environ.get("MINIO_ACCESS_KEY", "minioadmin")
MINIO_SECRET_KEY = os.environ.get("MINIO_SECRET_KEY", "minioadmin")
BUCKET_RAW = os.environ.get("MINIO_BUCKET_RAW", "lemkin-raw")
BUCKET_PROCESSED = os.environ.get("MINIO_BUCKET_PROCESSED", "lemkin-processed")
OLLAMA_BASE = os.environ.get("OLLAMA_BASE", "http://host.docker.internal:11434")
EMBED_MODEL = os.environ.get("OLLAMA_EMBED_MODEL", "nomic-embed-text")
CHUNK_WORD_LIMIT = 500
OLLAMA_EMBED_MAX_RETRIES = int(os.environ.get("OLLAMA_EMBED_MAX_RETRIES", "5"))
OLLAMA_EMBED_RETRY_BASE_SEC = float(os.environ.get("OLLAMA_EMBED_RETRY_BASE_SEC", "1.5"))
OLLAMA_EMBED_DELAY_SEC = float(os.environ.get("OLLAMA_EMBED_DELAY_SEC", "0.03"))
# Hard cap on characters sent to /api/embed (safety for model context)
OLLAMA_EMBED_MAX_CHARS = int(os.environ.get("OLLAMA_EMBED_MAX_CHARS", "12000"))
SPARK_EMBED_PARTITIONS = int(os.environ.get("SPARK_EMBED_PARTITIONS", "2"))
# Refuse to write chunks/ when more than this share of chunk rows has no embedding, so a
# dead Ollama cannot overwrite the last good output with empty vectors.
MAX_MISSING_EMBEDDING_FRACTION = float(os.environ.get("MAX_MISSING_EMBEDDING_FRACTION", "0.10"))

logger = logging.getLogger("lemkin.spark")


def preflight_ollama():
    """Fail in seconds, not after hours of empty embeddings, if Ollama cannot embed."""
    url = f"{OLLAMA_BASE}/api/embed"
    try:
        resp = req.post(url, json={"model": EMBED_MODEL, "input": "preflight"}, timeout=60)
    except Exception as e:
        raise RuntimeError(
            f"Ollama preflight failed: cannot reach {OLLAMA_BASE} ({e}). Start `ollama serve` on the "
            "host; from Docker the URL must be http://host.docker.internal:11434 (OLLAMA_BASE)."
        ) from e
    if resp.status_code == 404:
        raise RuntimeError(
            f"Ollama preflight failed: embed model '{EMBED_MODEL}' not found. "
            f"Run `ollama pull {EMBED_MODEL}` on the host."
        )
    if not resp.ok:
        raise RuntimeError(
            f"Ollama preflight failed: HTTP {resp.status_code} from {url}: {resp.text[:200]}"
        )
    try:
        vec = resp.json()["embeddings"][0]
    except (ValueError, KeyError, IndexError, TypeError) as e:
        raise RuntimeError(
            f"Ollama preflight failed: unusable /api/embed response for '{EMBED_MODEL}' "
            "(is it an embedding model?)."
        ) from e
    if not vec:
        raise RuntimeError(f"Ollama preflight failed: empty embedding from '{EMBED_MODEL}'.")
    logger.info("Ollama preflight ok: %s at %s (dim=%d)", EMBED_MODEL, OLLAMA_BASE, len(vec))


def check_embedding_coverage(total, with_embedding, max_missing_fraction=None):
    """Raise if too many chunk rows have no embedding. Called before anything is written."""
    if max_missing_fraction is None:
        max_missing_fraction = MAX_MISSING_EMBEDDING_FRACTION
    if total == 0:
        raise RuntimeError(
            "No chunk rows were produced. Check that raw/ in MinIO has documents "
            "(run extract_to_minio) and that they contain text."
        )
    missing = total - with_embedding
    if missing:
        logger.warning(
            "%d of %d chunk rows (%.1f%%) have no embedding (limit %.0f%%).",
            missing, total, 100.0 * missing / total, 100.0 * max_missing_fraction,
        )
    if with_embedding == 0 or missing / total > max_missing_fraction:
        raise RuntimeError(
            f"{missing} of {total} chunk rows have no embedding, above the "
            f"{max_missing_fraction:.0%} limit (MAX_MISSING_EMBEDDING_FRACTION). Not writing chunks/ "
            "so the previous good output is kept. Check `ollama serve`, lower "
            "SPARK_EMBED_PARTITIONS, and re-run."
        )


def strip_html(text):
    if not text:
        return text
    return BeautifulSoup(text, "html.parser").get_text(separator=" ")


def parse_date(date_str):
    if not date_str:
        return None
    s = str(date_str).strip()
    if not s:
        return None
    try:
        if s.endswith("Z"):
            s = s[:-1] + "+00:00"
        return datetime.fromisoformat(s)
    except ValueError:
        pass
    for fmt in ("%B %Y", "%Y", "%B %d, %Y", "%b %Y", "%m/%Y", "%Y-%m-%d"):
        try:
            return datetime.strptime(s, fmt)
        except ValueError:
            continue
    digits = re.search(r"\d{4}", s)
    if digits:
        try:
            return datetime.strptime(digits.group(), "%Y")
        except ValueError:
            pass
    return None


def chunk_text(text, limit=CHUNK_WORD_LIMIT):
    if not text:
        return []
    words = text.split()
    chunks = []
    for i in range(0, len(words), limit):
        chunk = " ".join(words[i : i + limit])
        chunks.append(chunk)
    return chunks


# Per-source fallback "topic" so LinkedIn/X posts (which have no real title)
# still produce a usable instruction prompt for SFT. Keep these short and natural.
_SOURCE_FALLBACK_TOPIC = {
    "blog": "a B2B SaaS topic",
    "linkedin": "a SaaS founder takeaway",
    "x": "a sharp SaaS observation",
    "youtube_jason": "a SaaStr lesson",
    "youtube_saastr": "a SaaStr conference talking point",
}


# Match host_finetune defaults (keep split logic + legacy truncate cap in sync).
_TRAINING_OUTPUT_MAX_CHARS = int(os.environ.get("DATASET_MAX_OUTPUT_CHARS", "2000"))
_MIN_TRAINING_OUTPUT_CHARS = 80
_JUNK_TITLE_FRAGMENTS = ("test post with ai chat",)
# Default 1600 chars/label pairs with host MAX_SEQ_LENGTH=512 (same 3200/1024 ratio).
# Set to 0 for legacy single-row export + truncate at _TRAINING_OUTPUT_MAX_CHARS.
SFT_CHUNK_OUTPUT_CHARS = int(os.environ.get("SFT_CHUNK_OUTPUT_CHARS", "1600"))


def _split_oversized_segment(chunk: str, max_chars: int) -> list[str]:
    """Split one segment into pieces each <= max_chars (paragraph cuts preferred)."""
    chunk = chunk.strip()
    if not chunk:
        return []
    if len(chunk) <= max_chars:
        return [chunk]
    out: list[str] = []
    rest = chunk
    while len(rest) > max_chars:
        window = rest[:max_chars]
        cut = window.rfind("\n\n")
        if cut < max_chars // 5:
            cut = max_chars
        piece = rest[:cut].strip()
        rest = rest[cut:].lstrip()
        if piece:
            out.append(piece)
    if rest.strip():
        out.append(rest.strip())
    return out


def split_long_body_for_sft(body: str, max_chars: int) -> list[str]:
    """Mirrors host_finetune/sft_chunk_utils.split_long_body_for_sft — keep in sync."""
    body = body.strip()
    if not body:
        return []
    if max_chars <= 0 or len(body) <= max_chars:
        return [body]

    parts: list[str] = []
    rest = body
    while rest:
        if len(rest) <= max_chars:
            p = rest.strip()
            if p:
                parts.append(p)
            break
        window = rest[:max_chars]
        cut = window.rfind("\n\n")
        if cut < max_chars // 5:
            cut = max_chars
        piece = rest[:cut].strip()
        rest = rest[cut:].lstrip()
        if piece:
            parts.append(piece)

    merged: list[str] = []
    for p in parts:
        if merged and len(p) < _MIN_TRAINING_OUTPUT_CHARS:
            cand = merged[-1] + "\n\n" + p
            if len(cand) <= max_chars:
                merged[-1] = cand.strip()
                continue
        merged.append(p)

    capped: list[str] = []
    for m in merged:
        if len(m) <= max_chars:
            capped.append(m)
        else:
            capped.extend(_split_oversized_segment(m, max_chars))

    final: list[str] = []
    for x in capped:
        if len(x) >= _MIN_TRAINING_OUTPUT_CHARS:
            final.append(x)
        elif final:
            cand = final[-1] + "\n\n" + x
            if len(cand) <= max_chars:
                final[-1] = cand.strip()
            else:
                final.append(x)
        else:
            final.append(x)

    return final if final else [body[:max_chars]]


def to_training_json_lines(title, source, text):
    """Return a list of Alpaca JSON strings (one per chunk for long documents)."""
    if not text:
        return []
    body = str(text).strip()
    if not body:
        return []
    if "\ufffd" in body:
        return []
    if len(body) < _MIN_TRAINING_OUTPUT_CHARS:
        return []
    raw_topic = (title or "").strip()
    if raw_topic:
        lt = raw_topic.lower()
        if any(f in lt for f in _JUNK_TITLE_FRAGMENTS):
            return []
        if "\ufffd" in raw_topic:
            return []
        topic = raw_topic[:200]
    else:
        topic = _SOURCE_FALLBACK_TOPIC.get(source or "", "B2B SaaS")

    if SFT_CHUNK_OUTPUT_CHARS <= 0:
        if len(body) > _TRAINING_OUTPUT_MAX_CHARS:
            cut = body[:_TRAINING_OUTPUT_MAX_CHARS]
            last_nl = cut.rfind("\n\n")
            if last_nl > len(cut) * 4 // 5:
                cut = cut[:last_nl]
            body = cut.rstrip() + "\n\n[Truncated for training.]"
        line = json.dumps(
            {
                "instruction": f"Write in the style of Jason Lemkin about: {topic}",
                "input": "",
                "output": body,
            },
            ensure_ascii=False,
        )
        return [line]

    chunks = split_long_body_for_sft(body, SFT_CHUNK_OUTPUT_CHARS)
    n = len(chunks)
    lines = []
    for i, chunk in enumerate(chunks):
        t = topic if n == 1 else f"{topic} (part {i + 1} of {n})"
        lines.append(
            json.dumps(
                {
                    "instruction": f"Write in the style of Jason Lemkin about: {t}",
                    "input": "",
                    "output": chunk,
                },
                ensure_ascii=False,
            )
        )
    return lines


def embed_text(text):
    """Call Ollama /api/embed with retries. Returns [] only after all attempts fail."""
    if text is None:
        return []
    t = str(text).strip()
    if not t:
        return []
    if len(t) > OLLAMA_EMBED_MAX_CHARS:
        t = t[:OLLAMA_EMBED_MAX_CHARS]

    url = f"{OLLAMA_BASE}/api/embed"
    last_err = None
    for attempt in range(OLLAMA_EMBED_MAX_RETRIES):
        try:
            resp = req.post(
                url,
                json={"model": EMBED_MODEL, "input": t},
                timeout=180,
            )
            resp.raise_for_status()
            data = resp.json()
            vecs = data.get("embeddings") or []
            if not vecs:
                last_err = "empty embeddings in response"
            else:
                emb = vecs[0]
                if isinstance(emb, list) and len(emb) > 0:
                    if OLLAMA_EMBED_DELAY_SEC > 0:
                        time.sleep(OLLAMA_EMBED_DELAY_SEC)
                    return emb
                last_err = "invalid embedding vector"
        except Exception as e:
            last_err = str(e)
        # backoff + jitter before retry
        if attempt < OLLAMA_EMBED_MAX_RETRIES - 1:
            sleep_s = OLLAMA_EMBED_RETRY_BASE_SEC * (2**attempt) + random.uniform(0, 0.75)
            time.sleep(sleep_s)

    # Runs inside Spark Python workers, which have no logging config: WARNING goes to stderr.
    logging.getLogger("lemkin.spark.embed").warning(
        "Embedding failed after %d tries: %s", OLLAMA_EMBED_MAX_RETRIES, last_err
    )
    return []


def main():
    logging.basicConfig(
        level=getattr(logging, os.environ.get("LOG_LEVEL", "INFO").upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    preflight_ollama()
    spark = (
        SparkSession.builder
        .appName("Lemkin content — clean, chunk, embed")
        .config("spark.hadoop.fs.s3a.endpoint", MINIO_ENDPOINT)
        .config("spark.hadoop.fs.s3a.access.key", MINIO_ACCESS_KEY)
        .config("spark.hadoop.fs.s3a.secret.key", MINIO_SECRET_KEY)
        .config("spark.hadoop.fs.s3a.path.style.access", "true")
        .config("spark.hadoop.fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem")
        .config("spark.hadoop.fs.s3a.connection.ssl.enabled", "false")
        .getOrCreate()
    )
    try:
        _run(spark)
    except Exception:
        logger.exception("Spark job failed (see the traceback above; Airflow marks the task failed)")
        raise
    finally:
        spark.stop()


def _run(spark):
    # UDFs are unpickled in separate Python workers; sibling ``corpus_footer_scrub.py``
    # is not on their sys.path unless we ship it with the job.
    _scrub = Path(__file__).resolve().parent / "corpus_footer_scrub.py"
    if _scrub.is_file():
        spark.sparkContext.addPyFile(str(_scrub))

    raw_schema = StructType([
        StructField("title", StringType(), True),
        StructField("date", StringType(), True),
        StructField("url", StringType(), True),
        StructField("text", StringType(), True),
        StructField("source", StringType(), True),
    ])

    df = spark.read.schema(raw_schema).json(f"s3a://{BUCKET_RAW}/raw/")

    strip_html_udf = F.udf(strip_html, StringType())
    df = df.withColumn("text", strip_html_udf(F.col("text")))
    footer_udf = F.udf(strip_footer_noise, StringType())
    df = df.withColumn("text", footer_udf(F.col("text")))

    parse_date_udf = F.udf(parse_date, TimestampType())
    df = df.withColumn("date_ts", parse_date_udf(F.col("date")))

    df = df.dropDuplicates(["source", "title", "url", "text"])

    df = df.withColumn("essay_id", F.monotonically_increasing_id())

    # Branch off a *document-grain* view before chunking. SFT training works better
    # on whole essays/posts than on 500-word chunks (which lose narrative arc).
    docs_df = df.select(
        "essay_id", "source", "title", "url", "date_ts", "text",
        F.size(F.split(F.col("text"), r"\s+")).alias("doc_word_count"),
    )

    chunk_udf = F.udf(chunk_text, ArrayType(StringType()))
    df = df.withColumn("chunks", chunk_udf(F.col("text")))

    df = df.select(
        "essay_id", "source", "title", "date", "date_ts", "url",
        F.posexplode("chunks").alias("chunk_index", "chunk_text"),
    )

    df = df.withColumn("word_count", F.size(F.split(F.col("chunk_text"), r"\s+")))

    # Drop empty chunk lines (nothing to embed or store)
    df = df.filter(F.length(F.trim(F.col("chunk_text"))) > 0)

    # Critical: limit parallel Ollama calls (one HTTP client storm per partition).
    parts = max(SPARK_EMBED_PARTITIONS, 1)
    logger.info(
        "Repartitioning to %d partition(s) before embedding "
        "(set SPARK_EMBED_PARTITIONS to tune Ollama concurrency).", parts,
    )
    df = df.repartition(parts)

    embed_udf = F.udf(embed_text, ArrayType(FloatType()))
    df = df.withColumn("embedding", embed_udf(F.col("chunk_text")))

    out_path = f"s3a://{BUCKET_PROCESSED}/chunks/"
    # Materialize once in memory/disk, then count + write — avoids a second Spark job
    # that re-reads all JSON from S3A (often slow or flaky vs. killing long embed work).
    df = df.persist(StorageLevel.MEMORY_AND_DISK)
    try:
        total = df.count()
        with_emb = df.filter(F.size(F.col("embedding")) > 0).count()
        logger.info(
            "Chunk rows: %d; with non-empty embedding: %d (missing: %d)",
            total, with_emb, total - with_emb,
        )
        # Gate before the overwrite below: a bad embed pass must not replace good chunks.
        check_embedding_coverage(total, with_emb)
        df.write.mode("overwrite").json(out_path)
        logger.info("Wrote to %s", out_path)
    finally:
        df.unpersist()

    # ── Training dataset export (StyleAdaptedLM-style SFT input) ──────────
    # Long essays → multiple JSON lines (see SFT_CHUNK_OUTPUT_CHARS). RAG chunk rows
    # above remain separate (~500-word embed chunks). coalesce(1) for one part file.
    train_lines_udf = F.udf(to_training_json_lines, ArrayType(StringType()))
    train_df = (
        docs_df.filter(F.col("doc_word_count") >= 50)
        .withColumn("_lines", train_lines_udf(F.col("title"), F.col("source"), F.col("text")))
        .filter(F.size(F.col("_lines")) > 0)
        .select(F.explode("_lines").alias("training_line"))
    )
    train_count = train_df.count()
    train_path = f"s3a://{BUCKET_PROCESSED}/training/dataset_jsonl"
    logger.info(
        "SFT export: SFT_CHUNK_OUTPUT_CHARS=%d "
        "(each label chunk ≤ this size; 0 = legacy single row + truncate)", SFT_CHUNK_OUTPUT_CHARS,
    )
    (
        train_df
        .coalesce(1)
        .write.mode("overwrite")
        .text(train_path)
    )
    logger.info("Wrote %d training rows to %s (one part-*.txt JSONL file)", train_count, train_path)


if __name__ == "__main__":
    main()
