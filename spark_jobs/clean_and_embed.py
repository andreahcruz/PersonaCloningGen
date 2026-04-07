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
import os
import random
import re
import time
from datetime import datetime

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

    print(f"Embedding failed after {OLLAMA_EMBED_MAX_RETRIES} tries: {last_err}")
    return []


def main():
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

    parse_date_udf = F.udf(parse_date, TimestampType())
    df = df.withColumn("date_ts", parse_date_udf(F.col("date")))

    df = df.dropDuplicates(["source", "title", "url", "text"])

    df = df.withColumn("essay_id", F.monotonically_increasing_id())

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
    print(
        f"Repartitioning to {parts} partition(s) before embedding "
        f"(set SPARK_EMBED_PARTITIONS to tune Ollama concurrency)."
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
        print(
            f"Chunk rows: {total}; with non-empty embedding: {with_emb} "
            f"(missing: {total - with_emb})"
        )
        df.write.mode("overwrite").json(out_path)
        print(f"Wrote to {out_path}")
    finally:
        df.unpersist()

    spark.stop()


if __name__ == "__main__":
    main()
