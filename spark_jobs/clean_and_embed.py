"""
PySpark job: reads raw PG essays from MinIO (pg-raw), applies six transforms,
writes cleaned + embedded chunks to MinIO (pg-processed).

Transforms:
  1. Strip HTML tags
  2. Parse date → timestamp
  3. Deduplicate rows
  4. Chunk text into ~500-word segments
  5. Add chunk_index
  6. Generate nomic-embed-text vectors via Ollama
"""
import os
import re
from datetime import datetime

import requests as req
from bs4 import BeautifulSoup
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
BUCKET_RAW = os.environ.get("MINIO_BUCKET_RAW", "pg-raw")
BUCKET_PROCESSED = os.environ.get("MINIO_BUCKET_PROCESSED", "pg-processed")
OLLAMA_BASE = os.environ.get("OLLAMA_BASE", "http://host.docker.internal:11434")
EMBED_MODEL = "nomic-embed-text"
CHUNK_WORD_LIMIT = 500


def strip_html(text):
    if not text:
        return text
    return BeautifulSoup(text, "html.parser").get_text(separator=" ")


def parse_date(date_str):
    if not date_str:
        return None
    for fmt in ("%B %Y", "%Y", "%B %d, %Y", "%b %Y", "%m/%Y"):
        try:
            return datetime.strptime(date_str.strip(), fmt)
        except (ValueError, TypeError):
            continue
    digits = re.search(r"\d{4}", str(date_str))
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
    try:
        resp = req.post(
            f"{OLLAMA_BASE}/api/embed",
            json={"model": EMBED_MODEL, "input": text},
            timeout=120,
        )
        resp.raise_for_status()
        return resp.json()["embeddings"][0]
    except Exception as e:
        print(f"Embedding failed: {e}")
        return []


def main():
    spark = (
        SparkSession.builder
        .appName("PG Essays — Clean, Chunk, Embed")
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
    ])

    df = spark.read.schema(raw_schema).json(f"s3a://{BUCKET_RAW}/essay_*.json")

    strip_html_udf = F.udf(strip_html, StringType())
    df = df.withColumn("text", strip_html_udf(F.col("text")))

    parse_date_udf = F.udf(parse_date, TimestampType())
    df = df.withColumn("date_ts", parse_date_udf(F.col("date")))

    df = df.dropDuplicates(["title", "text"])

    df = df.withColumn("essay_id", F.monotonically_increasing_id())

    chunk_udf = F.udf(chunk_text, ArrayType(StringType()))
    df = df.withColumn("chunks", chunk_udf(F.col("text")))

    df = df.select(
        "essay_id", "title", "date", "date_ts", "url",
        F.posexplode("chunks").alias("chunk_index", "chunk_text"),
    )

    df = df.withColumn("word_count", F.size(F.split(F.col("chunk_text"), r"\s+")))

    embed_udf = F.udf(embed_text, ArrayType(FloatType()))
    df = df.withColumn("embedding", embed_udf(F.col("chunk_text")))

    df.write.mode("overwrite").json(f"s3a://{BUCKET_PROCESSED}/chunks/")

    print(f"Wrote {df.count()} chunks to s3a://{BUCKET_PROCESSED}/chunks/")
    spark.stop()


if __name__ == "__main__":
    main()
