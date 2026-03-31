"""
Phase 2 — PySpark ETL: load raw JSON collectors output, normalize schemas,
clean text, extract style features, and save as Parquet.

Run: python persona_pipeline/jobs/spark_process.py
"""
from __future__ import annotations

import html as html_lib
import os
import re
import sys
import warnings
from pathlib import Path

REPO_ROOT = str(Path(__file__).resolve().parents[2])
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F
from pyspark.sql.types import (
    BooleanType,
    FloatType,
    IntegerType,
    StringType,
    StructField,
    StructType,
)

from persona_pipeline.config.config import (
    B2B_KEYWORDS,
    MIN_WORD_COUNT,
    PROCESSED_DIR,
    RAW_DIR,
    SPARK_APP_NAME,
    SPARK_MASTER,
)

# ── Schema that every source will be normalized to ────────────────────
UNIFIED_SCHEMA = [
    "id", "source", "url", "text", "title", "word_count", "score", "author",
]


# ── UDFs (plain Python — registered with Spark below) ────────────────

def _clean_text(text: str | None) -> str:
    if not text:
        return ""
    text = html_lib.unescape(text)
    text = re.sub(r"https?://\S+", "", text)
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)  # [text](url)->text
    text = re.sub(r"(\*\*|##|~~|`|__)", " ", text)
    text = re.sub(r"&\w+;", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _word_count(text: str | None) -> int:
    if not text:
        return 0
    return len(text.split())


def _avg_sentence_length(text: str | None) -> float:
    if not text:
        return 0.0
    sentences = [s.strip() for s in text.split(".") if s.strip()]
    if not sentences:
        return 0.0
    return sum(len(s.split()) for s in sentences) / len(sentences)


def _vocabulary_richness(text: str | None) -> float:
    if not text:
        return 0.0
    words = text.lower().split()
    if not words:
        return 0.0
    return len(set(words)) / len(words)


def _question_count(text: str | None) -> int:
    if not text:
        return 0
    return text.count("?")


def _formality_score(text: str | None) -> float:
    if not text:
        return 0.0
    words = text.split()
    if not words:
        return 0.0
    return sum(1 for w in words if len(w) >= 7) / len(words)


def _is_b2b(text: str | None) -> bool:
    if not text:
        return False
    lower = text.lower()
    return any(kw in lower for kw in B2B_KEYWORDS)


def _instruction_format(text: str | None, title: str | None) -> str:
    topic = title if title and title.strip() else "B2B SaaS and founder growth strategy"
    body = text or ""
    return (
        f"### Instruction:\nWrite in the style of Jason Lemkin about: {topic}\n\n"
        f"### Response:\n{body}"
    )


# ── Helpers ───────────────────────────────────────────────────────────

def _load_raw_json(spark: SparkSession, filename: str) -> DataFrame | None:
    path = str(RAW_DIR / filename)
    if not os.path.exists(path):
        warnings.warn(f"Raw file not found, skipping: {path}")
        return None
    try:
        df = spark.read.option("multiLine", True).json(path)
        if df.rdd.isEmpty():
            warnings.warn(f"Empty data in {path}, skipping.")
            return None
        return df
    except Exception as exc:
        warnings.warn(f"Failed to load {path}: {exc}")
        return None


def _normalize(df: DataFrame, source_name: str) -> DataFrame:
    """Ensure all UNIFIED_SCHEMA columns exist, filling missing ones with nulls."""
    for col_name in UNIFIED_SCHEMA:
        if col_name not in df.columns:
            df = df.withColumn(col_name, F.lit(None).cast(StringType()))
    # Cast id to string for uniformity across sources
    df = df.withColumn("id", F.col("id").cast(StringType()))
    df = df.withColumn("score", F.col("score").cast(IntegerType()))
    return df.select(*UNIFIED_SCHEMA)


# ── Main ETL ──────────────────────────────────────────────────────────

def run_etl() -> None:
    spark = (
        SparkSession.builder
        .appName(SPARK_APP_NAME)
        .master(SPARK_MASTER)
        .config("spark.driver.memory", "4g")
        .config("spark.sql.shuffle.partitions", "4")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")

    # 1. Register UDFs
    clean_udf = F.udf(_clean_text, StringType())
    wc_udf = F.udf(_word_count, IntegerType())
    asl_udf = F.udf(_avg_sentence_length, FloatType())
    vr_udf = F.udf(_vocabulary_richness, FloatType())
    qc_udf = F.udf(_question_count, IntegerType())
    fs_udf = F.udf(_formality_score, FloatType())
    b2b_udf = F.udf(_is_b2b, BooleanType())
    instr_udf = F.udf(_instruction_format, StringType())

    # 2. Load raw data
    sources = {
        "lemkin_blog.json": "blog",
        "lemkin_yt.json": "podcast",
        "saastr_yt.json": "saastr_video",
    }

    dfs: list[DataFrame] = []
    for filename, src in sources.items():
        df = _load_raw_json(spark, filename)
        if df is not None:
            dfs.append(_normalize(df, src))

    if not dfs:
        warnings.warn("No raw data loaded — nothing to process.")
        spark.stop()
        return

    # 3. Union
    combined = dfs[0]
    for df in dfs[1:]:
        combined = combined.unionByName(df)

    # 4. Clean text + recompute word_count
    combined = (
        combined
        .withColumn("cleaned_text", clean_udf(F.col("text")))
        .withColumn("word_count", wc_udf(F.col("cleaned_text")))
    )

    # 5. Filter short posts + deduplicate
    combined = combined.filter(F.col("word_count") >= MIN_WORD_COUNT)
    combined = combined.dropDuplicates(["cleaned_text"])

    # 6. Feature extraction
    combined = (
        combined
        .withColumn("avg_sentence_length", asl_udf(F.col("cleaned_text")))
        .withColumn("vocabulary_richness", vr_udf(F.col("cleaned_text")))
        .withColumn("question_count", qc_udf(F.col("cleaned_text")))
        .withColumn("formality_score", fs_udf(F.col("cleaned_text")))
        .withColumn("is_b2b", b2b_udf(F.col("cleaned_text")))
        .withColumn("instruction_format", instr_udf(F.col("cleaned_text"), F.col("title")))
    )

    # 7. Save Parquet
    os.makedirs(str(PROCESSED_DIR), exist_ok=True)
    all_path = str(PROCESSED_DIR / "all_posts.parquet")
    b2b_path = str(PROCESSED_DIR / "b2b_posts.parquet")

    combined.write.mode("overwrite").parquet(all_path)

    b2b_df = combined.filter(F.col("is_b2b") == True)  # noqa: E712
    b2b_df.write.mode("overwrite").parquet(b2b_path)

    # 8. Summary stats
    print("\n=== Spark ETL Summary ===")
    print(f"Total posts: {combined.count()}")
    print(f"B2B posts:   {b2b_df.count()}")
    print("\nPosts per source:")
    combined.groupBy("source").count().show()
    print("Average style features:")
    combined.select(
        F.avg("avg_sentence_length").alias("avg_sent_len"),
        F.avg("vocabulary_richness").alias("avg_vocab_rich"),
        F.avg("formality_score").alias("avg_formality"),
        F.avg("question_count").alias("avg_questions"),
    ).show()

    print(f"Saved all_posts.parquet -> {all_path}")
    print(f"Saved b2b_posts.parquet -> {b2b_path}")

    spark.stop()


if __name__ == "__main__":
    run_etl()
