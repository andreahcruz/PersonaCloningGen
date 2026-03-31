"""
Phase 3 — Load Parquet files into DuckDB, create analytics views,
and export training data as JSONL.

Run: python persona_pipeline/jobs/load_duckdb.py
"""
from __future__ import annotations

import json
import os
import sys
import warnings
from pathlib import Path

REPO_ROOT = str(Path(__file__).resolve().parents[2])
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import duckdb

from persona_pipeline.config.config import DB_PATH, PROCESSED_DIR, TRAINING_DIR


def _parquet_path(name: str) -> str:
    return str(PROCESSED_DIR / name)


def load_tables(con: duckdb.DuckDBPyConnection) -> None:
    all_path = _parquet_path("all_posts.parquet")
    b2b_path = _parquet_path("b2b_posts.parquet")

    if not os.path.exists(all_path):
        warnings.warn(f"all_posts.parquet not found at {all_path}; skipping table load.")
        return

    con.execute(
        f"CREATE OR REPLACE TABLE all_posts AS SELECT * FROM read_parquet('{all_path}/*.parquet')"
    )
    print(f"Loaded all_posts: {con.execute('SELECT count(*) FROM all_posts').fetchone()[0]} rows")

    if os.path.exists(b2b_path):
        con.execute(
            f"CREATE OR REPLACE TABLE b2b_posts AS SELECT * FROM read_parquet('{b2b_path}/*.parquet')"
        )
        print(f"Loaded b2b_posts: {con.execute('SELECT count(*) FROM b2b_posts').fetchone()[0]} rows")


def create_views(con: duckdb.DuckDBPyConnection) -> None:
    con.execute("""
        CREATE OR REPLACE VIEW persona_style_profile AS
        SELECT
            source,
            COUNT(*)               AS post_count,
            AVG(word_count)        AS avg_word_count,
            AVG(avg_sentence_length) AS avg_sentence_length,
            AVG(vocabulary_richness) AS avg_vocabulary_richness,
            AVG(formality_score)   AS avg_formality
        FROM all_posts
        GROUP BY source
    """)

    con.execute("""
        CREATE OR REPLACE VIEW top_quality_posts AS
        SELECT *
        FROM all_posts
        WHERE word_count > 100
        ORDER BY (COALESCE(score, 1) * word_count) DESC
        LIMIT 500
    """)

    con.execute("""
        CREATE OR REPLACE VIEW b2b_source_breakdown AS
        SELECT
            source,
            COUNT(*)            AS b2b_count,
            AVG(word_count)     AS avg_word_count,
            AVG(formality_score) AS avg_formality
        FROM all_posts
        WHERE is_b2b = true
        GROUP BY source
    """)

    print("\nAnalytics views created: persona_style_profile, top_quality_posts, b2b_source_breakdown")
    print("\npersona_style_profile:")
    for row in con.execute("SELECT * FROM persona_style_profile").fetchall():
        print("  ", row)


def export_training_data(con: duckdb.DuckDBPyConnection) -> None:
    os.makedirs(str(TRAINING_DIR), exist_ok=True)

    # completion_format.jsonl
    rows = con.execute("SELECT cleaned_text, source FROM all_posts").fetchall()
    comp_path = str(TRAINING_DIR / "completion_format.jsonl")
    with open(comp_path, "w", encoding="utf-8") as f:
        for cleaned_text, _ in rows:
            f.write(json.dumps({"text": cleaned_text}, ensure_ascii=False) + "\n")
    print(f"\nExported {len(rows)} rows -> {comp_path}")

    # instruction_format.jsonl
    INSTRUCTION_BY_SOURCE = {
        "blog": "Write a long-form B2B blog post about SaaS growth strategy",
        "podcast": "Write a detailed talk about B2B SaaS metrics and founder advice",
        "saastr_video": "Write a SaaStr keynote about scaling SaaS companies",
    }
    instr_path = str(TRAINING_DIR / "instruction_format.jsonl")
    with open(instr_path, "w", encoding="utf-8") as f:
        for cleaned_text, source in rows:
            instruction = INSTRUCTION_BY_SOURCE.get(
                source, "Write about B2B software and business strategy"
            )
            record = {
                "instruction": instruction,
                "input": "",
                "output": cleaned_text,
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"Exported {len(rows)} rows -> {instr_path}")


def main() -> None:
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    con = duckdb.connect(DB_PATH)

    try:
        load_tables(con)
        create_views(con)
        export_training_data(con)
    finally:
        con.close()

    print(f"\nDuckDB database: {DB_PATH}")


if __name__ == "__main__":
    main()
