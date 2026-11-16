"""Download the two usable Elon source datasets and load normalized DuckDB tables.

Zenodo's record description advertises CSV attachments, but its public API
currently exposes only the PDF report.  The catalog already includes the
small public ``dawood/elon-tweets`` mirror, so it is used as the tweet source
with explicit provenance.  Lex transcripts are downloaded from the listed
Hugging Face Parquet file and filtered to episodes whose title starts with
``Elon Musk:``.
"""

from __future__ import annotations

import csv
import io
import tempfile
from pathlib import Path

import duckdb
import requests


ROOT = Path(__file__).resolve().parents[2]
DB_PATH = ROOT / "persona_pipeline" / "data" / "elon_musk.duckdb"
TWEETS_URL = "https://huggingface.co/datasets/dawood/elon-tweets/resolve/main/elon_tweets.csv"
TRANSCRIPTS_URL = "https://huggingface.co/datasets/Aditya0619/lex-fridman-podcast/resolve/main/data/train-00000-of-00001.parquet"
TWEETS_SOURCE_URL = "https://huggingface.co/datasets/dawood/elon-tweets"
TRANSCRIPTS_SOURCE_URL = "https://huggingface.co/datasets/Aditya0619/lex-fridman-podcast"


def download(url: str, path: Path) -> None:
    response = requests.get(url, timeout=120)
    response.raise_for_status()
    path.write_bytes(response.content)
    print(f"Downloaded {len(response.content):,} bytes from {url}")


def load_tweets(con: duckdb.DuckDBPyConnection, path: Path) -> int:
    rows = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            text = (row.get("Cleaned_Tweets") or row.get("Tweets") or "").strip()
            if not text:
                continue
            rows.append([
                row.get("Date") or None,
                text,
                int(row.get("Likes") or 0),
                int(row.get("Retweets") or 0),
                TWEETS_SOURCE_URL,
            ])
    con.execute("DROP TABLE IF EXISTS elon_tweets")
    con.execute("""
        CREATE TABLE elon_tweets (
            tweet_date TIMESTAMP,
            text VARCHAR,
            likes BIGINT,
            retweets BIGINT,
            source_url VARCHAR
        )
    """)
    con.executemany("INSERT INTO elon_tweets VALUES (?, ?, ?, ?, ?)", rows)
    return len(rows)


def load_interviews(con: duckdb.DuckDBPyConnection, path: Path) -> int:
    con.execute("DROP TABLE IF EXISTS elon_interviews")
    con.execute("""
        CREATE TABLE elon_interviews AS
        SELECT
            title,
            transcript,
            ? AS source_url
        FROM read_parquet(?)
        WHERE title ILIKE 'Elon Musk:%'
    """, [TRANSCRIPTS_SOURCE_URL, str(path)])
    return con.execute("SELECT COUNT(*) FROM elon_interviews").fetchone()[0]


def main() -> None:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(DB_PATH))
    try:
        with tempfile.TemporaryDirectory(prefix="elon_sources_") as tmp:
            tmp_path = Path(tmp)
            tweets_path = tmp_path / "elon_tweets.csv"
            transcripts_path = tmp_path / "lex_transcripts.parquet"
            download(TWEETS_URL, tweets_path)
            download(TRANSCRIPTS_URL, transcripts_path)
            tweet_count = load_tweets(con, tweets_path)
            interview_count = load_interviews(con, transcripts_path)
        con.execute("CREATE OR REPLACE VIEW elon_rag_documents AS SELECT 'tweet' AS source_type, text, source_url, NULL::VARCHAR AS title FROM elon_tweets UNION ALL SELECT 'interview' AS source_type, transcript AS text, source_url, title FROM elon_interviews")
        print(f"Loaded {tweet_count:,} tweets into elon_tweets")
        print(f"Loaded {interview_count} Musk interview episodes into elon_interviews")
        print(f"RAG documents available: {con.execute('SELECT COUNT(*) FROM elon_rag_documents').fetchone()[0]:,}")
    finally:
        con.close()


if __name__ == "__main__":
    main()
