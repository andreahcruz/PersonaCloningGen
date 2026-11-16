"""Extract the Elon Musk source catalog and linked public pages into DuckDB.

The CSV is treated as a source registry.  Each CSV row is preserved in
``source_registry`` and each linked URL is fetched into ``source_documents``.
The complete response body is retained as a BLOB so downstream jobs can
re-parse a source without fetching it again.  Fetch failures are also stored,
which makes blocked/credentialed sources visible instead of silently dropping
them.

Run from the repository root::

    python persona_pipeline/jobs/elon_etl.py

The default output is ``persona_pipeline/data/elon_musk.duckdb``.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urljoin

import duckdb
import requests
from bs4 import BeautifulSoup


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CSV = ROOT / "Elon_Musk_Data_Sources.csv"
DEFAULT_DB = ROOT / "persona_pipeline" / "data" / "elon_musk.duckdb"
USER_AGENT = "PersonaPipeline/1.0 (research ETL; contact project owner)"


def read_catalog(path: Path) -> list[dict[str, str]]:
    """Read and validate the source catalog, preserving every CSV field."""
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    required = {"Type", "Source", "Link"}
    if not rows or not required.issubset(rows[0]):
        raise ValueError(f"{path} must contain columns: {', '.join(sorted(required))}")
    for row in rows:
        if not row.get("Link", "").strip():
            raise ValueError(f"Missing Link for source {row.get('Source', '<unknown>')}")
    return rows


def clean_text(html: bytes, content_type: str) -> tuple[str, str]:
    """Return a title and readable text for HTML/text responses."""
    decoded = html.decode("utf-8", errors="replace")
    if "html" not in content_type.lower():
        return "", re.sub(r"\s+", " ", decoded).strip()
    soup = BeautifulSoup(decoded, "html.parser")
    for tag in soup(["script", "style", "noscript", "svg"]):
        tag.decompose()
    title = soup.title.get_text(" ", strip=True) if soup.title else ""
    text = re.sub(r"\s+", " ", soup.get_text(" ", strip=True)).strip()
    return title[:1000], text


def extract_links(html: bytes, content_type: str, base_url: str) -> list[str]:
    if "html" not in content_type.lower():
        return []
    soup = BeautifulSoup(html.decode("utf-8", errors="replace"), "html.parser")
    links = {urljoin(base_url, tag["href"].split("#", 1)[0]) for tag in soup.find_all("a", href=True)}
    return sorted(link for link in links if link.startswith(("http://", "https://")))


def fetch(session: requests.Session, url: str, timeout: int) -> dict[str, Any]:
    fetched_at = datetime.now(timezone.utc).isoformat()
    try:
        response = session.get(url, timeout=timeout, allow_redirects=True)
        body = response.content
        content_type = response.headers.get("content-type", "")
        title, text = clean_text(body, content_type)
        return {
            "requested_url": url,
            "final_url": response.url,
            "status_code": response.status_code,
            "content_type": content_type,
            "title": title,
            "text": text,
            "raw_body": body,
            "content_sha256": hashlib.sha256(body).hexdigest(),
            "content_bytes": len(body),
            "extracted_links": json.dumps(extract_links(body, content_type, response.url)),
            "fetched_at": fetched_at,
            "error": None,
        }
    except requests.RequestException as exc:
        return {
            "requested_url": url,
            "final_url": None,
            "status_code": None,
            "content_type": None,
            "title": None,
            "text": None,
            "raw_body": None,
            "content_sha256": None,
            "content_bytes": 0,
            "extracted_links": "[]",
            "fetched_at": fetched_at,
            "error": f"{type(exc).__name__}: {exc}",
        }


def create_schema(con: duckdb.DuckDBPyConnection) -> None:
    con.execute("""
        CREATE TABLE IF NOT EXISTS source_registry (
            source_id INTEGER PRIMARY KEY,
            source_type VARCHAR,
            source_name VARCHAR,
            link VARCHAR,
            verification VARCHAR,
            what_it_is VARCHAR,
            license VARCHAR,
            caution VARCHAR,
            intended_use VARCHAR,
            catalog_json JSON,
            loaded_at TIMESTAMP
        )
    """)
    con.execute("""
        CREATE TABLE IF NOT EXISTS source_documents (
            source_id INTEGER PRIMARY KEY,
            requested_url VARCHAR,
            final_url VARCHAR,
            status_code INTEGER,
            content_type VARCHAR,
            title VARCHAR,
            extracted_text VARCHAR,
            raw_body BLOB,
            content_sha256 VARCHAR,
            content_bytes BIGINT,
            extracted_links JSON,
            fetched_at TIMESTAMP,
            error VARCHAR
        )
    """)


def run(csv_path: Path, db_path: Path, timeout: int, delay: float) -> None:
    rows = read_catalog(csv_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(db_path))
    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT, "Accept": "text/html,application/xhtml+xml,text/plain,application/json;q=0.9,*/*;q=0.8"})
    try:
        create_schema(con)
        loaded_at = datetime.now(timezone.utc).replace(tzinfo=None)
        for source_id, row in enumerate(rows, start=1):
            con.execute("""
                INSERT OR REPLACE INTO source_registry
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, [
                source_id, row.get("Type", ""), row.get("Source", ""), row.get("Link", ""),
                row.get("Verification", ""), row.get("What it is", ""), row.get("License", ""),
                row.get("Caution", ""), row.get("Use in our system", ""), json.dumps(row), loaded_at,
            ])
            result = fetch(session, row["Link"].strip(), timeout)
            con.execute("""
                INSERT OR REPLACE INTO source_documents
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, [
                source_id, result["requested_url"], result["final_url"], result["status_code"],
                result["content_type"], result["title"], result["text"], result["raw_body"],
                result["content_sha256"], result["content_bytes"], result["extracted_links"],
                result["fetched_at"], result["error"],
            ])
            status = result["status_code"] if result["status_code"] is not None else result["error"]
            print(f"[{source_id:02d}/{len(rows):02d}] {status} {row['Source']}")
            if delay:
                time.sleep(delay)
        con.execute("CREATE OR REPLACE VIEW successful_documents AS SELECT * FROM source_documents WHERE status_code BETWEEN 200 AND 299")
        con.execute("CREATE OR REPLACE VIEW fetch_summary AS SELECT status_code, COUNT(*) AS document_count FROM source_documents GROUP BY status_code ORDER BY status_code")
        print(f"\nLoaded {len(rows)} catalog rows into {db_path}")
        print(con.execute("SELECT * FROM fetch_summary").fetchall())
        print(con.execute("SELECT COUNT(*), COALESCE(SUM(content_bytes), 0) FROM successful_documents").fetchone())
    finally:
        con.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch Elon Musk source links into DuckDB")
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    parser.add_argument("--timeout", type=int, default=45)
    parser.add_argument("--delay", type=float, default=1.0)
    args = parser.parse_args()
    run(args.csv, args.db, args.timeout, args.delay)


if __name__ == "__main__":
    main()
