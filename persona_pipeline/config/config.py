"""
Central configuration for the persona pipeline.

All settings in one place so every script shares consistent paths, limits,
and constants.
"""

from __future__ import annotations

from pathlib import Path

# ── Persona identity ──────────────────────────────────────────────────
PERSONA_USERNAME = "patio11"
HN_USERNAME = "patio11"

# ── Base URLs ─────────────────────────────────────────────────────────
BLOG_BASE_URL = "https://www.kalzumeus.com"

# ── Collection limits ─────────────────────────────────────────────────
REDDIT_LIMIT = 1000
HN_LIMIT = 1000
BLOG_LIMIT = 100

# ── Thresholds ────────────────────────────────────────────────────────
MIN_WORD_COUNT = 50

# ── Collector throttles (polite crawling) ─────────────────────────────
BLOG_DELAY_SECONDS = 1.5
HN_ITEM_DELAY_SECONDS = 0.1
REDDIT_PAGE_DELAY_SECONDS = 2.0
REDDIT_429_RETRY_SECONDS = 60

# ── Request headers ───────────────────────────────────────────────────
BLOG_USER_AGENT = "Mozilla/5.0 (research project)"
REDDIT_USER_AGENT = "Mozilla/5.0 (persona-research-project/1.0)"

# ── Directory layout (resolved relative to this file) ─────────────────
PERSONA_PIPELINE_DIR = Path(__file__).resolve().parents[1]
RAW_DIR = PERSONA_PIPELINE_DIR / "data" / "raw"
PROCESSED_DIR = PERSONA_PIPELINE_DIR / "data" / "processed"
TRAINING_DIR = PERSONA_PIPELINE_DIR / "data" / "training"
DB_PATH = str(PERSONA_PIPELINE_DIR / "data" / "persona_db.duckdb")
CHROMA_PATH = str(PERSONA_PIPELINE_DIR / "data" / "chroma_patio11")
PERSONA_PROFILE_PATH = str(PERSONA_PIPELINE_DIR / "data" / "persona_profile_patio11.json")

# ── B2B keyword list ──────────────────────────────────────────────────
B2B_KEYWORDS = [
    "saas", "pricing", "sales", "marketing", "customer", "revenue",
    "business", "software", "startup", "product", "conversion",
    "churn", "growth", "b2b", "enterprise", "founder", "strategy",
    "email", "landing page",
]

# ── Spark settings ────────────────────────────────────────────────────
SPARK_APP_NAME = "Patio11PersonaETL"
SPARK_MASTER = "local[*]"

# ── Model / training settings ─────────────────────────────────────────
MODEL_NAME = "meta-llama/Meta-Llama-3.1-8B-Instruct"
LORA_OUTPUT = str(PERSONA_PIPELINE_DIR.parent / "models" / "patio11_lora")

# ── Airflow ───────────────────────────────────────────────────────────
SCHEDULE = "@weekly"


def persona_raw_path(filename: str) -> str:
    return str(RAW_DIR / filename)


if __name__ == "__main__":
    print("PERSONA_PIPELINE_DIR:", PERSONA_PIPELINE_DIR)
    print("RAW_DIR:", RAW_DIR)
    print("PROCESSED_DIR:", PROCESSED_DIR)
    print("DB_PATH:", DB_PATH)
    print("CHROMA_PATH:", CHROMA_PATH)

