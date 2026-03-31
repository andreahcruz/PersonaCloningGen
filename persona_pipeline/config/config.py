"""
Central configuration for the persona pipeline.

All settings in one place so every script shares consistent paths, limits,
and constants.
"""

from __future__ import annotations

import os
from pathlib import Path

# ── Persona identity ──────────────────────────────────────────────────
PERSONA_USERNAME = "jasonlemkin"

# ── Base URLs ─────────────────────────────────────────────────────────
BLOG_BASE_URL = "https://www.saastr.com"
BLOG_AUTHOR_SLUG = "jasonlkn"

# ── YouTube channels ─────────────────────────────────────────────────
LEMKIN_YT_CHANNEL = "https://www.youtube.com/@Jasonlk/videos"
SAASTR_YT_CHANNEL = "https://www.youtube.com/@Saastr/videos"
YT_MAX_VIDEOS = 200

# ── Collection limits ─────────────────────────────────────────────────
BLOG_LIMIT = 100

# ── Thresholds ────────────────────────────────────────────────────────
MIN_WORD_COUNT = 50

# ── Collector throttles (polite crawling) ─────────────────────────────
BLOG_DELAY_SECONDS = 1.5
YT_DELAY_SECONDS = 2.0

# ── Request headers ───────────────────────────────────────────────────
BLOG_USER_AGENT = "Mozilla/5.0 (research project)"

# ── Directory layout (resolved relative to this file) ─────────────────
PERSONA_PIPELINE_DIR = Path(__file__).resolve().parents[1]
RAW_DIR = PERSONA_PIPELINE_DIR / "data" / "raw"
PROCESSED_DIR = PERSONA_PIPELINE_DIR / "data" / "processed"
TRAINING_DIR = PERSONA_PIPELINE_DIR / "data" / "training"
DB_PATH = str(PERSONA_PIPELINE_DIR / "data" / "persona_db.duckdb")
CHROMA_PATH = str(PERSONA_PIPELINE_DIR / "data" / "chroma_lemkin")
PERSONA_PROFILE_PATH = str(PERSONA_PIPELINE_DIR / "data" / "persona_profile_lemkin.json")

# ── YouTube cookies (auto-detect at repo root, override with env var) ─
_default_cookies = PERSONA_PIPELINE_DIR.parent / "cookies.txt"
YT_COOKIES_PATH: str | None = os.environ.get(
    "YT_COOKIES_PATH",
    str(_default_cookies) if _default_cookies.exists() else None,
)

# ── B2B keyword list ──────────────────────────────────────────────────
B2B_KEYWORDS = [
    "saas", "pricing", "sales", "marketing", "customer", "revenue",
    "business", "software", "startup", "product", "conversion",
    "churn", "growth", "b2b", "enterprise", "founder", "strategy",
    "email", "landing page", "arr", "mrr", "fundraising", "venture",
    "go-to-market", "gtm", "pipeline", "quota", "vp of sales",
]

# ── Spark settings ────────────────────────────────────────────────────
SPARK_APP_NAME = "LemkinPersonaETL"
SPARK_MASTER = "local[*]"

# ── Model / training settings ─────────────────────────────────────────
MODEL_NAME = "meta-llama/Meta-Llama-3.1-8B-Instruct"
LORA_OUTPUT = str(PERSONA_PIPELINE_DIR.parent / "models" / "lemkin_lora")

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
