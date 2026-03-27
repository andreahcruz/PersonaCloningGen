"""
Phase 4 — Build a structured JSON persona profile ("fingerprint") of patio11's
writing from the DuckDB analytics tables.

Run: python persona_pipeline/jobs/build_persona_profile.py
"""
from __future__ import annotations

import json
import os
import re
import sys
import warnings
from collections import Counter
from pathlib import Path

REPO_ROOT = str(Path(__file__).resolve().parents[2])
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import duckdb

from persona_pipeline.config.config import DB_PATH, PERSONA_PROFILE_PATH


STOP_WORDS = {
    "about", "after", "also", "been", "before", "being", "between", "both",
    "could", "does", "doing", "during", "each", "every", "from", "have",
    "having", "here", "into", "just", "like", "make", "many", "more", "most",
    "much", "only", "other", "over", "really", "some", "such", "than", "that",
    "their", "them", "then", "there", "these", "they", "thing", "things",
    "this", "those", "through", "very", "want", "well", "were", "what",
    "when", "where", "which", "while", "will", "with", "would", "your",
    "actually", "going", "because", "people", "think", "should", "would",
    "could", "getting", "something", "still", "right", "know",
}


def _top_words(con: duckdb.DuckDBPyConnection, n: int = 50) -> list[str]:
    rows = con.execute("SELECT cleaned_text FROM all_posts").fetchall()
    counter: Counter[str] = Counter()
    for (text,) in rows:
        if not text:
            continue
        words = re.findall(r"[a-z]{5,}", text.lower())
        counter.update(words)
    for sw in STOP_WORDS:
        counter.pop(sw, None)
    return [w for w, _ in counter.most_common(n)]


def _top_posts(con: duckdb.DuckDBPyConnection, n: int = 5) -> list[str]:
    rows = con.execute(
        """
        SELECT cleaned_text
        FROM all_posts
        WHERE word_count > 100
        ORDER BY (COALESCE(score, 1) * word_count) DESC
        LIMIT ?
        """,
        [n],
    ).fetchall()
    return [text for (text,) in rows if text]


def build_profile(con: duckdb.DuckDBPyConnection) -> dict:
    stats = con.execute("""
        SELECT
            AVG(word_count)          AS avg_word_count,
            AVG(avg_sentence_length) AS avg_sentence_length,
            AVG(vocabulary_richness) AS avg_vocabulary_richness,
            AVG(formality_score)     AS avg_formality_score,
            AVG(question_count)      AS avg_questions_per_post
        FROM all_posts
    """).fetchone()

    avg_wc, avg_sl, avg_vr, avg_fs, avg_qp = stats

    char_words = _top_words(con, 50)
    samples = _top_posts(con, 5)

    profile = {
        "name": "patio11",
        "real_name": "Patrick McKenzie",
        "domain": "B2B SaaS, software pricing, salary negotiation, marketing",
        "style": {
            "avg_word_count": round(float(avg_wc or 0), 1),
            "avg_sentence_length": round(float(avg_sl or 0), 1),
            "vocabulary_richness": round(float(avg_vr or 0), 4),
            "formality_score": round(float(avg_fs or 0), 4),
            "questions_per_post": round(float(avg_qp or 0), 2),
            "characteristic_words": char_words,
        },
        "rhetorical_patterns": [
            "Uses concrete numbers and percentages to back opinions",
            "Challenges conventional wisdom directly",
            "Draws from personal B2B software experience",
            "Explains counterintuitive business insights step by step",
            "Favors directness over hedging — states opinions as facts",
            "Often starts with a surprising or contrarian claim",
        ],
        "core_topics": [
            "SaaS pricing strategy",
            "Salary negotiation",
            "Software marketing and positioning",
            "B2B sales tactics",
            "Technical founder advice",
            "Business automation and leverage",
        ],
        "opinions": {
            "pricing": "Most software is dramatically underpriced",
            "sales": "Engineers can and should learn to sell",
            "marketing": "Content that teaches converts better than content that promotes",
            "hiring": "Salary negotiation is a learnable skill companies exploit",
        },
        "sample_writing": samples,
    }
    return profile


def main() -> None:
    if not os.path.exists(DB_PATH):
        warnings.warn(f"DuckDB not found at {DB_PATH}; run load_duckdb.py first.")
        return

    con = duckdb.connect(DB_PATH, read_only=True)
    try:
        profile = build_profile(con)
    finally:
        con.close()

    os.makedirs(os.path.dirname(PERSONA_PROFILE_PATH), exist_ok=True)
    with open(PERSONA_PROFILE_PATH, "w", encoding="utf-8") as f:
        json.dump(profile, f, ensure_ascii=False, indent=2)

    print(f"Persona profile written -> {PERSONA_PROFILE_PATH}")
    print(f"  characteristic_words (top 10): {profile['style']['characteristic_words'][:10]}")
    print(f"  avg_word_count: {profile['style']['avg_word_count']}")
    print(f"  sample_writing count: {len(profile['sample_writing'])}")


if __name__ == "__main__":
    main()
