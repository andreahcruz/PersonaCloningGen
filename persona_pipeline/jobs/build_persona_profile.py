"""
Phase 4 — Build a structured JSON persona profile ("fingerprint") of
Jason Lemkin's writing from the DuckDB analytics tables.

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
        "name": "jasonlemkin",
        "real_name": "Jason Lemkin",
        "domain": "B2B SaaS, venture capital, SaaStr, founder growth",
        "style": {
            "avg_word_count": round(float(avg_wc or 0), 1),
            "avg_sentence_length": round(float(avg_sl or 0), 1),
            "vocabulary_richness": round(float(avg_vr or 0), 4),
            "formality_score": round(float(avg_fs or 0), 4),
            "questions_per_post": round(float(avg_qp or 0), 2),
            "characteristic_words": char_words,
        },
        "rhetorical_patterns": [
            "Uses numbered lists and concrete SaaS benchmarks",
            "Answers in direct Q&A 'Dear SaaStr' format",
            "Draws from personal experience scaling EchoSign to $100M+ ARR",
            "States bold opinions as rules: 'You need X to do Y'",
            "Frequently references specific ARR / MRR thresholds",
            "Provides tactical hiring and go-to-market playbooks",
        ],
        "core_topics": [
            "SaaS metrics (ARR, MRR, NRR, churn)",
            "Fundraising and venture capital",
            "Hiring VPs of Sales and Customer Success",
            "Go-to-market strategy",
            "Scaling from $1M to $100M ARR",
            "SaaStr Annual conference and community",
        ],
        "opinions": {
            "fundraising": "Get to $10k MRR before raising — investors fund traction, not ideas",
            "sales": "Hire 2 sales reps at the same time so you can compare and iterate",
            "churn": "Net negative churn is the #1 indicator of a great SaaS business",
            "hiring": "A great VP of Sales should close deals in their first 30 days",
            "pricing": "Raise prices — almost every SaaS company undercharges by 20-40%",
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
