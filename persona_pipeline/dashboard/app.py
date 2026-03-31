"""
Phase 11 — Streamlit dashboard with two tabs:
  Tab 1: Pipeline Status (metrics, charts, sample posts)
  Tab 2: Content Generator (topic input, format selector, generate button)

Run: streamlit run persona_pipeline/dashboard/app.py
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

import streamlit as st
import pandas as pd

from persona_pipeline.config.config import DB_PATH, PERSONA_PROFILE_PATH, TRAINING_DIR

st.set_page_config(page_title="Jason Lemkin Persona Pipeline", layout="wide")
st.title("Jason Lemkin Persona Pipeline Dashboard")


# ── Helpers ───────────────────────────────────────────────────────────

@st.cache_resource
def get_duckdb_connection():
    import duckdb
    if not os.path.exists(DB_PATH):
        return None
    return duckdb.connect(DB_PATH, read_only=True)


def load_persona_profile() -> dict | None:
    if not os.path.exists(PERSONA_PROFILE_PATH):
        return None
    with open(PERSONA_PROFILE_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


# ── Tabs ──────────────────────────────────────────────────────────────

tab1, tab2 = st.tabs(["Pipeline Status", "Content Generator"])

# ── Tab 1: Pipeline Status ───────────────────────────────────────────

with tab1:
    con = get_duckdb_connection()

    if con is None:
        st.warning("DuckDB not found. Run the pipeline first (`spark_process.py` then `load_duckdb.py`).")
    else:
        # Top-level metrics
        try:
            total = con.execute("SELECT count(*) FROM all_posts").fetchone()[0]
            b2b = con.execute("SELECT count(*) FROM all_posts WHERE is_b2b = true").fetchone()[0]
            sources = con.execute("SELECT count(DISTINCT source) FROM all_posts").fetchone()[0]
            avg_wc = con.execute("SELECT ROUND(AVG(word_count),1) FROM all_posts").fetchone()[0]

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Total Posts", total)
            c2.metric("B2B Posts", b2b)
            c3.metric("Sources", sources)
            c4.metric("Avg Word Count", avg_wc)
        except Exception as e:
            st.error(f"Could not load metrics: {e}")

        st.subheader("Posts per Source")
        try:
            df_source = con.execute(
                "SELECT source, count(*) AS count FROM all_posts GROUP BY source ORDER BY count DESC"
            ).fetchdf()
            st.bar_chart(df_source.set_index("source"))
        except Exception as e:
            st.error(f"Chart error: {e}")

        st.subheader("Style Features per Source")
        try:
            df_style = con.execute("SELECT * FROM persona_style_profile").fetchdf()
            st.dataframe(df_style, use_container_width=True)
        except Exception as e:
            st.error(f"Style profile error: {e}")

        st.subheader("Top Quality Posts (sample)")
        try:
            df_top = con.execute(
                "SELECT source, word_count, SUBSTRING(cleaned_text, 1, 200) AS snippet FROM top_quality_posts LIMIT 10"
            ).fetchdf()
            st.dataframe(df_top, use_container_width=True)
        except Exception as e:
            st.error(f"Top posts error: {e}")

        # Training data status
        st.subheader("Training Data Export")
        comp_path = Path(str(TRAINING_DIR)) / "completion_format.jsonl"
        instr_path = Path(str(TRAINING_DIR)) / "instruction_format.jsonl"
        if comp_path.exists():
            n_lines = sum(1 for _ in open(comp_path, encoding="utf-8"))
            st.success(f"completion_format.jsonl: {n_lines} records")
        else:
            st.info("completion_format.jsonl not yet exported.")
        if instr_path.exists():
            n_lines = sum(1 for _ in open(instr_path, encoding="utf-8"))
            st.success(f"instruction_format.jsonl: {n_lines} records")
        else:
            st.info("instruction_format.jsonl not yet exported.")


# ── Tab 2: Content Generator ─────────────────────────────────────────

with tab2:
    profile = load_persona_profile()

    if profile:
        with st.expander("Persona Profile Stats"):
            style = profile.get("style", {})
            st.json({
                "avg_word_count": style.get("avg_word_count"),
                "avg_sentence_length": style.get("avg_sentence_length"),
                "vocabulary_richness": style.get("vocabulary_richness"),
                "formality_score": style.get("formality_score"),
                "questions_per_post": style.get("questions_per_post"),
                "top_10_characteristic_words": style.get("characteristic_words", [])[:10],
            })
    else:
        st.info("Persona profile not built yet. Run `build_persona_profile.py` first.")

    topic = st.text_input("Topic", value="Why every SaaS founder should hire two sales reps at the same time")
    format_type = st.selectbox("Format", ["linkedin_post", "blog_post"])

    if st.button("Generate"):
        with st.spinner("Generating content..."):
            try:
                from persona_pipeline.inference.generate import generate_content

                text = generate_content(topic, format_type)
                st.subheader("Generated Output")
                st.markdown(text)
            except Exception as e:
                st.error(
                    f"Generation failed: {e}\n\n"
                    "This is expected if the LoRA model hasn't been trained yet. "
                    "Run `finetune_lora.py` on a GPU first."
                )

        # Show retrieved chunks
        if profile:
            st.subheader("Retrieved Context (PersonaRAG)")
            try:
                from persona_pipeline.jobs.persona_rag import persona_conditioned_retrieve

                chunks = persona_conditioned_retrieve(topic, profile, n_results=5)
                for i, chunk in enumerate(chunks, 1):
                    with st.expander(f"Chunk {i} (alignment={chunk['persona_alignment']:.3f})"):
                        st.text(chunk["text"][:500])
            except Exception as e:
                st.warning(f"Could not retrieve chunks: {e}")


if __name__ == "__main__":
    pass
