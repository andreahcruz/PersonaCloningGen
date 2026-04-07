#!/usr/bin/env python3
"""
Streamlit UI for Lemkin-style B2B content generation.
Run: streamlit run user_interface.py
"""
import os
from pathlib import Path

import streamlit as st

from generate import (
    build_prompt,
    format_context,
    load_format_spec,
    load_persona,
    ollama_embed,
    ollama_generate,
    retrieve,
    summarize_persona,
    CONTEXT_MAX_CHARS,
    DEFAULT_EMBED_MODEL,
    DEFAULT_GEN_MODEL,
    OLLAMA_BASE_DEFAULT,
)

BASE = Path(__file__).parent


def run_generation(
    format_name: str,
    topic: str,
    audience: str,
    goal: str,
    cta: str,
    k: int,
    index_path: Path,
    persona_path: Path,
    format_specs_path: Path,
    ollama_base: str,
    embed_model: str,
    gen_model: str,
) -> str:
    format_spec = load_format_spec(format_specs_path, format_name)
    profile = load_persona(persona_path)
    persona_summary = summarize_persona(profile)
    query_text = f"{topic}. {goal}. Audience: {audience}."
    query_embed = ollama_embed(query_text, ollama_base, embed_model)
    chunks = retrieve(index_path, query_embed, k, embed_model)
    context_block = format_context(chunks, max_chars=CONTEXT_MAX_CHARS)
    prompt = build_prompt(
        format_spec, persona_summary,
        topic, audience, goal, cta,
        context_block,
    )
    return ollama_generate(prompt, ollama_base, gen_model)


use_chroma_http = os.environ.get("CHROMA_USE_HTTP", "").lower() in ("1", "true", "yes")
default_ollama = os.environ.get("OLLAMA_BASE", OLLAMA_BASE_DEFAULT)

st.set_page_config(page_title="Lemkin-style Generator", layout="wide")
st.title("Jason Lemkin–style content generator")
st.caption(
    "RAG over your Lemkin corpus (blog, LinkedIn, X, YouTube) in Chroma + Ollama. "
    "Formats control output shape only — retrieval uses the full collection."
)

with st.sidebar:
    st.header("Content brief")
    format_name = st.selectbox(
        "Format",
        [
            "linkedin_post",
            "blog_draft",
            "x_thread",
            "youtube_script",
        ],
        help=(
            "linkedin_post / blog_draft: classic long-form social + article. "
            "x_thread: 3–7 short posts (thread). "
            "youtube_script: spoken outline + sections (~450–900 words)."
        ),
    )
    topic = st.text_input(
        "Topic",
        placeholder="e.g. Hiring a VP Sales in 2026",
        help="Subject or working title.",
    )
    audience = st.text_input(
        "Audience",
        placeholder="e.g. B2B founders at $5–20M ARR",
    )
    goal = st.text_input(
        "Goal",
        placeholder="e.g. share one sharp takeaway",
    )
    cta = st.text_input(
        "Call to action",
        value="none",
        placeholder="e.g. invite comments",
    )
    k = st.slider(
        "Chunks to retrieve",
        min_value=4,
        max_value=20,
        value=8,
        help="More chunks = more context, slower retrieval.",
    )

    with st.expander("Advanced"):
        index_path = st.text_input(
            "Local Chroma path (ignored when CHROMA_USE_HTTP is set)",
            value="data/index",
            help="PersistentClient path for local dev only.",
        )
        ollama_base = st.text_input("Ollama base URL", value=default_ollama)
        embed_model = st.text_input("Embed model", value=DEFAULT_EMBED_MODEL)
        gen_model = st.text_input("Generate model", value=DEFAULT_GEN_MODEL)

    generate_btn = st.button("Generate", type="primary", use_container_width=True)

index_path_resolved = BASE / index_path
persona_path = BASE / "data" / "persona_profile.json"
format_specs_path = BASE / "formats" / "format_specs.yaml"

if generate_btn:
    if not topic.strip():
        st.error("Please enter a topic.")
    elif not audience.strip():
        st.error("Please enter an audience.")
    elif not goal.strip():
        st.error("Please enter a goal.")
    elif not format_specs_path.exists():
        st.error(f"Format specs not found: {format_specs_path}")
    elif not persona_path.exists():
        st.error(
            f"Persona profile not found: {persona_path}. "
            "Run the Airflow DAG through extract_persona (or copy persona_profile.json into ./data)."
        )
    elif not use_chroma_http and not index_path_resolved.exists():
        st.error(
            f"Index not found at {index_path_resolved}. "
            "Set CHROMA_USE_HTTP=true for Docker, or build a local Chroma index."
        )
    else:
        with st.spinner("Embedding query, retrieving chunks, generating..."):
            try:
                draft = run_generation(
                    format_name=format_name,
                    topic=topic.strip(),
                    audience=audience.strip(),
                    goal=goal.strip(),
                    cta=cta.strip() or "none",
                    k=k,
                    index_path=index_path_resolved,
                    persona_path=persona_path,
                    format_specs_path=format_specs_path,
                    ollama_base=ollama_base,
                    embed_model=embed_model,
                    gen_model=gen_model,
                )
                st.success("Done!")
                st.subheader("Output")
                st.markdown(draft)
                st.download_button(
                    label="Download as .md",
                    data=draft,
                    file_name=f"lemkin_{format_name}_{topic[:30].replace(' ', '_')}.md",
                    mime="text/markdown",
                    use_container_width=True,
                )
            except Exception as e:
                st.error(f"Generation failed: {e}")
                if "Connection" in str(e) or "refused" in str(e).lower():
                    st.info("Make sure Ollama is running: ollama serve")
                raise
