#!/usr/bin/env python3
"""
Streamlit UI for Lemkin-style B2B content generation.
Run: streamlit run user_interface.py
"""
from __future__ import annotations

import logging
import os
from pathlib import Path

import requests
import streamlit as st

from generate import (
    DEFAULT_EMBED_MODEL,
    DEFAULT_GEN_MODEL,
    OLLAMA_BASE_DEFAULT,
    LemkinError,
    check_dependencies,
    configure_logging,
    run_rag_pipeline,
)

configure_logging()
logger = logging.getLogger("lemkin.ui")

BASE = Path(__file__).parent

# Shown first when those tags exist in Ollama (see ``register_ollama.py`` / ``ollama create``).
_FT_MODEL_ORDER = (
    "llama3.1",
    "lemkin-cleaned",
    "lemkin-balanced",
    "lemkin-r32",
    "lemkin-clone",
    "lemkin-smoke20",
    "lemkin-lora-v3",
)


def _ollama_model_tags(base_url: str) -> list[str]:
    """Names returned by Ollama (e.g. ``lemkin-clone:latest``). Empty if unreachable."""
    url = (base_url or "").strip().rstrip("/")
    if not url:
        return []
    try:
        r = requests.get(f"{url}/api/tags", timeout=5)
        r.raise_for_status()
        out: list[str] = []
        for m in r.json().get("models") or []:
            n = (m.get("name") or "").strip()
            if n:
                out.append(n)
        return sorted(set(out))
    except Exception as e:  # noqa: BLE001 - the UI falls back to common names
        logger.warning("could not list Ollama models at %s: %s", url, e)
        return []


def _gen_model_options(ollama_base: str) -> tuple[list[str], bool]:
    """Build selectbox options + whether Ollama listing succeeded."""
    tags = _ollama_model_tags(ollama_base)
    if not tags:
        # Offline / wrong URL: still offer common names + Other
        return list(_FT_MODEL_ORDER) + ["Other..."], False

    def base_name(tag: str) -> str:
        return tag.split(":", 1)[0]

    seen: set[str] = set()
    ordered: list[str] = []

    def add(tag: str) -> None:
        if tag not in seen:
            ordered.append(tag)
            seen.add(tag)

    hint_bases = {base_name(t): t for t in tags}
    for hint in _FT_MODEL_ORDER:
        if hint in hint_bases:
            add(hint_bases[hint])
    for t in tags:
        add(t)
    add("Other...")
    return ordered, True


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
    return run_rag_pipeline(
        format_name=format_name,
        topic=topic,
        audience=audience,
        goal=goal,
        cta=cta,
        k=k,
        index_path=index_path,
        persona_path=persona_path,
        format_specs_path=format_specs_path,
        ollama_base=ollama_base,
        embed_model=embed_model,
        gen_model=gen_model,
    )


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

        gen_opts, ollama_ok = _gen_model_options(ollama_base)
        if ollama_ok:
            st.caption(f"Models from Ollama at `{ollama_base}` ({len(gen_opts) - 1} tags).")
        else:
            st.caption(
                "Could not list models from Ollama — showing common names. "
                "Check the URL and that `ollama serve` is running."
            )

        # Fine-tuned GGUFs are registered via ``python -m host_finetune.register_ollama``.
        # Options refresh each run from GET /api/tags (same as ``ollama list``).
        gen_model_choice = st.selectbox(
            "Generate model",
            gen_opts,
            index=0,
            help=(
                f"`{DEFAULT_GEN_MODEL}` = stock base with persona prompt in the RAG prompt. "
                "`lemkin-*` = merged LoRA GGUF you created with merge_and_export + register_ollama. "
                "Pick `Other...` for any name from `ollama list`."
            ),
        )
        if gen_model_choice == "Other...":
            gen_model = st.text_input(
                "Custom Ollama model",
                value=DEFAULT_GEN_MODEL,
                help="Any model name visible in `ollama list`.",
            )
        else:
            gen_model = gen_model_choice

    generate_btn = st.button("Generate", type="primary", use_container_width=True)

    with st.expander("System check"):
        st.caption("Checks Ollama, the selected models, Chroma and the local files.")
        run_check = st.button("Run system check", use_container_width=True)

index_path_resolved = BASE / index_path
persona_path = BASE / "data" / "persona_profile.json"
format_specs_path = BASE / "formats" / "format_specs.yaml"

if run_check:
    with st.spinner("Checking dependencies..."):
        check_rows = check_dependencies(
            ollama_base, embed_model, gen_model,
            persona_path, format_specs_path, index_path_resolved,
        )
    for check_name, check_ok, check_detail in check_rows:
        (st.success if check_ok else st.error)(f"{check_name}: {check_detail}")

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
            except LemkinError as e:
                # Already logged with a request id inside run_rag_pipeline.
                st.error(f"Generation failed: {e}")
                if e.hint:
                    st.info(e.hint)
            except Exception as e:  # noqa: BLE001 - last resort; full traceback is in the logs
                logger.exception("unexpected error in Streamlit generate handler")
                st.error(f"Unexpected error: {e}")
                st.info("See the app log for the traceback: `docker compose logs streamlit`.")
