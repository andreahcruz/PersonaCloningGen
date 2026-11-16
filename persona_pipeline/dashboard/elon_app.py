"""Streamlit chat UI for the Elon Musk RAG pipeline.

Run from the repository root:
    streamlit run persona_pipeline/dashboard/elon_app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
JOBS = ROOT / "persona_pipeline" / "jobs"
if str(JOBS) not in sys.path:
    sys.path.insert(0, str(JOBS))

from elon_chat import build_prompt, generate  # noqa: E402
from elon_rag import retrieve  # noqa: E402


st.set_page_config(page_title="Elon Musk RAG", page_icon="🚀", layout="wide")
st.title("🚀 Elon Musk RAG")
st.caption("Conversational answers grounded in indexed Elon Musk tweets and Lex Fridman interviews.")

if "messages" not in st.session_state:
    st.session_state.messages = []

with st.sidebar:
    st.header("Settings")
    top_k = st.slider("Retrieved sources", min_value=2, max_value=8, value=5)
    model = st.text_input("Ollama model", value="llama3.1")
    ollama_base = st.text_input("Ollama URL", value="http://localhost:11434")
    if st.button("Clear conversation"):
        st.session_state.messages = []
        st.rerun()
    st.divider()
    st.write("Indexed data")
    st.write("• 388 tweets")
    st.write("• 5 Musk interview episodes")
    st.write("• 754 retrieval chunks")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

question = st.chat_input("Ask about Elon Musk's ideas, engineering, AI, or companies…")
if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"):
        with st.spinner("Retrieving context and generating answer…"):
            try:
                contexts = retrieve(question, top_k)
                if not contexts:
                    raise RuntimeError("No retrieval context found. Build the RAG index first.")
                prompt = build_prompt(question, contexts)
                answer = generate(prompt, ollama_base, model)
                st.markdown(answer)
                with st.expander("Retrieved context"):
                    for i, item in enumerate(contexts, start=1):
                        meta = item["metadata"]
                        st.markdown(
                            f"**Source {i}:** {meta.get('source_name', 'unknown')}  \n"
                            f"Distance: `{item['distance']:.4f}`"
                        )
                        st.write(item["text"][:1200])
                st.session_state.messages.append({"role": "assistant", "content": answer})
            except Exception as exc:
                error = f"Unable to generate an answer: {exc}"
                st.error(error)
                st.session_state.messages.append({"role": "assistant", "content": error})
