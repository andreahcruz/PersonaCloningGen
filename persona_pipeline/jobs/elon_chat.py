"""Conversational interface for the Elon Musk RAG index.

Requires Ollama running locally, for example::

    ollama serve
    python3 persona_pipeline/jobs/elon_chat.py "What is Elon Musk's approach to AI?"
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import requests

from elon_rag import retrieve


OLLAMA_BASE = os.environ.get("OLLAMA_BASE", "http://localhost:11434")
GEN_MODEL = os.environ.get("OLLAMA_GEN_MODEL", "llama3.1")
GEN_TIMEOUT = float(os.environ.get("OLLAMA_GENERATE_TIMEOUT", "900"))


def build_prompt(question: str, contexts: list[dict]) -> str:
    excerpts = []
    sources = []
    for i, item in enumerate(contexts, start=1):
        meta = item["metadata"]
        source = meta.get("source_name", meta.get("source_type", "unknown"))
        excerpts.append(f"[Source {i}: {source}]\n{item['text']}")
        sources.append(f"{i}. {source} — {meta.get('url', '')}")
    return f"""You are a careful conversational assistant answering questions using a retrieval index about Elon Musk.

Answer the user's question naturally and directly. Synthesize the retrieved material; do not pretend Elon Musk said something unless the context supports it. Distinguish reported context from interpretation, mention uncertainty when appropriate, and do not imitate or impersonate Elon Musk. Do not make up facts. If the context is insufficient, say so. Keep the answer concise but useful. Cite evidence inline as [Source 1], [Source 2], etc.

User question:
{question}

Retrieved context:
{'\n\n'.join(excerpts)}

After the answer, add a short 'Sources' section listing only the source names and URLs below:
{'\n'.join(sources)}
"""


def generate(prompt: str, base_url: str, model: str) -> str:
    response = requests.post(
        f"{base_url.rstrip('/')}/api/generate",
        json={"model": model, "prompt": prompt, "stream": False},
        timeout=GEN_TIMEOUT,
    )
    response.raise_for_status()
    answer = response.json().get("response", "").strip()
    if not answer:
        raise RuntimeError("Ollama returned an empty response.")
    return answer


def main() -> None:
    parser = argparse.ArgumentParser(description="Ask the Elon Musk RAG a conversational question")
    parser.add_argument("question", nargs="+")
    parser.add_argument("-k", type=int, default=5)
    parser.add_argument("--ollama-base", default=OLLAMA_BASE)
    parser.add_argument("--model", default=GEN_MODEL)
    parser.add_argument("--history", help="Optional JSON file containing prior {role, content} messages")
    args = parser.parse_args()
    question = " ".join(args.question).strip()
    contexts = retrieve(question, args.k)
    if not contexts:
        raise RuntimeError("No RAG context found. Build the index first.")
    prompt = build_prompt(question, contexts)
    if args.history:
        history = json.loads(Path(args.history).read_text(encoding="utf-8"))
        history_text = "\n".join(f"{m['role']}: {m['content']}" for m in history[-6:])
        prompt = f"Conversation so far:\n{history_text}\n\n{prompt}"
    print(generate(prompt, args.ollama_base, args.model))


if __name__ == "__main__":
    main()
