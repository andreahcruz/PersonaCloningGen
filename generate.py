#!/usr/bin/env python3
"""
Generate PG-style content from a format (linkedin_post, blog_draft), content brief,
and RAG retrieval. Uses Ollama /api/embed and /api/generate.

  python generate.py --format linkedin_post --topic "..." --audience "..." --goal "..." --cta "..." --k 8 --out outputs/pg_linkedin_001.md
  python generate.py --format blog_draft --topic "..." --audience "..." --goal "..." --k 10 --out outputs/pg_blog_001.md
  python generate.py --format x_thread --topic "..." --audience "..." --goal "..." --k 8 --out outputs/thread_001.md
  python generate.py --format youtube_script --topic "..." --audience "..." --goal "..." --k 10 --out outputs/script_001.md
"""
import argparse
import json
import os
import sys
from pathlib import Path

import requests
import yaml
import chromadb
from chromadb.config import DEFAULT_DATABASE, DEFAULT_TENANT, Settings
from chromadb.errors import NotFoundError


OLLAMA_BASE_DEFAULT = "http://localhost:11434"
COLLECTION_NAME = os.environ.get("CHROMA_COLLECTION_NAME", "lemkin_content")
DEFAULT_EMBED_MODEL = "nomic-embed-text"
DEFAULT_GEN_MODEL = "llama3.1"
CONTEXT_MAX_CHARS = 5000


def load_format_spec(path: Path, format_name: str) -> dict:
    with open(path, encoding="utf-8") as f:
        specs = yaml.safe_load(f)
    if format_name not in specs:
        raise KeyError(f"Unknown format: {format_name}. Available: {list(specs.keys())}")
    return specs[format_name]


def load_persona(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def summarize_persona(profile: dict) -> str:
    vocab = profile.get("vocabulary_tendencies", {})
    structure = profile.get("structure_patterns", {})
    framing = profile.get("favorite_framing", {})
    themes = profile.get("favorite_themes", {})
    meta = profile.get("meta", {})
    top_words = ", ".join(vocab.get("top_50_content_words", [])[:15])
    starters = ", ".join(structure.get("common_sentence_starters_bigrams", [])[:8])
    framing_phrases = ", ".join(framing.get("favorite_framing_phrases", [])[:10])
    theme_list = themes.get("top_content_words_and_bigrams", [])[:10]
    theme_examples = ", ".join(str(t) for t in theme_list) if theme_list else "B2B SaaS, GTM, fundraising"
    source_hint = meta.get("source", "Jason Lemkin — operator voice")
    lines = [
        f"You are writing in the persona of {source_hint}.",
        "Emulate: direct, experienced B2B / SaaS operator tone; concrete metrics and examples.",
        f"Vocabulary tendencies: {top_words}.",
        f"Sentence openers to echo (sparingly): {starters}.",
        f"Framing phrases: {framing_phrases}.",
        f"Themes to lean on: {theme_examples}.",
        "Favor clarity and specificity over generic hype.",
    ]
    return "\n".join(lines)


def format_spec_to_prompt(spec: dict) -> str:
    lines = ["[FORMAT RULES]"]
    lines.append(f"Output type: {spec.get('output_type', 'markdown')}.")
    if "length_target_words" in spec:
        lo, hi = spec["length_target_words"][0], spec["length_target_words"][1]
        lines.append(f"Target length: {lo}–{hi} words.")
    if "structure" in spec:
        lines.append("Structure:")
        for item in spec["structure"]:
            if isinstance(item, dict):
                for k, v in item.items():
                    lines.append(f"  - {k}: {v}")
            else:
                lines.append(f"  - {item}")
    if "banned" in spec:
        lines.append("Avoid: " + "; ".join(spec["banned"]))
    if "style" in spec:
        lines.append("Style: " + "; ".join(spec["style"]))
    return "\n".join(lines)


def ollama_embed(text: str, base_url: str, model: str) -> list[float]:
    url = f"{base_url.rstrip('/')}/api/embed"
    resp = requests.post(url, json={"model": model, "input": text}, timeout=60)
    resp.raise_for_status()
    return resp.json()["embeddings"][0]


def ollama_generate(prompt: str, base_url: str, model: str) -> str:
    url = f"{base_url.rstrip('/')}/api/generate"
    resp = requests.post(
        url,
        json={"model": model, "prompt": prompt, "stream": False},
        timeout=300,
    )
    resp.raise_for_status()
    return resp.json().get("response", "").strip()


def retrieve(index_path: Path, query_embedding: list[float], k: int, embed_model: str) -> list[dict]:
    use_http = os.environ.get("CHROMA_USE_HTTP", "").lower() in ("1", "true", "yes")
    if use_http:
        host = os.environ.get("CHROMA_HOST", "localhost")
        port = int(os.environ.get("CHROMA_PORT", "8000"))
        tenant = os.environ.get("CHROMA_TENANT", DEFAULT_TENANT)
        database = os.environ.get("CHROMA_DATABASE", DEFAULT_DATABASE)
        client = chromadb.HttpClient(
            host=host, port=port, tenant=tenant, database=database
        )
    else:
        client = chromadb.PersistentClient(path=str(index_path), settings=Settings(anonymized_telemetry=False))
    try:
        collection = client.get_collection(COLLECTION_NAME)
    except NotFoundError as err:
        try:
            existing = [c.name for c in client.list_collections()]
        except Exception:
            existing = []
        hint = (
            f"Chroma has no collection {COLLECTION_NAME!r}. "
            "In Airflow, run the DAG through **load_to_chroma** (MinIO → Chroma). "
            "If you removed the **chroma-data** Docker volume or upgraded Chroma, reload once."
        )
        if existing:
            hint += f" Existing collections: {existing}."
        raise RuntimeError(hint) from err
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=k,
        include=["documents", "metadatas"],
    )
    docs = results["documents"][0] if results["documents"] else []
    metas = results["metadatas"][0] if results["metadatas"] else []
    return [{"chunk_text": d, **m} for d, m in zip(docs, metas)]


def format_context(chunks: list[dict], max_chars: int = CONTEXT_MAX_CHARS) -> str:
    lines = ["[CONTEXT — EXCERPTS FROM JASON LEMKIN CORPUS]"]
    total = 0
    for i, ch in enumerate(chunks):
        text = str(ch.get("chunk_text", "")).strip()
        if not text:
            continue
        doc_id = ch.get("doc_id", "?")
        header = f"[Chunk {i+1} — doc_id={doc_id}]"
        if total + len(header) + len(text) + 2 > max_chars:
            remaining = max_chars - total - len(header) - 5
            if remaining <= 0:
                break
            text = text[:remaining] + "..."
        lines.append(header)
        lines.append(text)
        total += len(header) + len(text) + 2
        if total >= max_chars:
            break
    return "\n".join(lines)


def build_prompt(
    format_spec: dict,
    persona_summary: str,
    topic: str,
    audience: str,
    goal: str,
    cta: str,
    context_block: str,
) -> str:
    brief = "\n".join([
        "[CONTENT BRIEF]",
        f"Topic: {topic}",
        f"Audience: {audience}",
        f"Goal: {goal}",
        f"Call to action: {cta}",
    ])
    format_block = format_spec_to_prompt(format_spec)
    return "\n\n".join([
        "[PERSONA]",
        persona_summary,
        format_block,
        brief,
        context_block,
        "[INSTRUCTIONS]",
        "Write a single draft that matches the format rules and content brief. "
        "Use the context excerpts for tone and ideas; do not copy long verbatim passages. "
        "Output only the draft (markdown), with no preamble or commentary.",
        "[OUTPUT]",
    ])


def main():
    p = argparse.ArgumentParser(description="Generate Lemkin-style B2B content via RAG + Ollama.")
    p.add_argument("--format", required=True, help="Format name (e.g. linkedin_post, blog_draft)")
    p.add_argument("--topic", required=True, help="Topic or title")
    p.add_argument("--audience", required=True, help="Target audience")
    p.add_argument("--goal", required=True, help="Goal of the piece")
    p.add_argument("--cta", default="none", help="Call to action (default: none)")
    p.add_argument("--k", type=int, default=8, help="Number of chunks to retrieve (default: 8)")
    p.add_argument("--out", required=True, help="Output markdown file path")
    p.add_argument("--index", default="data/index", help="Chroma index directory (default: data/index)")
    p.add_argument("--persona", default="data/persona_profile.json", help="Persona JSON path")
    p.add_argument("--format-specs", default="formats/format_specs.yaml", help="Format specs YAML path")
    p.add_argument("--ollama-base", default=OLLAMA_BASE_DEFAULT, help="Ollama base URL")
    p.add_argument("--embed-model", default=DEFAULT_EMBED_MODEL, help="Ollama embed model")
    p.add_argument("--gen-model", default=DEFAULT_GEN_MODEL, help="Ollama generate model")
    args = p.parse_args()

    base = Path(".")
    format_specs_path = base / args.format_specs
    persona_path = base / args.persona
    index_path = base / args.index
    out_path = base / args.out

    if not format_specs_path.exists():
        print(f"Error: format specs not found: {format_specs_path}", file=sys.stderr)
        sys.exit(1)
    if not persona_path.exists():
        print(f"Error: persona profile not found: {persona_path}", file=sys.stderr)
        sys.exit(1)
    use_http = os.environ.get("CHROMA_USE_HTTP", "").lower() in ("1", "true", "yes")
    if not use_http and not index_path.exists():
        print(f"Error: index not found: {index_path}. Run build_index.py first.", file=sys.stderr)
        sys.exit(1)

    try:
        format_spec = load_format_spec(format_specs_path, args.format)
    except KeyError as e:
        print(e, file=sys.stderr)
        sys.exit(1)

    profile = load_persona(persona_path)
    persona_summary = summarize_persona(profile)

    query_text = f"{args.topic}. {args.goal}. Audience: {args.audience}."
    print("Embedding query...")
    query_embed = ollama_embed(query_text, args.ollama_base, args.embed_model)
    print(f"Retrieving top {args.k} chunks...")
    chunks = retrieve(index_path, query_embed, args.k, args.embed_model)
    context_block = format_context(chunks)

    prompt = build_prompt(
        format_spec, persona_summary,
        args.topic, args.audience, args.goal, args.cta,
        context_block,
    )
    print(f"Generating with {args.gen_model}...")
    draft = ollama_generate(prompt, args.ollama_base, args.gen_model)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(draft, encoding="utf-8")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
