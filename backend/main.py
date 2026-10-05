#!/usr/bin/env python3
"""FastAPI backend wrapping the RAG generation pipeline (generate.py) for the web frontend.

Run: uvicorn main:app --reload --port 8000
"""
from __future__ import annotations

import os
import time
import uuid
from pathlib import Path
from typing import Any, Literal

import requests
import yaml
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from generate import (
    CONTEXT_MAX_CHARS,
    DEFAULT_EMBED_MODEL,
    DEFAULT_GEN_MODEL,
    OLLAMA_BASE_DEFAULT,
    build_prompt,
    check_dependencies,
    format_context,
    load_format_spec,
    load_persona,
    ollama_embed,
    ollama_generate,
    retrieve,
    summarize_persona,
)

BASE = Path(__file__).resolve().parent.parent
FORMAT_SPECS_PATH = Path(os.environ.get("FORMAT_SPECS_PATH", BASE / "formats" / "format_specs.yaml"))
# Airflow writes this generated copy during a deploy. The tracked profile remains untouched.
PERSONA_PATH = Path(os.environ.get("PERSONA_PATH", BASE / "data" / "persona_profile.json"))
INDEX_PATH = Path(os.environ.get("INDEX_PATH", BASE / "data" / "index"))

app = FastAPI(title="Lemkin Content Generator API")


class GenerateRequest(BaseModel):
    format: str
    topic: str
    audience: str
    goal: str
    cta: str = "none"
    k: int = Field(default=8, ge=1, le=20)
    method: Literal["personarag", "qlora"] = "personarag"


class GenerateResponse(BaseModel):
    draft: str
    sources: list[dict[str, Any]]
    model: str
    latency_ms: int
    request_id: str
    method: Literal["personarag", "qlora"]


def _settings() -> tuple[str, str, str, str]:
    return (
        os.environ.get("OLLAMA_BASE", OLLAMA_BASE_DEFAULT),
        os.environ.get("EMBED_MODEL", DEFAULT_EMBED_MODEL),
        os.environ.get("GEN_MODEL", DEFAULT_GEN_MODEL),
        os.environ.get("QLORA_MODEL", "lemkin-qlora"),
    )


# Training rows are one user sentence. The Modelfile wraps this sentence in the
# Llama 3.1 Instruct chat template, so the application prompt stays plain text.
DIRECT_QLORA_PROMPTS = {
    "linkedin_post": "Write a LinkedIn post in the style of Jason Lemkin about: {topic}",
    "blog_draft": "Write a blog post in the style of Jason Lemkin about: {topic}",
    "x_thread": "Write an X post in the style of Jason Lemkin about: {topic}",
    "youtube_script": "Write a talk in the style of Jason Lemkin about: {topic}",
}
DIRECT_QLORA_MAX_TOKENS = {
    "linkedin_post": 320,
    "x_thread": 160,
    "blog_draft": 768,
    "youtube_script": 768,
}
DIRECT_QLORA_OPTIONS = {
    "temperature": 0.7,
    "top_p": 0.9,
    "repeat_penalty": 1.15,
    "stop": ["<|eot_id|>"],
}
# Negative keep_alive asks Ollama to leave the merged GGUF loaded after the reply.
DIRECT_QLORA_KEEP_ALIVE = -1


def direct_qlora_prompt(format_name: str, topic: str) -> str:
    """One training-style sentence. No brief, persona, format rules, or retrieved text."""
    template = DIRECT_QLORA_PROMPTS.get(format_name)
    if template is None:
        known = ", ".join(DIRECT_QLORA_PROMPTS)
        raise HTTPException(400, f"Unknown format for direct QLoRA: {format_name}. Expected: {known}")
    return template.format(topic=topic)


def _source_type(value: object) -> str:
    source = str(value or "unknown").lower()
    if source.startswith("youtube"):
        return "youtube"
    return source if source in {"blog", "linkedin", "x"} else "unknown"


def _source_payload(chunks: list[dict]) -> list[dict[str, Any]]:
    sources: list[dict[str, Any]] = []
    for index, chunk in enumerate(chunks):
        sources.append(
            {
                "id": str(chunk.get("id") or chunk.get("doc_id") or f"chunk-{index + 1}"),
                "text": str(chunk.get("chunk_text") or ""),
                "score": None,
                "source_type": _source_type(chunk.get("source")),
                "title": chunk.get("title"),
                "url": chunk.get("url"),
                "date": chunk.get("date"),
            }
        )
    return sources


@app.get("/health")
def health() -> dict:
    ollama_base, embed_model, gen_model, qlora_model = _settings()
    checks = check_dependencies(
        ollama_base, embed_model, gen_model, PERSONA_PATH, FORMAT_SPECS_PATH, INDEX_PATH
    )
    try:
        tags = {(model.get("name") or "") for model in requests.get(
            f"{ollama_base.rstrip('/')}/api/tags", timeout=5
        ).json().get("models") or []}
        qlora_ready = qlora_model in tags or f"{qlora_model}:latest" in tags
        checks.append(("qlora model", qlora_ready, qlora_model if qlora_ready else f"'{qlora_model}' not in `ollama list`"))
    except Exception as e:  # noqa: BLE001 - health must always be a response
        checks.append(("qlora model", False, f"cannot reach Ollama: {e}"))
    return {
        "status": "ok" if all(ok for _, ok, _ in checks) else "degraded",
        "model": gen_model,
        "services": [
            {"name": name, "status": "ok" if ok else "down", "detail": detail}
            for name, ok, detail in checks
        ],
    }


@app.get("/health/dependencies")
def health_dependencies() -> dict:
    return health()


@app.get("/formats")
def list_formats() -> list[str]:
    if not FORMAT_SPECS_PATH.exists():
        raise HTTPException(500, f"Format specs not found: {FORMAT_SPECS_PATH}")
    with open(FORMAT_SPECS_PATH, encoding="utf-8") as f:
        return list(yaml.safe_load(f).keys())


@app.post("/generate", response_model=GenerateResponse)
def generate(req: GenerateRequest) -> GenerateResponse:
    if req.method == "personarag" and not PERSONA_PATH.exists():
        raise HTTPException(500, f"Persona profile not found: {PERSONA_PATH}")

    try:
        format_spec = load_format_spec(FORMAT_SPECS_PATH, req.format)
    except (KeyError, FileNotFoundError) as e:
        raise HTTPException(400, str(e)) from e

    ollama_base, embed_model, gen_model, qlora_model = _settings()

    started = time.perf_counter()
    request_id = uuid.uuid4().hex
    chunks: list[dict] = []
    if req.method == "personarag":
        profile = load_persona(PERSONA_PATH)
        persona_summary = summarize_persona(profile)
        query_text = f"{req.topic}. {req.goal}. Audience: {req.audience}."
        try:
            query_embed = ollama_embed(query_text, ollama_base, embed_model)
            chunks = retrieve(INDEX_PATH, query_embed, req.k, embed_model)
        except Exception as e:
            raise HTTPException(502, f"Retrieval failed: {e}") from e
        context_block = format_context(chunks, max_chars=CONTEXT_MAX_CHARS)
        prompt = build_prompt(
            format_spec, persona_summary, req.topic, req.audience, req.goal, req.cta, context_block
        )
        model = gen_model
    else:
        prompt = direct_qlora_prompt(req.format, req.topic)
        model = qlora_model

    try:
        if req.method == "qlora":
            draft = ollama_generate(
                prompt,
                ollama_base,
                model,
                max_tokens=DIRECT_QLORA_MAX_TOKENS[req.format],
                options=DIRECT_QLORA_OPTIONS,
                keep_alive=DIRECT_QLORA_KEEP_ALIVE,
            )
        else:
            draft = ollama_generate(
                prompt,
                ollama_base,
                model,
                max_tokens=280,
            )
    except Exception as e:
        raise HTTPException(502, f"Generation failed: {e}") from e

    return GenerateResponse(
        draft=draft,
        sources=_source_payload(chunks),
        model=model,
        latency_ms=round((time.perf_counter() - started) * 1000),
        request_id=request_id,
        method=req.method,
    )
