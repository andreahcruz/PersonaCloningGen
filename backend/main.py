#!/usr/bin/env python3
"""FastAPI backend wrapping the RAG generation pipeline (generate.py) for the web frontend.

Run: uvicorn main:app --reload --port 8000
"""
from __future__ import annotations

import os
from pathlib import Path

import yaml
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from generate import (
    CONTEXT_MAX_CHARS,
    DEFAULT_EMBED_MODEL,
    DEFAULT_GEN_MODEL,
    OLLAMA_BASE_DEFAULT,
    build_prompt,
    format_context,
    load_format_spec,
    load_persona,
    ollama_embed,
    ollama_generate,
    retrieve,
    summarize_persona,
)

BASE = Path(__file__).resolve().parent.parent
FORMAT_SPECS_PATH = BASE / "formats" / "format_specs.yaml"
PERSONA_PATH = BASE / "data" / "persona_profile.json"
INDEX_PATH = BASE / "data" / "index"

app = FastAPI(title="Lemkin Content Generator API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class GenerateRequest(BaseModel):
    format: str
    topic: str
    audience: str
    goal: str
    cta: str = "none"
    k: int = Field(default=8, ge=1, le=20)


class GenerateResponse(BaseModel):
    draft: str


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/formats")
def list_formats() -> list[str]:
    if not FORMAT_SPECS_PATH.exists():
        raise HTTPException(500, f"Format specs not found: {FORMAT_SPECS_PATH}")
    with open(FORMAT_SPECS_PATH, encoding="utf-8") as f:
        return list(yaml.safe_load(f).keys())


@app.post("/generate", response_model=GenerateResponse)
def generate(req: GenerateRequest) -> GenerateResponse:
    if not PERSONA_PATH.exists():
        raise HTTPException(500, f"Persona profile not found: {PERSONA_PATH}")

    try:
        format_spec = load_format_spec(FORMAT_SPECS_PATH, req.format)
    except (KeyError, FileNotFoundError) as e:
        raise HTTPException(400, str(e)) from e

    ollama_base = os.environ.get("OLLAMA_BASE", OLLAMA_BASE_DEFAULT)
    embed_model = os.environ.get("EMBED_MODEL", DEFAULT_EMBED_MODEL)
    gen_model = os.environ.get("GEN_MODEL", DEFAULT_GEN_MODEL)

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

    try:
        draft = ollama_generate(prompt, ollama_base, gen_model)
    except Exception as e:
        raise HTTPException(502, f"Generation failed: {e}") from e

    return GenerateResponse(draft=draft)
