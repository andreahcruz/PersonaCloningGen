#!/usr/bin/env python3
"""FastAPI backend wrapping the RAG generation pipeline (generate.py) for the web frontend.

Run: uvicorn main:app --reload --port 8000
"""
from __future__ import annotations

import json
import os
import re
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
    ConfigError,
    DEFAULT_EMBED_MODEL,
    DEFAULT_GEN_MODEL,
    OLLAMA_BASE_DEFAULT,
    build_prompt,
    check_dependencies,
    format_context,
    load_format_spec,
    load_persona,
    ollama_chat,
    ollama_embed,
    ollama_generate,
    retrieve,
    summarize_persona,
)
try:  # Package import in tests; top-level import in the slim backend container.
    from backend.model_f import ModelFStatus, model_f_status
except ModuleNotFoundError:  # pragma: no cover - exercised by the container build
    from model_f import ModelFStatus, model_f_status

BASE = Path(__file__).resolve().parent.parent
FORMAT_SPECS_PATH = Path(os.environ.get("FORMAT_SPECS_PATH", BASE / "formats" / "format_specs.yaml"))
# Airflow writes this generated copy during a deploy. The tracked profile remains untouched.
PERSONA_PATH = Path(os.environ.get("PERSONA_PATH", BASE / "data" / "persona_profile.json"))
INDEX_PATH = Path(os.environ.get("INDEX_PATH", BASE / "data" / "index"))
EVALUATION_RESULTS_PATH = Path(
    os.environ.get("EVALUATION_RESULTS_PATH", BASE / "data" / "evaluation" / "batch_results.json")
)
LANGUAGETOOL_BASE = os.environ.get("LANGUAGETOOL_BASE", "http://languagetool:8081")
LANGUAGETOOL_TIMEOUT = float(os.environ.get("LANGUAGETOOL_TIMEOUT", "30"))

app = FastAPI(title="Lemkin Content Generator API")


class GenerateRequest(BaseModel):
    format: str
    topic: str
    audience: str
    goal: str
    cta: str = "none"
    k: int = Field(default=8, ge=1, le=20)
    method: Literal["baseline", "personarag", "qlora", "model_f"] = "personarag"
    # The visible application continues to use dense retrieval. These fields
    # reserve the Model F contract without changing existing browser requests.
    retrieval_backend: Literal["dense", "lightrag"] = "dense"
    model_f_context: Literal["off", "source", "relationships", "descriptions", "multihop"] = "off"


class GenerateResponse(BaseModel):
    draft: str
    sources: list[dict[str, Any]]
    model: str
    latency_ms: int
    request_id: str
    method: Literal["baseline", "personarag", "qlora", "model_f"]


class PersonaResponse(BaseModel):
    """The active persona profile, exposed for read-only frontend display."""

    name: str
    source: str
    summary: str


class GrammarCheckRequest(BaseModel):
    """Completed draft text submitted only when a user requests screening."""

    text: str = Field(min_length=1, max_length=12_000)


class GrammarIssue(BaseModel):
    message: str
    matched_text: str
    offset: int
    length: int
    rule_id: str
    replacements: list[str]


class GrammarCheckResponse(BaseModel):
    word_count: int
    issue_count: int
    issues_per_100_words: float
    issues: list[GrammarIssue]


class ModelFStatusResponse(BaseModel):
    enabled: bool
    ready: bool
    mode: str
    context_profile: str
    detail: str


# These are the conservative, published filters from the PUK-28 LanguageTool
# check.  They deliberately avoid treating stylistic repetition and informal
# Lemkin phrasing as grammar mistakes.  We do not surface spell-check-only
# flags because company names and SaaS terminology produced too many false
# positives in the batch audit.
GRAMMAR_CATEGORIES = {"GRAMMAR", "TYPOS", "CONFUSED_WORDS"}
GRAMMAR_RULES = {"ENGLISH_WORD_REPEAT_RULE", "EN_A_VS_AN"}
IGNORED_GRAMMAR_RULES = {
    "PHRASE_REPETITION",
    "THE_SUPERLATIVE",
    "CA_BRAND_NEW",
    "ANYMORE_ADVERB",
    "THESE_ONES",
    "A_INFINITIVE",
    "BEEN_PART_AGREEMENT",
}


def _settings() -> tuple[str, str, str, str]:
    return (
        os.environ.get("OLLAMA_BASE", OLLAMA_BASE_DEFAULT),
        os.environ.get("EMBED_MODEL", DEFAULT_EMBED_MODEL),
        os.environ.get("GEN_MODEL", DEFAULT_GEN_MODEL),
        os.environ.get("QLORA_MODEL", "lemkin-qlora"),
    )


def _model_f_response(status: ModelFStatus) -> ModelFStatusResponse:
    return ModelFStatusResponse(
        enabled=status.enabled,
        ready=status.ready,
        mode=status.mode,
        context_profile=status.context_profile,
        detail=status.detail,
    )


def _load_evaluation_manifest() -> dict[str, Any]:
    """Load published batch evidence; never calculate it on a draft request."""
    if not EVALUATION_RESULTS_PATH.exists():
        raise HTTPException(500, f"Evaluation manifest not found: {EVALUATION_RESULTS_PATH}")
    try:
        with EVALUATION_RESULTS_PATH.open(encoding="utf-8") as file:
            manifest = json.load(file)
    except (OSError, json.JSONDecodeError) as error:
        raise HTTPException(500, "Evaluation manifest is invalid or unreadable.") from error
    if not isinstance(manifest, dict) or not isinstance(manifest.get("groups"), list):
        raise HTTPException(500, "Evaluation manifest has an invalid format.")
    return manifest


def _language_category(match: dict[str, Any]) -> str:
    category = (match.get("rule") or {}).get("category", {})
    if isinstance(category, dict):
        return str(category.get("id", "")).upper()
    return str(category).upper()


def _keep_grammar_match(match: dict[str, Any], text: str) -> bool:
    """Suppress known style false positives without claiming an overall grade."""
    rule = match.get("rule") or {}
    rule_id = str(rule.get("id", ""))
    offset = int(match.get("offset", 0))
    length = int(match.get("length", 0))
    matched = text[offset : offset + length]
    if "\ufffd" in matched or rule_id in IGNORED_GRAMMAR_RULES:
        return False
    return _language_category(match) in GRAMMAR_CATEGORIES or rule_id in GRAMMAR_RULES


def _word_count(text: str) -> int:
    return len(re.findall(r"\b[\w]+(?:['\u2019][\w]+)?\b", text))


# The QLoRA path matches Kevin's medium-grid evaluation: one user sentence,
# Llama 3.1's default chat wrapper, and no custom system message.  LinkedIn
# and X rules intentionally appear *before* the topic so headline-shaped
# topics are not copied as the whole answer.
DIRECT_QLORA_PROMPTS = {
    "linkedin_post": (
        "About 80 words. The first line is a claim, not a headline. No hashtags. "
        "Write a LinkedIn post in the style of Jason Lemkin about {topic}."
    ),
    "blog_draft": (
        "Write a blog post in the style of Jason Lemkin about {topic}. "
        "Write the post itself in about 250 words, with no title line."
    ),
    "x_thread": (
        "Write one X post in the style of Jason Lemkin. "
        "Exactly 3 sentences, under 60 words, no headline, and no hashtags. "
        "Topic: {topic}"
    ),
    "youtube_script": (
        "Write a spoken talk in the style of Jason Lemkin about {topic}. "
        "Use short spoken sentences to founders, about 160 words, with no title and no headings."
    ),
}
DIRECT_QLORA_MAX_TOKENS = {
    "linkedin_post": 320,
    "x_thread": 120,
    "blog_draft": 500,
    "youtube_script": 400,
}
DIRECT_QLORA_REPEAT_PENALTIES = {
    "linkedin_post": 1.1,
    "x_thread": 1.1,
    "blog_draft": 1.1,
    "youtube_script": 1.15,
}
DIRECT_QLORA_OPTIONS = {
    "temperature": 0,
    "seed": 42,
    "top_p": 0.9,
    "num_ctx": 4096,
    "stop": ["<|eot_id|>"],
}
# This CPU-only demo has enough memory for one 8B writer at a time, not both
# the fine-tuned GGUF and base Llama indefinitely.  Unload QLoRA after each
# reply so switching back to PersonaRAG can load `llama3.1` reliably.
DIRECT_QLORA_KEEP_ALIVE = 0


BASELINE_FORMATS = {
    "linkedin_post": "LinkedIn post",
    "blog_draft": "blog post",
    "x_thread": "X post",
    "youtube_script": "spoken talk",
}


def direct_qlora_prompt(format_name: str, topic: str) -> str:
    """Build Kevin's evaluated direct-QLoRA user message for one medium."""
    template = DIRECT_QLORA_PROMPTS.get(format_name)
    if template is None:
        known = ", ".join(DIRECT_QLORA_PROMPTS)
        raise HTTPException(400, f"Unknown format for direct QLoRA: {format_name}. Expected: {known}")
    return template.format(topic=topic)


def direct_qlora_options(format_name: str) -> dict[str, object]:
    """Return QLoRA decoding settings, including its medium-specific penalty."""
    try:
        repeat_penalty = DIRECT_QLORA_REPEAT_PENALTIES[format_name]
    except KeyError as e:
        raise HTTPException(400, f"Unknown format for direct QLoRA: {format_name}") from e
    return {**DIRECT_QLORA_OPTIONS, "repeat_penalty": repeat_penalty, "num_predict": DIRECT_QLORA_MAX_TOKENS[format_name]}


def baseline_prompt(format_name: str, topic: str, audience: str, goal: str, cta: str) -> str:
    """Build the plain base-model control prompt without persona or RAG context."""
    try:
        medium = BASELINE_FORMATS[format_name]
    except KeyError as e:
        known = ", ".join(BASELINE_FORMATS)
        raise HTTPException(400, f"Unknown format for baseline: {format_name}. Expected: {known}") from e
    return "\n".join(
        [
            f"Write a {medium} about: {topic}",
            f"Audience: {audience}",
            f"Goal: {goal}",
            f"Call to action: {cta}",
        ]
    )


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


@app.get("/healthz")
def healthz() -> dict:
    """Fast liveness check for Docker.

    Keep this separate from ``/health``: the latter intentionally validates the
    Chroma collection and should be used for the UI/diagnostics, not every
    container-health interval.
    """
    return {"status": "ok"}


@app.get("/persona", response_model=PersonaResponse)
def persona() -> PersonaResponse:
    """Expose the profile PersonaRAG uses without changing generation behavior."""
    if not PERSONA_PATH.exists():
        raise HTTPException(500, f"Persona profile not found: {PERSONA_PATH}")
    try:
        profile = load_persona(PERSONA_PATH)
    except ConfigError as e:
        raise HTTPException(500, f"Could not load persona profile: {e}") from e

    meta = profile.get("meta", {})
    source = str(meta.get("source", "Jason Lemkin — operator voice"))
    name = source.split(" — ", 1)[0].strip() or "Jason Lemkin"
    return PersonaResponse(name=name, source=source, summary=summarize_persona(profile))


@app.get("/evaluation")
def evaluation() -> dict[str, Any]:
    """Serve versioned, precomputed evaluation evidence for the UI.

    This endpoint intentionally reads a committed manifest instead of invoking
    a judge, embedding model, or retrieval pipeline on EC2.
    """
    return _load_evaluation_manifest()


@app.get("/model-f/status", response_model=ModelFStatusResponse)
def model_f_state() -> ModelFStatusResponse:
    """Report Model F readiness without opening LightRAG storage."""
    return _model_f_response(model_f_status())


@app.post("/quality/grammar", response_model=GrammarCheckResponse)
def grammar_check(req: GrammarCheckRequest) -> GrammarCheckResponse:
    """Run a user-requested LanguageTool screen through the private network."""
    try:
        response = requests.post(
            f"{LANGUAGETOOL_BASE.rstrip('/')}/v2/check",
            data={"language": "en-US", "text": req.text},
            timeout=LANGUAGETOOL_TIMEOUT,
        )
        response.raise_for_status()
        matches = response.json().get("matches", [])
    except (requests.RequestException, ValueError) as error:
        raise HTTPException(503, "Grammar service unavailable. Try again shortly.") from error

    issues: list[GrammarIssue] = []
    for match in matches:
        if not isinstance(match, dict) or not _keep_grammar_match(match, req.text):
            continue
        rule = match.get("rule") or {}
        replacements = [str(item.get("value", "")) for item in match.get("replacements", []) if item.get("value")]
        issues.append(
            GrammarIssue(
                message=str(match.get("message", "LanguageTool issue")),
                matched_text=req.text[int(match.get("offset", 0)) : int(match.get("offset", 0)) + int(match.get("length", 0))],
                offset=int(match.get("offset", 0)),
                length=int(match.get("length", 0)),
                rule_id=str(rule.get("id", "unknown")),
                replacements=replacements[:5],
            )
        )

    word_count = _word_count(req.text)
    return GrammarCheckResponse(
        word_count=word_count,
        issue_count=len(issues),
        issues_per_100_words=round((len(issues) * 100 / word_count) if word_count else 0, 2),
        issues=issues,
    )


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
    if req.method == "model_f":
        if req.retrieval_backend != "lightrag":
            raise HTTPException(400, "Model F requires retrieval_backend='lightrag'.")
        status = model_f_status()
        raise HTTPException(503, f"Model F unavailable: {status.detail}")
    if req.retrieval_backend != "dense" or req.model_f_context != "off":
        raise HTTPException(400, "LightRAG context is available only through the Model F method.")
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
    elif req.method == "qlora":
        prompt = direct_qlora_prompt(req.format, req.topic)
        model = qlora_model
    else:
        prompt = baseline_prompt(req.format, req.topic, req.audience, req.goal, req.cta)
        model = gen_model

    try:
        if req.method == "qlora":
            draft = ollama_chat(
                prompt,
                ollama_base,
                model,
                options=direct_qlora_options(req.format),
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
