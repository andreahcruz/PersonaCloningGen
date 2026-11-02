"""Private HTTP boundary between the deployed FastAPI app and LightRAG."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .retriever import LightRAGUnavailable, retrieve, snapshot_status

WORKDIR = Path(os.environ.get("LIGHTRAG_WORKDIR", "/runtime"))
OLLAMA_BASE = os.environ.get("LIGHTRAG_OLLAMA_BASE", "http://ollama:11434")
EXTRACTION_MODEL = os.environ.get("LIGHTRAG_EXTRACTION_MODEL", "llama3.1")
EMBED_MODEL = os.environ.get("LIGHTRAG_EMBED_MODEL", "nomic-embed-text")
NUM_CTX = int(os.environ.get("LIGHTRAG_NUM_CTX", "16384"))

app = FastAPI(title="Lemkin LightRAG Retriever", docs_url=None, redoc_url=None)


class RetrieveRequest(BaseModel):
    query: str = Field(min_length=1, max_length=2000)
    k: int = Field(default=4, ge=1, le=8)
    mode: Literal["naive", "local", "global", "hybrid", "mix"] = "hybrid"


@app.get("/healthz")
def healthz() -> dict:
    """Return 200 even when artifacts are absent so disabled deployments stay stable."""
    return {"status": "ok", "snapshot": snapshot_status(WORKDIR)}


@app.post("/retrieve")
async def retrieve_sources(req: RetrieveRequest) -> dict:
    try:
        return await retrieve(
            req.query,
            WORKDIR,
            base_url=OLLAMA_BASE,
            extraction_model=EXTRACTION_MODEL,
            embed_model=EMBED_MODEL,
            mode=req.mode,
            evidence_budget=req.k,
            top_k=20,
            num_ctx=NUM_CTX,
        )
    except LightRAGUnavailable as error:
        raise HTTPException(503, str(error)) from error
