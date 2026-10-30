"""Shared retrieval identities for Model E and Model F.

Graph ranks and Chroma cosine distances are different quantities. Callers may
store both, and they must not treat them as one score.
"""

from __future__ import annotations

BACKENDS = ("dense", "lightrag")
LIGHTRAG_MODES = ("naive", "local", "global", "hybrid", "mix")
# LightRAG also accepts "bypass", which skips retrieval. Model F does not expose it.
EXPERIMENT_CELLS = (
    {"cell": "E0", "backend": "dense", "mode": None, "label": "Model E dense Chroma"},
    {"cell": "F1", "backend": "lightrag", "mode": "naive", "label": "LightRAG naive"},
    {"cell": "F2", "backend": "lightrag", "mode": "local", "label": "LightRAG local"},
    {"cell": "F3", "backend": "lightrag", "mode": "global", "label": "LightRAG global"},
    {"cell": "F4", "backend": "lightrag", "mode": "hybrid", "label": "LightRAG hybrid"},
    {"cell": "F5", "backend": "lightrag", "mode": "mix", "label": "LightRAG mix"},
)

# Verified against lightrag-hku 1.5.7 QueryParam and aquery_data.
# top_k counts entities in local mode and relationships in global mode.
# chunk_top_k counts text chunks. Neither number is Model E's document k.
MODE_SEMANTICS = {
    "naive": "Vector search over source chunks only. Entity and relationship lists are empty.",
    "local": "Entity-focused graph search from low-level keywords, plus the chunks those entities cite.",
    "global": "Relationship-focused graph search from high-level keywords, plus the chunks those relationships cite.",
    "hybrid": "Round-robin merge of the local and global graph searches.",
    "mix": "Knowledge-graph results plus an additional vector search over source chunks.",
}


class RetrievalConfigError(ValueError):
    """The requested backend or mode is not one this project exposes."""


def require_backend(backend: str) -> str:
    chosen = (backend or "").strip().lower()
    if chosen not in BACKENDS:
        known = ", ".join(BACKENDS)
        raise RetrievalConfigError(f"unknown retrieval backend {backend!r}; known: {known}")
    return chosen


def require_lightrag_mode(mode: str) -> str:
    chosen = (mode or "").strip().lower()
    if chosen not in LIGHTRAG_MODES:
        known = ", ".join(LIGHTRAG_MODES)
        raise RetrievalConfigError(f"unknown LightRAG mode {mode!r}; known: {known}")
    return chosen


def cell_for(backend: str, mode: str | None) -> str | None:
    backend = require_backend(backend)
    if backend == "lightrag":
        mode = require_lightrag_mode(mode or "")
    for cell in EXPERIMENT_CELLS:
        if cell["backend"] == backend and cell["mode"] == mode:
            return cell["cell"]
    return None
