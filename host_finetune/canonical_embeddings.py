"""Current nomic embedding calls used by the train-only index and retrieval audit.

Documents and queries both go to Ollama ``POST /api/embed``. The train-only
builder and ``audit_retrieval_corpus`` send ``options.num_gpu = 0``. The input
is the text itself: no ``search_document`` or ``search_query`` prefix, and no
extra normalization in this module.

The command that first wrote ``lemkin_train_only`` is still unrecorded. This
module reproduces the current call, not that missing command.
"""

from __future__ import annotations

import os

from host_finetune.rebuild_chroma import _embed_batch
from host_finetune.train_only_index import ollama_embed_body

MODEL = "nomic-embed-text"
DEFAULT_BASE = os.environ.get("OLLAMA_BASE", "http://127.0.0.1:11434")
# run_train_only caps batches at 16 after batches of 64 stalled Ollama.
DOCUMENT_BATCH = 16
RETRIEVAL_K = 4


def embed_texts(
    texts: list[str],
    *,
    base_url: str | None = None,
    model: str = MODEL,
    batch_size: int = DOCUMENT_BATCH,
    cpu: bool = True,
) -> list[list[float]]:
    """Embed document or query strings with the current CPU Ollama call."""
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    base = base_url or DEFAULT_BASE
    vectors: list[list[float]] = []
    for start in range(0, len(texts), batch_size):
        chunk = texts[start : start + batch_size]
        vectors.extend(_embed_batch(chunk, base, model, cpu=cpu))
    return vectors


def embed_documents(texts: list[str], **kwargs) -> list[list[float]]:
    """Embed stored document strings. The caller passes stripped output text."""
    return embed_texts(texts, **kwargs)


def embed_query(text: str, **kwargs) -> list[float]:
    """Embed one query with the same body as a one-item document batch."""
    return embed_texts([text], batch_size=1, **kwargs)[0]


def request_body(texts: list[str], *, model: str = MODEL, cpu: bool = True) -> dict:
    """The JSON body ``_embed_batch`` posts. Exposed for tests."""
    return ollama_embed_body(texts, model, cpu=cpu)
