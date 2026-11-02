"""Failure categories for LightRAG indexing.

Categories name the stage that failed. Retry policy is separate from the
category: a timeout is classified even when the scheduler declines to retry it.
"""

from __future__ import annotations

STAGES = (
    "source",
    "prepare",
    "chunk",
    "extract",
    "embed",
    "merge",
    "persist",
    "interrupt",
)

CATEGORIES = (
    "source_validation",
    "document_parsing_preparation",
    "tokenization_chunking",
    "context_window_overflow",
    "llm_extraction_timeout",
    "invalid_extraction_format",
    "entity_relation_merge",
    "embedding_model_failure",
    "embedding_dimension_mismatch",
    "vector_store_write",
    "graph_store_write",
    "persistence_failure",
    "process_interruption",
)

# Retried only when the graph has not recorded a mutation for the document
# and the attempt budget remains. Timeouts stay retryable in the ledger and
# are kept out of the default queue.
RETRYABLE_CATEGORIES = frozenset(
    {
        "document_parsing_preparation",
        "tokenization_chunking",
        "llm_extraction_timeout",
        "invalid_extraction_format",
        "entity_relation_merge",
        "embedding_model_failure",
        "vector_store_write",
        "graph_store_write",
        "persistence_failure",
        "process_interruption",
    }
)

PERMANENT_CATEGORIES = frozenset(
    {
        "source_validation",
        "context_window_overflow",
        "embedding_dimension_mismatch",
    }
)


def _blob(*parts: str | None) -> str:
    return " ".join(part.lower() for part in parts if part)


def classify_failure(
    message: str | None,
    *,
    stage_hint: str | None = None,
    exception_name: str | None = None,
) -> tuple[str, str]:
    """Return ``(stage, category)`` from a message and an optional stage hint.

    ``stage_hint`` is LightRAG's own label when the pipeline recorded one
    (``extract`` or ``merge``). The message is still read, because a worker
    timeout during extraction is not a generic extract failure.
    """
    text = _blob(exception_name, stage_hint, message)
    if any(
        token in text
        for token in (
            "user cancelled",
            "keyboardinterrupt",
            "process interruption",
            "cancellederror",
            "pipelinecancelledexception",
            "stop requested",
        )
    ):
        return "interrupt", "process_interruption"
    if "timeout" in text or "timed out" in text:
        if "embed" in text:
            return "embed", "embedding_model_failure"
        return "extract", "llm_extraction_timeout"
    if any(
        token in text
        for token in (
            "context length",
            "context window",
            "num_ctx",
            "maximum context",
            "input is too long",
        )
    ):
        return "extract", "context_window_overflow"
    if any(
        token in text
        for token in (
            "embedding dimension",
            "dimension mismatch",
            "expected 768",
        )
    ):
        return "embed", "embedding_dimension_mismatch"
    if any(
        token in text
        for token in (
            "embedding",
            "nomic",
            "ollama_embed",
        )
    ) and "extract" not in (stage_hint or ""):
        return "embed", "embedding_model_failure"
    if "indexflusherror" in text or "storage flush" in text or "persist" in text:
        return "persist", "persistence_failure"
    if any(token in text for token in ("chunks_vdb", "vector store", "nano-vectordb", "nanovectordb")):
        return "embed", "vector_store_write"
    if stage_hint == "merge" or any(
        token in text for token in ("merge_nodes", "merging", "graph store", "graphml")
    ):
        return "merge", "entity_relation_merge"
    if any(
        token in text
        for token in (
            "chunk",
            "tokeniz",
            "chunkblockmatch",
        )
    ):
        return "chunk", "tokenization_chunking"
    if any(
        token in text
        for token in (
            "jsondecode",
            "invalid extraction",
            "could not parse",
            "extraction format",
        )
    ):
        return "extract", "invalid_extraction_format"
    if any(
        token in text
        for token in (
            "empty text",
            "hash mismatch",
            "non-train",
            "exclusion",
            "source validation",
            "duplicate content",
        )
    ):
        return "source", "source_validation"
    if stage_hint == "extract":
        return "extract", "invalid_extraction_format"
    if any(token in text for token in ("prepare", "sanitiz", "enqueue")):
        return "prepare", "document_parsing_preparation"
    return "prepare", "document_parsing_preparation"


def retry_eligible(category: str, *, graph_mutated: bool) -> bool:
    """A mutated graph is not retried by deleting and inserting again."""
    if graph_mutated:
        return False
    if category in PERMANENT_CATEGORIES:
        return False
    return category in RETRYABLE_CATEGORIES


def terminal_status(category: str, *, eligible: bool, attempt: int, max_attempts: int) -> str:
    """Map a classified failure onto a ledger status."""
    if category == "source_validation":
        return "quarantined"
    if category == "process_interruption":
        return "interrupted"
    if category == "embedding_dimension_mismatch":
        return "failed_permanent"
    if not eligible or attempt >= max_attempts:
        return "failed_permanent"
    return "failed_retryable"
