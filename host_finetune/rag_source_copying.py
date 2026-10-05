"""Contiguous copy detection against retrieved excerpts.

This is not the training-memorization gate. Prompt text is not exempted,
because the retrieved excerpt is inside the prompt on purpose.
"""
from __future__ import annotations

import re

RAG_SOURCE_COPY_THRESHOLD = 12
RETRY_SEED_OFFSET = 10000
_TOKEN_RE = re.compile(r"[a-z0-9$%]+")


def rag_copy_tokens(text: str) -> list[str]:
    """Lowercased word sequence. Punctuation separates tokens and order is kept."""
    return _TOKEN_RE.findall((text or "").casefold())


def retry_seed(original_seed: int) -> int:
    """One deterministic alternate seed. Temperature and the prompt stay fixed."""
    return original_seed + RETRY_SEED_OFFSET


def copy_guard_seeds(original_seed: int, first_passed: bool) -> list[int]:
    """Attempt 1, plus one retry only when that attempt copied a retrieved excerpt."""
    if first_passed:
        return [original_seed]
    return [original_seed, retry_seed(original_seed)]


def _longest_span(left: list[str], right: list[str]) -> tuple[int, int, int]:
    best = 0
    left_end = -1
    right_end = -1
    previous = [0] * (len(right) + 1)
    for i, word in enumerate(left):
        current = [0]
        for j, other in enumerate(right):
            if word == other:
                span = previous[j] + 1
                current.append(span)
                if span > best:
                    best = span
                    left_end = i
                    right_end = j
            else:
                current.append(0)
        previous = current
    return best, left_end, right_end


def _span_text(tokens: list[str], end: int, length: int) -> str | None:
    if length <= 0 or end < 0:
        return None
    return " ".join(tokens[end - length + 1:end + 1])


def evaluate_rag_source_copying(
    answer: str,
    hits: list[dict] | None,
    *,
    enabled: bool,
    threshold: int = RAG_SOURCE_COPY_THRESHOLD,
) -> dict:
    """Score one completion against retrieved excerpts.

    A span of ``threshold`` words or longer fails. Retrieval-off calls return
    ``applicable`` false and do not compare text.
    """
    if not enabled:
        return {
            "applicable": False,
            "pass": None,
            "threshold": threshold,
            "max_contiguous_words": None,
            "document_id": None,
            "matched_generated_span": None,
            "matched_retrieved_span": None,
        }
    generated = rag_copy_tokens(answer)
    best = 0
    document_id = None
    generated_span = None
    retrieved_span = None
    for hit in hits or []:
        source = rag_copy_tokens(hit.get("text") or "")
        length, left_end, right_end = _longest_span(generated, source)
        if length > best:
            best = length
            document_id = hit.get("id")
            generated_span = _span_text(generated, left_end, length)
            retrieved_span = _span_text(source, right_end, length)
    return {
        "applicable": True,
        "pass": best < threshold,
        "threshold": threshold,
        "max_contiguous_words": best,
        "document_id": document_id,
        "matched_generated_span": generated_span,
        "matched_retrieved_span": retrieved_span,
    }


def assemble_copy_guard(
    first: dict,
    retry: dict | None,
    *,
    threshold: int = RAG_SOURCE_COPY_THRESHOLD,
) -> dict:
    """Keep attempt 1 even when a retry is accepted.

    ``first`` and ``retry`` each need ``seed``, ``answer``, and ``report``.
    A passing first attempt must not be paired with a retry. A third sample
    is not representable here.
    """
    report = first["report"]
    if not report["applicable"] and retry is not None:
        raise ValueError("retrieval-off output is not retried")
    if report["applicable"] and report["pass"] and retry is not None:
        raise ValueError("a passing first attempt is not retried")
    if retry is not None and retry["seed"] != retry_seed(first["seed"]):
        raise ValueError("retry seed must be the original seed plus the fixed offset")
    retry_passed = None if retry is None else bool(retry["report"]["pass"])
    if not report["applicable"] or report["pass"]:
        accepted_attempt = 1
        deployment_accepted = True
        answer = first["answer"]
    elif retry is not None and retry["report"]["pass"]:
        accepted_attempt = 2
        deployment_accepted = True
        answer = retry["answer"]
    else:
        accepted_attempt = None
        deployment_accepted = False
        answer = first["answer"]
    return {
        "answer": answer,
        "first_attempt_answer": first["answer"],
        "retry_answer": None if retry is None else retry["answer"],
        "seed": first["seed"],
        "retry_seed": None if retry is None else retry["seed"],
        "deployment_accepted": deployment_accepted,
        "rag_source_copying": {
            "applicable": report["applicable"],
            "threshold": threshold,
            "retry_seed_offset": RETRY_SEED_OFFSET,
            "retry_occurred": retry is not None,
            "retry_passed": retry_passed,
            "accepted_attempt": accepted_attempt,
            "pass": deployment_accepted if report["applicable"] else None,
            "first_attempt": report,
            "retry": None if retry is None else retry["report"],
        },
    }
