"""Metric wrappers. Ranked retrieval stays null without grades.

This module calls the existing retrieval and factual scorers. It does not
score BLEU, ROUGE, or an LLM judge, and it does not build a composite score.
Citation precision is computed only when the record carries an explicit
citation list. Prose that contains prompt tags is a leak, not a citation.
"""

from __future__ import annotations

import math
import re

from host_finetune.factual_reliability import assess_factual, instruction_leak
from host_finetune.model_f_context.prompt import marker_leak
from host_finetune.retrieval_eval import evaluate_query

_CITATION_TAG = re.compile(r"\[(?:E|R|P|D)\d+\]")
_DOC_ID = re.compile(r"\bgd_[0-9a-f]{6,}\b", re.IGNORECASE)


def _hits(record: dict) -> list[dict]:
    hits = []
    for item in record.get("evidence") or []:
        if not isinstance(item, dict):
            continue
        hit = dict(item)
        if not hit.get("id"):
            hit["id"] = hit.get("graph_document_id") or hit.get("source_id")
        hits.append(hit)
    return hits


def _qrels(record: dict) -> list[dict]:
    rows = []
    for item in record.get("qrels") or []:
        if isinstance(item, dict) and item.get("doc_id"):
            row = dict(item)
            row["query_id"] = record["question_id"]
            rows.append(row)
    return rows


def provenance_completeness(hits: list[dict]) -> float | None:
    """Share of hits that name a source and carry passage text plus split or provenance."""
    if not hits:
        return None
    complete = 0
    for hit in hits:
        has_id = bool(hit.get("id"))
        has_text = bool((hit.get("text") or hit.get("passage") or "").strip())
        has_lineage = hit.get("split") == "train" or bool(hit.get("provenance"))
        if has_id and has_text and has_lineage:
            complete += 1
    return complete / len(hits)


def designated_source_coverage(retrieved: list[str], designated: list[str]) -> dict:
    """Overlap with a preselected id list. This is not corpus recall."""
    wanted = [str(item) for item in designated if item]
    found = set(retrieved)
    if not wanted:
        return {"status": "not_applicable", "value": None, "interprets_as": "no_preselected_ids"}
    matched = [item for item in wanted if item in found]
    return {
        "status": "scored",
        "value": len(matched) / len(wanted),
        "matched": matched,
        "interprets_as": "overlap_with_preselected_ids_not_recall",
    }


def source_diversity(ids: list[str]) -> float | None:
    if not ids:
        return None
    return len(set(ids)) / len(ids)


def score_retrieval(record: dict, *, k_values: tuple[int, ...] = (1, 4)) -> dict:
    hits = _hits(record)
    qrels = _qrels(record)
    scope = "corpus" if qrels else "none"
    ranked = evaluate_query(
        record["question_id"],
        hits,
        qrels,
        k_values=k_values,
        relevance_scope=scope,
    )
    coverage = designated_source_coverage(
        record.get("retrieved_source_ids") or [str(hit.get("id") or "") for hit in hits],
        list(record.get("review_source_ids") or []),
    )
    return {
        "empty": ranked["empty"],
        "n_hits": ranked["n_hits"],
        "duplicate_source": ranked["duplicate_document_ids"],
        "source_diversity": source_diversity(record.get("retrieved_source_ids") or []),
        "unique_sources": len(set(record.get("retrieved_source_ids") or [])),
        "provenance_completeness": provenance_completeness(hits),
        "designated_source_coverage": coverage,
        "precision_at_1": ranked.get("precision_at_1"),
        "precision_at_4": ranked.get("precision_at_4"),
        "recall_at_1": ranked.get("recall_at_1"),
        "recall_at_4": ranked.get("recall_at_4"),
        "mrr": ranked.get("mrr"),
        "ndcg_at_4": ranked.get("ndcg_at_4"),
        "relevance_scope": scope,
        "limitation": (
            "Ranked metrics are null until a graded qrel exists. "
            "Preselected review ids are coverage against that list, not recall. "
            "A higher hit count is not itself a better score."
        ),
    }


def _citation_scores(citations: list | None, supporting: list[str]) -> dict:
    if citations is None:
        return {
            "status": "not_emitted",
            "reason": "This generator is told not to print evidence tags. Citation precision is not scored from the prose.",
            "precision": None,
            "recall": None,
        }
    cited = [str(item) for item in citations]
    support = set(supporting)
    if not cited and not support:
        return {"status": "not_applicable", "precision": None, "recall": None}
    if not cited:
        precision = None
        recall = 0.0 if support else None
    else:
        precision = len([item for item in cited if item in support]) / len(cited) if support or cited else None
        if not support:
            precision = 0.0
        recall = len(set(cited) & support) / len(support) if support else None
    return {"status": "scored", "precision": precision, "recall": recall}


def _contradictions(answer: str, pairs: list | None) -> dict:
    if not pairs:
        return {"status": "not_annotated", "count": None, "rate": None}
    hits = 0
    folded = (answer or "").casefold()
    for pair in pairs:
        both = [str(item) for item in (pair.get("must_not_both") or []) if item]
        if len(both) >= 2 and all(item.casefold() in folded for item in both):
            hits += 1
    return {"status": "scored", "count": hits, "rate": hits / len(pairs)}


def score_generation(record: dict) -> dict:
    answer = record.get("answer") or ""
    claims = list(record.get("claims") or [])
    hits = _hits(record)
    extra = [str(item) for item in (record.get("grounding_texts") or []) if item]
    factual = assess_factual(
        answer,
        str(record.get("question") or ""),
        claims,
        evidence_status="verified" if claims else "no_accepted_claims",
        hits=hits,
        retrieval_on=bool(hits),
        extra_texts=extra,
    )
    numeric = factual.get("numeric") or {}
    marker_copy = bool(instruction_leak(answer) or marker_leak(answer) or _CITATION_TAG.search(answer) or _DOC_ID.search(answer))
    supporting = [str(item) for item in (record.get("supporting_ids") or record.get("retrieved_source_ids") or [])]
    abstention_expected = bool(record.get("abstention_expected"))
    forbidden = [str(item) for item in (record.get("forbidden_assertions") or []) if item]
    forbidden_hits = [item for item in forbidden if item.casefold() in answer.casefold()]
    if abstention_expected:
        faithfulness = {
            "status": "scored_as_abstention",
            "pass": not forbidden_hits and numeric.get("pass") is not False and not marker_copy,
        }
    elif hits or claims or extra:
        faithfulness = {"status": "scored", "pass": bool(factual.get("pass"))}
    else:
        faithfulness = {"status": "not_scored", "reason": "no_retrieved_evidence", "pass": None}
    unsupported_numbers = list(numeric.get("unsupported") or [])
    return {
        "numeric_value_count": len(numeric.get("values") or []),
        "word_count": len(answer.split()),
        "faithfulness": faithfulness,
        "factual_reasons": list(factual.get("reasons") or []),
        "unsupported_numeric_count": len(unsupported_numbers),
        "numeric_pass": numeric.get("pass"),
        "numeric_applicable": numeric.get("applicable"),
        "role_unsupported_count": sum(
            1 for item in (factual.get("role_assertions") or []) if item.get("verdict") == "unsupported"
        ),
        "speaker": factual.get("speaker_attribution"),
        "copying": factual.get("retrieval_source_copying"),
        "citation_marker_leak": marker_copy,
        "citations": _citation_scores(record.get("citations"), supporting),
        "contradictions": _contradictions(answer, record.get("contradicted_pairs")),
        "reference_used": False,
        "limitation": (
            "Faithfulness uses deterministic quantity, role, speaker, and copying checks. "
            "It is not an LLM judgment and it is not n-gram overlap with a reference blog. "
            "Word count is recorded and is not a quality score. "
            "A missing reference leaves reference metrics unset."
        ),
    }


def _percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, math.ceil(fraction * len(ordered)) - 1))
    return ordered[index]


def score_efficiency(record: dict) -> dict:
    latency = record.get("latency") or {}
    retrieval = latency.get("retrieval_seconds")
    generation = latency.get("generation_seconds")
    end = latency.get("end_to_end_seconds")
    if end is None and isinstance(retrieval, (int, float)) and isinstance(generation, (int, float)):
        end = retrieval + generation
    tokens = record.get("tokens") or {}
    prompt_tokens = tokens.get("prompt")
    output_tokens = tokens.get("output")
    budget = record.get("context_budget_tokens")
    utilization = None
    if isinstance(prompt_tokens, (int, float)) and isinstance(budget, (int, float)) and budget > 0:
        utilization = prompt_tokens / budget
    return {
        "retrieval_seconds": retrieval if isinstance(retrieval, (int, float)) else None,
        "generation_seconds": generation if isinstance(generation, (int, float)) else None,
        "end_to_end_seconds": end if isinstance(end, (int, float)) else None,
        "prompt_tokens": prompt_tokens if isinstance(prompt_tokens, (int, float)) else None,
        "output_tokens": output_tokens if isinstance(output_tokens, (int, float)) else None,
        "context_utilization": utilization,
        "gpu_memory_bytes": record.get("gpu_memory_bytes") if isinstance(record.get("gpu_memory_bytes"), (int, float)) else None,
        "token_source": tokens.get("source") or ("recorded" if prompt_tokens is not None else "not_recorded"),
        "limitation": "Context utilization is not a quality score. Filling the budget is not a retrieval win.",
    }


def latency_summary(rows: list[dict]) -> dict:
    series = {
        "retrieval_seconds": [],
        "generation_seconds": [],
        "end_to_end_seconds": [],
    }
    for row in rows:
        efficiency = row.get("efficiency") or {}
        for key in series:
            value = efficiency.get(key)
            if isinstance(value, (int, float)):
                series[key].append(float(value))
    summary = {}
    for key, values in series.items():
        summary[key] = {
            "n": len(values),
            "p50": _percentile(values, 0.50),
            "p95": _percentile(values, 0.95),
        }
    prompt_tokens = [
        float(row["efficiency"]["prompt_tokens"])
        for row in rows
        if isinstance((row.get("efficiency") or {}).get("prompt_tokens"), (int, float))
    ]
    output_tokens = [
        float(row["efficiency"]["output_tokens"])
        for row in rows
        if isinstance((row.get("efficiency") or {}).get("output_tokens"), (int, float))
    ]
    ends = series["end_to_end_seconds"]
    summary["prompt_tokens_mean"] = sum(prompt_tokens) / len(prompt_tokens) if prompt_tokens else None
    summary["output_tokens_mean"] = sum(output_tokens) / len(output_tokens) if output_tokens else None
    summary["throughput_per_second"] = (len(ends) / sum(ends)) if ends and sum(ends) > 0 else None
    summary["gpu_memory"] = "not_measured_unless_a_record_already_contains_gpu_memory_bytes"
    return summary
