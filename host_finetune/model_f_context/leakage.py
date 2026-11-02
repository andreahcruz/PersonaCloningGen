"""Leakage checks for Model F evaluation. This module does not open an index.

Held-out validation and test documents, their source groups, gold-overlap
families, and duplicate families are rejected when a caller supplies them.
The untouched final test set is not read here.
"""

from __future__ import annotations

import hashlib
import json

from host_finetune.factual_reliability import assess_factual, instruction_leak
from host_finetune.model_f_context.budget import estimate_tokens
from host_finetune.model_f_context.prompt import marker_leak
from host_finetune.model_f_context.registry import CONFIGURATIONS

HELD_OUT_SPLITS = ("validation", "test")
# Counts recorded by the graph-ready measurement and this task's corpus note.
# They are not recomputed against the live store.
DOCUMENTED_CORPUS = {
    "training_eligible_blog_graph_documents": 2042,
    "training_eligible_blog_source": "docs/architecture/LIGHTRAG_MODEL_F.md included blog count",
    "scraped_blog_records_stated": 3502,
    "scraped_blog_status": "stated_for_this_task_not_recomputed_while_indexing",
    "held_out_policy": "validation and test posts stay out of the retrieval corpus",
}

# The registry is the authority. Letters A–D belong to EXP-20261031-001 and
# are not reused here. D is not multi-hop.
EVALUATION_CONFIGURATIONS = CONFIGURATIONS


class LeakageError(ValueError):
    """A held-out document or split was placed in an iterative corpus."""


class LeakagePolicy:
    """Caller-supplied exclusions. Split files are not opened."""

    def __init__(
        self,
        *,
        blocked_document_ids: set[str] | None = None,
        blocked_groups: set[str] | None = None,
        gold_overlap_groups: set[str] | None = None,
        blocked_families: set[str] | None = None,
        target_document_ids: set[str] | None = None,
    ) -> None:
        self.blocked_document_ids = set(blocked_document_ids or ())
        self.blocked_groups = set(blocked_groups or ())
        self.gold_overlap_groups = set(gold_overlap_groups or ())
        self.blocked_families = set(blocked_families or ())
        self.target_document_ids = set(target_document_ids or ())

    def groups_of(self, document: dict) -> set[str]:
        found = set()
        if document.get("group_id"):
            found.add(str(document["group_id"]))
        for item in document.get("group_ids") or []:
            if item:
                found.add(str(item))
        lineage = document.get("leakage_lineage") or {}
        if lineage.get("source_group"):
            found.add(str(lineage["source_group"]))
        for item in lineage.get("group_ids") or []:
            if item:
                found.add(str(item))
        return found

    def decision(self, document: dict) -> tuple[bool, str]:
        split = str(document.get("split") or (document.get("leakage_lineage") or {}).get("split") or "")
        if split in HELD_OUT_SPLITS:
            return False, "held_out_split"
        if split and split != "train":
            return False, "non_train_split"
        document_id = str(
            document.get("graph_document_id")
            or document.get("parent_graph_document_id")
            or document.get("id")
            or ""
        )
        if document_id and document_id in self.target_document_ids:
            return False, "exact_target"
        if document_id and document_id in self.blocked_document_ids:
            return False, "blocked_document"
        groups = self.groups_of(document)
        if groups & self.gold_overlap_groups:
            return False, "gold_overlap_family"
        if groups & self.blocked_groups:
            return False, "blocked_source_group"
        family = str(document.get("duplicate_family") or document.get("family_id") or "")
        if family and family in self.blocked_families:
            return False, "duplicate_family"
        return True, "eligible"


def screen_documents(documents: list[dict], policy: LeakagePolicy) -> dict:
    """Copy the eligibility decision. The input list is not modified."""
    kept = []
    excluded = []
    for document in documents:
        allowed, reason = policy.decision(document)
        if allowed:
            kept.append(dict(document))
        else:
            excluded.append({"id": document.get("graph_document_id") or document.get("id"), "reason": reason})
    return {"kept": kept, "excluded": excluded}


def corpus_identity(documents: list[dict]) -> str:
    ids = sorted(
        str(document.get("graph_document_id") or document.get("id") or "")
        for document in documents
    )
    encoded = json.dumps(ids, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def assert_same_eligible_corpus(left: list[dict], right: list[dict]) -> None:
    if corpus_identity(left) != corpus_identity(right):
        raise LeakageError("retrieval arms are not using the same eligible source ids")


def assert_iterative_selection_corpus(documents: list[dict], *, purpose: str) -> dict:
    """Refuse validation and test rows in a corpus used to choose a model."""
    if purpose != "iterative_model_selection":
        raise LeakageError("purpose must be iterative_model_selection")
    held = [
        document.get("graph_document_id") or document.get("id")
        for document in documents
        if str(document.get("split") or "") in HELD_OUT_SPLITS
    ]
    if held:
        raise LeakageError("held-out documents are in the iterative selection corpus: " + ", ".join(map(str, held[:8])))
    return {
        "purpose": purpose,
        "test_set_accessed": False,
        "held_out_in_corpus": 0,
        "documents": len(documents),
    }


def screen_hits(hits: list[dict], policy: LeakagePolicy) -> dict:
    return screen_documents(hits, policy)


def paired_comparison(query_id: str, arms: dict[str, dict], policy: LeakagePolicy) -> dict:
    """Compare saved retrieval payloads. Generation is recorded only when present."""
    compared = []
    for config in EVALUATION_CONFIGURATIONS:
        payload = arms.get(config["id"]) or {}
        hits = list(payload.get("hits") or [])
        screened = screen_hits(hits, policy)
        retrieved_ids = [
            str(hit.get("graph_document_id") or hit.get("id") or "")
            for hit in screened["kept"]
        ]
        contaminated = any(item["reason"] in {"exact_target", "duplicate_family", "held_out_split", "gold_overlap_family"} for item in screened["excluded"])
        draft = payload.get("draft")
        generation = None
        if draft is not None:
            generation = generation_observation(
                str(draft),
                claims=list(payload.get("claims") or []),
                extra_texts=list(payload.get("grounding_texts") or []),
                hits=screened["kept"],
                latency_seconds=payload.get("seconds"),
                prompt_tokens=payload.get("prompt_tokens"),
            )
        compared.append({
            "id": config["id"],
            "label": config["label"],
            "backend": payload.get("backend") or config["backend"],
            "context_profile": config["context_profile"],
            "retrieval": {
                "query_id": query_id,
                "retrieved_source_ids": retrieved_ids,
                "unique_sources": len(set(retrieved_ids)),
                "excluded": screened["excluded"],
                "contamination": contaminated,
                "latency_seconds": payload.get("seconds"),
            },
            "generation": generation,
        })
    return {
        "query_id": query_id,
        "arms": compared,
        "retrieval_separated_from_generation": True,
        "documented_corpus": dict(DOCUMENTED_CORPUS),
    }


def generation_observation(
    draft: str,
    *,
    claims: list[dict],
    extra_texts: list[str] | None = None,
    hits: list[dict] | None = None,
    latency_seconds: float | None = None,
    prompt_tokens: int | None = None,
) -> dict:
    factual = assess_factual(
        draft,
        "",
        claims,
        hits=hits or [],
        extra_texts=extra_texts or [],
        retrieval_on=bool(hits),
    )
    return {
        "factual_pass": factual["pass"],
        "factual_reasons": list(factual["reasons"]),
        "numeric": factual["numeric"],
        "marker_copy": bool(instruction_leak(draft) or marker_leak(draft)),
        "style": "not_scored",
        "latency_seconds": latency_seconds,
        "prompt_tokens": prompt_tokens,
        "draft_tokens_estimate": estimate_tokens(draft),
        "method": factual["method"],
    }


def withhold_graph_context(audit: list[dict]) -> list[str]:
    """Reasons graph context should stay out of the prompt."""
    graph_rows = [
        row for row in audit
        if row.get("channel") in {"source_supported_relationship", "derived_description", "multi_hop_path"}
    ]
    if not graph_rows:
        return ["no_graph_context"]
    selected = [row for row in graph_rows if row.get("decision") in {"selected", "selected_candidate"}]
    if selected:
        return []
    reasons = []
    for row in graph_rows:
        for reason in row.get("reasons") or []:
            if reason not in reasons:
                reasons.append(str(reason))
    return reasons or ["graph_context_excluded"]
