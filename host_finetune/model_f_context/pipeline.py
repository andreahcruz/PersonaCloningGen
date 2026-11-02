"""Compose Model F graph context without opening the live index.

``prepare_context`` normalizes, selects paths, and ranks. Prompt budgeting is
per medium so a short post and a blog do not share one cap. The default
profile for generation remains off; ``graph_context_callback`` returns None
until a caller selects a profile.
"""

from __future__ import annotations

import re

from host_finetune.evidence_capsule import PROMPT_CONTRACT
from host_finetune.model_f_context.budget import allocate_context
from host_finetune.model_f_context.multihop import select_paths
from host_finetune.model_f_context.normalize import normalize_graph_context
from host_finetune.model_f_context.prompt import render_context_prompt
from host_finetune.model_f_context.ranking import rank_evidence
from host_finetune.model_f_context.schema import SCHEMA_VERSION, bundle_dict

PROFILES = ("off", "source", "relationships", "descriptions", "multihop")


def require_profile(profile: str) -> str:
    chosen = (profile or "").strip().lower()
    if chosen not in PROFILES:
        known = ", ".join(PROFILES)
        raise ValueError(f"unknown model f context profile {profile!r}; known: {known}")
    return chosen


def prepare_context(
    payload: dict | None,
    *,
    profile: str,
    query: str = "",
    catalog: dict | None = None,
    failed_document_ids: set[str] | None = None,
    max_hops: int = 2,
    max_paths: int = 24,
    max_edges: int = 64,
) -> dict:
    """Build ranked graph context from a retrieval payload already in memory."""
    chosen = require_profile(profile)
    if chosen == "off":
        raise ValueError("profile off does not build graph context")
    normalized = normalize_graph_context(
        payload,
        catalog,
        failed_document_ids=failed_document_ids,
    )
    if query:
        normalized["query"] = query
    paths = select_paths(
        normalized["relationships"],
        max_hops=max_hops,
        max_paths=max_paths,
        max_edges=max_edges,
    )
    ranked = rank_evidence(
        query=normalized["query"],
        primary=normalized["primary"],
        relationships=normalized["relationships"],
        descriptions=normalized["descriptions"],
        paths=paths["paths"],
        profile=chosen,
    )
    audit = list(ranked["audit"])
    for item in paths["rejected"]:
        audit.append({
            "evidence_id": item.get("path_id") or "|".join(item.get("edge_ids") or []),
            "channel": "multi_hop_path",
            "decision": "excluded",
            "reasons": [item.get("reason") or "rejected_path"],
        })
    bundle = bundle_dict(
        query=normalized["query"],
        mode=normalized["mode"],
        primary=normalized["primary"],
        relationships=normalized["relationships"],
        descriptions=normalized["descriptions"],
        paths=paths["paths"],
        audit=[],
        diagnostics={
            **normalized["diagnostics"],
            "profile": chosen,
            "direct_edges": paths["direct_edge_count"],
            "paths": len(paths["paths"]),
            "withheld_edges": len(paths["withheld_edges"]),
        },
        raw_payload=normalized["raw_payload"],
    )
    return {
        "profile": chosen,
        "schema_version": SCHEMA_VERSION,
        "query": normalized["query"],
        "bundle": bundle,
        "ranked": ranked,
        "paths": paths,
        "audit": audit,
        "passages_retained": [item.evidence_id for item in normalized["primary"]],
    }


def _restates_claim(record, claims: list[dict]) -> bool:
    words = {
        token for token in re.findall(r"[a-z0-9]+", (record.predicate or "").casefold())
        if len(token) >= 4
    }
    if len(words) < 3:
        return False
    subject = record.subject.casefold()
    object_ = record.object.casefold()
    for claim in claims:
        folded = str(claim.get("claim") or "").casefold()
        if subject in folded and object_ in folded and all(word in folded for word in words):
            return True
    return False


def _claim_units(claims: list[dict]) -> list[dict]:
    units = []
    for index, claim in enumerate(claims):
        text = str(claim.get("claim") or "").strip()
        if not text:
            continue
        units.append({
            "evidence_id": f"claim:{index + 1}",
            "channel": "primary_source",
            "text": text,
            "priority": 4,
            "query_overlap": 0,
            "temporal": 0,
            "specificity": 0,
            "provenance": 1,
            "redundant": False,
            "retrieval_order": index,
            "weight_hint": 0,
            "score_type": "verified_claim",
            "exclude_reasons": [],
            "record": claim,
        })
    return units


def render_prepared(
    prepared: dict,
    medium: dict,
    topic: str,
    claims: list[dict],
    evidence_status: str = "verified",
    style_exemplars: list[str] | None = None,
    count_tokens=None,
    budget=None,
) -> dict:
    """Budget one medium and render its prompt. Passages stay in the bundle."""
    grouped = {
        channel: list(units)
        for channel, units in prepared["ranked"]["grouped"].items()
    }
    verified_claims = claims if evidence_status == "verified" else []
    grouped["primary_source"] = _claim_units(verified_claims)
    for unit in grouped["source_supported_relationship"]:
        record = unit["record"]
        if _restates_claim(record, verified_claims) and "redundant_with_primary_source" not in unit["exclude_reasons"]:
            unit["exclude_reasons"] = [*unit["exclude_reasons"], "redundant_with_primary_source"]
            unit["redundant"] = True
    allocation = allocate_context(
        grouped,
        medium["medium"],
        count_tokens=count_tokens,
        budget=budget,
    )
    selected_claims = [unit["record"] for unit in allocation["selected"]["primary_source"]]
    selected_claim_ids = {unit["evidence_id"] for unit in allocation["selected"]["primary_source"]}
    dropped_claims = [
        unit["record"].get("claim")
        for unit in grouped["primary_source"]
        if unit["evidence_id"] not in selected_claim_ids
    ]
    rendered = render_context_prompt(
        allocation,
        medium,
        topic,
        selected_claims,
        evidence_status,
        style_exemplars,
    )
    return {
        **rendered,
        "allocation": allocation,
        "selected_claims": selected_claims,
        "dropped_claims": dropped_claims,
        "passages_retained": list(prepared["passages_retained"]),
    }


def context_media_jobs(
    topic: str,
    hits: list[dict],
    claims: list[dict],
    evidence_status: str,
    prepared: dict,
    style_for=None,
    count_tokens=None,
    budget=None,
) -> list[dict]:
    """One prepared bundle, four media. Raw passages and source ids stay out."""
    from host_finetune.compare_adapters import MEDIUMS

    jobs = []
    for medium in MEDIUMS:
        exemplars = style_for(medium["medium"]) if style_for is not None else None
        medium_budget = budget(medium["medium"]) if callable(budget) else budget
        rendered = render_prepared(
            prepared,
            medium,
            topic,
            claims,
            evidence_status,
            exemplars,
            count_tokens=count_tokens,
            budget=medium_budget,
        )
        checked = rendered["instruction"]
        voice = rendered["voice"]
        if voice:
            if voice not in checked:
                raise ValueError("style block missing from the graph-context prompt")
            checked = checked.replace(voice, "", 1)
        for hit in hits:
            text = (hit.get("text") or "").strip()
            if text and text in checked:
                raise ValueError("raw retrieved prose leaked into the final prompt")
        for claim in rendered["selected_claims"]:
            if claim.get("claim") and claim["claim"] not in rendered["instruction"]:
                raise ValueError("verified claim missing from the final prompt")
            span = claim.get("support_span") or ""
            if span and span not in (claim.get("claim") or "") and span in checked:
                raise ValueError("support span leaked into the final prompt")
            source_id = str(claim.get("source_id") or "")
            if source_id and source_id not in (claim.get("claim") or "") and source_id in checked:
                raise ValueError("source id leaked into the final prompt")
        for unit in rendered["allocation"]["selected"]["source_supported_relationship"]:
            record = unit["record"]
            span = record.verbatim_evidence or ""
            if span and span not in (record.predicate or "") and span in checked:
                raise ValueError("relationship support span leaked into the final prompt")
            for source_id in record.supporting_document_ids:
                if source_id and source_id not in checked:
                    continue
                if source_id and source_id in checked:
                    raise ValueError("relationship source id leaked into the final prompt")
        supplementary = ""
        if "SUPPLEMENTARY GRAPH-DERIVED DESCRIPTIONS" in checked:
            supplementary = checked.split("SUPPLEMENTARY GRAPH-DERIVED DESCRIPTIONS", 1)[1]
        for description in rendered["description_texts"]:
            if not description:
                continue
            outside = checked.split("SUPPLEMENTARY GRAPH-DERIVED DESCRIPTIONS", 1)[0]
            if description in outside and description not in " ".join(rendered["grounding_texts"]):
                raise ValueError("derived description leaked above its section")
            if description in rendered["grounding_texts"]:
                raise ValueError("derived description was treated as a verified fact")
            if supplementary and description not in supplementary:
                raise ValueError("selected description missing from its section")
        if "gd_" in checked and any(token.startswith("gd_") for token in checked.split()):
            for token in checked.replace("\n", " ").split():
                if token.startswith("gd_") or "-chunk-" in token:
                    raise ValueError("internal document id leaked into the final prompt")
        selected_ids = [
            unit["evidence_id"]
            for rows in rendered["allocation"]["selected"].values()
            for unit in rows
        ]
        jobs.append({
            "medium": medium,
            "instruction": rendered["instruction"],
            "hits": hits,
            "claims": rendered["selected_claims"],
            "evidence_block": "PRIMARY SOURCE-GROUNDED CLAIMS",
            "evidence_status": evidence_status,
            "prompt_contract": PROMPT_CONTRACT,
            "graph_context_contract": rendered["graph_context_contract"],
            "style_applied": bool(voice),
            "style_exemplar_count": len([item for item in (exemplars or []) if item and item.strip()]),
            "relationships": [
                unit["record"].to_dict()
                for unit in rendered["allocation"]["selected"]["source_supported_relationship"]
            ],
            "relationship_block": "",
            "grounding_texts": list(rendered["grounding_texts"]),
            "context_pad": False,
            "model_f_context": {
                "profile": prepared["profile"],
                "schema_version": prepared["schema_version"],
                "selected_ids": selected_ids,
                "tokens_used": rendered["allocation"]["tokens_used"],
                "tokens_by_channel": {
                    channel: sum(unit.get("tokens") or 0 for unit in units)
                    for channel, units in rendered["allocation"]["selected"].items()
                },
                "token_estimate": rendered["allocation"]["budget"]["estimate"],
                "budget": dict(rendered["allocation"]["budget"]),
                "grounding_texts": list(rendered["grounding_texts"]),
                "passages_retained": list(rendered["passages_retained"]),
                "dropped_claims": list(rendered["dropped_claims"]),
            },
        })
    return jobs


def graph_context_callback(profile: str):
    """Return the generation hook, or None when the profile is off."""
    chosen = require_profile(profile)
    if chosen == "off":
        return None

    def _callback(topic_row, retrieval_payload):
        query = ""
        if isinstance(topic_row, dict):
            query = str(topic_row.get("topic") or "")
        if not query and isinstance(retrieval_payload, dict):
            query = str(retrieval_payload.get("query") or "")
        return prepare_context(retrieval_payload, profile=chosen, query=query)

    return _callback
