"""Compose Model F graph context without opening the live index.

``prepare_context`` normalizes, selects paths, and ranks. Prompt budgeting is
per medium so a short post and a blog do not share one cap. The default
profile for generation remains off; ``graph_context_callback`` returns None
until a caller selects a profile.
"""

from __future__ import annotations

import re

from host_finetune.evidence_capsule import PROMPT_CONTRACT, unauthorized_raw_passage
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


def _ranges(haystack: str, needle: str) -> list[tuple[int, int]]:
    if not needle:
        return []
    found = []
    start = 0
    while True:
        index = haystack.find(needle, start)
        if index < 0:
            return found
        found.append((index, index + len(needle)))
        start = index + 1


def _span_outside(haystack: str, span: str, allowed: list[tuple[int, int]]) -> bool:
    """True when a support span occurs outside the topic and the rendered claims."""
    if not span:
        return False
    start = 0
    while True:
        index = haystack.find(span, start)
        if index < 0:
            return False
        end = index + len(span)
        if not any(left <= index and end <= right for left, right in allowed):
            return True
        start = index + 1


def _unit_key(channel: str, unit: dict) -> str:
    return f"{channel}:{unit.get('evidence_id')}"


def _graph_tail_key(rendered: dict) -> str | None:
    selected = rendered["allocation"]["selected"]
    for channel in ("derived_description", "multi_hop_path", "source_supported_relationship"):
        units = selected.get(channel) or []
        if units:
            return _unit_key(channel, units[-1])
    return None


def _without_ids(prepared: dict, omit: set[str]) -> dict:
    if not omit:
        return prepared
    grouped = {}
    for channel, units in prepared["ranked"]["grouped"].items():
        copied = []
        for unit in units:
            item = dict(unit)
            if _unit_key(channel, unit) in omit:
                reasons = list(item.get("exclude_reasons") or [])
                if "window_item_does_not_fit" not in reasons:
                    reasons.append("window_item_does_not_fit")
                item["exclude_reasons"] = reasons
            copied.append(item)
        grouped[channel] = copied
    restricted = dict(prepared)
    restricted["ranked"] = {**prepared["ranked"], "grouped": grouped}
    return restricted


def _window_room(medium: dict) -> int:
    """Chat tokens left for the prompt after the requested output.

    The configured channel caps and ``max_new_tokens`` stay as they are. A
    graph item that does not fit this room is omitted whole.
    """
    from host_finetune.compare_adapters import MAX_SEQ_LENGTH

    return MAX_SEQ_LENGTH - int(medium["max_new_tokens"])


def _render_within_window(prepared, medium, topic, claims, evidence_status, exemplars, count_tokens, medium_budget):
    """Drop trailing graph items until the chat prompt fits. Claims stay."""
    measurer = prepared.get("window_tokens")
    omit: list[str] = []
    rendered = None
    while True:
        rendered = render_prepared(
            _without_ids(prepared, set(omit)),
            medium,
            topic,
            claims,
            evidence_status,
            exemplars,
            count_tokens=count_tokens,
            budget=medium_budget,
        )
        if measurer is None:
            return rendered, omit
        size = int(measurer(rendered["instruction"]))
        if size <= _window_room(medium):
            return rendered, omit
        tail = _graph_tail_key(rendered)
        if tail is None or tail in omit or len(omit) >= 64:
            return rendered, omit
        omit.append(tail)


def _assert_rendered_prompt(rendered: dict, hits: list[dict], topic: str) -> None:
    checked = rendered["instruction"]
    voice = rendered["voice"]
    if voice:
        if voice not in checked:
            raise ValueError("style block missing from the graph-context prompt")
        checked = checked.replace(voice, "", 1)
    leaked = unauthorized_raw_passage(checked, hits, rendered["selected_claims"])
    if leaked:
        raise ValueError(leaked)
    allowed = _ranges(checked, topic)
    for claim in rendered["selected_claims"]:
        text = str(claim.get("claim") or "")
        allowed.extend(_ranges(checked, text))
        allowed.extend(_ranges(checked, re.sub(r"\s+", " ", text).strip()))
    for claim in rendered["selected_claims"]:
        if claim.get("claim") and claim["claim"] not in rendered["instruction"]:
            raise ValueError("verified claim missing from the final prompt")
        span = claim.get("support_span") or ""
        if span and span not in (claim.get("claim") or "") and _span_outside(checked, span, allowed):
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
        rendered, omitted = _render_within_window(
            prepared,
            medium,
            topic,
            claims,
            evidence_status,
            exemplars,
            count_tokens,
            medium_budget,
        )
        _assert_rendered_prompt(rendered, hits, topic)
        voice = rendered["voice"]
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
                "window_omitted": list(omitted),
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
