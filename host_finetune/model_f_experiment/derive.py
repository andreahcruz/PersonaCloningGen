"""Build four prompt plans from one frozen retrieval payload."""

from __future__ import annotations

from host_finetune.model_f_context.pipeline import prepare_context, render_prepared
from host_finetune.model_f_experiment.arms import (
    PREPARE_PROFILE,
    RENDERED_CHANNELS,
    experiment_budget,
)


def restrict_channels(prepared: dict, allowed: frozenset[str]) -> dict:
    """Copy the plan and keep only the sections this arm is allowed to show."""
    ranked = prepared["ranked"]
    grouped = {}
    for channel, units in ranked["grouped"].items():
        copied = []
        for unit in units:
            item = dict(unit)
            reasons = list(item.get("exclude_reasons") or [])
            if channel not in allowed and "arm_excludes_channel" not in reasons:
                reasons.append("arm_excludes_channel")
            record = item.get("record")
            hops = getattr(record, "hops", None)
            if channel == "multi_hop_path" and hops is not None and hops < 2:
                if "direct_edge_is_not_a_multi_hop_path" not in reasons:
                    reasons.append("direct_edge_is_not_a_multi_hop_path")
            item["exclude_reasons"] = reasons
            copied.append(item)
        grouped[channel] = copied
    restricted = dict(prepared)
    restricted["ranked"] = {**ranked, "grouped": grouped}
    return restricted


def derive_arm(
    payload: dict,
    query: str,
    arm: str,
    claims: list[dict],
    medium: dict,
    *,
    catalog: dict | None = None,
    failed_document_ids: set[str] | None = None,
    count_tokens=None,
    evidence_status: str = "verified",
) -> dict:
    prepared = prepare_context(
        payload,
        profile=PREPARE_PROFILE[arm],
        query=query,
        catalog=catalog,
        failed_document_ids=failed_document_ids,
    )
    prepared = restrict_channels(prepared, RENDERED_CHANNELS[arm])
    prepared["experiment_budget"] = experiment_budget(medium["medium"])
    prepared["count_tokens"] = count_tokens
    rendered = render_prepared(
        prepared,
        medium,
        query,
        claims,
        evidence_status,
        count_tokens=count_tokens,
        budget=prepared["experiment_budget"],
    )
    allocation = rendered["allocation"]
    selected = allocation["selected"]
    return {
        "arm": arm,
        "prepare_profile": PREPARE_PROFILE[arm],
        "rendered_channels": sorted(RENDERED_CHANNELS[arm]),
        "passages_retained": list(prepared["passages_retained"]),
        "instruction": rendered["instruction"],
        "grounding_texts": list(rendered["grounding_texts"]),
        "tokens_used": allocation["tokens_used"],
        "tokens_by_channel": {
            channel: sum(unit.get("tokens") or 0 for unit in units)
            for channel, units in selected.items()
        },
        "budget": prepared["experiment_budget"].__dict__,
        "token_source": "caller" if count_tokens is not None else "characters_div_4",
        "entity_descriptions_not_retained": prepared["bundle"]["diagnostics"].get("entity_descriptions_not_retained", 0),
        "paths": len(prepared["paths"]["paths"]),
        "prepared": prepared,
    }


def arm_callback(
    arm: str,
    *,
    catalog: dict | None = None,
    failed_document_ids: set[str] | None = None,
    count_tokens=None,
):
    """Hook for compare_adapters.generate. It does not retrieve."""

    def _callback(topic_row, payload):
        query = ""
        if isinstance(topic_row, dict):
            query = str(topic_row.get("topic") or "")
        if not query and isinstance(payload, dict):
            query = str(payload.get("query") or "")
        prepared = prepare_context(
            payload,
            profile=PREPARE_PROFILE[arm],
            query=query,
            catalog=catalog,
            failed_document_ids=failed_document_ids,
        )
        prepared = restrict_channels(prepared, RENDERED_CHANNELS[arm])
        prepared["count_tokens"] = count_tokens
        prepared["experiment_budget"] = experiment_budget
        return prepared

    return _callback
