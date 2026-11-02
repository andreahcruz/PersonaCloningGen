"""Token budgets for Model F context. Primary claims are reserved first.

The default counter is the project's conservative characters/4 estimate from
``indexing_schedule.CHARS_PER_TOKEN_ESTIMATE``. It is not a Llama tokenizer
count. A caller that already holds the generation tokenizer may pass it.
An item that does not fit is omitted whole. Source evidence is not removed
to make room for a generated description.
"""

from __future__ import annotations

from dataclasses import dataclass

CHARS_PER_TOKEN_ESTIMATE = 4


@dataclass(frozen=True)
class ChannelBudget:
    total: int
    primary: int
    relationships: int
    paths: int
    descriptions: int


BUDGETS = {
    "x": ChannelBudget(total=480, primary=320, relationships=80, paths=40, descriptions=40),
    "linkedin": ChannelBudget(total=800, primary=480, relationships=160, paths=80, descriptions=80),
    "blog": ChannelBudget(total=1400, primary=800, relationships=280, paths=160, descriptions=120),
    "talk": ChannelBudget(total=1400, primary=800, relationships=280, paths=160, descriptions=120),
}


def estimate_tokens(text: str) -> int:
    """Characters divided by 4, rounded up. Empty text costs nothing."""
    if not text:
        return 0
    return max(1, (len(text) + CHARS_PER_TOKEN_ESTIMATE - 1) // CHARS_PER_TOKEN_ESTIMATE)


def allocate_context(
    grouped: dict,
    medium: str,
    *,
    count_tokens=None,
    budget: ChannelBudget | None = None,
) -> dict:
    """Fill primary, then relationships, paths, and derived descriptions."""
    if budget is None and medium not in BUDGETS:
        raise ValueError(f"unknown medium {medium!r}")
    counter = count_tokens or estimate_tokens
    caps = budget or BUDGETS[medium]
    channel_caps = {
        "primary_source": caps.primary,
        "source_supported_relationship": caps.relationships,
        "multi_hop_path": caps.paths,
        "derived_description": caps.descriptions,
    }
    order = (
        "primary_source",
        "source_supported_relationship",
        "multi_hop_path",
        "derived_description",
    )
    selected = {name: [] for name in order}
    audit = []
    used = 0
    spent = {name: 0 for name in order}
    for channel in order:
        cap = channel_caps[channel]
        heading = {
            "primary_source": "PRIMARY SOURCE-GROUNDED CLAIMS",
            "source_supported_relationship": "VERIFIED RELATIONSHIPS",
            "multi_hop_path": "MULTI-HOP EVIDENCE PATHS",
            "derived_description": "SUPPLEMENTARY GRAPH-DERIVED DESCRIPTIONS",
        }[channel]
        heading_cost = counter(heading + "\n")
        heading_charged = False
        for unit in grouped.get(channel) or []:
            if unit.get("exclude_reasons"):
                audit.append({
                    "evidence_id": unit["evidence_id"],
                    "channel": channel,
                    "decision": "excluded",
                    "reasons": list(unit["exclude_reasons"]),
                })
                continue
            text = str(unit.get("display") or unit.get("text") or "")
            cost = counter(text)
            extra = 0 if heading_charged else heading_cost
            if cost <= 0:
                audit.append({
                    "evidence_id": unit["evidence_id"],
                    "channel": channel,
                    "decision": "excluded",
                    "reasons": ["empty_evidence"],
                })
                continue
            if spent[channel] + extra + cost > cap or used + extra + cost > caps.total:
                audit.append({
                    "evidence_id": unit["evidence_id"],
                    "channel": channel,
                    "decision": "excluded",
                    "reasons": ["budget_item_does_not_fit"],
                })
                continue
            spent[channel] += extra + cost
            used += extra + cost
            heading_charged = True
            chosen = dict(unit)
            chosen["tokens"] = cost
            selected[channel].append(chosen)
            audit.append({
                "evidence_id": unit["evidence_id"],
                "channel": channel,
                "decision": "selected",
                "reasons": ["fits_channel_budget", f"tokens:{cost}"],
            })
    return {
        "medium": medium,
        "budget": {
            "total": caps.total,
            "primary": caps.primary,
            "relationships": caps.relationships,
            "paths": caps.paths,
            "descriptions": caps.descriptions,
            "estimate": "characters_divided_by_4" if counter is estimate_tokens else "caller_tokenizer",
        },
        "selected": selected,
        "tokens_used": used,
        "tokens_by_channel": spent,
        "audit": audit,
    }
