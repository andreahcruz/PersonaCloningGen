"""The four LightRAG arms. Retrieval is shared. Only the prompt sections change.

The generator profiles ``descriptions`` and ``multihop`` also keep other
channels. This experiment restricts the rendered sections so each arm adds
one kind of graph context to the same source passages.
"""

from __future__ import annotations

import hashlib
import json

from host_finetune.model_f_context.budget import BUDGETS, ChannelBudget

EXPERIMENT_ID = "EXP-20261101-009-model-f-four-arm"
ARMS = ("F-SOURCE", "F-REL", "F-DESC", "F-HOP")

# Profile used to build candidates. A later restriction removes extra channels.
PREPARE_PROFILE = {
    "F-SOURCE": "source",
    "F-REL": "relationships",
    "F-DESC": "descriptions",
    "F-HOP": "multihop",
}

RENDERED_CHANNELS = {
    "F-SOURCE": frozenset({"primary_source"}),
    "F-REL": frozenset({"primary_source", "source_supported_relationship"}),
    "F-DESC": frozenset({"primary_source", "derived_description"}),
    "F-HOP": frozenset({"primary_source", "multi_hop_path"}),
}

DECODING = {
    "temperature": 0.7,
    "top_p": 0.9,
    "repetition_penalty": 1.15,
    "seed_base": 42,
}
RETRIEVAL = {
    "backend": "lightrag",
    "mode": "hybrid",
    "evidence_budget": 4,
    "top_k": 20,
}
ADAPTER = "host_finetune/output/lemkin_lora_relabel"
PROMPT_FAMILY = "model-f-graph-context-v1"

# Channel caps match the generator defaults. The shared total is the sum of
# those caps plus room for section headings, so a graph section cannot push a
# source claim out and a heading cannot consume the source cap.
_HEADING_ALLOWANCE = 80


# Blog and talk ask for 768 new tokens. On the longest development question
# the chat prompt, a 40-token retry line, and label overhead leave about 640
# tokens for evidence inside the 2048 window. The default 800 + 120 split does
# not fit. Both caps are scaled by the same factor, and both media use the
# tighter result so neither arm gets a larger source or graph allowance.
_WINDOW_FITTED = {
    "blog": (557, 83),
    "talk": (557, 83),
}


def experiment_budget(medium: str) -> ChannelBudget:
    """Same source cap, and the same graph cap, for every arm.

    The generator's default relationship cap is larger than its path and
    description caps. Using those defaults would let F-REL add more tokens
    than F-DESC or F-HOP. The experiment uses the smallest of the three,
    then lowers blog and talk so the full prompt plus the requested output
    stays inside the configured 2048-token window.
    """
    base = BUDGETS[medium]
    graph = min(base.relationships, base.paths, base.descriptions)
    primary = base.primary
    fitted = _WINDOW_FITTED.get(medium)
    if fitted is not None:
        primary, graph = fitted
    return ChannelBudget(
        total=primary + graph + _HEADING_ALLOWANCE,
        primary=primary,
        relationships=graph,
        paths=graph,
        descriptions=graph,
    )


def fingerprint(question_ids: list[str]) -> str:
    payload = {
        "arms": list(ARMS),
        "rendered_channels": {arm: sorted(RENDERED_CHANNELS[arm]) for arm in ARMS},
        "decoding": DECODING,
        "retrieval": RETRIEVAL,
        "adapter": ADAPTER,
        "prompt_family": PROMPT_FAMILY,
        "budgets": {
            medium: experiment_budget(medium).__dict__
            for medium in ("blog", "linkedin", "x", "talk")
        },
        "question_ids": list(question_ids),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()
