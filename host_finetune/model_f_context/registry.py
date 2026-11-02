"""Authoritative Model E and Model F evaluation arms.

``pipeline.PROFILES`` is the set the generator accepts. This registry is the
set an experiment may run. Model E is not the ``off`` profile: both leave
the graph-context callback empty, and they use different retrievers.
"""

from __future__ import annotations

from host_finetune.model_f_context.pipeline import PROFILES

CONFIGURATIONS = (
    {
        "id": "E-DENSE",
        "architecture": "model_e",
        "backend": "dense",
        "lightrag_mode": None,
        "context_profile": None,
        "graph_context_callback": None,
        "prompt_family": "factual-reliability-v1",
        "label": "Model E dense baseline",
    },
    {
        "id": "F-OFF",
        "architecture": "model_f",
        "backend": "lightrag",
        "lightrag_mode": "hybrid",
        "context_profile": "off",
        "graph_context_callback": None,
        "prompt_family": "factual-reliability-v1",
        "label": "Model F retrieval with graph context off",
    },
    {
        "id": "F-SOURCE",
        "architecture": "model_f",
        "backend": "lightrag",
        "lightrag_mode": "hybrid",
        "context_profile": "source",
        "graph_context_callback": "source",
        "prompt_family": "model-f-graph-context-v1",
        "label": "Model F source evidence",
    },
    {
        "id": "F-REL",
        "architecture": "model_f",
        "backend": "lightrag",
        "lightrag_mode": "hybrid",
        "context_profile": "relationships",
        "graph_context_callback": "relationships",
        "prompt_family": "model-f-graph-context-v1",
        "label": "Model F verified relationships",
    },
    {
        "id": "F-DESC",
        "architecture": "model_f",
        "backend": "lightrag",
        "lightrag_mode": "hybrid",
        "context_profile": "descriptions",
        "graph_context_callback": "descriptions",
        "prompt_family": "model-f-graph-context-v1",
        "label": "Model F graph descriptions",
    },
    {
        "id": "F-HOP",
        "architecture": "model_f",
        "backend": "lightrag",
        "lightrag_mode": "hybrid",
        "context_profile": "multihop",
        "graph_context_callback": "multihop",
        "prompt_family": "model-f-graph-context-v1",
        "label": "Model F multi-hop context",
    },
)


def validate_registry() -> dict:
    """Reject a registry that collapses two profiles or mislabels Model E."""
    ids = [row["id"] for row in CONFIGURATIONS]
    if len(ids) != len(set(ids)):
        raise ValueError("evaluation ids must be unique")
    profiles = [row["context_profile"] for row in CONFIGURATIONS if row["architecture"] == "model_f"]
    if len(profiles) != len(set(profiles)):
        raise ValueError("each Model F profile may appear once")
    if set(profiles) != set(PROFILES):
        missing = set(PROFILES) - set(profiles)
        raise ValueError("registry is missing Model F profiles: " + ", ".join(sorted(missing)))
    by_id = {row["id"]: row for row in CONFIGURATIONS}
    dense = by_id["E-DENSE"]
    off = by_id["F-OFF"]
    descriptions = by_id["F-DESC"]
    multihop = by_id["F-HOP"]
    if dense["backend"] != "dense" or dense["context_profile"] is not None:
        raise ValueError("Model E is dense retrieval and is not the off profile")
    if off["architecture"] != "model_f" or off["backend"] != "lightrag" or off["context_profile"] != "off":
        raise ValueError("off is a Model F LightRAG arm")
    if dense["graph_context_callback"] is None and off["graph_context_callback"] is None:
        if dense["backend"] == off["backend"] and dense["prompt_family"] == off["prompt_family"]:
            raise ValueError("Model E and Model F off must differ by retrieval backend")
    if descriptions["context_profile"] != "descriptions":
        raise ValueError("descriptions id must select the descriptions profile")
    if multihop["context_profile"] != "multihop" or multihop["id"] == descriptions["id"]:
        raise ValueError("multi-hop must not replace descriptions")
    return {
        "ids": ids,
        "model_f_profiles": profiles,
        "descriptions_id": descriptions["id"],
        "multihop_id": multihop["id"],
        "model_e_id": dense["id"],
        "off_id": off["id"],
    }
