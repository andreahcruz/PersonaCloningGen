"""Controlled comparisons derived from the configuration registry.

A new LightRAG query against the blog expansion store is not. The arms share
question ids, the Relabel generator, and the recorded decoding settings. They
do not share one composite score.
"""

from __future__ import annotations

from host_finetune.model_f_context.registry import CONFIGURATIONS, validate_registry

_NOTES = {
    "E-DENSE": {
        "score_saved_traces_now": "Schema can score a dense trace. EXP-20261031-001 arm C is a 400-character pilot preview, not this arm.",
        "live_retrieval_now": "dense index only; do not retune on the corpus test split",
        "waits_for_blog_snapshot": False,
    },
    "F-OFF": {
        "score_saved_traces_now": "No saved generation used LightRAG with graph context left off.",
        "live_retrieval_now": "frozen hybrid payloads can be rescored only after a trace records this id",
        "waits_for_blog_snapshot": True,
    },
    "F-SOURCE": {
        "score_saved_traces_now": "A trace with the source profile can be scored. EXP-20261031-001 arm A is not this profile.",
        "live_retrieval_now": "pilot and frozen payloads only",
        "waits_for_blog_snapshot": True,
    },
    "F-REL": {
        "score_saved_traces_now": "A trace with the relationships profile can be scored. EXP-20261031-001 arm B is not this profile.",
        "live_retrieval_now": "saved relationship bundles only",
        "waits_for_blog_snapshot": True,
    },
    "F-DESC": {
        "score_saved_traces_now": "Current normalized payloads omit entity description text.",
        "live_retrieval_now": "blocked until query_documents keeps entity descriptions",
        "waits_for_blog_snapshot": True,
    },
    "F-HOP": {
        "score_saved_traces_now": "A trace with the multihop profile can be scored. No saved generation used this profile.",
        "live_retrieval_now": "selector is ready; the expansion graph is still being written",
        "waits_for_blog_snapshot": True,
    },
}

_DECODING = {"temperature": 0.7, "top_p": 0.9, "repetition_penalty": 1.15, "seed_base": 42}

EXPERIMENT_MATRIX = tuple(
    {
        "arm": row["id"],
        "architecture": row["architecture"] if row["context_profile"] is None else f"model_f_{row['context_profile']}",
        "uses_graph": row["context_profile"] in {"relationships", "descriptions", "multihop"},
        "retrieval_backend": row["backend"],
        "lightrag_mode": row["lightrag_mode"],
        "context_profile": row["context_profile"],
        "prompt_family": row["prompt_family"],
        "evidence_budget": 4,
        "generator": "relabel_qlora",
        "decoding": dict(_DECODING),
        **_NOTES[row["id"]],
    }
    for row in CONFIGURATIONS
)

CONTROLS = {
    "questions": "Same question ids on every arm. Development questions are the pilot and graph-relationship review files. The 30-topic gold file is a generation development set, not the corpus test split.",
    "generator": "Frozen Relabel adapter. Do not change the adapter between arms.",
    "decoding": "temperature 0.7, top_p 0.9, repetition penalty 1.15, seed base 42, as compare_adapters records them.",
    "context_budget": "Model E passes four source passages. Model F profiles use the per-medium budgets in model_f_context.budget. Utilization is reported and is not a quality metric.",
    "retrieval_difference": "Dense k counts passages. LightRAG top_k counts entities or relationships, and chunk_top_k counts chunks. Both arms then keep an evidence budget of four passages. Those ranks are not the same score.",
    "pairing": "Compare arms on the same question_id and medium. Report a paired difference and a bootstrap interval. Do not average those differences into one rank.",
    "held_out": "Do not read the corpus validation or test split to choose a profile, a prompt, or a budget.",
}


def matrix_by_architecture() -> dict[str, dict]:
    return {row["architecture"]: row for row in EXPERIMENT_MATRIX}


def profile_alignment() -> dict:
    """Confirm the matrix, the leakage list, and the generator profiles agree."""
    from host_finetune.model_f_context.leakage import EVALUATION_CONFIGURATIONS
    from host_finetune.model_f_context.pipeline import PROFILES

    checked = validate_registry()
    matrix_ids = [row["arm"] for row in EXPERIMENT_MATRIX]
    leakage_ids = [row["id"] for row in EVALUATION_CONFIGURATIONS]
    model_f_profiles = [row["context_profile"] for row in EXPERIMENT_MATRIX if row["architecture"] != "model_e"]
    dense = next(row for row in EXPERIMENT_MATRIX if row["arm"] == "E-DENSE")
    off = next(row for row in EXPERIMENT_MATRIX if row["arm"] == "F-OFF")
    return {
        "model_f_profiles": list(PROFILES),
        "registry": checked,
        "matrix_ids": matrix_ids,
        "leakage_ids": leakage_ids,
        "registries_agree": matrix_ids == leakage_ids == checked["ids"],
        "every_profile_has_one_arm": model_f_profiles == list(PROFILES),
        "descriptions_is_its_own_arm": checked["descriptions_id"] == "F-DESC",
        "multihop_id": checked["multihop_id"],
        "model_e_is_not_the_off_profile": dense["context_profile"] is None and off["context_profile"] == "off" and dense["retrieval_backend"] != off["retrieval_backend"],
        "leakage_has_separate_descriptions_arm": "descriptions" in [row["context_profile"] for row in EVALUATION_CONFIGURATIONS],
        "historical_october_31_cells_are_these_profiles": False,
    }
