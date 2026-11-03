"""Refuse a Model F cell that did not actually use LightRAG graph context.

F-SOURCE may omit graph sections. It still has to be a LightRAG hybrid
retrieval with the graph-context prompt. ``model-f-context off`` and dense
retrieval are not arms of this experiment.
"""

from __future__ import annotations

from host_finetune.model_f_context.prompt import GRAPH_CONTEXT_CONTRACT
from host_finetune.model_f_experiment.arms import PREPARE_PROFILE


class ModelFGuardError(RuntimeError):
    """The cell would not be a LightRAG Model F generation."""


def require_lightrag_trace(trace: dict, arm: str) -> None:
    """Raise before a cell is saved when the trace is not the requested arm."""
    backend = trace.get("retrieval_backend")
    mode = trace.get("retrieval_mode")
    if backend != "lightrag" or mode != "hybrid":
        raise ModelFGuardError(
            f"{arm} retrieval was {backend!r} mode {mode!r}; "
            "this experiment requires LightRAG hybrid and does not fall back to dense retrieval"
        )
    contract = trace.get("graph_context_contract")
    if contract != GRAPH_CONTEXT_CONTRACT:
        raise ModelFGuardError(
            f"{arm} graph context contract was {contract!r}; "
            "model-f-context off is not an arm of this experiment"
        )
    profile = (trace.get("model_f_context") or {}).get("profile")
    expected = PREPARE_PROFILE[arm]
    if profile != expected:
        raise ModelFGuardError(
            f"{arm} context profile was {profile!r}; expected {expected!r}"
        )
    if arm != "F-SOURCE" and profile in {None, "off", "source"}:
        raise ModelFGuardError(
            f"{arm} did not receive its graph channel; profile {profile!r} is not a graph arm"
        )
