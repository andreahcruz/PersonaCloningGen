"""Safe configuration boundary for the future Model F LightRAG integration.

This module intentionally does *not* import LightRAG or open a graph store.
It lets the running API report why Model F is unavailable while the external
index is still being built.  The eventual query adapter must accept only a
verified, read-only snapshot that passes ``model_f_status`` first.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


CONTEXT_PROFILES = ("off", "source", "relationships", "descriptions", "multihop")
LIGHTRAG_MODES = ("naive", "local", "global", "hybrid", "mix")


@dataclass(frozen=True)
class ModelFStatus:
    enabled: bool
    ready: bool
    mode: str
    context_profile: str
    detail: str


def _enabled(value: str | None) -> bool:
    return (value or "false").strip().lower() in {"1", "true", "yes", "on"}


def model_f_status(environ: dict[str, str] | None = None) -> ModelFStatus:
    """Validate configuration without loading a package, model, or index file."""
    env = os.environ if environ is None else environ
    enabled = _enabled(env.get("MODEL_F_ENABLED"))
    mode = (env.get("LIGHTRAG_MODE") or "hybrid").strip().lower()
    profile = (env.get("MODEL_F_CONTEXT") or "off").strip().lower()
    workdir_value = (env.get("LIGHTRAG_WORKDIR") or "").strip()

    if mode not in LIGHTRAG_MODES:
        return ModelFStatus(enabled, False, mode, profile, f"Unsupported LightRAG mode: {mode}.")
    if profile not in CONTEXT_PROFILES:
        return ModelFStatus(enabled, False, mode, profile, f"Unsupported Model F context profile: {profile}.")
    if not enabled:
        return ModelFStatus(False, False, mode, profile, "Model F is disabled pending a verified LightRAG snapshot.")
    if not workdir_value:
        return ModelFStatus(True, False, mode, profile, "Model F needs LIGHTRAG_WORKDIR to reference a verified snapshot.")

    workdir = Path(workdir_value)
    if workdir.name == "storage_expand_v1":
        return ModelFStatus(True, False, mode, profile, "Refusing the live LightRAG indexing directory.")
    if not workdir.is_dir():
        return ModelFStatus(True, False, mode, profile, "Configured LightRAG snapshot directory is unavailable.")
    if (workdir / "indexing.lock").exists():
        return ModelFStatus(True, False, mode, profile, "Configured LightRAG snapshot is locked by an indexer.")

    # Do not return ready until a query adapter, snapshot catalog verification,
    # and live hybrid-query validation are added in the post-index phase.
    return ModelFStatus(
        True,
        False,
        mode,
        profile,
        "Snapshot path is safe to inspect, but Model F query integration is not enabled yet.",
    )
