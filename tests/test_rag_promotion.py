"""RAG v1/v2 selection. These tests do not open the production index."""

from __future__ import annotations

import json

import pytest

from host_finetune.rag_runtime import (
    LEGACY_COLLECTION,
    V1_COLLECTION,
    V1_DIR,
    V2_COLLECTION,
    V2_DIR,
    RagConfigError,
    catalog,
    promote_shadow_directory,
    read_active_version,
    resolve_rag,
)
from host_finetune.spark_v2_embed import compare_rankings


def test_default_active_file_is_v1_before_cutover(tmp_path):
    config = tmp_path / "rag_active.json"
    config.write_text(json.dumps({"version": "v1"}), encoding="utf-8")
    selected = resolve_rag(config_path=config)
    assert selected["version"] == "v1"
    assert selected["collection"] == V1_COLLECTION
    assert selected["path"] == V1_DIR


def test_explicit_v2_selection_uses_the_promoted_directory():
    selected = resolve_rag("v2")
    assert selected["collection"] == V2_COLLECTION
    assert selected["path"] == V2_DIR
    assert selected["path"] != V1_DIR


def test_invalid_version_and_legacy_collection_are_refused(tmp_path):
    config = tmp_path / "rag_active.json"
    config.write_text(json.dumps({"version": "v3"}), encoding="utf-8")
    with pytest.raises(RagConfigError):
        read_active_version(config)
    with pytest.raises(RagConfigError):
        resolve_rag(LEGACY_COLLECTION)
    assert "v1" in catalog() and "v2" in catalog()
    assert all(item["collection"] != LEGACY_COLLECTION for item in catalog().values())


def test_rollback_file_restores_v1(tmp_path):
    config = tmp_path / "rag_active.json"
    config.write_text(json.dumps({"version": "v2"}), encoding="utf-8")
    assert resolve_rag(config_path=config)["version"] == "v2"
    config.write_text(json.dumps({"version": "v1"}), encoding="utf-8")
    assert resolve_rag(config_path=config)["version"] == "v1"


def test_promotion_refuses_to_write_v1(tmp_path):
    source = tmp_path / "shadow"
    source.mkdir()
    (source / "chroma.sqlite3").write_bytes(b"not-opened")
    with pytest.raises(RagConfigError):
        promote_shadow_directory(source, V1_DIR)


def test_cutoff_tie_is_retrieval_equivalent_and_a_real_swap_is_not():
    tied = compare_rankings(
        [{"id": "a", "distance": 0.1}, {"id": "b", "distance": 0.2}],
        [{"id": "a", "distance": 0.1}, {"id": "c", "distance": 0.2}],
    )
    assert tied["cutoff_tie"] is True
    assert tied["exact_rank"] is False
    swapped = compare_rankings(
        [{"id": "a", "distance": 0.1}, {"id": "b", "distance": 0.2}],
        [{"id": "a", "distance": 0.1}, {"id": "c", "distance": 0.3}],
    )
    assert swapped["cutoff_tie"] is False
    assert swapped["exact_rank"] is False
