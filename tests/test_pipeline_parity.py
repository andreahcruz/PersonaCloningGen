"""File-based parity for the frozen Relabel dataset and the RAG selection.

Selection is replayed in memory. The persisted collection is read with SQLite
mode=ro. Balance bytes are an explicit CRLF contract, not the host newline.
"""
import json
from pathlib import Path

import pytest

from host_finetune.verify_frozen_parity import (
    BALANCED_SHA256,
    evaluate,
    explicit_crlf_jsonl_bytes,
)

BLOCKING = (
    "relabel_counts",
    "relabel_dataset_hash",
    "relabel_split_hash",
    "exp008_split_not_relabel_split",
    "group_single_split",
    "sft_train_count",
    "sft_command_source",
    "rag_family_selection",
    "rag_final_count",
    "rag_held_out_absent",
    "rag_exclusion_split",
    "rag_all_train",
    "rag_text_is_output",
    "rag_roles",
    "rag_metadata",
    "historical_index_guard",
    "raw_cleaned_gold_hashes",
    "balanced_file_hash",
    "balance_policy",
    "balance_membership",
    "balance_order",
    "balance_byte_parity",
    "exp009_recorded_corpus",
    "persisted_chroma",
)

GAPS = (
    "balance_invocation",
    "embedding_build_provenance",
    "dirty_worktrees",
    "completed_inference_provenance",
)


@pytest.fixture(scope="module")
def checks() -> dict:
    return {row.name: row for row in evaluate()}


@pytest.mark.parametrize("name", BLOCKING)
def test_blocking_parity_check(checks, name):
    row = checks[name]
    assert row.blocking is True
    assert row.status == "PASS", row.detail()


@pytest.mark.parametrize("name", GAPS)
def test_gap_is_reported_and_not_blocking(checks, name):
    row = checks[name]
    assert row.blocking is False
    assert row.status in {"PASS", "FAIL", "UNVERIFIED"}, row.detail()


def test_persisted_chroma_matches_the_memory_selection(checks):
    row = checks["persisted_chroma"]
    assert row.blocking is True
    assert row.status == "PASS", row.detail()
    assert "missing=0" in row.actual
    assert "unexpected=0" in row.actual
    assert "train_26148=False" in row.actual
    assert "mode=ro" in row.evidence


def test_balance_invocation_stays_unrecorded(checks):
    invocation = checks["balance_invocation"]
    byte_parity = checks["balance_byte_parity"]
    assert invocation.status == "UNVERIFIED", invocation.detail()
    assert "unrecorded" in invocation.actual
    assert byte_parity.status == "PASS", byte_parity.detail()
    assert BALANCED_SHA256 in byte_parity.actual


def test_explicit_crlf_contract_does_not_use_text_mode():
    payload = explicit_crlf_jsonl_bytes([{"a": "é"}, {"b": "c"}])
    first = json.dumps({"a": "é"}, ensure_ascii=False).encode("utf-8")
    second = json.dumps({"b": "c"}, ensure_ascii=False).encode("utf-8")
    assert payload == first + b"\r\n" + second + b"\r\n"
    assert b"\n" not in payload.replace(b"\r\n", b"")
    source = Path(__file__).resolve().parents[1] / "host_finetune" / "verify_frozen_parity.py"
    text = source.read_text(encoding="utf-8")
    assert 'b"\\r\\n".join' in text
    assert "os.linesep" not in text


def test_verifier_does_not_construct_chroma_or_import_training():
    source = Path(__file__).resolve().parents[1] / "host_finetune" / "verify_frozen_parity.py"
    text = source.read_text(encoding="utf-8")
    assert "import chromadb" not in text
    assert "PersistentClient(" not in text
    assert "from host_finetune.audit_retrieval_corpus" not in text
    assert "import host_finetune.audit_retrieval_corpus" not in text
    assert "from host_finetune.rebuild_chroma" not in text
    assert "import host_finetune.rebuild_chroma" not in text
    assert "host_finetune.finetune" not in text
    assert "unsloth" not in text
