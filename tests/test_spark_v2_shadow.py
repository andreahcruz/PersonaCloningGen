"""Shadow v2 stage contracts. The corpus test replays raw through fit."""

import json
from pathlib import Path

import pytest

from host_finetune.canonical_cleaning import clean_record
from host_finetune.spark_v2 import (
    EXPECTED_FIT_SHA256,
    FROZEN_FIT,
    ParityError,
    assert_staging_dir,
    clean_indexed,
    clean_partition,
    file_sha256,
    gate,
    restore_order,
)
import lemkin_canonical_v2_shadow as shadow_dag

ROOT = Path(__file__).resolve().parents[1]


def _item(index: int, text: str, reply: bool = False) -> dict:
    return {
        "source_ordinal": 2,
        "raw_record_index": index,
        "source_file": "jasonlk_originals.jsonl",
        "medium": "x",
        "record": {"text": text, "is_reply": reply, "is_repost_or_quote": False},
    }


def test_clean_partition_calls_the_canonical_cleaner():
    dropped = clean_indexed(_item(1, "hello there", reply=True))
    assert dropped["kept"] is False
    assert dropped["frozen_reason"] == "reply_or_repost"
    assert dropped["frozen_reason"] == clean_record("x", _item(1, "hello there", reply=True)["record"]).frozen_reason


def test_partition_order_is_restored_explicitly():
    items = [_item(index, f"This is long enough post number {index} for the floor.") for index in (1, 2, 3, 4, 5)]
    partitions = [items[3:], items[:1], items[1:3]]
    scrambled = [row for part in partitions for row in clean_partition(part)]
    ordered = restore_order(scrambled)
    assert [row["raw_record_index"] for row in ordered] == [1, 2, 3, 4, 5]
    assert [row["raw_record_index"] for row in scrambled] != [1, 2, 3, 4, 5]


def test_parity_gate_fails_the_stage():
    with pytest.raises(ParityError):
        gate("canonical_clean", {"content": 2, "ordering": 0})
    gate("canonical_clean", {"content": 0, "ordering": 0})


def test_staging_directory_cannot_be_a_frozen_corpus_path():
    with pytest.raises(ParityError):
        assert_staging_dir(ROOT / "host_finetune" / "data")
    with pytest.raises(ParityError):
        assert_staging_dir(ROOT / "data" / "cleaned")


def test_shadow_dag_stops_at_fit_and_keeps_later_stages_out_of_the_graph():
    assert shadow_dag.DAG_ID == "lemkin_canonical_v2_shadow"
    assert shadow_dag.FUTURE_STAGES == (
        "balance",
        "split",
        "relabel",
        "rag_governance",
        "embeddings",
        "chroma_publish",
    )
    assert "embeddings" not in {task.task_id for task in shadow_dag.dag.tasks} if hasattr(shadow_dag.dag, "tasks") else True


def test_shadow_pipeline_matches_the_python_oracle_through_fit(tmp_path):
    import os
    import subprocess

    before = file_sha256(FROZEN_FIT)
    venv = ROOT / "host_finetune" / ".venv" / "Scripts" / "python.exe"
    staging = tmp_path / "shadow"
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = ""
    env["PYTHONIOENCODING"] = "utf-8"
    completed = subprocess.run(
        [
            str(venv),
            "-m",
            "host_finetune.spark_v2",
            "--staging",
            str(staging),
            "--run-id",
            "test-run",
            "--through",
            "parity_summary",
            "--executor",
            "ordered",
        ],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    after = file_sha256(FROZEN_FIT)
    assert before == after == EXPECTED_FIT_SHA256
    summary = json.loads((staging / "manifests" / "parity_summary.json").read_text(encoding="utf-8"))
    fit = json.loads((staging / "manifests" / "canonical_fit.json").read_text(encoding="utf-8"))
    cleaned = json.loads((staging / "manifests" / "canonical_clean.json").read_text(encoding="utf-8"))
    nopromo = json.loads((staging / "manifests" / "build_nopromo.json").read_text(encoding="utf-8"))
    assert summary["status"] == "PASS"
    assert summary["config_version"] == "spark-v2-shadow-fit-v1"
    assert cleaned["output_rows"] == 30370
    assert cleaned["mismatches"] == {
        "membership": 0,
        "drop_counts": 0,
        "kept_counts": 0,
        "ordering": 0,
        "content": 0,
        "identity": 0,
    }
    assert nopromo["output_rows"] == 42771
    assert all(value == 0 for value in nopromo["mismatches"].values())
    assert fit["output_rows"] == 43012
    assert fit["output_sha256"] == EXPECTED_FIT_SHA256
    assert all(value == 0 for value in fit["mismatches"].values())
    assert "host_finetune/data" not in staging.as_posix()
