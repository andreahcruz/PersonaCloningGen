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


def test_spark_executor_does_not_fall_back_when_pyspark_is_missing():
    with pytest.raises(ParityError, match="pyspark"):
        from host_finetune.spark_v2 import clean_with_spark

        clean_with_spark([_item(1, "This is long enough post number 1 for the floor.")])


def test_unknown_executor_is_rejected(tmp_path):
    from host_finetune.spark_v2 import stage_clean

    with pytest.raises(ParityError, match="unknown executor"):
        stage_clean(tmp_path / "stage", "run", [], executor="mystery")


def test_parity_gate_fails_the_stage():
    with pytest.raises(ParityError):
        gate("canonical_clean", {"content": 2, "ordering": 0})
    gate("canonical_clean", {"content": 0, "ordering": 0})


def test_staging_directory_cannot_be_a_frozen_corpus_path():
    with pytest.raises(ParityError):
        assert_staging_dir(ROOT / "host_finetune" / "data")
    with pytest.raises(ParityError):
        assert_staging_dir(ROOT / "data" / "cleaned")


def test_shadow_dag_orders_parity_gates_ahead_of_the_next_stage():
    assert shadow_dag.DAG_ID == "lemkin_canonical_v2_shadow"
    assert shadow_dag.FUTURE_STAGES == ("production_cutover",)
    order = shadow_dag.TASK_ORDER
    assert order[0] == "validate_raw"
    assert order[-1] == "parity_summary"
    assert "production_cutover" not in order
    assert "lemkin_train_only" not in order
    for gate, downstream in (
        ("clean_parity", "build_nopromo"),
        ("nopromo_parity", "canonical_fit"),
        ("fit_parity", "balance"),
        ("balance_parity", "split"),
        ("split_parity", "relabel"),
        ("relabel_parity", "rag_governance"),
        ("rag_parity", "embed_rag"),
        ("embedding_parity", "build_chroma_shadow"),
        ("retrieval_parity", "parity_summary"),
    ):
        assert order.index(gate) < order.index(downstream)


def test_recorded_parity_failure_stops_the_stage(tmp_path):
    from host_finetune.spark_v2 import stage_recorded_parity, write_manifest

    staging = tmp_path / "shadow"
    write_manifest(
        staging,
        "canonical_clean",
        {
            "run_id": "gate",
            "executor": "spark",
            "status": "FAIL",
            "mismatches": {"content": 2},
        },
    )
    with pytest.raises(ParityError):
        stage_recorded_parity(staging, "gate", "canonical_clean", "clean_parity")


def test_manifest_records_the_executor(tmp_path):
    from host_finetune.spark_v2 import write_manifest

    path = write_manifest(
        tmp_path / "shadow",
        "canonical_clean",
        {"run_id": "exec", "executor": "spark", "status": "PASS", "mismatches": {}},
    )
    document = json.loads(path.read_text(encoding="utf-8"))
    assert document["executor"] == "spark"
    assert document["config_version"] == "spark-v2-shadow-embed-v1"


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
            "fit_parity",
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
    summary = json.loads((staging / "manifests" / "fit_parity.json").read_text(encoding="utf-8"))
    fit = json.loads((staging / "manifests" / "canonical_fit.json").read_text(encoding="utf-8"))
    cleaned = json.loads((staging / "manifests" / "canonical_clean.json").read_text(encoding="utf-8"))
    nopromo = json.loads((staging / "manifests" / "build_nopromo.json").read_text(encoding="utf-8"))
    assert summary["status"] == "PASS"
    assert summary["config_version"] == "spark-v2-shadow-embed-v1"
    assert cleaned["executor"] == "ordered"
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


def test_balance_split_relabel_and_rag_match_the_frozen_oracle(tmp_path):
    import os
    import subprocess

    from host_finetune.spark_v2 import (
        EXPECTED_BALANCED_SHA256,
        EXPECTED_RELABEL_SHA256,
        FROZEN_BALANCED,
        FROZEN_FIT,
        RELABEL_DATASET,
        file_sha256,
    )

    before = {
        "fit": file_sha256(FROZEN_FIT),
        "balanced": file_sha256(FROZEN_BALANCED),
        "relabel": file_sha256(RELABEL_DATASET),
    }
    venv = ROOT / "host_finetune" / ".venv" / "Scripts" / "python.exe"
    staging = tmp_path / "downstream"
    script = r"""
import json, os, sys
from pathlib import Path
os.environ["CUDA_VISIBLE_DEVICES"] = ""
from host_finetune.spark_v2 import (
    FROZEN_FIT, HELD_OUT_RAG_ID, _read_jsonl, stage_balance, stage_rag,
    stage_relabel, stage_split,
)
staging = Path(sys.argv[1])
fit_rows = _read_jsonl(FROZEN_FIT)
balanced = stage_balance(staging, "downstream", fit_rows)
assignments = stage_split(staging, "downstream", balanced)
relabel_rows, relabel_assignments = stage_relabel(staging, "downstream", balanced, assignments)
rag_rows = stage_rag(staging, "downstream", relabel_rows, relabel_assignments)
print(json.dumps({
    "balanced": len(balanced),
    "split": len(assignments),
    "relabel": len(relabel_rows),
    "rag": len(rag_rows),
    "held_out": HELD_OUT_RAG_ID in {row["row_id"] for row in rag_rows},
}))
"""
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = ""
    env["PYTHONIOENCODING"] = "utf-8"
    env["HF_HUB_OFFLINE"] = "1"
    env["TRANSFORMERS_OFFLINE"] = "1"
    completed = subprocess.run(
        [str(venv), "-c", script, str(staging)],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    payload = json.loads(completed.stdout.strip().splitlines()[-1])
    assert payload == {"balanced": 31328, "split": 31328, "relabel": 26545, "rag": 21193, "held_out": False}
    balance = json.loads((staging / "manifests" / "balance.json").read_text(encoding="utf-8"))
    split = json.loads((staging / "manifests" / "split.json").read_text(encoding="utf-8"))
    relabel = json.loads((staging / "manifests" / "relabel.json").read_text(encoding="utf-8"))
    rag = json.loads((staging / "manifests" / "rag_governance.json").read_text(encoding="utf-8"))
    assert balance["output_sha256"] == EXPECTED_BALANCED_SHA256
    assert balance["executor"] == "ordered-python"
    assert all(value == 0 for value in balance["mismatches"].values())
    assert split["details"]["split_counts"] == {"train": 25062, "validation": 3133, "test": 3133}
    assert split["mismatches"]["crossing_groups"] == 0
    assert split["mismatches"]["ownership"] == 0
    assert relabel["output_sha256"] == EXPECTED_RELABEL_SHA256
    assert relabel["details"]["split_counts"] == {"train": 21377, "validation": 2588, "test": 2580}
    assert rag["output_rows"] == 21193
    assert rag["details"]["headline_removed_ids"] == ["train_26148"]
    assert rag["details"]["roles"] == {"unchanged": 11611, "opening": 1687, "continuation": 7895}
    assert rag["mismatches"]["missing"] == 0
    assert rag["mismatches"]["unexpected"] == 0
    assert rag["mismatches"]["document"] == 0
    assert file_sha256(FROZEN_FIT) == before["fit"]
    assert file_sha256(FROZEN_BALANCED) == before["balanced"]
    assert file_sha256(RELABEL_DATASET) == before["relabel"]
    assert "host_finetune/data" not in staging.as_posix()


def test_embedding_request_uses_the_document_text_and_cpu_body():
    from host_finetune.canonical_embeddings import request_body
    from host_finetune.spark_v2_embed import embedding_input_is_document, select_sample

    row = {"row_id": "train_1", "document": "the output", "instruction": "write a post", "split": "train"}
    assert embedding_input_is_document(row) == "the output"
    body = request_body(["the output"])
    assert body == {"model": "nomic-embed-text", "input": ["the output"], "options": {"num_gpu": 0}}
    rows = [
        {"row_id": "train_26148", "document": "held", "medium": "blog", "sft_role": "unchanged", "split": "train"},
        {"row_id": "train_1", "document": "a", "medium": "blog", "sft_role": "unchanged", "split": "train"},
        {"row_id": "train_2", "document": "b", "medium": "x", "sft_role": "opening", "split": "train"},
    ]
    assert "train_26148" not in {item["row_id"] for item in select_sample(rows)}


def test_vector_contract_accepts_float32_storage_and_rejects_a_different_vector():
    from host_finetune.spark_v2_embed import (
        ParityError,
        assert_shadow_target,
        compare_rankings,
        contract_from_sample,
        vector_delta,
    )

    fresh = [0.1, -0.2, 0.3]
    stored = [struct_f32(value) for value in fresh]
    pair = {
        "raw": vector_delta(fresh, stored),
        "float32": vector_delta([struct_f32(value) for value in fresh], stored),
    }
    contract = contract_from_sample([pair])
    assert contract["mode"] == "float32_storage"
    assert pair["float32"]["exact"] is True
    far = vector_delta(fresh, [1.0, 1.0, 1.0])
    with pytest.raises(ParityError):
        contract_from_sample([{"raw": far, "float32": far, "float32_ulps": 100}])
    nudged = list(stored)
    nudged[0] = struct_f32(stored[0] + 1.5e-7)
    one_ulp = {
        "raw": vector_delta(fresh, nudged),
        "float32": vector_delta(stored, nudged),
        "float32_ulps": 1,
    }
    assert contract_from_sample([one_ulp])["mode"] == "float32_one_ulp"
    assert compare_rankings(
        [{"id": "a", "distance": 0.1}, {"id": "b", "distance": 0.2}],
        [{"id": "a", "distance": 0.1}, {"id": "b", "distance": 0.25}],
    )["exact_rank"] is True
    tied = compare_rankings(
        [{"id": "a", "distance": 0.1}, {"id": "b", "distance": 0.2}],
        [{"id": "a", "distance": 0.1}, {"id": "c", "distance": 0.2}],
    )
    assert tied["cutoff_tie"] is True
    assert tied["exact_rank"] is False
    assert compare_rankings(
        [{"id": "a", "distance": 0.1}, {"id": "b", "distance": 0.2}],
        [{"id": "b", "distance": 0.2}, {"id": "a", "distance": 0.1}],
    )["same_set"] is True
    with pytest.raises(ParityError):
        assert_shadow_target(ROOT / "host_finetune" / "output" / "chroma_lemkin_train_only", "lemkin_train_only_v2_shadow")
    with pytest.raises(ParityError):
        assert_shadow_target(ROOT / "artifacts" / "refactor" / "spark_v2" / "shadow", "lemkin_train_only")


def struct_f32(value: float) -> float:
    import struct

    return struct.unpack("<f", struct.pack("<f", value))[0]
