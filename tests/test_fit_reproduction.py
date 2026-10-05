"""Locks the in-memory keep-breaks replay. Does not load a tokenizer or rewrite frozen JSONL."""

import hashlib
import json
from pathlib import Path

from host_finetune.fit_sft_to_context import merge_lead_in_rows
from host_finetune.replay_fit_lineage import merge_with_parents

ROOT = Path(__file__).resolve().parents[1]
PROV = ROOT / "artifacts" / "refactor" / "provenance"
NOPROMO = ROOT / "host_finetune" / "data" / "dataset_from_cleaned_sources_nopromo.jsonl"
FIT = ROOT / "host_finetune" / "data" / "dataset_from_cleaned_sources_fit512_keepbreaks.jsonl"
NOPROMO_SHA256 = "55b9680657ed72a424d47ea7102214d9b2c6693a535572e02520e406d57f8241"
FIT_SHA256 = "bf00e53fd2401f2d6bea6ac39cb69829fc21761de3a4f7225fed7ecd75082fba"
TOKENIZER_REVISION = "f15c379fb32bb402fa06a7ae9aecb1febf4b79ec"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _rows(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _report() -> dict:
    return json.loads((PROV / "fit_lineage_exact.report.json").read_text(encoding="utf-8"))


def test_merge_parent_tracking_matches_the_frozen_function_and_is_deterministic():
    rows = [
        {"instruction": "lead", "output": "Intro:", "source": "blog", "source_file": "a.jsonl", "source_line": 1},
        {"instruction": "next", "output": "Body:", "source": "blog", "source_file": "a.jsonl", "source_line": 1},
        {"instruction": "last", "output": " Tail", "source": "blog", "source_file": "a.jsonl", "source_line": 1},
        {"instruction": "other", "output": "Alone", "source": "blog", "source_file": "a.jsonl", "source_line": 2},
    ]
    first = merge_with_parents(rows)
    second = merge_with_parents(rows)
    assert first == second
    reference = merge_lead_in_rows([dict(row) for row in rows])
    assert [row for row, _parents in first] == reference
    assert [parents for _row, parents in first] == [[0, 1, 2], [3]]
    assert rows[1]["output"] == "Body:"


def test_frozen_fit_hash_and_replay_report():
    report = _report()
    assert _sha256(NOPROMO) == NOPROMO_SHA256
    assert _sha256(FIT) == FIT_SHA256
    assert report["status"] == "PASS"
    assert report["input_rows"] == 42771
    assert report["output_rows"] == 43012
    assert report["input_sha256"] == NOPROMO_SHA256
    assert report["output_sha256"] == FIT_SHA256
    parity = report["parity"]
    assert parity["membership"] is True
    assert parity["order"] is True
    assert parity["key_order"] is True
    assert parity["content_mismatches"] == 0
    assert parity["frozen_has_bom"] is False
    assert parity["byte_crlf"] is True
    assert parity["byte_lf"] is False
    assert parity["replay_crlf_sha256"] == FIT_SHA256
    assert parity["frozen_sha256"] == FIT_SHA256
    tokenizer = report["tokenizer"]
    assert tokenizer["id"] == "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit"
    assert tokenizer["revision"] == TOKENIZER_REVISION
    assert tokenizer["local_files_only"] is True
    assert tokenizer["add_special_tokens"] is False
    assert tokenizer["max_seq_length"] == 512
    assert tokenizer["max_prompt_tokens"] == 120
    assert tokenizer["packing_limit"] == 504
    assert report["fit_version"] == "fit512-keepbreaks-v1"


def test_parent_accounting_reconciles():
    accounting = _report()["accounting"]
    assert accounting["input_parents"] == 42771
    assert accounting["final_fit_rows"] == 43012
    assert accounting["lead_in_rows_consumed"] == 706
    assert accounting["merged_rows"] == 42065
    assert accounting["merged_rows_emitting_zero"] == 0
    assert accounting["parts_rejected_after_label"] == 0
    assert accounting["additional_parts"] == 947
    assert accounting["parents_with_zero_outputs"] == 0
    assert accounting["net_change"] == 241
    assert accounting["reconciled_final"] == 43012
    assert (
        accounting["input_parents"]
        - accounting["lead_in_rows_consumed"]
        - accounting["merged_rows_emitting_zero"]
        + accounting["additional_parts"]
    ) == 43012
    parent_counts = {int(key): value for key, value in accounting["parent_histogram"].items()}
    assert sum(parent_counts.values()) == 42771
    assert sum(outputs * count for outputs, count in parent_counts.items()) == (
        1 * 41733 + 2 * 956 + 3 * 71 + 4 * 11
    )
    merged_counts = {int(key): value for key, value in accounting["merged_histogram"].items()}
    assert sum(merged_counts.values()) == 42065
    assert sum(outputs * count for outputs, count in merged_counts.items()) == 43012
    assert sum(_report()["relationship_counts"].values()) == 43012


def test_every_fit_row_has_exact_parent_lineage_and_stable_model_row_id():
    lineage = _rows(PROV / "fit_lineage_exact.preview.jsonl")
    fit_manifest = _rows(PROV / "fit_manifest.preview.jsonl")
    frozen = _rows(FIT)
    nopromo = _rows(NOPROMO)
    assert len(lineage) == len(fit_manifest) == len(frozen) == 43012
    assert len(nopromo) == 42771
    seen_parents = set()
    for index, (row, manifest_row, frozen_row) in enumerate(zip(lineage, fit_manifest, frozen)):
        assert row["fit_row_index"] == index
        parents = row["parent_sft_row_indices"]
        assert parents
        assert row["raw_record_id"]
        assert row["document_id"]
        assert row["source_id"]
        assert row["model_row_id"] == manifest_row["model_row_id"]
        assert manifest_row["chunk_alignment"] == "exact_replay"
        assert manifest_row["parent_sft_row_indices"] == parents
        assert row["fit_version"] == "fit512-keepbreaks-v1"
        assert row["tokenizer_revision"] == TOKENIZER_REVISION
        if len(parents) == 1:
            assert row["parent_sft_row_index"] == parents[0]
        else:
            assert row["parent_sft_row_index"] is None
        survivor = nopromo[parents[-1]]
        assert (row["source_file"], row["source_line"]) == (
            frozen_row["source_file"],
            frozen_row["source_line"],
        )
        assert (row["source_file"], row["source_line"]) == (
            survivor["source_file"],
            survivor["source_line"],
        )
        assert hashlib.sha256(str(frozen_row["output"]).encode("utf-8")).hexdigest() == row["fit_output_sha256"]
        if row["relationship_type"] == "REFLOWED":
            assert len(parents) == 1
            assert frozen_row["output"].split() == nopromo[parents[0]]["output"].split()
        seen_parents.update(parents)
    assert seen_parents == set(range(42771))
    coverage = json.loads((PROV / "coverage.json").read_text(encoding="utf-8"))
    fit = next(item for item in coverage["transitions"] if item["transition"] == "SFT → FIT")
    assert fit["input"] == 42771
    assert fit["output"] == 43012
    assert fit["ambiguous"] == 0
    assert fit["unmapped"] == 0
    assert fit["dropped"] == 0
