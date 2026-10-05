"""Replay the frozen keep-breaks fit in memory and record exact SFT parents.

This module does not write ``host_finetune/data`` and does not load model weights.
The tokenizer is loaded from the local snapshot with ``local_files_only=True``.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path

from host_finetune.canonical_ids import CONTEXT_FIT_VERSION, model_row_id
from host_finetune.fit_sft_to_context import (
    _n_tokens,
    _short_topic,
    _split_output,
    merge_lead_in_rows,
)
from host_finetune.sft_chunk_utils import extract_base_topic, style_prefix

ROOT = Path(__file__).resolve().parents[1]
NOPROMO = ROOT / "host_finetune" / "data" / "dataset_from_cleaned_sources_nopromo.jsonl"
FIT_PATH = ROOT / "host_finetune" / "data" / "dataset_from_cleaned_sources_fit512_keepbreaks.jsonl"
PROV = ROOT / "artifacts" / "refactor" / "provenance"
SFT_MANIFEST = PROV / "sft_manifest.preview.jsonl"
FIT_MANIFEST = PROV / "fit_manifest.preview.jsonl"
COVERAGE = PROV / "coverage.json"
LINEAGE_PATH = PROV / "fit_lineage_exact.preview.jsonl"
REPORT_PATH = PROV / "fit_lineage_exact.report.json"

TOKENIZER_ID = "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit"
TOKENIZER_REVISION = "f15c379fb32bb402fa06a7ae9aecb1febf4b79ec"
TOKENIZER_SNAPSHOT = (
    Path.home()
    / ".cache"
    / "huggingface"
    / "hub"
    / "models--unsloth--Meta-Llama-3.1-8B-Instruct-bnb-4bit"
    / "snapshots"
    / TOKENIZER_REVISION
)
MAX_SEQ_LENGTH = 512
MAX_PROMPT_TOKENS = 120
FIT_VERSION = CONTEXT_FIT_VERSION
NOPROMO_SHA256 = "55b9680657ed72a424d47ea7102214d9b2c6693a535572e02520e406d57f8241"
FIT_SHA256 = "bf00e53fd2401f2d6bea6ac39cb69829fc21761de3a4f7225fed7ecd75082fba"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def load_rows(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def merge_with_parents(rows: list[dict]) -> list[tuple[dict, list[int]]]:
    """Same control flow as ``merge_lead_in_rows``, keeping every consumed index.

    The frozen function writes a lead-in onto the next row and skips the lead-in.
    Parents accumulate until a row is actually emitted.
    """
    rows = [dict(row) for row in rows]
    merged: list[tuple[dict, list[int]]] = []
    pending: list[int] = []
    index = 0
    while index < len(rows):
        row = dict(rows[index])
        output = (row.get("output") or "").rstrip()
        nxt = rows[index + 1] if index + 1 < len(rows) else None
        same_document = (
            nxt is not None
            and row.get("source_file")
            and (row.get("source_file"), str(row.get("source_line")))
            == (nxt.get("source_file"), str(nxt.get("source_line")))
        )
        if output.endswith(":") and same_document:
            attached = dict(nxt)
            attached["output"] = output + "\n\n" + (attached.get("output") or "").lstrip()
            rows[index + 1] = attached
            pending.append(index)
            index += 1
            continue
        row["output"] = output
        merged.append((row, pending + [index]))
        pending = []
        index += 1
    return merged


def serialize_fit_rows(records: list[dict], newline: bytes) -> bytes:
    lines = [json.dumps(record, ensure_ascii=False).encode("utf-8") for record in records]
    if not lines:
        return newline
    return newline.join(lines) + newline


def _relationship(parent_count: int, emitted: int, raw_parts: int, same_text: bool) -> str:
    if parent_count > 1 and emitted > 1:
        return "COMBINED_SPLIT"
    if parent_count > 1:
        return "COMBINED"
    if emitted > 1:
        return "SPLIT"
    if raw_parts > emitted or not same_text:
        return "REFLOWED"
    return "UNCHANGED"


def load_tokenizer():
    if not TOKENIZER_SNAPSHOT.is_dir():
        raise FileNotFoundError(f"local tokenizer snapshot missing: {TOKENIZER_SNAPSHOT}")
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(
        str(TOKENIZER_SNAPSHOT),
        trust_remote_code=True,
        local_files_only=True,
    )


def replay(tokenizer, rows: list[dict]) -> tuple[list[dict], list[dict], dict]:
    tracked = merge_with_parents(rows)
    reference = merge_lead_in_rows([dict(row) for row in rows])
    if len(tracked) != len(reference):
        raise RuntimeError("parent-tracking merge diverged from merge_lead_in_rows")
    for (row, _parents), expected in zip(tracked, reference):
        if row != expected:
            raise RuntimeError("parent-tracking merge changed a merged row")

    fitted: list[dict] = []
    lineage: list[dict] = []
    merged_histogram: Counter[int] = Counter()
    parent_outputs: Counter[int] = Counter()
    lead_ins_consumed = 0
    parts_rejected = 0
    empty_splits = 0
    for merged_index, (row, parents) in enumerate(tracked):
        if merged_index and merged_index % 5000 == 0:
            print(f"replayed {merged_index} of {len(tracked)} merged rows", flush=True)
        lead_ins_consumed += len(parents) - 1
        prefix = style_prefix(str(row.get("instruction") or ""))
        base_topic = extract_base_topic(str(row.get("instruction") or ""))
        topic = _short_topic(tokenizer, prefix, base_topic, MAX_PROMPT_TOKENS)
        merged_output = str(row.get("output") or "")
        before_tokens = _n_tokens(tokenizer, prefix + topic, merged_output)
        raw_parts = _split_output(tokenizer, prefix, topic, merged_output, MAX_SEQ_LENGTH)
        if not raw_parts:
            empty_splits += 1
            merged_histogram[0] += 1
            continue
        emitted_here: list[tuple[str, str, int]] = []
        for index, output in enumerate(raw_parts, 1):
            part_topic = topic if len(raw_parts) == 1 else f"{topic} (part {index} of {len(raw_parts)})"
            part_instruction = prefix + part_topic
            after_tokens = _n_tokens(tokenizer, part_instruction, output)
            if after_tokens > MAX_SEQ_LENGTH:
                parts_rejected += 1
                continue
            emitted_here.append((part_instruction, output, after_tokens))
        merged_histogram[len(emitted_here)] += 1
        if not emitted_here:
            continue
        same_text = len(emitted_here) == 1 and emitted_here[0][1] == merged_output.rstrip()
        relationship = _relationship(len(parents), len(emitted_here), len(raw_parts), same_text)
        for ordinal, (part_instruction, output, after_tokens) in enumerate(emitted_here):
            record = {key: value for key, value in row.items() if key not in {"instruction", "output"}}
            record["instruction"] = part_instruction
            record["output"] = output
            fitted.append(record)
            for parent in parents:
                parent_outputs[parent] += 1
            lineage.append(
                {
                    "parents": parents,
                    "relationship_type": relationship,
                    "fit_chunk_ordinal": ordinal,
                    "fit_chunk_count": len(emitted_here),
                    "parent_output_sha256": _sha256_text(merged_output),
                    "parent_output_sha256s": [
                        _sha256_text(str(rows[parent].get("output") or "")) for parent in parents
                    ],
                    "fit_output_sha256": _sha256_text(output),
                    "token_count_before": before_tokens,
                    "token_count_after": after_tokens,
                    "source_key": (record.get("source_file"), record.get("source_line")),
                }
            )
    for parent in range(len(rows)):
        parent_outputs.setdefault(parent, 0)
    accounting = {
        "input_parents": len(rows),
        "lead_in_rows_consumed": lead_ins_consumed,
        "merged_rows": len(tracked),
        "merged_rows_emitting_zero": merged_histogram[0],
        "empty_splits": empty_splits,
        "parts_rejected_after_label": parts_rejected,
        "merged_histogram": {str(key): merged_histogram[key] for key in sorted(merged_histogram)},
    }
    parent_histogram = Counter(parent_outputs.values())
    accounting["parent_histogram"] = {str(key): parent_histogram[key] for key in sorted(parent_histogram)}
    accounting["parents_with_zero_outputs"] = parent_histogram[0]
    accounting["parents_with_one_output"] = parent_histogram[1]
    accounting["parents_with_two_outputs"] = parent_histogram[2]
    accounting["parents_with_three_or_more"] = sum(
        count for outputs, count in parent_histogram.items() if outputs >= 3
    )
    extra_parts = sum(max(0, outputs - 1) * count for outputs, count in merged_histogram.items())
    accounting["additional_parts"] = extra_parts
    accounting["final_fit_rows"] = len(fitted)
    accounting["net_change"] = len(fitted) - len(rows)
    accounting["reconciled_final"] = (
        len(rows) - lead_ins_consumed - merged_histogram[0] + extra_parts
    )
    return fitted, lineage, accounting


def _load_sft_identity() -> dict[int, dict]:
    identity = {}
    with SFT_MANIFEST.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            identity[int(row["sft_row_index"])] = row
    return identity


def _load_fit_model_ids() -> list[str | None]:
    ids = []
    with FIT_MANIFEST.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                ids.append(json.loads(line).get("model_row_id"))
    return ids


def build_sidecar_rows(
    source_rows: list[dict],
    frozen_rows: list[dict],
    lineage: list[dict],
) -> list[dict]:
    identity = _load_sft_identity()
    existing_ids = _load_fit_model_ids()
    if len(existing_ids) != len(lineage) or len(frozen_rows) != len(lineage):
        raise RuntimeError("fit manifest row count does not match replay")
    seen: Counter[tuple] = Counter()
    rows = []
    for fit_index, item in enumerate(lineage):
        parents = item["parents"]
        parent_rows = [identity[parent] for parent in parents]
        document_ids = {row.get("document_id") for row in parent_rows}
        raw_ids = {row.get("raw_record_id") for row in parent_rows}
        source_ids = {row.get("source_id") for row in parent_rows}
        if len(document_ids) != 1 or len(raw_ids) != 1 or len(source_ids) != 1:
            raise RuntimeError(f"fit row {fit_index} parents disagree on capture identity")
        document_id = parent_rows[-1].get("document_id")
        key = item["source_key"]
        ordinal = seen[key]
        seen[key] += 1
        output = str(frozen_rows[fit_index].get("output") or "")
        fitted_id = model_row_id(
            document_id,
            fit_version=FIT_VERSION,
            chunk_ordinal=ordinal,
            output=output,
        )
        rows.append(
            {
                "fit_row_index": fit_index,
                "model_row_id": fitted_id,
                "parent_sft_row_index": parents[0] if len(parents) == 1 else None,
                "parent_sft_row_indices": parents,
                "parent_sft_row_id": None,
                "source_id": parent_rows[-1].get("source_id"),
                "raw_record_id": parent_rows[-1].get("raw_record_id"),
                "document_id": document_id,
                "source": parent_rows[-1].get("source"),
                "source_file": source_rows[parents[-1]].get("source_file"),
                "source_line": source_rows[parents[-1]].get("source_line"),
                "fit_chunk_ordinal": ordinal,
                "part_ordinal": item["fit_chunk_ordinal"],
                "part_count": item["fit_chunk_count"],
                "relationship_type": item["relationship_type"],
                "parent_output_sha256": item["parent_output_sha256"],
                "parent_output_sha256s": item["parent_output_sha256s"],
                "fit_output_sha256": item["fit_output_sha256"],
                "token_count_before": item["token_count_before"],
                "token_count_after": item["token_count_after"],
                "fit_version": FIT_VERSION,
                "tokenizer_id": TOKENIZER_ID,
                "tokenizer_revision": TOKENIZER_REVISION,
            }
        )
    return rows


def compare(replayed: list[dict], frozen: list[dict], frozen_bytes: bytes) -> dict:
    membership = len(replayed) == len(frozen)
    order = replayed == frozen
    key_order = all(list(left) == list(right) for left, right in zip(replayed, frozen))
    content_mismatches = 0
    if len(replayed) == len(frozen):
        for left, right in zip(replayed, frozen):
            if left != right:
                content_mismatches += 1
    crlf = serialize_fit_rows(replayed, b"\r\n")
    lf = serialize_fit_rows(replayed, b"\n")
    return {
        "membership": membership,
        "order": order and key_order,
        "content_mismatches": content_mismatches,
        "key_order": key_order,
        "frozen_has_bom": frozen_bytes.startswith(b"\xef\xbb\xbf"),
        "byte_crlf": crlf == frozen_bytes,
        "byte_lf": lf == frozen_bytes,
        "replay_crlf_sha256": hashlib.sha256(crlf).hexdigest(),
        "replay_lf_sha256": hashlib.sha256(lf).hexdigest(),
        "frozen_sha256": hashlib.sha256(frozen_bytes).hexdigest(),
        "frozen_size": len(frozen_bytes),
    }


def publish(sidecar: list[dict], report: dict) -> None:
    PROV.mkdir(parents=True, exist_ok=True)
    with LINEAGE_PATH.open("w", encoding="utf-8", newline="\n") as handle:
        for row in sidecar:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    REPORT_PATH.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    by_index = {row["fit_row_index"]: row for row in sidecar}
    updated = []
    with FIT_MANIFEST.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            exact = by_index[row["fit_row_index"]]
            row["parent_sft_row_index"] = exact["parent_sft_row_index"]
            row["parent_sft_row_indices"] = exact["parent_sft_row_indices"]
            row["relationship_type"] = exact["relationship_type"]
            row["chunk_alignment"] = "exact_replay"
            row["parent_sft_chunk_count"] = len(exact["parent_sft_row_indices"])
            updated.append(row)
    with FIT_MANIFEST.open("w", encoding="utf-8", newline="\n") as handle:
        for row in updated:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    coverage = json.loads(COVERAGE.read_text(encoding="utf-8"))
    accounting = report["accounting"]
    for item in coverage["transitions"]:
        if item["transition"] != "SFT → FIT":
            continue
        item["dropped"] = accounting["parents_with_zero_outputs"]
        item["expanded"] = accounting["additional_parts"]
        item["ambiguous"] = 0
        item["unmapped"] = 0
        item["note"] = (
            "Exact tokenizer replay. "
            f"Lead-in rows consumed={accounting['lead_in_rows_consumed']}. "
            f"Merged rows={accounting['merged_rows']}. "
            f"Merged rows with zero outputs={accounting['merged_rows_emitting_zero']}. "
            f"Additional parts={accounting['additional_parts']}. "
            f"Net={accounting['net_change']}."
        )
    COVERAGE.write_text(json.dumps(coverage, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main() -> None:
    nopromo_hash = file_sha256(NOPROMO)
    fit_hash = file_sha256(FIT_PATH)
    if nopromo_hash != NOPROMO_SHA256 or fit_hash != FIT_SHA256:
        raise SystemExit(
            f"frozen hash mismatch nopromo={nopromo_hash} fit={fit_hash}"
        )
    source = load_rows(NOPROMO)
    frozen = load_rows(FIT_PATH)
    tokenizer = load_tokenizer()
    replayed, lineage, accounting = replay(tokenizer, source)
    frozen_bytes = FIT_PATH.read_bytes()
    parity = compare(replayed, frozen, frozen_bytes)
    if not (parity["membership"] and parity["order"] and parity["content_mismatches"] == 0):
        report = {
            "status": "FAIL",
            "parity": parity,
            "accounting": accounting,
            "tokenizer": {
                "id": TOKENIZER_ID,
                "revision": TOKENIZER_REVISION,
                "local_files_only": True,
            },
        }
        REPORT_PATH.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        raise SystemExit("fit replay content parity failed")
    sidecar = build_sidecar_rows(source, frozen, lineage)
    for row, frozen_row in zip(sidecar, frozen):
        if row["fit_output_sha256"] != _sha256_text(str(frozen_row.get("output") or "")):
            raise SystemExit(f"output hash mismatch at fit row {row['fit_row_index']}")
        if (row["source_file"], row["source_line"]) != (
            frozen_row.get("source_file"),
            frozen_row.get("source_line"),
        ):
            raise SystemExit(f"source identity mismatch at fit row {row['fit_row_index']}")
    existing_ids = _load_fit_model_ids()
    renamed = sum(row["model_row_id"] != existing_ids[row["fit_row_index"]] for row in sidecar)
    ambiguous = sum(not row["parent_sft_row_indices"] for row in sidecar)
    missing_raw = sum(not row["raw_record_id"] for row in sidecar)
    report = {
        "status": "PASS",
        "input_path": str(NOPROMO.relative_to(ROOT)).replace("\\", "/"),
        "input_rows": len(source),
        "input_sha256": nopromo_hash,
        "output_path": str(FIT_PATH.relative_to(ROOT)).replace("\\", "/"),
        "output_rows": len(frozen),
        "output_sha256": fit_hash,
        "historical_command": (
            "host_finetune\\.venv\\Scripts\\python.exe -u -m host_finetune.fit_sft_to_context "
            "--dataset host_finetune/data/dataset_from_cleaned_sources_nopromo.jsonl "
            "--tokenizer unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit "
            "--out host_finetune/data/dataset_from_cleaned_sources_fit512_keepbreaks.jsonl"
        ),
        "tokenizer": {
            "id": TOKENIZER_ID,
            "revision": TOKENIZER_REVISION,
            "local_source": str(TOKENIZER_SNAPSHOT),
            "local_files_only": True,
            "cpu_only": True,
            "trust_remote_code": True,
            "add_special_tokens": False,
            "chat_template": "apply_chat_template tokenize=False add_generation_prompt=False",
            "max_seq_length": MAX_SEQ_LENGTH,
            "max_prompt_tokens": MAX_PROMPT_TOKENS,
            "packing_limit": MAX_SEQ_LENGTH - 8,
        },
        "parity": parity,
        "accounting": accounting,
        "lineage": {
            "fit_rows": len(sidecar),
            "rows_without_parents": ambiguous,
            "rows_missing_raw_record_id": missing_raw,
            "model_row_id_renames_versus_existing_manifest": renamed,
        },
        "fit_version": FIT_VERSION,
        "relationship_counts": dict(Counter(row["relationship_type"] for row in sidecar)),
    }
    if accounting["reconciled_final"] != len(frozen) or ambiguous or missing_raw or renamed:
        report["status"] = "FAIL"
        REPORT_PATH.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        raise SystemExit("fit lineage accounting or identity failed")
    publish(sidecar, report)
    print(json.dumps({"status": "PASS", "rows": len(sidecar), "net": accounting["net_change"]}))


if __name__ == "__main__":
    main()
