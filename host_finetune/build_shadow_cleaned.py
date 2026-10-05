"""Build the shadow cleaned preview and compare it to the frozen cleaned files.

Reads raw and cleaned JSONL. Writes only under ``artifacts/refactor/cleaning/``.
Does not call ``clean_file`` or ``clean_scraped``. Does not load a tokenizer.
"""
from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path

from host_finetune.build_sft_from_cleaned_sources import (
    DEFAULT_MIN_WORDS,
    MIN_WORDS,
    SOURCES,
    _title,
)
from host_finetune.canonical_cleaning import SOURCE_SPECS, clean_record
from host_finetune.canonical_ids import IDENTITY_VERSION
from host_finetune.filter_event_promos import _is_direct_event_announcement, _strip_trailing_cta
from host_finetune.sft_chunk_utils import instruction_for, split_long_document

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts" / "refactor" / "cleaning"
PREVIEW_PATH = OUT / "canonical_cleaned.preview.jsonl"
MANIFEST_PATH = OUT / "canonical_cleaned_manifest.preview.jsonl"
REPORT_PATH = OUT / "parity_report.json"
NOPROMO = ROOT / "host_finetune" / "data" / "dataset_from_cleaned_sources_nopromo.jsonl"
SFT_CHUNK_CHARS = 1600


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _explicit_bytes(rows: list[dict], newline: bytes) -> bytes:
    encoded = [json.dumps(row, ensure_ascii=False).encode("utf-8") for row in rows]
    if not encoded:
        return b""
    return newline.join(encoded) + newline


def _load_cleaned(path: Path) -> list[tuple[int, dict]]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            if line.strip():
                rows.append((number, json.loads(line)))
    return rows


def _sft_from_kept(kept_by_file: dict[str, list[tuple[int, dict]]]) -> list[dict]:
    """Existing SFT build and promo filter, fed by shadow cleaned rows."""
    rows = []
    for filename, (source, text_key, title_key) in SOURCES.items():
        floor = MIN_WORDS.get(source, DEFAULT_MIN_WORDS)
        for line_no, record in kept_by_file[filename]:
            text = str(record.get(text_key) or "").strip()
            if len(text.split()) < floor:
                continue
            title = _title(record, title_key, source, text)
            pieces = split_long_document(text, SFT_CHUNK_CHARS)
            if not pieces:
                continue
            built = []
            for part, (heading, chunk) in enumerate(pieces, 1):
                topic = title
                if heading and len(pieces) > 1:
                    topic = f"{title} — {heading}"
                if len(pieces) > 1:
                    topic = f"{topic} (part {part} of {len(pieces)})"
                built.append(
                    {
                        "instruction": instruction_for(source, topic),
                        "output": chunk,
                        "source": source,
                        "source_file": filename,
                        "source_line": line_no,
                    }
                )
            for piece in built:
                if _is_direct_event_announcement(piece):
                    continue
                clean_output, changed = _strip_trailing_cta(str(piece["output"] or ""))
                if changed:
                    piece["output"] = clean_output
                    if len(clean_output.split()) < 20:
                        continue
                rows.append(piece)
    return rows


def _nopromo_matches(replayed: list[dict]) -> dict:
    mismatches = 0
    seen = 0
    with NOPROMO.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if seen >= len(replayed):
                mismatches += 1
            else:
                expected = replayed[seen]
                actual = (
                    row.get("source_file"),
                    row.get("source_line"),
                    row.get("source"),
                    row.get("instruction"),
                    row.get("output"),
                )
                wanted = (
                    expected["source_file"],
                    expected["source_line"],
                    expected["source"],
                    expected["instruction"],
                    expected["output"],
                )
                if actual != wanted:
                    mismatches += 1
            seen += 1
    return {
        "frozen_rows": seen,
        "shadow_rows": len(replayed),
        "mismatches": mismatches,
        "membership_order_content": seen == len(replayed) and mismatches == 0,
    }


def evaluate() -> dict:
    """Compare the shadow cleaner to the frozen cleaned files. No writes."""
    preview_rows: list[dict] = []
    manifest: list[dict] = []
    per_source = []
    frozen_line_numbers: dict[str, list[int]] = {}
    drop_counts: Counter = Counter()
    identity_breaks = 0
    for filename, _kind, medium, text_field, _title_field, _scrub in SOURCE_SPECS:
        raw_path = ROOT / "data" / filename
        cleaned_path = ROOT / "data" / "cleaned" / filename
        frozen = _load_cleaned(cleaned_path)
        frozen_records = [row for _number, row in frozen]
        frozen_line_numbers[filename] = [number for number, _row in frozen]
        shadow_kept: list[dict] = []
        raw_count = 0
        with raw_path.open(encoding="utf-8") as handle:
            for raw_index, line in enumerate(line for line in handle if line.strip()):
                record = json.loads(line)
                result = clean_record(medium, record)
                again = clean_record(medium, record)
                if (result.kept, result.cleaned_text, result.drop_reason, result.source_id, result.raw_record_id) != (
                    again.kept,
                    again.cleaned_text,
                    again.drop_reason,
                    again.source_id,
                    again.raw_record_id,
                ):
                    identity_breaks += 1
                if result.source_id is None or result.raw_record_id is None:
                    identity_breaks += 1
                raw_count += 1
                cleaned_line = len(shadow_kept) + 1 if result.kept else None
                manifest.append(
                    {
                        "source_id": result.source_id,
                        "source_identity_kind": result.source_identity_kind,
                        "external_id": result.external_id,
                        "raw_record_id": result.raw_record_id,
                        "source": medium,
                        "medium": medium,
                        "source_file": filename,
                        "raw_record_index": raw_index,
                        "cleaned_line": cleaned_line,
                        "status": "kept" if result.kept else "dropped",
                        "drop_reason": result.drop_reason,
                        "frozen_reason": result.frozen_reason,
                        "cleaned_text_sha256": _sha256_text(result.cleaned_text) if result.cleaned_text else None,
                        "identity_version": IDENTITY_VERSION,
                    }
                )
                if result.frozen_reason:
                    drop_counts[result.frozen_reason] += 1
                if result.kept and result.cleaned_record is not None:
                    shadow_kept.append(result.cleaned_record)
                    preview_rows.append(result.cleaned_record)
        same_records = shadow_kept == frozen_records
        same_keys = all(list(left.keys()) == list(right.keys()) for left, right in zip(shadow_kept, frozen_records))
        disk = cleaned_path.read_bytes()
        lf = _explicit_bytes(shadow_kept, b"\n")
        crlf = _explicit_bytes(shadow_kept, b"\r\n")
        per_source.append(
            {
                "source_file": filename,
                "medium": medium,
                "raw_count": raw_count,
                "frozen_cleaned_count": len(frozen_records),
                "shadow_cleaned_count": len(shadow_kept),
                "membership": len(shadow_kept) == len(frozen_records) and same_records,
                "order": same_records,
                "content": same_records and same_keys,
                "byte_lf": lf == disk,
                "byte_crlf": crlf == disk,
                "byte_status": "PASS" if lf == disk or crlf == disk else "FAIL",
                "byte_newline": "CRLF" if crlf == disk else ("LF" if lf == disk else "NEITHER"),
            }
        )
    kept_by_file: dict[str, list[tuple[int, dict]]] = {spec[0]: [] for spec in SOURCE_SPECS}
    cursor = 0
    for source in per_source:
        count = source["shadow_cleaned_count"]
        chunk = preview_rows[cursor : cursor + count]
        numbers = frozen_line_numbers[source["source_file"]]
        if len(numbers) != len(chunk):
            numbers = list(range(1, len(chunk) + 1))
        kept_by_file[source["source_file"]] = list(zip(numbers, chunk))
        cursor += count
    sft = _nopromo_matches(_sft_from_kept(kept_by_file))
    report = {
        "artifact": "shadow_cleaned_preview",
        "canonical_production_artifact": False,
        "per_source": per_source,
        "drop_reasons": dict(drop_counts),
        "dropped_total": sum(drop_counts.values()),
        "kept_total": sum(row["shadow_cleaned_count"] for row in per_source),
        "raw_total": sum(row["raw_count"] for row in per_source),
        "identity_breaks": identity_breaks,
        "membership_parity": all(row["membership"] for row in per_source),
        "order_parity": all(row["order"] for row in per_source),
        "content_parity": all(row["content"] for row in per_source),
        "byte_parity": all(row["byte_status"] == "PASS" for row in per_source),
        "sft_nopromo": sft,
        "preview_rows": preview_rows,
        "manifest": manifest,
    }
    return report


def publish() -> dict:
    report = evaluate()
    OUT.mkdir(parents=True, exist_ok=True)
    with PREVIEW_PATH.open("w", encoding="utf-8", newline="\n") as handle:
        for row in report["preview_rows"]:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    with MANIFEST_PATH.open("w", encoding="utf-8", newline="\n") as handle:
        for row in report["manifest"]:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    stored = {key: value for key, value in report.items() if key not in {"preview_rows", "manifest"}}
    with REPORT_PATH.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(stored, indent=2, ensure_ascii=False) + "\n")
    report["stored"] = stored
    return report


def main() -> int:
    report = publish()
    stored = report["stored"]
    print(json.dumps(stored["per_source"], indent=2))
    print("drops", json.dumps(stored["drop_reasons"], sort_keys=True))
    print("sft", json.dumps(stored["sft_nopromo"]))
    print(f"wrote {OUT}")
    ok = (
        stored["membership_parity"]
        and stored["order_parity"]
        and stored["content_parity"]
        and stored["sft_nopromo"]["membership_order_content"]
        and stored["dropped_total"] == 2465
        and stored["identity_breaks"] == 0
    )
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
