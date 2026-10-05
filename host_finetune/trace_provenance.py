"""Read-only lookup over the provenance sidecars.

Does not open Chroma, call Ollama, or write datasets. Prints no source text.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DIR = ROOT / "artifacts" / "refactor" / "provenance"


def _rows(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def trace_relabel_index(index: int, provenance_dir: Path | None = None) -> dict:
    """Lineage for one EXP-004 row index, such as 26148. No text bodies."""
    if index < 0:
        raise ValueError("index must be >= 0")
    base = provenance_dir or DEFAULT_DIR
    relabel = _rows(base / "relabel_manifest.preview.jsonl")
    matches = [row for row in relabel if row.get("legacy_exp004_row_index") == index]
    if len(matches) != 1:
        raise ValueError(f"expected one relabel row for index {index}, found {len(matches)}")
    row = matches[0]
    rag_hits = [
        item
        for item in _rows(base / "rag_manifest.preview.jsonl")
        if item.get("legacy_exp004_row_index") == index
    ]
    rag = rag_hits[0] if rag_hits else None
    return {
        "legacy_exp004_row_index": index,
        "legacy_experiment_row_id": row["legacy_experiment_row_id"],
        "medium": row["medium"],
        "legacy_source_file": row["legacy_source_file"],
        "legacy_source_line": row["legacy_source_line"],
        "source_id": row["source_id"],
        "source_identity_kind": row["source_identity_kind"],
        "raw_record_id": row["raw_record_id"],
        "document_id": row["document_id"],
        "model_row_id": row["model_row_id"],
        "group_id": row["group_id"],
        "split": row["split"],
        "relabel_row_id": row["relabel_row_id"],
        "rag_row_id": row["rag_row_id"],
        "sft_role": row["sft_role"],
        "rag_eligible": None if rag is None else rag["eligible"],
        "exclusion_reasons": [] if rag is None else list(rag["exclusion_reasons"]),
    }


def traces_for_source(source_id: str, provenance_dir: Path | None = None) -> dict:
    """Captures, model rows, and Relabel indexes for one external source id."""
    if not source_id:
        raise ValueError("source_id is required")
    base = provenance_dir or DEFAULT_DIR
    captures = []
    for row in _rows(base / "raw_manifest.preview.jsonl"):
        if row.get("source_id") != source_id:
            continue
        captures.append(
            {
                "raw_record_id": row["raw_record_id"],
                "raw_payload_sha256": row["raw_payload_sha256"],
                "source_file": row["source_file"],
                "raw_record_index": row["raw_record_index"],
                "source_identity_kind": row["source_identity_kind"],
            }
        )
    relabel_rows = []
    model_ids = []
    for row in _rows(base / "relabel_manifest.preview.jsonl"):
        if row.get("source_id") != source_id:
            continue
        model_ids.append(row["model_row_id"])
        relabel_rows.append(
            {
                "legacy_exp004_row_index": row["legacy_exp004_row_index"],
                "legacy_experiment_row_id": row["legacy_experiment_row_id"],
                "model_row_id": row["model_row_id"],
                "relabel_row_id": row["relabel_row_id"],
                "raw_record_id": row["raw_record_id"],
                "split": row["split"],
            }
        )
    return {
        "source_id": source_id,
        "captures": captures,
        "capture_count": len(captures),
        "distinct_raw_record_id": len({row["raw_record_id"] for row in captures}),
        "model_row_ids": sorted(set(model_ids)),
        "relabel_rows": relabel_rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=int, default=None)
    parser.add_argument("--source-id", default=None)
    parser.add_argument("--provenance-dir", type=Path, default=DEFAULT_DIR)
    args = parser.parse_args()
    if args.index is None and not args.source_id:
        raise SystemExit("pass --index or --source-id")
    if args.index is not None:
        print(json.dumps(trace_relabel_index(args.index, args.provenance_dir), indent=2))
    if args.source_id:
        print(json.dumps(traces_for_source(args.source_id, args.provenance_dir), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
