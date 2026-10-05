"""Forensic comparison of the frozen balanced file and an in-memory replay.

Reads two JSONL files and does not write either of them. Does not import
Chroma, the retrieval auditor, or the trainer. The parity harness reproduces
the frozen SHA-256 with explicit CRLF bytes in
``verify_frozen_parity.explicit_crlf_jsonl_bytes``. This module still
enumerates the other serializers so the LF hash stays visible as a
counterexample.
"""
from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path

from host_finetune.queue_followup_trains import select_balanced_rows

ROOT = Path(__file__).resolve().parents[1]
FROZEN = ROOT / "host_finetune" / "data" / "dataset_fit512_keepbreaks_balanced.jsonl"
PARENT = ROOT / "host_finetune" / "data" / "dataset_from_cleaned_sources_fit512_keepbreaks.jsonl"
EXPECTED_SHA256 = "6e99b414733eafdfa27a5e94dd26beff1498bbfd58af132251e24044fef4f057"
SAMPLE = 5


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _read_bytes(path: Path) -> bytes:
    return path.read_bytes()


def _logical_lines(payload: bytes) -> list[bytes]:
    """Non-empty physical lines without their newline bytes."""
    text = payload
    if text.startswith(b"\xef\xbb\xbf"):
        text = text[3:]
    lines = []
    for raw in text.splitlines():
        if raw.strip():
            lines.append(raw)
    return lines


def _parse(line: bytes) -> dict:
    return json.loads(line.decode("utf-8"))


def _identity(row: dict) -> tuple:
    """Provenance plus the two text fields that distinguish chunks of one source line."""
    return (
        row.get("source"),
        row.get("source_file"),
        row.get("source_line"),
        row.get("instruction"),
        row.get("output"),
    )


def _preview(row: dict) -> dict:
    return {
        "source": row.get("source"),
        "source_file": row.get("source_file"),
        "source_line": row.get("source_line"),
        "instruction_chars": len(row.get("instruction") or ""),
        "output_chars": len(row.get("output") or ""),
    }


def _newline_facts(payload: bytes) -> dict:
    return {
        "bom": payload.startswith(b"\xef\xbb\xbf"),
        "crlf": payload.count(b"\r\n"),
        "lf": payload.count(b"\n"),
        "cr_not_in_crlf": payload.count(b"\r") - payload.count(b"\r\n"),
        "endswith_lf": payload.endswith(b"\n"),
        "endswith_crlf": payload.endswith(b"\r\n"),
        "nbytes": len(payload),
    }


def _candidate_payloads(rows: list[dict], raw_lines: list[bytes] | None) -> list[tuple[str, bytes]]:
    """Serializers justified by the fitter and by queue_followup_trains.build_balanced."""
    dumps = {
        "ensure_ascii_false_lf": lambda row: json.dumps(row, ensure_ascii=False) + "\n",
        "ensure_ascii_false_crlf": lambda row: json.dumps(row, ensure_ascii=False) + "\r\n",
        "ensure_ascii_true_lf": lambda row: json.dumps(row, ensure_ascii=True) + "\n",
        "compact_separators_lf": lambda row: (
            json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
        ),
        "sort_keys_ensure_ascii_false_lf": lambda row: (
            json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
        ),
        "sort_keys_ensure_ascii_true_lf": lambda row: (
            json.dumps(row, ensure_ascii=True, sort_keys=True) + "\n"
        ),
    }
    found = []
    for name, render in dumps.items():
        found.append((name, "".join(render(row) for row in rows).encode("utf-8")))
    if raw_lines is not None:
        found.append(("parent_line_bytes_plus_lf", b"\n".join(raw_lines) + b"\n"))
        found.append(("parent_line_bytes_plus_crlf", b"\r\n".join(raw_lines) + b"\r\n"))
        found.append(("parent_line_bytes_no_final_newline", b"\n".join(raw_lines)))
    return found


def diagnose() -> dict:
    """Compare the frozen file with select_balanced_rows on the keep-breaks parent."""
    frozen_bytes = _read_bytes(FROZEN)
    parent_bytes = _read_bytes(PARENT)
    frozen_lines = _logical_lines(frozen_bytes)
    parent_lines = _logical_lines(parent_bytes)
    frozen_rows = [_parse(line) for line in frozen_lines]
    parent_rows = [_parse(line) for line in parent_lines]
    replay_rows, stats = select_balanced_rows(parent_rows)
    parent_index = {id(row): index for index, row in enumerate(parent_rows)}
    replay_indexes = [parent_index[id(row)] for row in replay_rows]

    frozen_ids = [_identity(row) for row in frozen_rows]
    replay_ids = [_identity(row) for row in replay_rows]
    frozen_counts = Counter(frozen_ids)
    replay_counts = Counter(replay_ids)
    frozen_only = list((frozen_counts - replay_counts).elements())
    replay_only = list((replay_counts - frozen_counts).elements())
    intersection = sum((frozen_counts & replay_counts).values())
    membership_identical = not frozen_only and not replay_only and len(frozen_rows) == len(replay_rows)
    order_identical = frozen_ids == replay_ids
    parsed_identical = frozen_rows == replay_rows

    key_orders = Counter(tuple(row) for row in frozen_rows)
    replay_key_orders = Counter(tuple(row) for row in replay_rows)
    frozen_keys = set().union(*(row.keys() for row in frozen_rows)) if frozen_rows else set()
    replay_keys = set().union(*(row.keys() for row in replay_rows)) if replay_rows else set()

    value_diffs = 0
    key_order_diffs = 0
    first_value_diff = None
    first_key_order_diff = None
    if len(frozen_rows) == len(replay_rows):
        for index, (left, right) in enumerate(zip(frozen_rows, replay_rows)):
            if left != right and first_value_diff is None:
                value_diffs += 1
                left_only = sorted(set(left) - set(right))
                right_only = sorted(set(right) - set(left))
                changed = sorted(key for key in left.keys() & right.keys() if left[key] != right[key])
                first_value_diff = {
                    "index": index,
                    "frozen_only_fields": left_only,
                    "replay_only_fields": right_only,
                    "changed_fields": changed,
                    "frozen": _preview(left),
                    "replay": _preview(right),
                }
            elif left != right:
                value_diffs += 1
            if list(left) != list(right):
                key_order_diffs += 1
                if first_key_order_diff is None:
                    first_key_order_diff = {
                        "index": index,
                        "frozen_keys": list(left),
                        "replay_keys": list(right),
                    }

    first_order_divergence = None
    if not order_identical:
        limit = min(len(frozen_ids), len(replay_ids))
        for index in range(limit):
            if frozen_ids[index] != replay_ids[index]:
                first_order_divergence = {
                    "index": index,
                    "frozen": _preview(frozen_rows[index]),
                    "replay": _preview(replay_rows[index]),
                }
                break
        if first_order_divergence is None:
            first_order_divergence = {
                "index": limit,
                "reason": "common prefix matches; lengths differ",
                "frozen_len": len(frozen_ids),
                "replay_len": len(replay_ids),
            }

    selected_lines = [parent_lines[index] for index in replay_indexes]
    candidates = []
    if membership_identical and order_identical:
        for name, payload in _candidate_payloads(replay_rows, selected_lines):
            digest = _sha256(payload)
            candidates.append(
                {
                    "name": name,
                    "sha256": digest,
                    "matches_frozen_hash": digest == EXPECTED_SHA256,
                    "matches_frozen_bytes": payload == frozen_bytes,
                }
            )

    frozen_hash = _sha256(frozen_bytes)
    exact = any(item["matches_frozen_hash"] for item in candidates)
    if exact and frozen_hash == EXPECTED_SHA256:
        level = 1
    elif membership_identical and order_identical and parsed_identical:
        level = 2
    elif membership_identical:
        level = 3
    elif stats.get("rows_out") == len(frozen_rows):
        level = 4
    else:
        level = 5

    return {
        "frozen_sha256": frozen_hash,
        "expected_sha256": EXPECTED_SHA256,
        "frozen_hash_matches_inventory": frozen_hash == EXPECTED_SHA256,
        "stats": stats,
        "frozen_rows": len(frozen_rows),
        "replay_rows": len(replay_rows),
        "intersection": intersection,
        "frozen_only": len(frozen_only),
        "replay_only": len(replay_only),
        "duplicate_frozen_identities": sum(1 for count in frozen_counts.values() if count > 1),
        "duplicate_replay_identities": sum(1 for count in replay_counts.values() if count > 1),
        "membership_identical": membership_identical,
        "order_identical": order_identical,
        "parsed_identical": parsed_identical,
        "value_diffs_in_paired_order": value_diffs,
        "key_order_diffs_in_paired_order": key_order_diffs,
        "first_order_divergence": first_order_divergence,
        "first_value_diff": first_value_diff,
        "first_key_order_diff": first_key_order_diff,
        "frozen_only_sample": [_preview(dict(zip(
            ("source", "source_file", "source_line", "instruction", "output"), item
        ))) for item in frozen_only[:SAMPLE]],
        "replay_only_sample": [_preview(dict(zip(
            ("source", "source_file", "source_line", "instruction", "output"), item
        ))) for item in replay_only[:SAMPLE]],
        "frozen_fields": sorted(frozen_keys),
        "replay_fields": sorted(replay_keys),
        "fields_only_frozen": sorted(frozen_keys - replay_keys),
        "fields_only_replay": sorted(replay_keys - frozen_keys),
        "frozen_key_orders": ["|".join(order) for order in key_orders],
        "replay_key_orders": ["|".join(order) for order in replay_key_orders],
        "frozen_newlines": _newline_facts(frozen_bytes),
        "parent_newlines": _newline_facts(parent_bytes),
        "candidates": candidates,
        "exact_hash_reproduced": exact,
        "level": level,
    }


def main() -> int:
    report = diagnose()
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
