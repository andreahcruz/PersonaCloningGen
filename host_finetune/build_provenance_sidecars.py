"""Sidecar provenance for the frozen stages. Does not rewrite those stages.

Reads JSONL, calls ``clean_record`` and the SFT chunk functions in memory, and
writes only under ``artifacts/refactor/provenance/``. Does not call
``clean_file``, ``clean_scraped``, or ``fit_sft_to_context.main``. Does not
load a tokenizer.
"""
from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

from host_finetune.build_identity_crosswalk import CLEAN_SPECS, align_kept_rows, _locator
from host_finetune.build_sft_from_cleaned_sources import (
    DEFAULT_MIN_WORDS,
    MIN_WORDS,
    SOURCES,
    _title,
)
from host_finetune.canonical_ids import (
    BALANCE_VERSION,
    CLEANING_VERSION,
    CONTEXT_FIT_VERSION,
    GROUP_VERSION,
    IDENTITY_VERSION,
    RAG_GOVERNANCE_VERSION,
    RAW_SCHEMA_VERSION,
    RELABEL_VERSION,
    SFT_BUILD_VERSION,
    SPLIT_VERSION,
    document_id,
    experiment_row_id,
    model_row_id,
    rag_row_id,
    raw_payload_sha256,
    raw_record_id,
    relabel_row_id,
    source_identity_for_raw,
)
from host_finetune.canonical_cleaning import clean_record
from host_finetune.filter_event_promos import _is_direct_event_announcement, _strip_trailing_cta
from host_finetune.queue_followup_trains import select_balanced_rows
from host_finetune.sft_chunk_utils import instruction_for, split_long_document
from host_finetune.train_only_index import (
    DEFAULT_GOLD_DISPOSITIONS,
    DEFAULT_GOLD_EVAL,
    blocked_group_ids,
    headline_overlap_groups,
    load_jsonl,
    select_train_documents,
)

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts" / "refactor" / "provenance"
PROMO_FILTER_VERSION = "event-promo-v1"
SFT_CHUNK_CHARS = 1600

RAW_MANIFEST = OUT / "raw_manifest.preview.jsonl"
CLEANED_MANIFEST = OUT / "cleaned_manifest.preview.jsonl"
SFT_MANIFEST = OUT / "sft_manifest.preview.jsonl"
FIT_MANIFEST = OUT / "fit_manifest.preview.jsonl"
BALANCED_MANIFEST = OUT / "balanced_manifest.preview.jsonl"
SPLIT_MANIFEST = OUT / "split_manifest.preview.jsonl"
RELABEL_MANIFEST = OUT / "relabel_manifest.preview.jsonl"
RAG_MANIFEST = OUT / "rag_manifest.preview.jsonl"
COVERAGE_PATH = OUT / "coverage.json"

NOPROMO = ROOT / "host_finetune" / "data" / "dataset_from_cleaned_sources_nopromo.jsonl"
FIT_PATH = ROOT / "host_finetune" / "data" / "dataset_from_cleaned_sources_fit512_keepbreaks.jsonl"
BALANCED_PATH = ROOT / "host_finetune" / "data" / "dataset_fit512_keepbreaks_balanced.jsonl"
EXP008_ASSIGNMENTS = (
    ROOT / "experiments" / "EXP-20261001-008-repaired-balanced-split" / "raw" / "split_assignments.jsonl"
)
RELABEL_DATASET = (
    ROOT / "experiments" / "EXP-20261003-004-relabel-continuations" / "raw" / "dataset.jsonl"
)
RELABEL_ASSIGNMENTS = (
    ROOT / "experiments" / "EXP-20261003-004-relabel-continuations" / "raw" / "split_assignments.jsonl"
)
RELABEL_DISPOSITIONS = (
    ROOT / "experiments" / "EXP-20261003-004-relabel-continuations" / "raw" / "dispositions.jsonl"
)


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def iter_stored_records(path: Path):
    """Yield ``(index, object_bytes, record)`` for each non-empty JSONL line.

    ``object_bytes`` are the stored JSON bytes with the line terminator removed.
    They are not rebuilt from the parsed dict.
    """
    payload = path.read_bytes()
    if payload.startswith(b"\xef\xbb\xbf"):
        payload = payload[3:]
    index = 0
    for line in payload.splitlines():
        if not line.strip():
            continue
        yield index, line, json.loads(line.decode("utf-8"))
        index += 1


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _identity(medium: str, record: dict, raw_index: int) -> dict:
    try:
        identity = source_identity_for_raw(medium, record)
        capture = raw_record_id(medium, record)
    except (ValueError, TypeError):
        return {
            "source_id": None,
            "source_identity_kind": None,
            "external_id": None,
            "raw_record_id": None,
            "document_id": None,
            "raw_record_index": raw_index,
        }
    return {
        "source_id": identity.source_id,
        "source_identity_kind": identity.source_identity_kind,
        "external_id": identity.external_id,
        "raw_record_id": capture,
        "document_id": document_id(capture),
        "raw_record_index": raw_index,
    }


def _line_number(value) -> int | None:
    if isinstance(value, int):
        return value
    text = str(value).strip()
    return int(text) if text.isdigit() else None


def build_raw_and_cleaned() -> tuple[list[dict], list[dict], dict]:
    raw_rows: list[dict] = []
    cleaned_rows: list[dict] = []
    by_line: dict[tuple, dict] = {}
    anomalies = {
        "replay_failures": [],
        "one_raw_to_multiple_cleaned": 0,
        "multiple_raw_to_one_cleaned": 0,
        "unmapped_cleaned_lines": 0,
    }
    for spec in CLEAN_SPECS:
        filename, _kind, medium, text_field, _title_field, _scrub_kwargs = spec
        raw_path = ROOT / "data" / filename
        cleaned_path = ROOT / "data" / "cleaned" / filename
        stored = list(iter_stored_records(raw_path))
        disk = []
        with cleaned_path.open(encoding="utf-8") as handle:
            for number, line in enumerate(handle, 1):
                if line.strip():
                    disk.append((number, json.loads(line)))
        kept = []
        for raw_index, blob, record in stored:
            identity = _identity(medium, record, raw_index)
            payload_sha = raw_payload_sha256(blob)
            raw_rows.append(
                {
                    **identity,
                    "raw_payload_sha256": payload_sha,
                    "source": medium,
                    "medium": medium,
                    "source_file": filename,
                    "mapping_method": "canonical_content_fields",
                    "identity_version": IDENTITY_VERSION,
                    "raw_schema_version": RAW_SCHEMA_VERSION,
                    "file_sha256": None,
                }
            )
            result = clean_record(medium, record)
            replay, reason = result.cleaned_record, result.frozen_reason
            if reason or replay is None:
                cleaned_rows.append(
                    {
                        **identity,
                        "source_file": filename,
                        "medium": medium,
                        "status": "dropped",
                        "drop_reason": reason or "cleaner_rejected",
                        "cleaned_line": None,
                        "cleaned_text_sha256": None,
                        "cleaning_version": CLEANING_VERSION,
                        "mapping_status": "EXPLICIT_DROP",
                    }
                )
                continue
            kept.append((raw_index, record, replay, identity))
        aligned = align_kept_rows(
            [(raw_index, record, replay) for raw_index, record, replay, _identity_row in kept],
            disk,
            text_field=text_field,
            locator=_locator(medium),
        )
        if not aligned["ok"]:
            anomalies["replay_failures"].append({"source_file": filename, "reason": aligned["reason"]})
            for raw_index, record, replay, identity in kept:
                cleaned_rows.append(
                    {
                        **identity,
                        "source_file": filename,
                        "medium": medium,
                        "status": "ambiguous",
                        "drop_reason": None,
                        "cleaned_line": None,
                        "cleaned_text_sha256": None,
                        "cleaning_version": CLEANING_VERSION,
                        "mapping_status": "AMBIGUOUS",
                    }
                )
            continue
        if len(kept) != len(disk):
            anomalies["replay_failures"].append({"source_file": filename, "reason": "count"})
        for (raw_index, record, replay, identity), (line_no, on_disk) in zip(kept, disk):
            text = str(on_disk.get(text_field) or "")
            cleaned_rows.append(
                {
                    **identity,
                    "source_file": filename,
                    "medium": medium,
                    "status": "kept",
                    "drop_reason": None,
                    "cleaned_line": line_no,
                    "cleaned_text_sha256": _sha256_text(text),
                    "cleaning_version": CLEANING_VERSION,
                    "mapping_status": "EXACT_RAW_REPLAY",
                }
            )
            by_line[(filename, line_no)] = identity
    file_hashes = {}
    for spec in CLEAN_SPECS:
        filename = spec[0]
        file_hashes[f"data/{filename}"] = _file_sha256(ROOT / "data" / filename)
        file_hashes[f"data/cleaned/{filename}"] = _file_sha256(ROOT / "data" / "cleaned" / filename)
    for row in raw_rows:
        row["file_sha256"] = file_hashes[f"data/{row['source_file']}"]
    anomalies["kept_cleaned_lines"] = sum(1 for row in cleaned_rows if row["status"] == "kept")
    anomalies["dropped_raw"] = sum(1 for row in cleaned_rows if row["status"] == "dropped")
    return raw_rows, cleaned_rows, {"by_line": by_line, "anomalies": anomalies, "file_hashes": file_hashes}


def replay_sft_rows(by_line: dict[tuple, dict]) -> tuple[list[dict], Counter]:
    """In-memory SFT build plus promo filter. Does not write the nopromo file."""
    rows: list[dict] = []
    drops: Counter = Counter()
    cleaned_dir = ROOT / "data" / "cleaned"
    for filename, (source, text_key, title_key) in SOURCES.items():
        floor = MIN_WORDS.get(source, DEFAULT_MIN_WORDS)
        with (cleaned_dir / filename).open(encoding="utf-8") as handle:
            for line_no, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                record = json.loads(line)
                identity = by_line.get((filename, line_no))
                text = str(record.get(text_key) or "").strip()
                if len(text.split()) < floor:
                    drops["cleaned_lines_dropped_short"] += 1
                    continue
                title = _title(record, title_key, source, text)
                pieces = split_long_document(text, SFT_CHUNK_CHARS)
                if not pieces:
                    drops["cleaned_lines_dropped_empty"] += 1
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
                            "sft_chunk_ordinal": part - 1,
                            "sft_chunk_count": len(pieces),
                        }
                    )
                kept_here = []
                for piece in built:
                    if _is_direct_event_announcement(piece):
                        drops["chunks_dropped_promo"] += 1
                        continue
                    clean_output, changed = _strip_trailing_cta(str(piece["output"] or ""))
                    if changed:
                        piece["output"] = clean_output
                        if len(clean_output.split()) < 20:
                            drops["chunks_dropped_after_cta"] += 1
                            continue
                    kept_here.append(piece)
                if not kept_here:
                    drops["cleaned_lines_dropped_all_chunks"] += 1
                    continue
                if len(kept_here) > 1:
                    drops["cleaned_lines_expanded"] += 1
                for ordinal, piece in enumerate(kept_here):
                    parent = identity or {}
                    rows.append(
                        {
                            "source_id": parent.get("source_id"),
                            "source_identity_kind": parent.get("source_identity_kind"),
                            "raw_record_id": parent.get("raw_record_id"),
                            "document_id": parent.get("document_id"),
                            "source": piece["source"],
                            "medium": piece["source"],
                            "legacy_source_file": filename,
                            "legacy_source_line": line_no,
                            "sft_row_index": len(rows),
                            "sft_chunk_ordinal": ordinal,
                            "sft_chunk_count": len(kept_here),
                            "instruction_sha256": _sha256_text(piece["instruction"]),
                            "output_sha256": _sha256_text(piece["output"]),
                            "output": piece["output"],
                            "sft_build_version": SFT_BUILD_VERSION,
                            "promo_filter_version": PROMO_FILTER_VERSION,
                            "mapping_status": "EXACT_REPLAY" if parent else "UNMAPPED",
                        }
                    )
    return rows, drops


def _confirm_nopromo(replayed: list[dict]) -> int:
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
                    row.get("output"),
                )
                wanted = (
                    expected["legacy_source_file"],
                    expected["legacy_source_line"],
                    expected["source"],
                    expected["output"],
                )
                if actual != wanted:
                    mismatches += 1
            seen += 1
    if seen != len(replayed) or mismatches:
        raise RuntimeError(f"nopromo replay mismatch rows={seen} replay={len(replayed)} mismatches={mismatches}")
    return seen


def build_fit_rows(sft_rows: list[dict]) -> tuple[list[dict], dict]:
    fit_source = load_jsonl(FIT_PATH)
    sft_groups: dict[tuple, list[int]] = defaultdict(list)
    for index, row in enumerate(sft_rows):
        sft_groups[(row["legacy_source_file"], row["legacy_source_line"])].append(index)
    fit_groups: dict[tuple, list[int]] = defaultdict(list)
    for index, row in enumerate(fit_source):
        fit_groups[(row.get("source_file"), _line_number(row.get("source_line")))].append(index)
    stats = Counter()
    manifest = []
    for index, row in enumerate(fit_source):
        key = (row.get("source_file"), _line_number(row.get("source_line")))
        parents = sft_groups.get(key) or []
        siblings = fit_groups[key]
        ordinal = siblings.index(index)
        parent_outputs = [sft_rows[item]["output"] for item in parents]
        fit_outputs = [fit_source[item].get("output") or "" for item in siblings]
        if not parents:
            alignment = "unmapped"
            stats["unmapped_fit_rows"] += 1
            parent_index = None
            document = {}
        elif parent_outputs == fit_outputs:
            alignment = "exact"
            parent_index = parents[ordinal]
            document = sft_rows[parent_index]
            stats["exact_fit_rows"] += 1
        else:
            alignment = "unpaired_chunks"
            parent_index = None
            document = sft_rows[parents[0]]
            stats["unpaired_fit_rows"] += 1
        if parents and len(siblings) > len(parents):
            stats["expanded_documents"] += 1 if ordinal == 0 else 0
        elif parents and len(siblings) < len(parents):
            stats["contracted_documents"] += 1 if ordinal == 0 else 0
        output = row.get("output") or ""
        document_key = document.get("document_id")
        fitted = None
        if document_key:
            fitted = model_row_id(
                document_key,
                fit_version=CONTEXT_FIT_VERSION,
                chunk_ordinal=ordinal,
                output=output,
            )
        manifest.append(
            {
                "model_row_id": fitted,
                "document_id": document_key,
                "source_id": document.get("source_id"),
                "raw_record_id": document.get("raw_record_id"),
                "legacy_source_file": key[0],
                "legacy_source_line": key[1],
                "fit_row_index": index,
                "chunk_ordinal": ordinal,
                "parent_sft_row_index": parent_index,
                "parent_sft_chunk_count": len(parents),
                "fit_chunk_count": len(siblings),
                "chunk_alignment": alignment,
                "output_sha256": _sha256_text(output),
                "context_fit_version": CONTEXT_FIT_VERSION,
                "medium": row.get("source"),
            }
        )
    stats["sft_rows"] = len(sft_rows)
    stats["fit_rows"] = len(fit_source)
    stats["expanded_fit_rows"] = sum(
        max(0, len(fit_groups[key]) - len(sft_groups.get(key) or [])) for key in fit_groups
    )
    stats["contracted_sft_rows"] = sum(
        max(0, len(parents) - len(fit_groups.get(key) or [])) for key, parents in sft_groups.items()
    )
    return _overlay_exact_fit_lineage(manifest, fit_source, stats)


def _overlay_exact_fit_lineage(
    manifest: list[dict],
    fit_source: list[dict],
    stats: Counter,
) -> tuple[list[dict], list[dict], Counter]:
    """Replace heuristic fit parentage when the tokenizer replay sidecar is present."""
    lineage_path = OUT / "fit_lineage_exact.preview.jsonl"
    report_path = OUT / "fit_lineage_exact.report.json"
    if not lineage_path.exists() or not report_path.exists():
        return manifest, fit_source, stats
    exact = []
    with lineage_path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                exact.append(json.loads(line))
    if len(exact) != len(manifest):
        raise RuntimeError("exact fit lineage row count does not match the fit manifest")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("status") != "PASS":
        raise RuntimeError("exact fit lineage report is not PASS")
    for row, item in zip(manifest, exact):
        if row["model_row_id"] != item["model_row_id"]:
            raise RuntimeError("exact fit lineage would rename model_row_id")
        if row["fit_row_index"] != item["fit_row_index"]:
            raise RuntimeError("exact fit lineage is not in frozen fit order")
        row["parent_sft_row_index"] = item["parent_sft_row_index"]
        row["parent_sft_row_indices"] = item["parent_sft_row_indices"]
        row["relationship_type"] = item["relationship_type"]
        row["chunk_alignment"] = "exact_replay"
        row["parent_sft_chunk_count"] = len(item["parent_sft_row_indices"])
    accounting = report["accounting"]
    stats["unpaired_fit_rows"] = 0
    stats["unmapped_fit_rows"] = 0
    stats["expanded_fit_rows"] = accounting["additional_parts"]
    stats["contracted_sft_rows"] = accounting["parents_with_zero_outputs"]
    stats["exact_lineage"] = True
    stats["lead_in_rows_consumed"] = accounting["lead_in_rows_consumed"]
    stats["merged_rows"] = accounting["merged_rows"]
    stats["additional_parts"] = accounting["additional_parts"]
    stats["dropped_parents"] = accounting["parents_with_zero_outputs"]
    return manifest, fit_source, stats


def build_balance(fit_manifest: list[dict], fit_source: list[dict]) -> tuple[list[dict], list[dict], dict]:
    kept, policy = select_balanced_rows(fit_source)
    frozen = load_jsonl(BALANCED_PATH)
    if kept != frozen:
        raise RuntimeError("in-memory balance selection does not match the frozen balanced rows")
    kept_ids = {id(row) for row in kept}
    rows = []
    for fit_row, source in zip(fit_manifest, fit_source):
        selected = id(source) in kept_ids
        if selected:
            reason = "kept"
        elif source.get("source") == "youtube_saastr":
            reason = "dropped_youtube_saastr"
        elif source.get("source") == "x":
            reason = "dropped_x_cap"
        else:
            reason = "dropped_other"
        rows.append(
            {
                "model_row_id": fit_row["model_row_id"],
                "document_id": fit_row["document_id"],
                "source_id": fit_row["source_id"],
                "raw_record_id": fit_row["raw_record_id"],
                "fit_row_index": fit_row["fit_row_index"],
                "chunk_ordinal": fit_row["chunk_ordinal"],
                "selected": selected,
                "balance_reason": reason,
                "balance_version": BALANCE_VERSION,
                "medium": fit_row["medium"],
                "legacy_source_file": fit_row["legacy_source_file"],
                "legacy_source_line": fit_row["legacy_source_line"],
            }
        )
    selected_ids = [row["model_row_id"] for row in rows if row["selected"]]
    fit_ids = {row["fit_row_index"]: row["model_row_id"] for row in fit_manifest}
    renamed = [
        row["fit_row_index"]
        for row in rows
        if row["selected"] and row["model_row_id"] != fit_ids[row["fit_row_index"]]
    ]
    return rows, kept, {
        "policy": policy,
        "selected": len(selected_ids),
        "dropped": sum(1 for row in rows if not row["selected"]),
        "renamed_model_row_ids": len(renamed),
        "reasons": dict(Counter(row["balance_reason"] for row in rows)),
    }


def build_split(balance_rows: list[dict]) -> tuple[list[dict], list[dict], dict]:
    assignments = [row for row in load_jsonl(EXP008_ASSIGNMENTS) if row.get("record_type") != "meta"]
    selected = [row for row in balance_rows if row["selected"]]
    if len(assignments) != len(selected):
        raise RuntimeError(f"EXP-008 assignments {len(assignments)} != selected {len(selected)}")
    crossed = defaultdict(set)
    rows = []
    for index, (parent, assignment) in enumerate(zip(selected, assignments)):
        if assignment.get("row_index") != index:
            raise RuntimeError(f"EXP-008 row_index gap at {index}")
        split = assignment.get("split")
        group_id = assignment.get("group_id")
        crossed[group_id].add(split)
        rows.append(
            {
                "model_row_id": parent["model_row_id"],
                "document_id": parent["document_id"],
                "source_id": parent["source_id"],
                "raw_record_id": parent["raw_record_id"],
                "group_id": group_id,
                "split": split,
                "balanced_row_index": index,
                "group_version": GROUP_VERSION,
                "split_version": SPLIT_VERSION,
                "medium": assignment.get("source_platform") or parent["medium"],
                "legacy_source_file": assignment.get("source_file"),
                "legacy_source_line": _line_number(assignment.get("source_line")),
            }
        )
    cross_count = sum(1 for splits in crossed.values() if len(splits) > 1)
    return rows, assignments, {"rows": len(rows), "groups": len(crossed), "groups_crossing_splits": cross_count}


def build_relabel(split_rows: list[dict]) -> tuple[list[dict], list[dict], list[dict], dict]:
    dataset = load_jsonl(RELABEL_DATASET)
    assignments = [row for row in load_jsonl(RELABEL_ASSIGNMENTS) if row.get("record_type") != "meta"]
    dispositions = load_jsonl(RELABEL_DISPOSITIONS)
    if len(dispositions) != len(split_rows):
        raise RuntimeError("relabel dispositions do not cover every balanced row")
    by_prior = {row["prior_row_index"]: row for row in dispositions}
    if len(by_prior) != len(dispositions):
        raise RuntimeError("duplicate relabel prior_row_index")
    manifest = []
    dropped = 0
    for index, (row, assignment) in enumerate(zip(dataset, assignments)):
        if assignment.get("row_index") != index:
            raise RuntimeError(f"EXP-004 row_index gap at {index}")
        prior = assignment.get("prior_row_index")
        parent = split_rows[prior]
        disposition = by_prior[prior]
        if disposition.get("new_row_index") != index:
            raise RuntimeError(f"disposition new_row_index mismatch at {index}")
        if parent["model_row_id"] is None:
            raise RuntimeError(f"relabel row {index} has no model_row_id")
        relabel_id = relabel_row_id(
            parent["model_row_id"],
            relabel_version=RELABEL_VERSION,
            sft_role=str(row.get("sft_role") or ""),
            instruction=str(row.get("instruction") or ""),
        )
        manifest.append(
            {
                "legacy_exp004_row_index": index,
                "legacy_experiment_row_id": experiment_row_id(index),
                "legacy_prior_row_index": prior,
                "legacy_source_file": row.get("source_file"),
                "legacy_source_line": _line_number(row.get("source_line")),
                "source_id": parent["source_id"],
                "source_identity_kind": None,
                "raw_record_id": parent["raw_record_id"],
                "document_id": parent["document_id"],
                "model_row_id": parent["model_row_id"],
                "relabel_row_id": relabel_id,
                "rag_row_id": rag_row_id(relabel_id),
                "group_id": assignment.get("group_id"),
                "split": assignment.get("split"),
                "sft_role": row.get("sft_role"),
                "prior_group": row.get("prior_group"),
                "medium": row.get("source"),
                "relabel_version": RELABEL_VERSION,
                "identity_version": IDENTITY_VERSION,
                "instruction_sha256": _sha256_text(str(row.get("instruction") or "")),
                "output_sha256": _sha256_text(str(row.get("output") or "")),
            }
        )
        if assignment.get("group_id") != parent["group_id"]:
            raise RuntimeError(f"group_id changed at relabel row {index}")
    dropped = sum(1 for row in dispositions if "new_row_index" not in row)
    if len(manifest) + dropped != len(split_rows):
        raise RuntimeError("relabel kept plus dropped does not cover the split")
    return manifest, dataset, assignments, {"kept": len(manifest), "dropped": dropped}


def _fill_identity_kind(relabel_rows: list[dict], raw_rows: list[dict]) -> None:
    kinds = {row["raw_record_id"]: row["source_identity_kind"] for row in raw_rows if row["raw_record_id"]}
    for row in relabel_rows:
        row["source_identity_kind"] = kinds.get(row["raw_record_id"])


def build_rag(dataset: list[dict], assignments: list[dict], relabel_rows: list[dict]) -> tuple[list[dict], dict]:
    dispositions = load_jsonl(DEFAULT_GOLD_DISPOSITIONS)
    gold_rows = load_jsonl(DEFAULT_GOLD_EVAL)
    family = blocked_group_ids(assignments, dispositions)
    _family_docs, family_stats = select_train_documents(dataset, assignments, family)
    headline, _details = headline_overlap_groups(dataset, assignments, gold_rows, family)
    final_docs, final_stats = select_train_documents(dataset, assignments, family | headline)
    eligible_ids = {doc["id"] for doc in final_docs}
    family_only_ids = {doc["id"] for doc in _family_docs}
    rows = []
    for row in relabel_rows:
        if row["split"] != "train":
            continue
        experiment_id = row["legacy_experiment_row_id"]
        reasons = []
        if experiment_id not in family_only_ids and experiment_id not in eligible_ids:
            reasons.append("gold_overlap_family")
        elif experiment_id in family_only_ids and experiment_id not in eligible_ids:
            reasons.append("headline_overlap")
        rows.append(
            {
                "relabel_row_id": row["relabel_row_id"],
                "rag_row_id": row["rag_row_id"],
                "eligible": experiment_id in eligible_ids,
                "exclusion_reasons": reasons,
                "group_id": row["group_id"],
                "split": row["split"],
                "medium": row["medium"],
                "legacy_experiment_row_id": experiment_id,
                "legacy_exp004_row_index": row["legacy_exp004_row_index"],
                "source_id": row["source_id"],
                "raw_record_id": row["raw_record_id"],
                "document_id": row["document_id"],
                "model_row_id": row["model_row_id"],
                "rag_governance_version": RAG_GOVERNANCE_VERSION,
            }
        )
    summary = {
        "train_candidates": family_stats["train_rows"],
        "gold_family_exclusions": family_stats["excluded_gold"],
        "headline_exclusions": final_stats["excluded_gold"] - family_stats["excluded_gold"],
        "eligible": final_stats["indexed"],
        "headline_removed_ids": sorted(family_only_ids - eligible_ids),
    }
    return rows, summary


def _transition(name, input_count, output_count, dropped, expanded, ambiguous, unmapped, note) -> dict:
    return {
        "transition": name,
        "input": input_count,
        "output": output_count,
        "dropped": dropped,
        "expanded": expanded,
        "ambiguous": ambiguous,
        "unmapped": unmapped,
        "note": note,
    }


def build_coverage(
    raw_rows,
    cleaned_rows,
    sft_rows,
    sft_drops,
    fit_stats,
    balance_stats,
    split_stats,
    relabel_stats,
    rag_summary,
    anomalies,
) -> dict:
    kept_cleaned = anomalies["kept_cleaned_lines"]
    sft_unmapped = sum(1 for row in sft_rows if row["mapping_status"] != "EXACT_REPLAY")
    cleaned_drop_reasons = dict(Counter(row["drop_reason"] for row in cleaned_rows if row["status"] == "dropped"))
    transitions = [
        _transition(
            "RAW → CLEANED",
            len(raw_rows),
            kept_cleaned,
            anomalies["dropped_raw"],
            anomalies["one_raw_to_multiple_cleaned"],
            len(anomalies["replay_failures"]),
            anomalies["unmapped_cleaned_lines"],
            "One kept raw capture becomes one cleaned line. Drop reasons are the cleaner reasons.",
        ),
        _transition(
            "CLEANED → SFT",
            kept_cleaned,
            len(sft_rows),
            sft_drops["cleaned_lines_dropped_short"]
            + sft_drops["cleaned_lines_dropped_empty"]
            + sft_drops["cleaned_lines_dropped_all_chunks"],
            sft_drops["cleaned_lines_expanded"],
            0,
            sft_unmapped,
            "Expanded counts cleaned lines that become more than one nopromo chunk. "
            f"Promo chunk drops={sft_drops['chunks_dropped_promo']}. "
            f"CTA chunk drops={sft_drops['chunks_dropped_after_cta']}.",
        ),
        _transition(
            "SFT → FIT",
            fit_stats["sft_rows"],
            fit_stats["fit_rows"],
            fit_stats["dropped_parents"] if fit_stats.get("exact_lineage") else 0,
            fit_stats["expanded_fit_rows"],
            fit_stats["unpaired_fit_rows"],
            fit_stats["unmapped_fit_rows"],
            (
                "Exact tokenizer replay. "
                f"Lead-in rows consumed={fit_stats['lead_in_rows_consumed']}. "
                f"Merged rows={fit_stats['merged_rows']}. "
                f"Parents with zero outputs={fit_stats['dropped_parents']}. "
                f"Additional parts={fit_stats['additional_parts']}. "
                "No fit row is left without parent SFT indices."
            )
            if fit_stats.get("exact_lineage")
            else (
                "Every fit row keeps the nopromo source_file and source_line. "
                "Ambiguous counts fit rows whose document chunk texts are not an exact sequence match, "
                "so those rows do not receive a single parent SFT index. "
                f"Contracted SFT rows inside those documents={fit_stats['contracted_sft_rows']}. "
                "The tokenizer was not loaded."
            ),
        ),
        _transition(
            "FIT → BALANCE",
            fit_stats["fit_rows"],
            balance_stats["selected"],
            balance_stats["dropped"],
            0,
            0,
            0,
            "Balance selects rows. Selected model_row_id values are copied from the fit sidecar. "
            f"Renamed ids={balance_stats['renamed_model_row_ids']}.",
        ),
        _transition(
            "BALANCE → SPLIT",
            balance_stats["selected"],
            split_stats["rows"],
            0,
            0,
            0,
            0,
            f"EXP-008 labels the balanced rows. Groups crossing splits={split_stats['groups_crossing_splits']}.",
        ),
        _transition(
            "SPLIT → RELABEL",
            split_stats["rows"],
            relabel_stats["kept"],
            relabel_stats["dropped"],
            0,
            0,
            0,
            "Each kept balanced row becomes one Relabel row. model_row_id is the parent fit id.",
        ),
        _transition(
            "RELABEL train → RAG",
            rag_summary["train_candidates"],
            rag_summary["eligible"],
            rag_summary["gold_family_exclusions"] + rag_summary["headline_exclusions"],
            0,
            0,
            0,
            "Eligibility is governance state. relabel_row_id is unchanged. Chroma was not opened.",
        ),
    ]
    return {
        "artifact": "provenance_preview",
        "canonical_production_artifact": False,
        "identity_version": IDENTITY_VERSION,
        "raw_record_id_semantics": (
            "raw_record_id hashes the canonical content-bearing fields. "
            "raw_payload_sha256 hashes the stored JSON object bytes from the JSONL line, "
            "excluding the line terminator. File SHA-256 remains the container byte guarantee."
        ),
        "transitions": transitions,
        "cleaned_drop_reasons": cleaned_drop_reasons,
        "sft_drops": dict(sft_drops),
        "balance_reasons": balance_stats["reasons"],
        "rag": rag_summary,
        "split_groups_crossing": split_stats["groups_crossing_splits"],
        "balance_renamed_model_row_ids": balance_stats["renamed_model_row_ids"],
        "replay_failures": anomalies["replay_failures"],
        "file_hashes": anomalies.get("file_hashes") or {},
    }


def build_all() -> dict:
    raw_rows, cleaned_rows, cleaned_state = build_raw_and_cleaned()
    if cleaned_state["anomalies"]["replay_failures"]:
        raise RuntimeError(f"cleaner replay failed: {cleaned_state['anomalies']['replay_failures']}")
    sft_rows, sft_drops = replay_sft_rows(cleaned_state["by_line"])
    _confirm_nopromo(sft_rows)
    fit_manifest, fit_source, fit_stats = build_fit_rows(sft_rows)
    for row in sft_rows:
        row.pop("output", None)
    balance_rows, _kept, balance_stats = build_balance(fit_manifest, fit_source)
    split_rows, _assignments, split_stats = build_split(balance_rows)
    relabel_rows, dataset, relabel_assignments, relabel_stats = build_relabel(split_rows)
    _fill_identity_kind(relabel_rows, raw_rows)
    rag_rows, rag_summary = build_rag(dataset, relabel_assignments, relabel_rows)
    coverage = build_coverage(
        raw_rows,
        cleaned_rows,
        sft_rows,
        sft_drops,
        fit_stats,
        balance_stats,
        split_stats,
        relabel_stats,
        rag_summary,
        {**cleaned_state["anomalies"], "file_hashes": cleaned_state["file_hashes"]},
    )
    coverage["file_hashes"] = cleaned_state["file_hashes"]
    write_jsonl(RAW_MANIFEST, raw_rows)
    write_jsonl(CLEANED_MANIFEST, cleaned_rows)
    write_jsonl(SFT_MANIFEST, sft_rows)
    write_jsonl(FIT_MANIFEST, fit_manifest)
    write_jsonl(BALANCED_MANIFEST, balance_rows)
    write_jsonl(SPLIT_MANIFEST, split_rows)
    write_jsonl(RELABEL_MANIFEST, relabel_rows)
    write_jsonl(RAG_MANIFEST, rag_rows)
    COVERAGE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with COVERAGE_PATH.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(coverage, indent=2, ensure_ascii=False) + "\n")
    return coverage


def main() -> int:
    coverage = build_all()
    print(json.dumps(coverage["transitions"], indent=2))
    print("rag", json.dumps(coverage["rag"]))
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
