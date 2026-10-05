"""Read-only sidecar from frozen Relabel rows to canonical ids.

Calls ``clean_record`` in memory. Does not call ``clean_file`` or ``clean_scraped``.
Does not write ``data/cleaned``, Chroma, or experiment directories.

The preview written by ``main`` is not a production artifact.
"""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

from host_finetune.canonical_ids import (
    CONTENT_FALLBACK,
    CONTEXT_FIT_VERSION,
    IDENTITY_VERSION,
    RELABEL_VERSION,
    document_id,
    experiment_row_id,
    model_row_id,
    rag_row_id,
    raw_record_id,
    relabel_row_id,
    source_identity_for_raw,
)
from host_finetune.canonical_cleaning import clean_record

ROOT = Path(__file__).resolve().parents[1]
PREVIEW_PATH = ROOT / "artifacts" / "refactor" / "canonical_identity_crosswalk.preview.jsonl"
COVERAGE_PATH = ROOT / "artifacts" / "refactor" / "canonical_identity_crosswalk.coverage.json"

RELABEL_DATASET = (
    ROOT / "experiments" / "EXP-20261003-004-relabel-continuations" / "raw" / "dataset.jsonl"
)
RELABEL_ASSIGNMENTS = (
    ROOT / "experiments" / "EXP-20261003-004-relabel-continuations" / "raw" / "split_assignments.jsonl"
)
FIT_PATH = ROOT / "host_finetune" / "data" / "dataset_from_cleaned_sources_fit512_keepbreaks.jsonl"
BALANCED_PATH = ROOT / "host_finetune" / "data" / "dataset_fit512_keepbreaks_balanced.jsonl"

# filename, clean_row kind, identity medium, text field, title field, scrub kwargs
CLEAN_SPECS = (
    ("jasonlemkin_blog.jsonl", "blog", "blog", "content", "title", {}),
    ("jasonlemkinlinkedin.jsonl", "linkedin", "linkedin", "content", None, {"linkedin": True}),
    ("jasonlk_originals.jsonl", "x", "x", "text", None, {}),
    (
        "jasonmlemkinyoutubetranscripts.jsonl",
        "youtube",
        "youtube_jason",
        "transcript_text",
        "video_title",
        {"unescape": True},
    ),
    (
        "saastryoutubetranscripts.jsonl",
        "youtube",
        "youtube_saastr",
        "transcript_text",
        "video_title",
        {"unescape": True},
    ),
)

EXACT_NATURAL_KEY = "EXACT_NATURAL_KEY"
EXACT_RAW_REPLAY = "EXACT_RAW_REPLAY"
CONTENT_FALLBACK_STATUS = "CONTENT_FALLBACK"
AMBIGUOUS = "AMBIGUOUS"
UNMAPPED = "UNMAPPED"
NATURAL_KINDS = frozenset({"platform_url", "platform_tweet_id", "platform_urn", "platform_video_id"})


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def load_assignments(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            if record.get("record_type") == "meta":
                continue
            rows.append(record)
    rows.sort(key=lambda row: row["row_index"])
    return rows


def load_cleaned_lines(path: Path) -> list[tuple[int, dict]]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            if line.strip():
                rows.append((number, json.loads(line)))
    return rows


def _line_number(value) -> int | None:
    if isinstance(value, int):
        return value
    text = str(value).strip()
    if text.isdigit():
        return int(text)
    return None


def align_kept_rows(
    kept: list[tuple[int, dict, dict]],
    cleaned_rows: list[tuple[int, dict]],
    *,
    text_field: str,
    locator,
) -> dict:
    """Pair in-memory kept rows with cleaned-file lines.

    A count or text mismatch rejects the whole file. Partial zip alignment
    would hide a shift.
    """
    if len(kept) != len(cleaned_rows):
        return {
            "ok": False,
            "reason": "count_mismatch",
            "kept": len(kept),
            "cleaned": len(cleaned_rows),
            "by_line": {},
            "captures": [],
        }
    by_line = {}
    captures = []
    for (raw_index, raw, replay), (line_no, on_disk) in zip(kept, cleaned_rows):
        text_ok = replay.get(text_field) == on_disk.get(text_field)
        locator_ok = locator(raw) == locator(on_disk)
        if not text_ok or not locator_ok:
            return {
                "ok": False,
                "reason": "text_or_locator_mismatch",
                "kept": len(kept),
                "cleaned": len(cleaned_rows),
                "by_line": {},
                "captures": [],
            }
        by_line[line_no] = {"raw_index": raw_index, "raw": raw}
        captures.append({"line_no": line_no, "raw_index": raw_index, "raw": raw})
    return {
        "ok": True,
        "reason": "replay_match",
        "kept": len(kept),
        "cleaned": len(cleaned_rows),
        "by_line": by_line,
        "captures": captures,
    }


def _locator(medium: str):
    def blog_or_empty(row: dict) -> str:
        if medium == "blog":
            return str(row.get("url") or "")
        if medium == "x":
            return str(row.get("id") or "")
        if medium in {"youtube_jason", "youtube_saastr"}:
            return str(row.get("video_id") or "")
        return str(row.get("urn") or "") + "\n" + str(row.get("url") or "")

    return blog_or_empty


def replay_source(raw_rows: list[dict], cleaned_rows: list[tuple[int, dict]], spec: tuple) -> dict:
    _filename, _kind, medium, text_field, _title_field, _scrub_kwargs = spec
    kept = []
    for raw_index, raw in enumerate(raw_rows):
        result = clean_record(medium, raw)
        if result.frozen_reason or result.cleaned_record is None:
            continue
        kept.append((raw_index, raw, result.cleaned_record))
    aligned = align_kept_rows(
        kept,
        cleaned_rows,
        text_field=text_field,
        locator=_locator(medium),
    )
    aligned["medium"] = medium
    return aligned


def _fit_ordinals(fit_rows: list[dict]) -> dict[tuple, list[str]]:
    grouped: dict[tuple, list[str]] = defaultdict(list)
    for row in fit_rows:
        key = (row.get("source_file"), _line_number(row.get("source_line")))
        grouped[key].append(row.get("output") or "")
    return grouped


def _balanced_fit_indexes(fit_rows: list[dict], balanced_rows: list[dict]) -> list[int | None]:
    cursor = 0
    indexes: list[int | None] = []
    for balanced in balanced_rows:
        found = None
        while cursor < len(fit_rows):
            if fit_rows[cursor] == balanced:
                found = cursor
                cursor += 1
                break
            cursor += 1
        indexes.append(found)
    return indexes


def _status_for(identity_kind: str | None, aligned: bool) -> str:
    if not aligned or identity_kind is None:
        return UNMAPPED
    if identity_kind == CONTENT_FALLBACK:
        return CONTENT_FALLBACK_STATUS
    if identity_kind in NATURAL_KINDS:
        return EXACT_NATURAL_KEY
    return AMBIGUOUS


def _empty_identity() -> dict:
    return {
        "source_id": None,
        "source_identity_kind": None,
        "external_id": None,
        "raw_record_id": None,
        "document_id": None,
        "raw_record_index": None,
    }


def _identity_from_raw(medium: str, raw: dict, raw_index: int) -> dict:
    try:
        identity = source_identity_for_raw(medium, raw)
        capture = raw_record_id(medium, raw)
    except (ValueError, TypeError):
        return _empty_identity()
    return {
        "source_id": identity.source_id,
        "source_identity_kind": identity.source_identity_kind,
        "external_id": identity.external_id,
        "raw_record_id": capture,
        "document_id": document_id(capture),
        "raw_record_index": raw_index,
    }


def build_crosswalk_rows(
    relabel_rows: list[dict],
    assignments: list[dict],
    *,
    alignments: dict[str, dict],
    fit_groups: dict[tuple, list[str]],
    balanced_indexes: list[int | None],
    fit_rows: list[dict],
) -> list[dict]:
    if len(relabel_rows) != len(assignments):
        raise ValueError("relabel dataset and assignment lengths differ")
    preview = []
    for index, (row, assignment) in enumerate(zip(relabel_rows, assignments)):
        if assignment.get("row_index") != index:
            raise ValueError(f"assignment row_index gap at {index}")
        source_file = row.get("source_file") or ""
        source_line = _line_number(row.get("source_line"))
        medium = row.get("source") or ""
        alignment = alignments.get(source_file) or {}
        aligned = bool(alignment.get("ok"))
        located = alignment.get("by_line", {}).get(source_line) if aligned else None
        if aligned and located is not None:
            identity = _identity_from_raw(alignment["medium"], located["raw"], located["raw_index"])
            status = _status_for(identity["source_identity_kind"], True)
            method = "raw_cleaner_replay"
            confidence = "high" if status in {EXACT_NATURAL_KEY, CONTENT_FALLBACK_STATUS} else "low"
        elif aligned:
            identity = _empty_identity()
            status = UNMAPPED
            method = "cleaned_line_absent"
            confidence = "none"
        else:
            identity = _empty_identity()
            status = AMBIGUOUS
            method = "cleaner_replay_" + str(alignment.get("reason") or "missing_source")
            confidence = "low"

        prior = assignment.get("prior_row_index")
        chunk_ordinal = None
        fitted_output = None
        model_mapping = UNMAPPED
        if (
            status in {EXACT_NATURAL_KEY, CONTENT_FALLBACK_STATUS}
            and isinstance(prior, int)
            and 0 <= prior < len(balanced_indexes)
            and balanced_indexes[prior] is not None
        ):
            fit_index = balanced_indexes[prior]
            parent = fit_rows[fit_index]
            parent_key = (parent.get("source_file"), _line_number(parent.get("source_line")))
            outputs = fit_groups.get(parent_key) or []
            fitted_output = parent.get("output") or ""
            matches = [ordinal for ordinal, text in enumerate(outputs) if text == fitted_output]
            if len(matches) == 1 and parent_key == (source_file, source_line):
                chunk_ordinal = matches[0]
                model_mapping = EXACT_RAW_REPLAY
            else:
                model_mapping = AMBIGUOUS

        model_id = None
        relabel_id = None
        rag_id = None
        if chunk_ordinal is not None and identity["document_id"] and fitted_output is not None:
            model_id = model_row_id(
                identity["document_id"],
                fit_version=CONTEXT_FIT_VERSION,
                chunk_ordinal=chunk_ordinal,
                output=fitted_output,
            )
            relabel_id = relabel_row_id(
                model_id,
                relabel_version=RELABEL_VERSION,
                sft_role=str(row.get("sft_role") or ""),
                instruction=str(row.get("instruction") or ""),
            )
            rag_id = rag_row_id(relabel_id)

        preview.append(
            {
                "legacy_source_file": source_file,
                "legacy_source_line": source_line,
                "legacy_exp004_row_index": index,
                "legacy_experiment_row_id": experiment_row_id(index),
                "legacy_group_id": assignment.get("group_id"),
                "legacy_prior_row_index": prior,
                "legacy_split": assignment.get("split"),
                "source_id": identity["source_id"],
                "source_identity_kind": identity["source_identity_kind"],
                "external_id": identity["external_id"],
                "raw_record_id": identity["raw_record_id"],
                "document_id": identity["document_id"],
                "model_row_id": model_id,
                "relabel_row_id": relabel_id,
                "rag_row_id": rag_id,
                "chunk_ordinal": chunk_ordinal,
                "raw_record_index": identity["raw_record_index"],
                "source": medium,
                "medium": medium,
                "mapping_status": status,
                "mapping_confidence": confidence,
                "mapping_method": method,
                "model_mapping_status": model_mapping,
                "identity_version": IDENTITY_VERSION,
            }
        )
    return preview


def _bucket(rows: list[dict]) -> dict:
    counts = Counter(row["mapping_status"] for row in rows)
    natural = counts[EXACT_NATURAL_KEY]
    fallback = counts[CONTENT_FALLBACK_STATUS]
    ambiguous = counts[AMBIGUOUS]
    unmapped = counts[UNMAPPED]
    exact_replay = sum(
        1
        for row in rows
        if row["mapping_method"] == "raw_cleaner_replay"
        and row["mapping_status"] in {EXACT_NATURAL_KEY, CONTENT_FALLBACK_STATUS}
    )
    return {
        "rows": len(rows),
        "natural_source_id": natural,
        "fallback_source_identity": fallback,
        "exact_raw_replay": exact_replay,
        "ambiguous": ambiguous,
        "unmapped": unmapped,
        "with_model_row_id": sum(1 for row in rows if row["model_row_id"]),
        "with_relabel_row_id": sum(1 for row in rows if row["relabel_row_id"]),
        "with_raw_record_id": sum(1 for row in rows if row["raw_record_id"]),
    }


def _group_captures(pairs: list[tuple[str, dict]]) -> dict[str, dict]:
    grouped: dict[str, dict] = {}
    for medium, raw in pairs:
        try:
            identity = source_identity_for_raw(medium, raw)
            capture = raw_record_id(medium, raw)
        except (ValueError, TypeError):
            continue
        slot = grouped.setdefault(
            identity.source_id,
            {
                "medium": medium,
                "source_identity_kind": identity.source_identity_kind,
                "external_id": identity.external_id,
                "raw_record_ids": set(),
            },
        )
        slot["raw_record_ids"].add(capture)
    return grouped


def capture_stats(pairs: list[tuple[str, dict]]) -> dict:
    """Repeated external items. ``pairs`` is ``(medium, raw_record)``."""
    grouped = _group_captures(pairs)
    repeated = [slot for slot in grouped.values() if len(slot["raw_record_ids"]) > 1]
    return {
        "distinct_source_id": len(grouped),
        "distinct_raw_record_id": len({value for slot in grouped.values() for value in slot["raw_record_ids"]}),
        "source_id_with_multiple_raw_records": len(repeated),
    }


def repeated_capture_rows(raw_pairs: list[tuple[str, dict]], kept_pairs: list[tuple[str, dict]]) -> list[dict]:
    raw_grouped = _group_captures(raw_pairs)
    kept_grouped = _group_captures(kept_pairs)
    rows = []
    for source, slot in raw_grouped.items():
        count = len(slot["raw_record_ids"])
        if count < 2:
            continue
        kept = kept_grouped.get(source)
        rows.append(
            {
                "medium": slot["medium"],
                "source_identity_kind": slot["source_identity_kind"],
                "external_id": slot["external_id"],
                "raw_capture_count": count,
                "kept_capture_count": len(kept["raw_record_ids"]) if kept else 0,
            }
        )
    rows.sort(key=lambda row: (row["medium"], row["external_id"] or "", -row["raw_capture_count"]))
    return rows


def coverage_report(
    rows: list[dict],
    *,
    raw_pairs: list[tuple[str, dict]],
    kept_pairs: list[tuple[str, dict]],
) -> dict:
    media = ["blog", "linkedin", "x", "youtube_jason", "youtube_saastr"]
    by_medium = {medium: _bucket([row for row in rows if row["medium"] == medium]) for medium in media}
    total = _bucket(rows)
    crosswalk_sources: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        if row["source_id"] and row["raw_record_id"]:
            crosswalk_sources[row["source_id"]].add(row["raw_record_id"])
    return {
        "artifact": "preview",
        "canonical_production_artifact": False,
        "identity_version": IDENTITY_VERSION,
        "total_relabel_rows": len(rows),
        "totals": total,
        "by_medium": by_medium,
        "by_medium_note": (
            "youtube_saastr has zero Relabel rows because select_balanced_rows drops that source "
            "before the split. Those raw rows are still in raw_records_before_cleaning."
        ),
        "crosswalk_distinct_source_id": len(crosswalk_sources),
        "crosswalk_distinct_raw_record_id": len(
            {value for values in crosswalk_sources.values() for value in values}
        ),
        "crosswalk_source_id_with_multiple_raw_records": sum(
            1 for values in crosswalk_sources.values() if len(values) > 1
        ),
        "raw_records_before_cleaning": capture_stats(raw_pairs),
        "kept_cleaned_captures": capture_stats(kept_pairs),
        "repeated_raw_captures": repeated_capture_rows(raw_pairs, kept_pairs),
        "status_sum_equals_rows": total["natural_source_id"]
        + total["fallback_source_identity"]
        + total["ambiguous"]
        + total["unmapped"]
        == len(rows),
    }


def build_preview(root: Path | None = None) -> tuple[list[dict], dict]:
    base = root or ROOT
    data = base / "data"
    cleaned = data / "cleaned"
    alignments = {}
    raw_pairs: list[tuple[str, dict]] = []
    for spec in CLEAN_SPECS:
        filename = spec[0]
        medium = spec[2]
        raw_rows = load_jsonl(data / filename)
        raw_pairs.extend((medium, row) for row in raw_rows)
        cleaned_rows = load_cleaned_lines(cleaned / filename)
        alignments[filename] = replay_source(raw_rows, cleaned_rows, spec)
    fit_rows = load_jsonl(base / "host_finetune" / "data" / "dataset_from_cleaned_sources_fit512_keepbreaks.jsonl")
    balanced_rows = load_jsonl(base / "host_finetune" / "data" / "dataset_fit512_keepbreaks_balanced.jsonl")
    relabel_rows = load_jsonl(
        base / "experiments" / "EXP-20261003-004-relabel-continuations" / "raw" / "dataset.jsonl"
    )
    assignments = load_assignments(
        base / "experiments" / "EXP-20261003-004-relabel-continuations" / "raw" / "split_assignments.jsonl"
    )
    rows = build_crosswalk_rows(
        relabel_rows,
        assignments,
        alignments=alignments,
        fit_groups=_fit_ordinals(fit_rows),
        balanced_indexes=_balanced_fit_indexes(fit_rows, balanced_rows),
        fit_rows=fit_rows,
    )
    kept_pairs = [
        (alignment["medium"], capture["raw"])
        for alignment in alignments.values()
        if alignment.get("ok")
        for capture in alignment["captures"]
    ]
    return rows, coverage_report(rows, raw_pairs=raw_pairs, kept_pairs=kept_pairs)


def write_preview(rows: list[dict], report: dict, preview_path: Path, coverage_path: Path) -> None:
    preview_path.parent.mkdir(parents=True, exist_ok=True)
    with preview_path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    with coverage_path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(report, indent=2, ensure_ascii=False) + "\n")


def main() -> int:
    rows, report = build_preview()
    write_preview(rows, report, PREVIEW_PATH, COVERAGE_PATH)
    print(f"rows={report['total_relabel_rows']} status_sum_ok={report['status_sum_equals_rows']}")
    print(json.dumps(report["totals"], sort_keys=True))
    print(json.dumps(report["by_medium"], sort_keys=True))
    print("raw", json.dumps(report["raw_records_before_cleaning"], sort_keys=True))
    print("kept", json.dumps(report["kept_cleaned_captures"], sort_keys=True))
    print(f"repeated_raw_captures={len(report['repeated_raw_captures'])}")
    print(f"wrote {PREVIEW_PATH}")
    return 0 if report["status_sum_equals_rows"] and report["total_relabel_rows"] == len(rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
