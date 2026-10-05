"""Shadow stages for the Airflow/Spark v2 pipeline.

These functions are the only transformation path. The DAG and the local
command both call them. They write under a staging directory and refuse
frozen corpus paths.

Cleaning calls ``canonical_cleaning.clean_record``. SFT uses the existing
chunk and promo helpers. Fit calls ``canonical_fit.fit_rows`` on the full
ordered nopromo sequence, because lead-in merges are adjacency-sensitive.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from collections import Counter
from pathlib import Path

from host_finetune.build_sft_from_cleaned_sources import (
    DEFAULT_MIN_WORDS,
    MIN_WORDS,
    SOURCES,
    _title,
)
from host_finetune.canonical_cleaning import SOURCE_SPECS, clean_record
from host_finetune.canonical_fit import (
    FitConfig,
    fit_rows,
    load_compatibility_tokenizer,
    serialize_fit_records,
)
from host_finetune.filter_event_promos import _is_direct_event_announcement, _strip_trailing_cta
from host_finetune.sft_chunk_utils import instruction_for, split_long_document

ROOT = Path(__file__).resolve().parents[1]
CONFIG_VERSION = "spark-v2-shadow-fit-v1"
SFT_CHUNK_CHARS = 1600
STAGING_ROOT = ROOT / "artifacts" / "refactor" / "spark_v2"

RAW_SHA256 = {
    "jasonlemkin_blog.jsonl": "8492ef11acacd767ce6bcdbdf6215467b6ba74d3e57dc50ffcd0af7deebc34a8",
    "jasonlemkinlinkedin.jsonl": "a2899576ac75ebb5fe05af15bdd0425eef9636de42bb058e82b9c456ae71ef09",
    "jasonlk_originals.jsonl": "a2021528bf4a34cd34c074d252d6bf246fddbe422033e7b88ebc0e11c74044c1",
    "jasonmlemkinyoutubetranscripts.jsonl": "a943eb049b79b75e6aa17073cfcd4a7431e7a163b3a4276f4f8116cc31a1e337",
    "saastryoutubetranscripts.jsonl": "89e4de78a7e3e947625523b27fbc29a0b75ef217f5c6e7a76d1aaca41916560d",
}
EXPECTED_RAW_ROWS = 32835
EXPECTED_CLEANED_ROWS = 30370
EXPECTED_DROPS = {
    "event_promo": 450,
    "linkedin_feed_chrome": 623,
    "too_short": 1203,
    "empty": 150,
    "empty_transcript": 39,
}
EXPECTED_KEPT = {
    "blog": 3466,
    "linkedin": 2006,
    "x": 24277,
    "youtube_jason": 428,
    "youtube_saastr": 193,
}
EXPECTED_NOPROMO_ROWS = 42771
EXPECTED_NOPROMO_SHA256 = "55b9680657ed72a424d47ea7102214d9b2c6693a535572e02520e406d57f8241"
EXPECTED_FIT_ROWS = 43012
EXPECTED_FIT_SHA256 = "bf00e53fd2401f2d6bea6ac39cb69829fc21761de3a4f7225fed7ecd75082fba"
FROZEN_CLEANED = ROOT / "data" / "cleaned"
FROZEN_NOPROMO = ROOT / "host_finetune" / "data" / "dataset_from_cleaned_sources_nopromo.jsonl"
FROZEN_FIT = ROOT / "host_finetune" / "data" / "dataset_from_cleaned_sources_fit512_keepbreaks.jsonl"
LINEAGE = ROOT / "artifacts" / "refactor" / "provenance" / "fit_lineage_exact.preview.jsonl"
FORBIDDEN_OUTPUT_PARTS = (
    "host_finetune/data",
    "host_finetune/output",
    "data/cleaned",
    "experiments",
)


class ParityError(RuntimeError):
    """A shadow stage did not match the Python oracle. Downstream stages must not run."""


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_state() -> tuple[str, bool]:
    sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip())
    return sha, dirty


def assert_staging_dir(path: Path) -> Path:
    resolved = path.resolve()
    relative = resolved.relative_to(ROOT.resolve()).as_posix() if _is_relative_to(resolved, ROOT.resolve()) else resolved.as_posix()
    for part in FORBIDDEN_OUTPUT_PARTS:
        if relative == part or relative.startswith(part + "/"):
            raise ParityError(f"refusing to write frozen path: {relative}")
    if "lemkin_content" in relative or "lemkin_train_only" in relative:
        raise ParityError(f"refusing to write a production collection path: {relative}")
    return resolved


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _write_jsonl(path: Path, rows: list[dict]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows).encode("utf-8")
    path.write_bytes(payload)
    return hashlib.sha256(payload).hexdigest()


def write_manifest(staging: Path, stage: str, body: dict) -> Path:
    staging = assert_staging_dir(staging)
    sha, dirty = git_state()
    document = {
        "run_id": body["run_id"],
        "stage": stage,
        "input_rows": body.get("input_rows"),
        "output_rows": body.get("output_rows"),
        "input_sha256": body.get("input_sha256"),
        "output_sha256": body.get("output_sha256"),
        "git_sha": sha,
        "dirty": dirty,
        "config_version": CONFIG_VERSION,
        "status": body["status"],
        "mismatches": body.get("mismatches") or {},
        "details": body.get("details") or {},
    }
    path = staging / "manifests" / f"{stage}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


def gate(stage: str, mismatches: dict) -> None:
    bad = {key: value for key, value in mismatches.items() if value}
    if bad:
        raise ParityError(f"{stage} parity failed: {bad}")


def iter_raw_records(data_dir: Path | None = None) -> list[dict]:
    """Read the five frozen raw files in canonical source order."""
    data_dir = data_dir or (ROOT / "data")
    rows = []
    for ordinal, spec in enumerate(SOURCE_SPECS):
        filename, _kind, medium, _text, _title, _scrub = spec
        path = data_dir / filename
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                rows.append(
                    {
                        "source_ordinal": ordinal,
                        "raw_record_index": line_number,
                        "source_file": filename,
                        "medium": medium,
                        "record": json.loads(line),
                    }
                )
    return rows


def assert_raw_hashes(data_dir: Path | None = None) -> dict[str, str]:
    data_dir = data_dir or (ROOT / "data")
    actual = {}
    for filename, expected in RAW_SHA256.items():
        digest = file_sha256(data_dir / filename)
        actual[filename] = digest
        if digest != expected:
            raise ParityError(f"raw hash mismatch for {filename}: {digest}")
    return actual


def clean_indexed(item: dict) -> dict:
    """One raw row through the canonical cleaner. Order keys are preserved."""
    result = clean_record(item["medium"], item["record"])
    return {
        "source_ordinal": item["source_ordinal"],
        "raw_record_index": item["raw_record_index"],
        "source_file": item["source_file"],
        "medium": result.medium,
        "kept": result.kept,
        "frozen_reason": result.frozen_reason,
        "source_id": result.source_id,
        "raw_record_id": result.raw_record_id,
        "cleaned_record": result.cleaned_record,
    }


def clean_partition(items) -> list[dict]:
    """Worker body for mapPartitions. Safe to call on one unordered partition."""
    return [clean_indexed(item) for item in items]


def restore_order(rows: list[dict]) -> list[dict]:
    """Explicit global order. Partition arrival order is not the corpus order."""
    return sorted(rows, key=lambda row: (row["source_ordinal"], row["raw_record_index"]))


def clean_ordered(items: list[dict]) -> list[dict]:
    return restore_order(clean_partition(items))


def clean_with_spark(items: list[dict], slices: int = 4) -> list[dict]:
    """Distribute ``clean_partition``, then restore source order on the driver."""
    from pyspark.sql import SparkSession

    spark = SparkSession.builder.master("local[*]").appName("lemkin-v2-shadow-clean").getOrCreate()
    try:
        cleaned = spark.sparkContext.parallelize(items, numSlices=slices).mapPartitions(clean_partition).collect()
    finally:
        spark.stop()
    return restore_order(list(cleaned))


def kept_by_file(cleaned: list[dict]) -> dict[str, list[dict]]:
    grouped = {spec[0]: [] for spec in SOURCE_SPECS}
    for row in cleaned:
        if row["kept"] and row["cleaned_record"] is not None:
            grouped[row["source_file"]].append(row["cleaned_record"])
    return grouped


def compare_cleaned(cleaned: list[dict]) -> dict:
    ordered = restore_order(cleaned)
    kept = [row for row in ordered if row["kept"]]
    drops = Counter(row["frozen_reason"] for row in ordered if not row["kept"])
    kept_counts = Counter(row["medium"] for row in kept)
    mismatches = {
        "membership": int(len(ordered) != EXPECTED_RAW_ROWS),
        "drop_counts": int(dict(drops) != EXPECTED_DROPS),
        "kept_counts": int(dict(kept_counts) != EXPECTED_KEPT),
        "ordering": 0,
        "content": 0,
        "identity": 0,
    }
    grouped = kept_by_file(ordered)
    for filename, rows in grouped.items():
        frozen = _read_jsonl(FROZEN_CLEANED / filename)
        if len(rows) != len(frozen):
            mismatches["membership"] += abs(len(rows) - len(frozen))
            continue
        for actual, expected in zip(rows, frozen):
            if list(actual) != list(expected):
                mismatches["ordering"] += 1
            if actual != expected:
                mismatches["content"] += 1
    manifest = ROOT / "artifacts" / "refactor" / "provenance" / "cleaned_manifest.preview.jsonl"
    if not manifest.is_file():
        mismatches["identity"] += 1
    else:
        by_file: dict[str, list[dict]] = {spec[0]: [] for spec in SOURCE_SPECS}
        with manifest.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                if row.get("status") == "kept":
                    by_file[row["source_file"]].append(row)
        kept_rows = [row for row in ordered if row["kept"]]
        produced: dict[str, list[dict]] = {spec[0]: [] for spec in SOURCE_SPECS}
        for row in kept_rows:
            produced[row["source_file"]].append(row)
        for filename, rows in produced.items():
            expected_rows = by_file[filename]
            if len(rows) != len(expected_rows):
                mismatches["identity"] += abs(len(rows) - len(expected_rows))
                continue
            for actual, expected in zip(rows, expected_rows):
                if actual["source_id"] != expected["source_id"] or actual["raw_record_id"] != expected["raw_record_id"]:
                    mismatches["identity"] += 1
    mismatches["details_drops"] = dict(drops)
    mismatches["details_kept"] = dict(kept_counts)
    return mismatches


def build_nopromo_rows(grouped: dict[str, list[dict]]) -> list[dict]:
    """Same chunk, word-floor, and promo rules as the frozen nopromo file."""
    rows = []
    for filename, (source, text_key, title_key) in SOURCES.items():
        floor = MIN_WORDS.get(source, DEFAULT_MIN_WORDS)
        for line_no, record in enumerate(grouped[filename], 1):
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
                        "input": "",
                        "output": chunk,
                        "source": source,
                        "source_file": filename,
                        "source_line": line_no,
                    }
                )
            for piece in built:
                if _is_direct_event_announcement(piece):
                    continue
                cleaned_output, changed = _strip_trailing_cta(str(piece["output"] or ""))
                if changed:
                    piece["output"] = cleaned_output
                    if len(cleaned_output.split()) < 20:
                        continue
                rows.append(piece)
    return rows


def compare_nopromo(rows: list[dict]) -> dict:
    frozen = _read_jsonl(FROZEN_NOPROMO)
    mismatches = {
        "membership": int(len(rows) != len(frozen) or len(rows) != EXPECTED_NOPROMO_ROWS),
        "ordering": 0,
        "content": 0,
    }
    if len(rows) == len(frozen):
        for actual, expected in zip(rows, frozen):
            if list(actual) != list(expected):
                mismatches["ordering"] += 1
            if actual != expected:
                mismatches["content"] += 1
            elif (
                actual.get("instruction"),
                actual.get("output"),
                actual.get("source"),
                actual.get("source_file"),
                actual.get("source_line"),
            ) != (
                expected.get("instruction"),
                expected.get("output"),
                expected.get("source"),
                expected.get("source_file"),
                expected.get("source_line"),
            ):
                mismatches["content"] += 1
    return mismatches


def compare_fit(parts, records: list[dict], encoded: bytes) -> dict:
    frozen = _read_jsonl(FROZEN_FIT)
    digest = hashlib.sha256(encoded).hexdigest()
    mismatches = {
        "membership": int(len(records) != len(frozen) or len(records) != EXPECTED_FIT_ROWS),
        "ordering": 0,
        "content": 0,
        "bytes": int(digest != EXPECTED_FIT_SHA256 or digest != file_sha256(FROZEN_FIT)),
        "lineage": 0,
    }
    if len(records) == len(frozen):
        for actual, expected in zip(records, frozen):
            if list(actual) != list(expected):
                mismatches["ordering"] += 1
            if actual != expected:
                mismatches["content"] += 1
    if LINEAGE.is_file() and len(parts) == EXPECTED_FIT_ROWS:
        with LINEAGE.open(encoding="utf-8") as handle:
            for part, line in zip(parts, handle):
                sidecar = json.loads(line)
                if list(part.parent_indices) != sidecar["parent_sft_row_indices"]:
                    mismatches["lineage"] += 1
                elif part.fit_chunk_ordinal != sidecar["fit_chunk_ordinal"]:
                    mismatches["lineage"] += 1
                elif part.relationship_type != sidecar["relationship_type"]:
                    mismatches["lineage"] += 1
    elif not LINEAGE.is_file():
        mismatches["lineage"] += 1
    mismatches["fit_sha256"] = digest
    return mismatches


def _input_hash(items_or_rows) -> str:
    payload = json.dumps(items_or_rows, ensure_ascii=False, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def stage_validate(staging: Path, run_id: str, data_dir: Path | None = None) -> dict:
    hashes = assert_raw_hashes(data_dir)
    rows = iter_raw_records(data_dir)
    mismatches = {"raw_rows": int(len(rows) != EXPECTED_RAW_ROWS)}
    status = "FAIL" if mismatches["raw_rows"] else "PASS"
    write_manifest(
        staging,
        "validate_raw",
        {
            "run_id": run_id,
            "input_rows": len(rows),
            "output_rows": len(rows),
            "input_sha256": hashes["jasonlemkin_blog.jsonl"],
            "output_sha256": hashlib.sha256("".join(f"{name}:{digest}" for name, digest in hashes.items()).encode()).hexdigest(),
            "status": status,
            "mismatches": mismatches,
            "details": {"file_sha256": hashes},
        },
    )
    gate("validate_raw", mismatches)
    return {"rows": rows, "hashes": hashes}


def stage_clean(staging: Path, run_id: str, raw_rows: list[dict], executor: str = "ordered") -> list[dict]:
    if executor == "spark":
        cleaned = clean_with_spark(raw_rows)
    else:
        cleaned = clean_ordered(raw_rows)
    mismatches = compare_cleaned(cleaned)
    comparable = {key: mismatches[key] for key in ("membership", "drop_counts", "kept_counts", "ordering", "content", "identity")}
    status = "FAIL" if any(comparable.values()) else "PASS"
    grouped = kept_by_file(cleaned)
    digest = hashlib.sha256()
    for filename, rows in grouped.items():
        payload = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows).encode("utf-8")
        path = staging / "cleaned" / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        assert_staging_dir(path)
        path.write_bytes(payload)
        digest.update(payload)
    output_hash = digest.hexdigest()
    write_manifest(
        staging,
        "canonical_clean",
        {
            "run_id": run_id,
            "input_rows": len(raw_rows),
            "output_rows": sum(len(rows) for rows in grouped.values()),
            "input_sha256": _input_hash([row["raw_record_index"] for row in raw_rows]),
            "output_sha256": output_hash,
            "status": status,
            "mismatches": comparable,
            "details": {
                "executor": executor,
                "ordering": "source_ordinal, raw_record_index",
                "drops": mismatches["details_drops"],
                "kept": mismatches["details_kept"],
            },
        },
    )
    gate("canonical_clean", comparable)
    return cleaned


def stage_nopromo(staging: Path, run_id: str, cleaned: list[dict]) -> list[dict]:
    rows = build_nopromo_rows(kept_by_file(cleaned))
    mismatches = compare_nopromo(rows)
    line_ids = {}
    counts: Counter[str] = Counter()
    for row in restore_order(cleaned):
        if not row["kept"] or not row["raw_record_id"]:
            continue
        counts[row["source_file"]] += 1
        line_ids[(row["source_file"], counts[row["source_file"]])] = row["raw_record_id"]
    mismatches["lineage"] = sum((row["source_file"], row["source_line"]) not in line_ids for row in rows)
    status = "FAIL" if any(mismatches.values()) else "PASS"
    output_hash = _write_jsonl(staging / "nopromo" / "rows.jsonl", rows)
    parents = [row for row in cleaned if row["kept"] and row["raw_record_id"]]
    write_manifest(
        staging,
        "build_nopromo",
        {
            "run_id": run_id,
            "input_rows": sum(1 for row in cleaned if row["kept"]),
            "output_rows": len(rows),
            "input_sha256": file_sha256(FROZEN_CLEANED / "jasonlemkin_blog.jsonl"),
            "output_sha256": output_hash,
            "status": status,
            "mismatches": mismatches,
            "details": {
                "kept_rows_with_raw_record_id": len(parents),
                "nopromo_rows_missing_source_line": sum(row.get("source_line") in (None, "") for row in rows),
            },
        },
    )
    gate("build_nopromo", mismatches)
    return rows


def stage_fit(staging: Path, run_id: str, nopromo_rows: list[dict], tokenizer=None) -> dict:
    tokenizer = tokenizer or load_compatibility_tokenizer()
    result = fit_rows(nopromo_rows, tokenizer, FitConfig())
    records = [part.record for part in result.parts]
    encoded = serialize_fit_records(records)
    mismatches = compare_fit(result.parts, records, encoded)
    comparable = {key: mismatches[key] for key in ("membership", "ordering", "content", "bytes", "lineage")}
    status = "FAIL" if any(comparable.values()) else "PASS"
    fit_path = staging / "fit" / "rows.jsonl"
    fit_path.parent.mkdir(parents=True, exist_ok=True)
    assert_staging_dir(fit_path)
    fit_path.write_bytes(encoded)
    write_manifest(
        staging,
        "canonical_fit",
        {
            "run_id": run_id,
            "input_rows": len(nopromo_rows),
            "output_rows": len(records),
            "input_sha256": EXPECTED_NOPROMO_SHA256,
            "output_sha256": mismatches["fit_sha256"],
            "status": status,
            "mismatches": comparable,
            "details": {
                "execution": "single ordered canonical_fit.fit_rows",
                "lead_ins_consumed": result.lead_ins_consumed,
                "merged_rows": result.merged_rows,
                "tokenizer_revision": FitConfig().compatibility_revision,
                "historical_revision": FitConfig().historical_revision,
            },
        },
    )
    gate("canonical_fit", comparable)
    return {"result": result, "sha256": mismatches["fit_sha256"]}


def stage_summary(staging: Path, run_id: str) -> dict:
    names = ("validate_raw", "canonical_clean", "build_nopromo", "canonical_fit")
    documents = []
    mismatches = {}
    for name in names:
        path = staging / "manifests" / f"{name}.json"
        if not path.is_file():
            mismatches[name] = 1
            continue
        document = json.loads(path.read_text(encoding="utf-8"))
        documents.append(document)
        mismatches[name] = int(document.get("status") != "PASS")
    status = "FAIL" if any(mismatches.values()) else "PASS"
    write_manifest(
        staging,
        "parity_summary",
        {
            "run_id": run_id,
            "input_rows": EXPECTED_RAW_ROWS,
            "output_rows": EXPECTED_FIT_ROWS,
            "input_sha256": RAW_SHA256["jasonlemkin_blog.jsonl"],
            "output_sha256": EXPECTED_FIT_SHA256,
            "status": status,
            "mismatches": mismatches,
            "details": {"stages": [item["stage"] for item in documents]},
        },
    )
    gate("parity_summary", mismatches)
    return {"status": status, "stages": documents}


def run_shadow(
    staging: Path,
    run_id: str,
    *,
    through: str = "fit",
    executor: str = "ordered",
    tokenizer=None,
) -> dict:
    """Run the shadow stages in order. A parity failure stops the run."""
    staging = assert_staging_dir(staging)
    staging.mkdir(parents=True, exist_ok=True)
    validated = stage_validate(staging, run_id)
    if through == "validate_raw":
        return validated
    cleaned = stage_clean(staging, run_id, validated["rows"], executor=executor)
    if through == "canonical_clean":
        return {"cleaned": len(cleaned)}
    nopromo = stage_nopromo(staging, run_id, cleaned)
    if through == "build_nopromo":
        return {"nopromo": len(nopromo)}
    fitted = stage_fit(staging, run_id, nopromo, tokenizer=tokenizer)
    if through == "canonical_fit":
        return fitted
    return stage_summary(staging, run_id)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Run the v2 shadow pipeline through fit.")
    parser.add_argument("--staging", type=Path, default=None)
    parser.add_argument("--run-id", default="local")
    parser.add_argument(
        "--through",
        choices=("validate_raw", "canonical_clean", "build_nopromo", "canonical_fit", "parity_summary"),
        default="parity_summary",
    )
    parser.add_argument("--executor", choices=("ordered", "spark"), default="ordered")
    args = parser.parse_args()
    staging = args.staging or (STAGING_ROOT / args.run_id)
    result = run_shadow(staging, args.run_id, through=args.through, executor=args.executor)
    print(json.dumps({"status": "PASS", "staging": str(staging), "through": args.through, "result_keys": list(result)}))


if __name__ == "__main__":
    main()
