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
import inspect
import json
import os
import subprocess
import sys
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
from host_finetune.llama_chat_format import training_text_from_row
from host_finetune.queue_followup_trains import select_balanced_rows
from host_finetune.relabel_continuations import EXPECTED_TRAIN_GROUPS, transform_rows
from host_finetune.sft_chunk_utils import instruction_for, split_long_document
from host_finetune.split_groups import assign_groups
from host_finetune.train_only_index import (
    DEFAULT_GOLD_DISPOSITIONS,
    DEFAULT_GOLD_EVAL,
    DEFAULT_PERSIST_DIR,
    TRAIN_ONLY_COLLECTION,
    assignment_rows,
    blocked_group_ids,
    headline_overlap_groups,
    load_jsonl as load_index_jsonl,
    select_train_documents,
)
from host_finetune.verify_frozen_parity import explicit_crlf_jsonl_bytes

ROOT = Path(__file__).resolve().parents[1]
CONFIG_VERSION = "spark-v2-shadow-embed-v1"
ORDERED_EXECUTOR = "ordered-python"
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
FROZEN_BALANCED = ROOT / "host_finetune" / "data" / "dataset_fit512_keepbreaks_balanced.jsonl"
EXP008_ASSIGNMENTS = (
    ROOT / "experiments" / "EXP-20261001-008-repaired-balanced-split" / "raw" / "split_assignments.jsonl"
)
RELABEL_DATASET = (
    ROOT / "experiments" / "EXP-20261003-004-relabel-continuations" / "raw" / "dataset.jsonl"
)
RELABEL_ASSIGNMENTS = (
    ROOT / "experiments" / "EXP-20261003-004-relabel-continuations" / "raw" / "split_assignments.jsonl"
)
LINEAGE = ROOT / "artifacts" / "refactor" / "provenance" / "fit_lineage_exact.preview.jsonl"
EXPECTED_BALANCED_ROWS = 31328
EXPECTED_BALANCED_SHA256 = "6e99b414733eafdfa27a5e94dd26beff1498bbfd58af132251e24044fef4f057"
EXPECTED_BALANCE_SOURCES = {"blog": 14315, "x": 14316, "youtube_saastr": 0}
EXPECTED_SPLIT_COUNTS = {"train": 25062, "validation": 3133, "test": 3133}
EXPECTED_RELABEL_ROWS = 26545
EXPECTED_RELABEL_SHA256 = "1231835ecc5dcb608fefa0357d325affb2afbdc3aeade2a63b1dc42b2920897b"
EXPECTED_RELABEL_SPLITS = {"train": 21377, "validation": 2588, "test": 2580}
EXPECTED_RAG_ROWS = 21193
EXPECTED_RAG_EXCLUSIONS = {"gold_overlap_family": 183, "headline": 1}
EXPECTED_RAG_ROLES = {"unchanged": 11611, "opening": 1687, "continuation": 7895}
HELD_OUT_RAG_ID = "train_26148"
SUMMARY_STAGES = (
    "validate_raw",
    "canonical_clean",
    "clean_parity",
    "build_nopromo",
    "nopromo_parity",
    "canonical_fit",
    "fit_parity",
    "balance",
    "balance_parity",
    "split",
    "split_parity",
    "relabel",
    "relabel_parity",
    "rag_governance",
    "rag_parity",
    "embed_rag",
    "embedding_parity",
    "build_chroma_shadow",
    "logical_chroma_parity",
    "retrieval_parity",
)
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
    """Read HEAD without writing git config. ``safe.directory`` is a one-shot flag."""
    def _git(*args: str) -> str:
        return subprocess.check_output(
            [
                "git",
                "-c",
                "safe.directory=*",
                "-c",
                "filter.lfs.required=false",
                "-c",
                "filter.lfs.process=",
                "-c",
                "filter.lfs.smudge=",
                "-c",
                "filter.lfs.clean=",
                *args,
            ],
            cwd=ROOT,
            text=True,
        )

    sha = _git("rev-parse", "HEAD").strip()
    dirty = bool(_git("status", "--porcelain").strip())
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
        "executor": body.get("executor") or ORDERED_EXECUTOR,
        "input_artifact": body.get("input_artifact"),
        "output_artifact": body.get("output_artifact"),
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


def _require_spark():
    """Import PySpark or fail. There is no in-process fallback."""
    try:
        import pyspark
        from pyspark.sql import SparkSession
    except ImportError as exc:
        raise ParityError("spark executor requested but pyspark is not installed") from exc
    return pyspark, SparkSession


def _ensure_worker_pythonpath() -> str:
    """Spark workers do not inherit a parent ``sys.path`` edit. They read PYTHONPATH."""
    root = str(ROOT)
    parts = [part for part in os.environ.get("PYTHONPATH", "").split(os.pathsep) if part]
    if root not in parts:
        parts.insert(0, root)
    path = os.pathsep.join(parts)
    os.environ["PYTHONPATH"] = path
    return path


def _spark_session(SparkSession):
    pythonpath = _ensure_worker_pythonpath()
    return (
        SparkSession.builder.master("local[*]")
        .appName("lemkin-v2-shadow-clean")
        .config("spark.ui.enabled", "false")
        .config("spark.executorEnv.PYTHONPATH", pythonpath)
        .config("spark.pyspark.python", sys.executable)
        .config("spark.pyspark.driver.python", sys.executable)
        .getOrCreate()
    )


def _clean_on_spark(spark, pyspark_module, items: list[dict], slices: int) -> tuple[list[dict], dict]:
    """Run ``clean_partition`` as Spark tasks, then restore source order."""
    if slices < 1:
        raise ParityError(f"spark partition count must be positive, got {slices}")
    context = spark.sparkContext
    rdd = context.parallelize(items, numSlices=slices)
    actual = rdd.getNumPartitions()
    if actual != slices:
        raise ParityError(f"spark produced {actual} partitions, requested {slices}")
    partitions_ran = context.accumulator(0)

    def _partition(iterator):
        partitions_ran.add(1)
        return clean_partition(list(iterator))

    cleaned = rdd.mapPartitions(_partition).collect()
    if partitions_ran.value != slices:
        raise ParityError(
            f"spark worker partitions ran {partitions_ran.value}, requested {slices}"
        )
    info = {
        "executor": "spark",
        "fallback": False,
        "spark_version": spark.version,
        "pyspark_version": pyspark_module.__version__,
        "master": context.master,
        "requested_partitions": slices,
        "actual_partitions": actual,
        "worker_partitions_ran": partitions_ran.value,
        "default_parallelism": context.defaultParallelism,
        "python": sys.executable,
        "pythonpath": os.environ.get("PYTHONPATH", ""),
        "canonical_module": inspect.getfile(clean_record),
        "worker": "mapPartitions(clean_partition)",
    }
    return restore_order(list(cleaned)), info


def clean_with_spark(items: list[dict], slices: int = 4) -> tuple[list[dict], dict]:
    """Distribute ``clean_partition``, then restore source order on the driver."""
    pyspark_module, SparkSession = _require_spark()
    spark = _spark_session(SparkSession)
    try:
        return _clean_on_spark(spark, pyspark_module, items, slices)
    finally:
        spark.stop()


def clean_signature(rows: list[dict]) -> list[tuple]:
    """Semantic identity of a cleaned sequence, independent of partition arrival."""
    signature = []
    for row in rows:
        record = row["cleaned_record"]
        signature.append(
            (
                row["source_ordinal"],
                row["raw_record_index"],
                row["kept"],
                row["frozen_reason"],
                row["source_id"],
                row["raw_record_id"],
                json.dumps(record, ensure_ascii=False, sort_keys=True) if record is not None else None,
            )
        )
    return signature


def prove_spark_partitions(raw_rows: list[dict], counts: list[int]) -> dict:
    """Every partition count must match the ordered cleaner and the frozen oracle."""
    baseline = clean_ordered(raw_rows)
    baseline_signature = clean_signature(baseline)
    baseline_mismatches = _comparable_clean(compare_cleaned(baseline))
    gate("spark_partition_baseline", baseline_mismatches)
    pyspark_module, SparkSession = _require_spark()
    spark = _spark_session(SparkSession)
    reports = []
    try:
        for count in counts:
            cleaned, info = _clean_on_spark(spark, pyspark_module, raw_rows, count)
            mismatches = _comparable_clean(compare_cleaned(cleaned))
            signature = clean_signature(cleaned)
            versus_ordered = sum(
                left != right for left, right in zip(signature, baseline_signature)
            ) + abs(len(signature) - len(baseline_signature))
            mismatches["versus_ordered"] = versus_ordered
            gate(f"spark_partitions_{count}", mismatches)
            info["versus_ordered"] = versus_ordered
            info["oracle_mismatches"] = mismatches
            reports.append(info)
    finally:
        spark.stop()
    return {
        "fallback": False,
        "partition_counts": counts,
        "reports": reports,
        "identical_across_partitions": True,
    }


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


def _comparable_clean(mismatches: dict) -> dict:
    return {
        key: mismatches[key]
        for key in ("membership", "drop_counts", "kept_counts", "ordering", "content", "identity")
    }


def stage_clean(
    staging: Path,
    run_id: str,
    raw_rows: list[dict],
    executor: str = "ordered",
    slices: int = 4,
) -> list[dict]:
    if executor not in {"ordered", "spark"}:
        raise ParityError(f"unknown executor: {executor}")
    spark_info = None
    if executor == "spark":
        cleaned, spark_info = clean_with_spark(raw_rows, slices=slices)
    else:
        cleaned = clean_ordered(raw_rows)
    mismatches = compare_cleaned(cleaned)
    comparable = _comparable_clean(mismatches)
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
    indexed_hash = _write_jsonl(staging / "cleaned" / "indexed.jsonl", cleaned)
    output_hash = digest.hexdigest()
    write_manifest(
        staging,
        "canonical_clean",
        {
            "run_id": run_id,
            "executor": executor,
            "input_artifact": "data/*.jsonl",
            "output_artifact": "cleaned/indexed.jsonl",
            "input_rows": len(raw_rows),
            "output_rows": sum(len(rows) for rows in grouped.values()),
            "input_sha256": _input_hash([row["raw_record_index"] for row in raw_rows]),
            "output_sha256": output_hash,
            "indexed_sha256": indexed_hash,
            "status": status,
            "mismatches": comparable,
            "details": {
                "executor": executor,
                "fallback": False,
                "ordering": "source_ordinal, raw_record_index",
                "drops": mismatches["details_drops"],
                "kept": mismatches["details_kept"],
                "spark": spark_info,
                "indexed_sha256": indexed_hash,
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
    return {"result": result, "sha256": mismatches["fit_sha256"], "records": records}


def load_cleaned(staging: Path) -> list[dict]:
    return restore_order(_read_jsonl(staging / "cleaned" / "indexed.jsonl"))


def load_nopromo(staging: Path) -> list[dict]:
    return _read_jsonl(staging / "nopromo" / "rows.jsonl")


def load_fit(staging: Path) -> list[dict]:
    return _read_jsonl(staging / "fit" / "rows.jsonl")


def load_balanced(staging: Path) -> list[dict]:
    return _read_jsonl(staging / "balance" / "rows.jsonl")


def load_split(staging: Path) -> list[dict]:
    return assignment_rows(_read_jsonl(staging / "split" / "assignments.jsonl"))


def load_rag(staging: Path) -> list[dict]:
    return _read_jsonl(staging / "rag" / "rows.jsonl")


def load_relabel(staging: Path) -> tuple[list[dict], list[dict]]:
    return (
        _read_jsonl(staging / "relabel" / "dataset.jsonl"),
        assignment_rows(_read_jsonl(staging / "relabel" / "assignments.jsonl")),
    )


def stage_recorded_parity(staging: Path, run_id: str, source_stage: str, gate_name: str) -> dict:
    """Fail the run when the recorded stage manifest is missing or not PASS."""
    path = staging / "manifests" / f"{source_stage}.json"
    mismatches: dict[str, int] = {"missing_manifest": 0, "status": 0}
    if not path.is_file():
        mismatches["missing_manifest"] = 1
    else:
        document = json.loads(path.read_text(encoding="utf-8"))
        mismatches["status"] = int(document.get("status") != "PASS")
        for key, value in (document.get("mismatches") or {}).items():
            if value:
                mismatches[f"{source_stage}_{key}"] = int(value)
    status = "FAIL" if any(mismatches.values()) else "PASS"
    write_manifest(
        staging,
        gate_name,
        {
            "run_id": run_id,
            "executor": ORDERED_EXECUTOR,
            "input_artifact": f"manifests/{source_stage}.json",
            "output_artifact": f"manifests/{gate_name}.json",
            "input_rows": None,
            "output_rows": None,
            "input_sha256": file_sha256(path) if path.is_file() else None,
            "output_sha256": None,
            "status": status,
            "mismatches": mismatches,
            "details": {"source_stage": source_stage},
        },
    )
    gate(gate_name, mismatches)
    return {"status": status}


def _lf_jsonl_bytes(rows: list[dict]) -> bytes:
    return "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows).encode("utf-8")


def stage_balance(staging: Path, run_id: str, fit_rows: list[dict]) -> list[dict]:
    """Drop SaaStr and cap X with the proven selector. Bytes are explicit CRLF."""
    kept, stats = select_balanced_rows(fit_rows)
    encoded = explicit_crlf_jsonl_bytes(kept)
    digest = hashlib.sha256(encoded).hexdigest()
    frozen = _read_jsonl(FROZEN_BALANCED)
    source_counts = Counter(row.get("source") for row in kept)
    mismatches = {
        "membership": int(len(kept) != EXPECTED_BALANCED_ROWS or len(kept) != len(frozen)),
        "ordering": 0,
        "content": 0,
        "bytes": int(digest != EXPECTED_BALANCED_SHA256 or digest != file_sha256(FROZEN_BALANCED)),
        "sources": int(
            source_counts["blog"] != EXPECTED_BALANCE_SOURCES["blog"]
            or source_counts["x"] != EXPECTED_BALANCE_SOURCES["x"]
            or source_counts["youtube_saastr"] != 0
        ),
    }
    if len(kept) == len(frozen):
        for actual, expected in zip(kept, frozen):
            if list(actual) != list(expected):
                mismatches["ordering"] += 1
            if actual != expected:
                mismatches["content"] += 1
    status = "FAIL" if any(mismatches.values()) else "PASS"
    path = staging / "balance" / "rows.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    assert_staging_dir(path)
    path.write_bytes(encoded)
    write_manifest(
        staging,
        "balance",
        {
            "run_id": run_id,
            "executor": ORDERED_EXECUTOR,
            "input_artifact": "fit/rows.jsonl",
            "output_artifact": "balance/rows.jsonl",
            "input_rows": len(fit_rows),
            "output_rows": len(kept),
            "input_sha256": EXPECTED_FIT_SHA256,
            "output_sha256": digest,
            "status": status,
            "mismatches": mismatches,
            "details": {"policy": "select_balanced_rows", "stats": stats, "sources": dict(source_counts)},
        },
    )
    gate("balance", mismatches)
    return kept


def stage_split(staging: Path, run_id: str, balanced_rows: list[dict]) -> list[dict]:
    """Assign groups with the frozen splitter and compare to EXP-008."""
    assignments, pairs = assign_groups(balanced_rows)
    frozen = assignment_rows(load_index_jsonl(EXP008_ASSIGNMENTS))
    split_counts = Counter(row["split"] for row in assignments)
    groups: dict[str, set[str]] = {}
    for row in assignments:
        groups.setdefault(row["group_id"], set()).add(row["split"])
    crossing = sum(len(names) > 1 for names in groups.values())
    mismatches = {
        "membership": int(len(assignments) != len(frozen) or len(assignments) != EXPECTED_BALANCED_ROWS),
        "ordering": 0,
        "ownership": 0,
        "counts": int(dict(split_counts) != EXPECTED_SPLIT_COUNTS),
        "crossing_groups": crossing,
    }
    if len(assignments) == len(frozen):
        for actual, expected in zip(assignments, frozen):
            if actual.get("row_index") != expected.get("row_index"):
                mismatches["ordering"] += 1
            same_owner = (
                actual.get("group_id") == expected.get("group_id")
                and actual.get("split") == expected.get("split")
                and str(actual.get("source_file") or "") == str(expected.get("source_file") or "")
                and str(actual.get("source_line") or "") == str(expected.get("source_line") or "")
            )
            if not same_owner:
                mismatches["ownership"] += 1
    status = "FAIL" if any(mismatches.values()) else "PASS"
    payload_rows = [{"record_type": "meta", "n_rows": len(assignments)}, *assignments]
    output_hash = _write_jsonl(staging / "split" / "assignments.jsonl", payload_rows)
    write_manifest(
        staging,
        "split",
        {
            "run_id": run_id,
            "executor": ORDERED_EXECUTOR,
            "input_artifact": "balance/rows.jsonl",
            "output_artifact": "split/assignments.jsonl",
            "input_rows": len(balanced_rows),
            "output_rows": len(assignments),
            "input_sha256": EXPECTED_BALANCED_SHA256,
            "output_sha256": output_hash,
            "status": status,
            "mismatches": mismatches,
            "details": {
                "policy": "split_groups.assign_groups",
                "split_counts": dict(split_counts),
                "groups": len(groups),
                "near_duplicate_pairs": len(pairs),
                "oracle": str(EXP008_ASSIGNMENTS.relative_to(ROOT)),
            },
        },
    )
    gate("split", mismatches)
    return assignments


def _relabel_token_counter(tokenizer):
    def n_tokens(instruction: str, input_text: str, output: str) -> int:
        text = training_text_from_row(tokenizer, instruction, input_text, output)
        return len(tokenizer(text, add_special_tokens=False)["input_ids"])

    return n_tokens


def stage_relabel(
    staging: Path,
    run_id: str,
    balanced_rows: list[dict],
    assignments: list[dict],
    tokenizer=None,
) -> tuple[list[dict], list[dict]]:
    """Relabel with the proven continuation rules. The tokenizer is local only."""
    tokenizer = tokenizer or load_compatibility_tokenizer()
    rows, new_assignments, dispositions, stats = transform_rows(
        balanced_rows,
        assignments,
        _relabel_token_counter(tokenizer),
        512,
    )
    if stats["train_groups"] != EXPECTED_TRAIN_GROUPS:
        raise ParityError(f"relabel train groups changed: {stats['train_groups']}")
    encoded = _lf_jsonl_bytes(rows)
    digest = hashlib.sha256(encoded).hexdigest()
    frozen_rows = _read_jsonl(RELABEL_DATASET)
    frozen_assignments = assignment_rows(load_index_jsonl(RELABEL_ASSIGNMENTS))
    split_counts = Counter(row["split"] for row in new_assignments)
    mismatches = {
        "membership": int(len(rows) != EXPECTED_RELABEL_ROWS or len(rows) != len(frozen_rows)),
        "ordering": 0,
        "content": 0,
        "bytes": int(digest != EXPECTED_RELABEL_SHA256 or digest != file_sha256(RELABEL_DATASET)),
        "ownership": 0,
        "splits": int(dict(split_counts) != EXPECTED_RELABEL_SPLITS),
    }
    if len(rows) == len(frozen_rows):
        for actual, expected in zip(rows, frozen_rows):
            if list(actual) != list(expected):
                mismatches["ordering"] += 1
            if actual != expected:
                mismatches["content"] += 1
    if len(new_assignments) == len(frozen_assignments):
        for actual, expected in zip(new_assignments, frozen_assignments):
            if (
                actual.get("group_id") != expected.get("group_id")
                or actual.get("split") != expected.get("split")
                or actual.get("sft_role") != expected.get("sft_role")
                or actual.get("row_index") != expected.get("row_index")
            ):
                mismatches["ownership"] += 1
    else:
        mismatches["ownership"] += abs(len(new_assignments) - len(frozen_assignments))
    status = "FAIL" if any(mismatches.values()) else "PASS"
    dataset_path = staging / "relabel" / "dataset.jsonl"
    dataset_path.parent.mkdir(parents=True, exist_ok=True)
    assert_staging_dir(dataset_path)
    dataset_path.write_bytes(encoded)
    _write_jsonl(staging / "relabel" / "assignments.jsonl", new_assignments)
    _write_jsonl(staging / "relabel" / "dispositions.jsonl", dispositions)
    write_manifest(
        staging,
        "relabel",
        {
            "run_id": run_id,
            "executor": ORDERED_EXECUTOR,
            "input_artifact": "balance/rows.jsonl",
            "output_artifact": "relabel/dataset.jsonl",
            "input_rows": len(balanced_rows),
            "output_rows": len(rows),
            "input_sha256": EXPECTED_BALANCED_SHA256,
            "output_sha256": digest,
            "status": status,
            "mismatches": mismatches,
            "details": {
                "policy": "relabel_continuations.transform_rows",
                "tokenizer_revision": FitConfig().compatibility_revision,
                "local_files_only": True,
                "split_counts": dict(split_counts),
                "train_groups": stats["train_groups"],
            },
        },
    )
    gate("relabel", mismatches)
    return rows, new_assignments


def _read_persisted_rag(db_path: Path) -> dict[str, dict]:
    """Read lemkin_train_only through SQLite. No client, no upsert, no embed."""
    import sqlite3

    if not db_path.is_file():
        raise ParityError(f"persisted chroma database missing: {db_path}")
    uri = "file:" + db_path.resolve().as_posix() + "?mode=ro"
    connection = sqlite3.connect(uri, uri=True)
    try:
        names = [row[0] for row in connection.execute("SELECT name FROM collections")]
        if names != [TRAIN_ONLY_COLLECTION]:
            raise ParityError(f"unexpected persisted collections: {names}")
        rows: dict[str, dict] = {}
        query = """
            SELECT e.embedding_id, m.key, m.string_value, m.int_value
            FROM embeddings e
            JOIN embedding_metadata m ON m.id = e.id
        """
        for embedding_id, key, string_value, int_value in connection.execute(query):
            slot = rows.setdefault(embedding_id, {})
            if key == "chroma:document":
                slot["document"] = string_value or ""
            elif key == "row_id":
                slot["row_id"] = int_value
            elif key in {"group_id", "medium", "split"}:
                slot[key] = string_value
        return rows
    finally:
        connection.close()


def stage_rag(staging: Path, run_id: str, dataset: list[dict], assignments: list[dict]) -> list[dict]:
    """Apply the proven train-only selectors. Do not embed or open a Chroma client."""
    dispositions = load_index_jsonl(DEFAULT_GOLD_DISPOSITIONS)
    gold_rows = load_index_jsonl(DEFAULT_GOLD_EVAL)
    family = blocked_group_ids(assignments, dispositions)
    family_docs, family_stats = select_train_documents(dataset, assignments, family)
    headline, _details = headline_overlap_groups(dataset, assignments, gold_rows, family)
    final_docs, final_stats = select_train_documents(dataset, assignments, family | headline)
    eligible = {doc["id"] for doc in final_docs}
    family_ids = {doc["id"] for doc in family_docs}
    headline_removed = sorted(family_ids - eligible)
    by_index = {row["row_index"]: row for row in assignment_rows(assignments)}
    artifacts = []
    roles: Counter[str] = Counter()
    for doc in final_docs:
        meta = doc["metadata"]
        assignment = by_index[int(meta["row_id"])]
        roles[str(assignment.get("sft_role") or "")] += 1
        artifacts.append(
            {
                "row_id": doc["id"],
                "document": doc["text"],
                "group_id": meta["group_id"],
                "medium": meta["medium"],
                "split": meta["split"],
                "sft_role": assignment.get("sft_role"),
                "source_file": assignment.get("source_file"),
                "source_line": assignment.get("source_line"),
                "prior_row_index": assignment.get("prior_row_index"),
            }
        )
    persisted = _read_persisted_rag(DEFAULT_PERSIST_DIR / "chroma.sqlite3")
    produced = {row["row_id"]: row for row in artifacts}
    mismatches = {
        "eligible": int(len(artifacts) != EXPECTED_RAG_ROWS),
        "candidates": int(family_stats["train_rows"] != EXPECTED_RELABEL_SPLITS["train"]),
        "gold_exclusions": int(family_stats["excluded_gold"] != EXPECTED_RAG_EXCLUSIONS["gold_overlap_family"]),
        "headline_exclusions": int(
            final_stats["excluded_gold"] - family_stats["excluded_gold"] != EXPECTED_RAG_EXCLUSIONS["headline"]
        ),
        "held_out": int(HELD_OUT_RAG_ID in eligible or HELD_OUT_RAG_ID not in headline_removed),
        "roles": int({name: roles.get(name, 0) for name in EXPECTED_RAG_ROLES} != EXPECTED_RAG_ROLES),
        "missing": len(set(persisted) - set(produced)),
        "unexpected": len(set(produced) - set(persisted)),
        "document": 0,
        "group_id": 0,
        "medium": 0,
        "split": 0,
    }
    for row_id, actual in produced.items():
        expected = persisted.get(row_id)
        if expected is None:
            continue
        if actual["document"] != expected.get("document"):
            mismatches["document"] += 1
        if actual["group_id"] != expected.get("group_id"):
            mismatches["group_id"] += 1
        if actual["medium"] != expected.get("medium"):
            mismatches["medium"] += 1
        if actual["split"] != expected.get("split"):
            mismatches["split"] += 1
    status = "FAIL" if any(mismatches.values()) else "PASS"
    output_hash = _write_jsonl(staging / "rag" / "rows.jsonl", artifacts)
    write_manifest(
        staging,
        "rag_governance",
        {
            "run_id": run_id,
            "executor": ORDERED_EXECUTOR,
            "input_artifact": "relabel/dataset.jsonl",
            "output_artifact": "rag/rows.jsonl",
            "input_rows": len(dataset),
            "output_rows": len(artifacts),
            "input_sha256": EXPECTED_RELABEL_SHA256,
            "output_sha256": output_hash,
            "status": status,
            "mismatches": mismatches,
            "details": {
                "policy": "blocked_group_ids + headline_overlap_groups + select_train_documents",
                "embedded": False,
                "chroma_client": False,
                "candidates": family_stats["train_rows"],
                "gold_exclusions": family_stats["excluded_gold"],
                "headline_exclusions": final_stats["excluded_gold"] - family_stats["excluded_gold"],
                "headline_removed_ids": headline_removed,
                "roles": {name: roles.get(name, 0) for name in EXPECTED_RAG_ROLES},
                "held_out": HELD_OUT_RAG_ID,
            },
        },
    )
    gate("rag_governance", mismatches)
    return artifacts


def stage_summary(staging: Path, run_id: str) -> dict:
    names = SUMMARY_STAGES
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
            "executor": ORDERED_EXECUTOR,
            "input_artifact": "manifests/",
            "output_artifact": "manifests/parity_summary.json",
            "input_rows": EXPECTED_RAW_ROWS,
            "output_rows": EXPECTED_RAG_ROWS,
            "input_sha256": RAW_SHA256["jasonlemkin_blog.jsonl"],
            "output_sha256": next(
                (item.get("output_sha256") for item in documents if item.get("stage") == "rag_governance"),
                None,
            ),
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
    slices: int = 4,
    tokenizer=None,
) -> dict:
    """Run the shadow stages in order. A parity failure stops the run."""
    staging = assert_staging_dir(staging)
    staging.mkdir(parents=True, exist_ok=True)
    validated = stage_validate(staging, run_id)
    if through == "validate_raw":
        return validated
    cleaned = stage_clean(staging, run_id, validated["rows"], executor=executor, slices=slices)
    if through == "canonical_clean":
        return {"cleaned": len(cleaned)}
    stage_recorded_parity(staging, run_id, "canonical_clean", "clean_parity")
    if through == "clean_parity":
        return {"cleaned": len(cleaned)}
    nopromo = stage_nopromo(staging, run_id, cleaned)
    if through == "build_nopromo":
        return {"nopromo": len(nopromo)}
    stage_recorded_parity(staging, run_id, "build_nopromo", "nopromo_parity")
    if through == "nopromo_parity":
        return {"nopromo": len(nopromo)}
    fitted = stage_fit(staging, run_id, nopromo, tokenizer=tokenizer)
    if through == "canonical_fit":
        return fitted
    stage_recorded_parity(staging, run_id, "canonical_fit", "fit_parity")
    if through == "fit_parity":
        return fitted
    balanced = stage_balance(staging, run_id, fitted["records"])
    if through == "balance":
        return {"balanced": len(balanced)}
    stage_recorded_parity(staging, run_id, "balance", "balance_parity")
    if through == "balance_parity":
        return {"balanced": len(balanced)}
    assignments = stage_split(staging, run_id, balanced)
    if through == "split":
        return {"split": len(assignments)}
    stage_recorded_parity(staging, run_id, "split", "split_parity")
    if through == "split_parity":
        return {"split": len(assignments)}
    relabel_rows, relabel_assignments = stage_relabel(
        staging, run_id, balanced, assignments, tokenizer=tokenizer
    )
    if through == "relabel":
        return {"relabel": len(relabel_rows)}
    stage_recorded_parity(staging, run_id, "relabel", "relabel_parity")
    if through == "relabel_parity":
        return {"relabel": len(relabel_rows)}
    rag_rows = stage_rag(staging, run_id, relabel_rows, relabel_assignments)
    if through == "rag_governance":
        return {"rag": len(rag_rows)}
    stage_recorded_parity(staging, run_id, "rag_governance", "rag_parity")
    if through == "rag_parity":
        return {"rag": len(rag_rows)}
    from host_finetune.spark_v2_embed import stage_build_chroma, stage_embed, stage_logical_chroma, stage_retrieval

    embedded = stage_embed(staging, run_id, rag_rows)
    if through == "embed_rag":
        return embedded
    stage_recorded_parity(staging, run_id, "embed_rag", "embedding_parity")
    if through == "embedding_parity":
        return embedded
    built = stage_build_chroma(staging, run_id, rag_rows)
    if through == "build_chroma_shadow":
        return built
    logical = stage_logical_chroma(staging, run_id, rag_rows)
    if through == "logical_chroma_parity":
        return logical
    retrieved = stage_retrieval(staging, run_id)
    if through == "retrieval_parity":
        return retrieved
    return stage_summary(staging, run_id)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Run the v2 shadow pipeline through RAG governance.")
    parser.add_argument("--staging", type=Path, default=None)
    parser.add_argument("--run-id", default="local")
    parser.add_argument("--through", choices=SUMMARY_STAGES + ("parity_summary",), default="parity_summary")
    parser.add_argument("--executor", choices=("ordered", "spark"), default="ordered")
    parser.add_argument("--slices", type=int, default=4)
    parser.add_argument(
        "--partition-counts",
        default="",
        help="Comma-separated Spark partition counts. Proves order, then exits.",
    )
    args = parser.parse_args()
    staging = args.staging or (STAGING_ROOT / args.run_id)
    if args.partition_counts:
        if args.executor != "spark":
            raise SystemExit("partition proof requires --executor spark")
        counts = [int(part) for part in args.partition_counts.split(",") if part.strip()]
        validated = stage_validate(staging, args.run_id)
        proof = prove_spark_partitions(validated["rows"], counts)
        write_manifest(
            staging,
            "spark_partition_proof",
            {
                "run_id": args.run_id,
                "input_rows": len(validated["rows"]),
                "output_rows": EXPECTED_CLEANED_ROWS,
                "input_sha256": validated["hashes"]["jasonlemkin_blog.jsonl"],
                "output_sha256": "",
                "status": "PASS",
                "mismatches": {"versus_ordered": 0, "oracle": 0},
                "details": proof,
            },
        )
        print(json.dumps({"status": "PASS", "staging": str(staging), "partition_counts": counts}))
        return
    result = run_shadow(
        staging,
        args.run_id,
        through=args.through,
        executor=args.executor,
        slices=args.slices,
    )
    print(json.dumps({"status": "PASS", "staging": str(staging), "through": args.through, "result_keys": list(result)}))


if __name__ == "__main__":
    main()
