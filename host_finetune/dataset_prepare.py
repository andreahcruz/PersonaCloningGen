"""Prepare ``dataset.jsonl`` after MinIO download: chunk long rows, then clean.

**Order matters:** chunking must run *before* ``clean_dataset`` truncates labels to
``DATASET_MAX_OUTPUT_CHARS`` — otherwise megabyte tails are lost as a single 8k prefix.

Environment:
    SKIP_CHUNK_DATASET=1     — skip local chunk expand (Spark-only chunking).
    SKIP_DATASET_CLEAN=1     — skip junk filter + per-row truncation (see clean_dataset).
"""
from __future__ import annotations

from pathlib import Path

from host_finetune.config import (
    CHUNK_DATASET_AFTER_DOWNLOAD,
    DATASET_LOCAL,
    SKIP_DATASET_CLEAN,
)


def expand_long_outputs(path: Path | None = None) -> dict | None:
    """Split very long ``output`` fields into multiple Alpaca rows. Returns stats or None if skipped."""
    if not CHUNK_DATASET_AFTER_DOWNLOAD:
        return None
    path = path or DATASET_LOCAL
    if not path.is_file():
        return None
    from host_finetune.sft_chunk_utils import expand_jsonl_text

    text = path.read_text(encoding="utf-8")
    lines = text.splitlines()
    new_lines, stats = expand_jsonl_text(lines)
    path.write_text("\n".join(new_lines) + "\n", encoding="utf-8")
    return stats


def prepare_dataset_file(path: Path | None = None) -> dict:
    """Chunk (optional) + clean (optional). Returns combined stats for logging."""
    path = path or DATASET_LOCAL
    out: dict = {"path": str(path)}
    st_exp = expand_long_outputs(path)
    if st_exp is not None:
        out["chunk_expand"] = st_exp
    if not SKIP_DATASET_CLEAN:
        from host_finetune.clean_dataset import clean_dataset_file

        out["clean"] = clean_dataset_file(path)
    return out
