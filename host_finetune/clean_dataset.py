"""Filter and truncate ``dataset.jsonl`` before training (junk rows, mojibake).

Run standalone: ``python -m host_finetune.clean_dataset``
Called automatically from ``finetune`` unless ``SKIP_DATASET_CLEAN=1``.
"""
from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path

from host_finetune.config import DATASET_LOCAL

REPLACEMENT_CHAR = "\ufffd"

# Rows matching these are dropped (substring match on instruction or output).
_JUNK_SUBSTRINGS = (
    "test post with ai chat",
    "asdad\nrelated posts",
)

# Minimum significant output length after strip (below = drop).
_MIN_OUTPUT_CHARS = 80


def _truncate_output(text: str, max_chars: int) -> tuple[str, bool]:
    if max_chars <= 0 or len(text) <= max_chars:
        return text, False
    cut = text[:max_chars]
    last_nl = cut.rfind("\n\n")
    if last_nl > max_chars * 4 // 5:
        cut = cut[:last_nl]
    cut = cut.rstrip() + "\n\n[Truncated for training.]"
    return cut, True


def _is_junk_row(instr: str, out: str) -> bool:
    il = instr.lower()
    ol = out.lower()
    for s in _JUNK_SUBSTRINGS:
        if s in il or s in ol:
            return True
    if REPLACEMENT_CHAR in instr or REPLACEMENT_CHAR in out:
        return True
    if len(out.strip()) < _MIN_OUTPUT_CHARS:
        return True
    return False


def clean_dataset_records(
    rows: list[dict],
    max_output_chars: int,
) -> tuple[list[dict], dict]:
    stats = {
        "rows_in": len(rows),
        "rows_out": 0,
        "dropped_junk": 0,
        "truncated_outputs": 0,
    }
    out_rows: list[dict] = []
    for row in rows:
        instr = str(row.get("instruction", ""))
        inp = str(row.get("input", ""))
        output = str(row.get("output", ""))
        if _is_junk_row(instr, output):
            stats["dropped_junk"] += 1
            continue
        output, did_trunc = _truncate_output(output, max_output_chars)
        if did_trunc:
            stats["truncated_outputs"] += 1
        out_rows.append(
            {"instruction": instr, "input": inp, "output": output}
        )
    stats["rows_out"] = len(out_rows)
    return out_rows, stats


def clean_dataset_file(
    path: Path | None = None,
    *,
    max_output_chars: int | None = None,
) -> dict:
    """Rewrite ``path`` in place with filtered rows. Returns stats dict."""
    path = path or DATASET_LOCAL
    if max_output_chars is None:
        from host_finetune.config import DATASET_MAX_OUTPUT_CHARS

        max_output_chars = DATASET_MAX_OUTPUT_CHARS

    if not path.is_file():
        return {"error": f"missing {path}", "rows_in": 0, "rows_out": 0}

    rows: list[dict] = []
    bad_lines = 0
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                bad_lines += 1

    cleaned, stats = clean_dataset_records(rows, max_output_chars)
    stats["bad_json_lines"] = bad_lines
    stats["path"] = str(path)

    tmp_fd, tmp_name = tempfile.mkstemp(
        suffix=".jsonl", dir=path.parent, text=True
    )
    try:
        with os.fdopen(tmp_fd, "w", encoding="utf-8") as tmp:
            for row in cleaned:
                tmp.write(json.dumps(row, ensure_ascii=False) + "\n")
        os.replace(tmp_name, path)
    except Exception:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise

    return stats


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "path",
        nargs="?",
        default=str(DATASET_LOCAL),
        help=f"JSONL path (default: {DATASET_LOCAL})",
    )
    p.add_argument(
        "--max-output-chars",
        type=int,
        default=None,
        help="Override DATASET_MAX_OUTPUT_CHARS (0 = no truncation)",
    )
    args = p.parse_args()
    path = Path(args.path)
    mo = args.max_output_chars
    stats = clean_dataset_file(path, max_output_chars=mo if mo is not None else None)
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
