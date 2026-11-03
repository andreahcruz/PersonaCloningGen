"""Read-only readiness audit of the frozen Relabel SFT file.

This does not modify the dataset, the splits, or the adapter.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
DATASET = ROOT / "experiments" / "EXP-20261003-004-relabel-continuations" / "raw" / "dataset.jsonl"
ASSIGNMENTS = ROOT / "experiments" / "EXP-20261003-004-relabel-continuations" / "raw" / "split_assignments.jsonl"
OUT = Path(__file__).resolve().parents[1] / "metrics" / "sft_readiness.json"

QUOTE_RE = re.compile(r"""^\s*["“'].{40,}["”']""")
PROMO_RE = re.compile(r"saastrannual|saastrdeploy|register now|use code|% off|sponsor", re.I)
BOILER_RE = re.compile(r"\b(click here|sign up now|learn more at)\b", re.I)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    assignments = []
    with ASSIGNMENTS.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                if row.get("record_type") == "meta":
                    meta = row
                    continue
                assignments.append(row)
    outputs = []
    with DATASET.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                outputs.append(json.loads(line))
    if len(outputs) != len(assignments):
        raise SystemExit(f"row mismatch dataset={len(outputs)} assignments={len(assignments)}")
    split_role = Counter()
    medium_split = Counter()
    exact_targets = Counter()
    short = 0
    promo = 0
    quoted_open = 0
    continuation_missing_prior = 0
    opening_with_prior = 0
    unchanged = 0
    group_splits: dict[str, set[str]] = {}
    for row, assignment in zip(outputs, assignments):
        text = (row.get("output") or "").strip()
        role = row.get("sft_role")
        split = assignment.get("split")
        medium = row.get("source")
        split_role[(split, role)] += 1
        medium_split[(medium, split)] += 1
        exact_targets[hashlib.sha256(text.encode("utf-8")).hexdigest()] += 1
        words = len(text.split())
        if words < 20:
            short += 1
        if PROMO_RE.search(text) or BOILER_RE.search(text):
            promo += 1
        if QUOTE_RE.search(text):
            quoted_open += 1
        instruction = row.get("instruction") or ""
        if role == "continuation" and "Previous section ending:" not in instruction:
            continuation_missing_prior += 1
        if role == "opening" and "Previous section ending:" in instruction:
            opening_with_prior += 1
        if role == "unchanged":
            unchanged += 1
        group_splits.setdefault(assignment.get("group_id"), set()).add(split)
    leaked_groups = {group: sorted(splits) for group, splits in group_splits.items() if len(splits) > 1}
    duplicate_targets = sum(count - 1 for count in exact_targets.values() if count > 1)
    duplicate_groups = sum(1 for count in exact_targets.values() if count > 1)
    report = {
        "dataset": str(DATASET.relative_to(ROOT)),
        "dataset_sha256": _sha256(DATASET),
        "assignment_sha256": _sha256(ASSIGNMENTS),
        "assignment_meta_sha256": meta.get("dataset_sha256"),
        "rows": len(outputs),
        "split_role": {f"{split}:{role}": count for (split, role), count in sorted(split_role.items())},
        "medium_split": {f"{medium}:{split}": count for (medium, split), count in sorted(medium_split.items())},
        "exact_duplicate_extra_rows": duplicate_targets,
        "exact_duplicate_target_groups": duplicate_groups,
        "targets_under_20_words": short,
        "promo_or_boilerplate_regex_hits": promo,
        "opens_with_a_long_quotation": quoted_open,
        "continuation_missing_prior_marker": continuation_missing_prior,
        "opening_contains_prior_marker": opening_with_prior,
        "unchanged_rows": unchanged,
        "groups_in_more_than_one_split": len(leaked_groups),
        "decision": (
            "do_not_retrain"
        ),
        "decision_reason": (
            "The frozen adapter was trained on this Relabel file. "
            "No newer versioned SFT derivative is the training source for a replacement adapter. "
            "Exact duplicate targets, short rows, promotional lines, and quotation openings are quality risks, "
            "but they are not a prepared replacement dataset with a held-out evaluation plan."
        ),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: report[key] for key in report if key != "split_role"}, indent=2))


if __name__ == "__main__":
    main()
