"""Document-grouped train/validation/test assignment for the frozen SFT jsonl.

Chunks of one source line stay in one group. Near-duplicate outputs are merged
into that group so they cannot cross splits. When rows carry a ``source``
medium, each medium is assigned its own 80/10/10 quota. Assignment is greedy
on group size inside that medium and does not call ``datasets.train_test_split``.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

JACCARD_T = 0.8
TARGET_SHARES = {"train": 0.80, "validation": 0.10, "test": 0.10}
SPLIT_NAMES = ("train", "validation", "test")
TOKEN_RE = re.compile(r"[a-z0-9$%]+")
TOPIC_PREFIX = "Write in the style of Jason Lemkin about:"
_INSTRUCTION_RE = re.compile(
    r"^Write (?:(?:a blog post|a LinkedIn post|an X post|a talk) )?"
    r"in the style of Jason Lemkin about:\s*(.*)$",
    re.IGNORECASE | re.DOTALL,
)
PART_RE = re.compile(r"\s+\(part\s+(\d+)\s+of\s+(\d+)\)\s*$", re.IGNORECASE)

# Frozen host_finetune/data/dataset.jsonl measured in EXP-20260927-001.
FROZEN_DATASET_SHA256 = "a4bfe637b9c178a29bb885ef8502ff45d348fe743d8d2938ba56f8ea3f71e4f5"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text or "")
    text = text.casefold()
    return re.sub(r"\s+", " ", text).strip()


# Gold-overlap comparison only. Near-duplicate clustering still uses normalize_text
# and JACCARD_T. This does not lower that threshold.
OVERLAP_JACCARD = JACCARD_T
_OVERLAP_URL_RE = re.compile(
    r"(?:https?://|www\.)\S+|\b(?:[\w-]+\.)+(?:com|me|ly|be|io|org|net)/\S*",
    re.IGNORECASE,
)
_OVERLAP_DASH_RE = re.compile(r"[\u2010\u2011\u2012\u2013\u2014\u2015\u2212]|--")


def normalize_overlap_text(text: str) -> str:
    """Strip URLs and fold dashes and punctuation so a headline repost can match.

    Whitespace is collapsed last. ``$`` and ``%`` stay, matching the token
    pattern used for the 0.80 Jaccard check.
    """
    text = unicodedata.normalize("NFKC", text or "")
    text = _OVERLAP_URL_RE.sub(" ", text)
    text = _OVERLAP_DASH_RE.sub("-", text)
    text = text.casefold()
    text = re.sub(r"[^\w\s$%]", " ", text, flags=re.UNICODE)
    return re.sub(r"\s+", " ", text).strip()


def matches_gold_field(text: str, title: str, value: str, field: str) -> tuple[bool, float]:
    """Return whether overlap-normalized text hits one gold field, and the Jaccard.

    The predicates are the existing ones: exact text, exact topic title,
    an 8-word-or-longer value contained in the text, or token Jaccard at or
    above 0.80. The threshold is not lowered.
    """
    text_n = normalize_overlap_text(text)
    title_n = normalize_overlap_text(title)
    value_n = normalize_overlap_text(value)
    if not value_n:
        return False, 0.0
    score = _jaccard(tokens(text_n), tokens(value_n))
    matched = (
        text_n == value_n
        or (field == "topic" and bool(title_n) and title_n == value_n)
        or (len(value_n.split()) >= 8 and value_n in text_n)
        or score >= OVERLAP_JACCARD
    )
    return matched, score


def tokens(text: str) -> set[str]:
    return set(TOKEN_RE.findall(text))


def parse_instruction(instruction: str) -> tuple[str, int | None, int | None]:
    """Return base title, part index, and part count. Unparsed text is the base title."""
    text = (instruction or "").strip()
    prefixed = _INSTRUCTION_RE.match(text)
    if prefixed:
        text = prefixed.group(1).strip()
    elif text.lower().startswith(TOPIC_PREFIX.lower()):
        text = text[len(TOPIC_PREFIX) :].strip()
    match = PART_RE.search(text)
    if not match:
        return text, None, None
    return text[: match.start()].strip(), int(match.group(1)), int(match.group(2))


def load_rows(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


class _UnionFind:
    def __init__(self, n: int) -> None:
        self.parent = list(range(n))

    def find(self, item: int) -> int:
        parent = self.parent
        while parent[item] != item:
            parent[item] = parent[parent[item]]
            item = parent[item]
        return item

    def union(self, left: int, right: int) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root != right_root:
            self.parent[right_root] = left_root


def _prefix_len(n_tokens: int) -> int:
    return max(1, n_tokens - math.ceil(JACCARD_T * n_tokens) + 1)


def _jaccard(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    inter = len(left & right)
    if inter == 0:
        return 0.0
    return inter / (len(left) + len(right) - inter)


def _sizes_allow(left: int, right: int) -> bool:
    if left == 0 or right == 0:
        return False
    lo, hi = (left, right) if left <= right else (right, left)
    return lo / hi >= JACCARD_T


def near_duplicate_pairs(norm_outputs: list[str]) -> list[tuple[int, int]]:
    """Pairs with token-set Jaccard >= 0.8, excluding normalized-exact matches."""
    docs: list[dict] = []
    freq: dict[str, int] = defaultdict(int)
    for index, text in enumerate(norm_outputs):
        toks = tokens(text)
        docs.append({"i": index, "tokens": toks, "norm": text})
        for tok in toks:
            freq[tok] += 1
    index: dict[str, list[int]] = defaultdict(list)
    for doc in docs:
        toks = doc["tokens"]
        if not toks:
            doc["prefix"] = []
            continue
        ordered = sorted(toks, key=lambda tok: (freq[tok], tok))
        pref = ordered[: _prefix_len(len(ordered))]
        doc["prefix"] = pref
        for tok in pref:
            index[tok].append(doc["i"])
    by_i = {doc["i"]: doc for doc in docs}
    seen: set[tuple[int, int]] = set()
    pairs: list[tuple[int, int]] = []
    for doc in docs:
        candidates: set[int] = set()
        for tok in doc["prefix"]:
            candidates.update(index.get(tok, ()))
        for other_i in candidates:
            if other_i <= doc["i"]:
                continue
            pair = (doc["i"], other_i)
            if pair in seen:
                continue
            seen.add(pair)
            other = by_i[other_i]
            if doc["norm"] and doc["norm"] == other["norm"]:
                continue
            if not _sizes_allow(len(doc["tokens"]), len(other["tokens"])):
                continue
            if _jaccard(doc["tokens"], other["tokens"]) >= JACCARD_T:
                pairs.append(pair)
    return pairs


def assign_groups(
    rows: list[dict],
    pairs: list[tuple[int, int]] | None = None,
) -> tuple[list[dict], list[tuple[int, int]]]:
    """Return assignments in input order, and the near-duplicate pairs used to merge."""
    parsed = [parse_instruction(str(row.get("instruction", ""))) for row in rows]
    norm_outputs = [normalize_text(str(row.get("output", ""))) for row in rows]
    title_keys = [normalize_text(title) or f"row-{i}" for i, (title, _, _) in enumerate(parsed)]

    uf = _UnionFind(len(rows))
    by_source: dict[tuple[str, str], list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        source_file = str(row.get("source_file") or "").strip()
        source_line = str(row.get("source_line") or "").strip()
        if source_file and source_line:
            by_source[(source_file, source_line)].append(index)
    for members in by_source.values():
        head = members[0]
        for other in members[1:]:
            uf.union(head, other)

    by_title: dict[str, list[int]] = defaultdict(list)
    for index, key in enumerate(title_keys):
        by_title[key].append(index)
    for members in by_title.values():
        head = members[0]
        for other in members[1:]:
            uf.union(head, other)

    by_norm: dict[str, list[int]] = defaultdict(list)
    for index, text in enumerate(norm_outputs):
        if text:
            by_norm[text].append(index)
    for members in by_norm.values():
        head = members[0]
        for other in members[1:]:
            uf.union(head, other)

    if pairs is None:
        pairs = near_duplicate_pairs(norm_outputs)
    for left, right in pairs:
        uf.union(left, right)

    components: dict[int, list[int]] = defaultdict(list)
    for index in range(len(rows)):
        components[uf.find(index)].append(index)

    group_ids: dict[int, str] = {}
    canonical_of: dict[int, str] = {}
    for root, members in components.items():
        canonical = min(title_keys[i] for i in members)
        canonical_of[root] = canonical
        group_ids[root] = hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    groups = [
        (len(members), canonical_of[root], root)
        for root, members in components.items()
    ]

    def _group_medium(members: list[int]) -> str:
        counts: dict[str, int] = defaultdict(int)
        for index in members:
            name = str(rows[index].get("source") or "").strip() or "unavailable"
            counts[name] += 1
        best = max(counts.values())
        return min(name for name, count in counts.items() if count == best)

    by_medium: dict[str, list[tuple[int, str, int]]] = defaultdict(list)
    medium_rows: dict[str, int] = defaultdict(int)
    for size, key, root in groups:
        medium = _group_medium(components[root])
        by_medium[medium].append((size, key, root))
        medium_rows[medium] += size

    # Each medium gets its own 80/10/10 quota so validation and test are not
    # filled by leftover short posts from the largest medium.
    split_of_root: dict[int, str] = {}
    for medium in sorted(by_medium):
        medium_groups = sorted(by_medium[medium], key=lambda item: (-item[0], item[1], item[2]))
        remaining = {name: TARGET_SHARES[name] * medium_rows[medium] for name in SPLIT_NAMES}
        for size, _key, root in medium_groups:
            choice = min(SPLIT_NAMES, key=lambda name: (-remaining[name], name))
            split_of_root[root] = choice
            remaining[choice] -= size

    assignments: list[dict] = []
    for index, (title, part_index, part_count) in enumerate(parsed):
        root = uf.find(index)
        source_row = rows[index]
        source_name = str(source_row.get("source") or "").strip()
        assignments.append(
            {
                "row_index": index,
                "split": split_of_root[root],
                "group_id": group_ids[root],
                "base_title": title,
                "part_index": part_index,
                "part_count": part_count,
                "source_platform": source_name or "unavailable",
                "source_file": str(source_row.get("source_file") or ""),
                "source_line": source_row.get("source_line") or "",
            }
        )
    return assignments, pairs


def _percentile(ordered: list[int], pct: float) -> int:
    index = round((pct / 100) * (len(ordered) - 1))
    return ordered[min(len(ordered) - 1, max(0, index))]


def _word_stats(counts: list[int]) -> dict:
    if not counts:
        return {"n": 0, "min": 0, "p25": 0, "p50": 0, "p75": 0, "mean": 0.0, "max": 0}
    ordered = sorted(counts)
    return {
        "n": len(counts),
        "min": ordered[0],
        "p25": _percentile(ordered, 25),
        "p50": _percentile(ordered, 50),
        "p75": _percentile(ordered, 75),
        "mean": round(sum(ordered) / len(ordered), 2),
        "max": ordered[-1],
    }


def build_report(
    rows: list[dict],
    assignments: list[dict],
    dataset_sha256: str,
    pairs: list[tuple[int, int]],
) -> dict:
    norm_outputs = [normalize_text(str(row.get("output", ""))) for row in rows]
    by_index = {row["row_index"]: row for row in assignments}
    cross = 0
    within = {name: 0 for name in SPLIT_NAMES}
    for left, right in pairs:
        left_split = by_index[left]["split"]
        right_split = by_index[right]["split"]
        if left_split == right_split:
            within[left_split] += 1
        else:
            cross += 1

    norm_groups: dict[str, set[str]] = defaultdict(set)
    for index, text in enumerate(norm_outputs):
        if text:
            norm_groups[text].add(by_index[index]["split"])
    cross_exact = sum(1 for splits in norm_groups.values() if len(splits) > 1)

    per_split = {}
    n_rows = len(rows)
    for name in SPLIT_NAMES:
        members = [row for row in assignments if row["split"] == name]
        title_counts: dict[str, int] = defaultdict(int)
        words = []
        for row in members:
            title_counts[row["base_title"]] += 1
            words.append(len(str(rows[row["row_index"]].get("output", "")).split()))
        counts = list(title_counts.values())
        per_split[name] = {
            "n_rows": len(members),
            "share": round(len(members) / n_rows, 4) if n_rows else 0.0,
            "n_groups": len({row["group_id"] for row in members}),
            "n_base_titles": len(title_counts),
            "rows_per_base_title": _word_stats(counts),
            "word_count": _word_stats(words),
            "near_duplicate_pairs_within_split": within[name],
            "medium_counts": dict(sorted(Counter(row["source_platform"] for row in members).items())),
        }
    residuals = {
        name: round(per_split[name]["share"] - TARGET_SHARES[name], 4) for name in SPLIT_NAMES
    }
    row_remainder = {
        name: round(per_split[name]["n_rows"] - TARGET_SHARES[name] * n_rows, 4)
        for name in SPLIT_NAMES
    }
    within_one_point = all(abs(value) <= 0.01 for value in residuals.values())
    group_splits: dict[str, set[str]] = defaultdict(set)
    for row in assignments:
        group_splits[row["group_id"]].add(row["split"])
    groups_in_multiple_splits = sum(1 for splits in group_splits.values() if len(splits) > 1)
    return {
        "dataset_sha256": dataset_sha256,
        "n_rows": n_rows,
        "n_groups": len({row["group_id"] for row in assignments}),
        "target_shares": TARGET_SHARES,
        "source_platform": (
            "source_file+source_line"
            if any(row.get("source_file") for row in assignments)
            else "unavailable"
        ),
        "source_note": (
            "Chunks that share source_file and source_line are one document and stay in one split. "
            "Each source medium is assigned its own 80/10/10 quota, so validation and test "
            "keep that medium's share of the rows. source_platform on each assignment is the scraped medium."
            if any(row.get("source_file") for row in assignments)
            else
            "source, url, date, and essay_id are dropped by to_training_json_lines "
            "and clean_dataset_records. No platform or B2B source tag is stored on these rows. "
            "base_title is the only document key that survives, via the instruction string."
        ),
        "near_duplicate_pairs_crossing_splits": cross,
        "normalized_exact_groups_crossing_splits": cross_exact,
        "groups_in_multiple_splits": groups_in_multiple_splits,
        "share_residual": residuals,
        "row_remainder_vs_quota": row_remainder,
        "within_one_percentage_point": within_one_point,
        "residual_note": (
            None
            if within_one_point
            else "Share differs from 80/10/10 by more than one percentage point because groups are assigned whole."
        ),
        "splits": per_split,
    }


def write_assignment_file(path: Path, dataset_sha256: str, assignments: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    has_source = any(row.get("source_file") for row in assignments)
    meta = {
        "record_type": "meta",
        "dataset_sha256": dataset_sha256,
        "n_rows": len(assignments),
        "target_shares": TARGET_SHARES,
        "jaccard_threshold": JACCARD_T,
        "source_platform": "source_file+source_line" if has_source else "unavailable",
    }
    lines = [json.dumps(meta, ensure_ascii=False)]
    for row in assignments:
        lines.append(json.dumps(row, ensure_ascii=False))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def load_assignment_file(path: Path) -> tuple[dict, list[dict]]:
    meta = None
    rows: list[dict] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            if record.get("record_type") == "meta":
                meta = record
            else:
                rows.append(record)
    if meta is None:
        raise RuntimeError(f"{path} is missing a record_type=meta header.")
    rows.sort(key=lambda row: row["row_index"])
    return meta, rows


def assert_manifest_matches(manifest_path: Path, dataset_path: Path) -> tuple[dict, list[dict]]:
    """Raise if the manifest was not built for these exact dataset bytes."""
    meta, rows = load_assignment_file(manifest_path)
    digest = file_sha256(dataset_path)
    expected = meta.get("dataset_sha256")
    if digest != expected:
        raise RuntimeError(
            f"SPLIT_MANIFEST {manifest_path} records dataset sha256 {expected}, "
            f"but {dataset_path} is {digest}. Refusing to train on a different file."
        )
    if len(rows) != meta.get("n_rows"):
        raise RuntimeError(
            f"SPLIT_MANIFEST {manifest_path} header says {meta.get('n_rows')} rows "
            f"but {len(rows)} assignments are present."
        )
    return meta, rows


def indices_for_split(assignments: list[dict], split: str) -> list[int]:
    return [row["row_index"] for row in assignments if row["split"] == split]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    digest = file_sha256(args.dataset)
    rows = load_rows(args.dataset)
    assignments, pairs = assign_groups(rows)
    report = build_report(rows, assignments, digest, pairs)
    raw_dir = args.out_dir / "raw"
    metrics_dir = args.out_dir / "metrics"
    write_assignment_file(raw_dir / "split_assignments.jsonl", digest, assignments)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    (metrics_dir / "split_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps({
        "dataset_sha256": digest,
        "n_rows": len(rows),
        "n_groups": report["n_groups"],
        "shares": {name: report["splits"][name]["share"] for name in SPLIT_NAMES},
        "crossing_near": report["near_duplicate_pairs_crossing_splits"],
        "crossing_exact": report["normalized_exact_groups_crossing_splits"],
    }))


if __name__ == "__main__":
    main()
