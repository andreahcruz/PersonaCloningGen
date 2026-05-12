"""Stitch the preclean dataset, classify groups, estimate repunct workload."""
from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path

PATH = Path(__file__).resolve().parent / "data" / "dataset.preclean.jsonl"

_TITLE_RE = re.compile(r"^Write in the style of Jason Lemkin about:\s*", re.IGNORECASE)
_PART_RE = re.compile(r"\s*\(part\s+(\d+)\s+of\s+(\d+)\)\s*$", re.IGNORECASE)


def parse_title(instr: str) -> tuple[str, int]:
    t = _TITLE_RE.sub("", (instr or "").strip())
    m = _PART_RE.search(t)
    if m:
        return _PART_RE.sub("", t).strip(), int(m.group(1))
    return t.strip(), 1


def cap_rate(s: str) -> float:
    w = s.split()
    return sum(1 for x in w if x[:1].isupper()) / max(1, len(w))


def punct_rate(s: str) -> float:
    return sum(s.count(p) for p in ".!?") / max(1, len(s.split()))


groups: dict[str, list[tuple[int, str]]] = defaultdict(list)
n_rows = 0
with PATH.open(encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        n_rows += 1
        instr = r.get("instruction") or ""
        out = (r.get("output") or "").strip()
        if not out:
            continue
        base, k = parse_title(instr)
        if not base:
            base = "untitled"
        groups[base].append((k, out))

print(f"rows: {n_rows}")
print(f"unique base titles (groups): {len(groups)}")

# Classify each group
bins = defaultdict(lambda: {"n": 0, "total_chars": 0, "max_chars": 0})
for base, parts in groups.items():
    parts.sort(key=lambda x: x[0])
    body = "".join(p for _, p in parts)
    c = cap_rate(body)
    p = punct_rate(body)
    if c < 0.05 and p < 0.02:
        klass = "transcript"
    elif c >= 0.05 and p >= 0.02:
        klass = "blog_or_tweet"
    else:
        klass = "other"
    bins[klass]["n"] += 1
    bins[klass]["total_chars"] += len(body)
    bins[klass]["max_chars"] = max(bins[klass]["max_chars"], len(body))

print("\nGroup classification after re-stitch:")
for k, v in bins.items():
    avg = v["total_chars"] / max(1, v["n"])
    print(f"  {k:15s}  n={v['n']:5d}  total_chars={v['total_chars']:>10d}  avg={int(avg):>7d}  max={v['max_chars']}")

# Estimate Ollama windows + time
WINDOW_CHARS = 2500
SECS_PER_WINDOW = 8.0  # conservative
n_windows = sum(
    (v["total_chars"] // WINDOW_CHARS + (1 if v["total_chars"] % WINDOW_CHARS else 0))
    for k, v in bins.items()
    if k == "transcript"
)
print(f"\nEstimated transcript repunct work:")
print(f"  windows ({WINDOW_CHARS} chars each): {n_windows}")
print(f"  est. time at {SECS_PER_WINDOW}s/window: {n_windows * SECS_PER_WINDOW / 60:.1f} min")
print(f"  est. time at {SECS_PER_WINDOW * 2}s/window: {n_windows * SECS_PER_WINDOW * 2 / 60:.1f} min")

# Top transcript groups by size
print("\nTop 5 transcript groups by chars (largest = most windows):")
transcript_groups = []
for base, parts in groups.items():
    parts.sort(key=lambda x: x[0])
    body = "".join(p for _, p in parts)
    if cap_rate(body) < 0.05 and punct_rate(body) < 0.02:
        transcript_groups.append((len(body), base))
transcript_groups.sort(reverse=True)
for n, t in transcript_groups[:5]:
    print(f"  {n:>8d} chars  {t[:80]}")
