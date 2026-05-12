"""Sample rows classified as 'other' (mixed style) to understand the bucket."""
from __future__ import annotations

import json
import random
import re
from pathlib import Path

PATH = Path(__file__).resolve().parent / "data" / "dataset.jsonl"


def cap_rate(s: str) -> float:
    words = s.split()
    return sum(1 for w in words if w[:1].isupper()) / max(1, len(words))


def punct_rate(s: str) -> float:
    return sum(s.count(p) for p in ".!?") / max(1, len(s.split()))


rows = []
with PATH.open(encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            pass

other = []
for r in rows:
    out = (r.get("output") or "").strip()
    c = cap_rate(out)
    p = punct_rate(out)
    if (c < 0.05) != (p < 0.02):  # XOR: exactly one is low
        other.append((c, p, r))

print(f"'Other' rows (mixed cap/punct): {len(other)}")
random.seed(7)
sample = random.sample(other, min(10, len(other)))
for c, p, r in sample:
    out = (r.get("output") or "").strip()
    print(f"\n--- cap={c:.2f} punct={p:.3f} chars={len(out)}")
    print(f"INSTR: {r.get('instruction', '')[:100]}")
    print(f"OUT  : {out[:280]}")
