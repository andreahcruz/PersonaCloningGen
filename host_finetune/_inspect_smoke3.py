"""Compare smoke-test repunct output vs original transcript chunks."""
from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent / "data"
PRE = ROOT / "dataset.preclean.jsonl"
POST = ROOT / "dataset.clean_v2_smoke3.jsonl"

TITLES = [
    "Decacorns & Unicorns: Founders Fund Partner Keith Rabois and SaaStr CEO Jason Le",
    "Q&A with Maria Pergolino, Anthony Kennada, Aaron Ross and Jason Lemkin | SaaStr",
    "Live Workshop Wednesday: How to Do Company All-Hands Meetings Right with Guru",
]

_TITLE_RE = re.compile(r"^Write in the style of Jason Lemkin about:\s*", re.IGNORECASE)
_PART_RE = re.compile(r"\s*\(part\s+(\d+)\s+of\s+(\d+)\)\s*$", re.IGNORECASE)


def parse(instr: str) -> tuple[str, int]:
    t = _TITLE_RE.sub("", (instr or "").strip())
    m = _PART_RE.search(t)
    if m:
        return _PART_RE.sub("", t).strip(), int(m.group(1))
    return t.strip(), 1


def load(path: Path) -> dict[str, list[tuple[int, str]]]:
    groups: dict[str, list[tuple[int, str]]] = defaultdict(list)
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            base, k = parse(r.get("instruction", ""))
            groups[base].append((k, (r.get("output") or "").strip()))
    for v in groups.values():
        v.sort(key=lambda x: x[0])
    return groups


pre = load(PRE)
post = load(POST)


def find_match(needle: str, table: dict) -> str | None:
    for k in table:
        if k.startswith(needle):
            return k
    return None


for title_needle in TITLES:
    pre_key = find_match(title_needle, pre)
    post_key = find_match(title_needle, post)
    print("\n" + "=" * 80)
    print(f"TITLE: {title_needle[:80]}")
    print(f"  pre match : {pre_key[:90] if pre_key else None}")
    print(f"  post match: {post_key[:90] if post_key else None}")
    if not pre_key or not post_key:
        continue
    pre_body = "".join(p for _, p in pre[pre_key])
    post_body = "\n\n".join(p for _, p in post[post_key])

    print(f"\n  BEFORE ({len(pre_body)} chars):")
    print(f"  {pre_body[:600]!r}")
    print(f"\n  AFTER  ({len(post_body)} chars):")
    print(f"  {post_body[:600]!r}")
    print(f"\n  AFTER (mid):")
    mid = len(post_body) // 2
    print(f"  {post_body[mid:mid + 600]!r}")
