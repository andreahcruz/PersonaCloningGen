"""Remove direct event promotions and trailing event CTAs from SFT JSONL.

This is intentionally conservative.  It keeps advice that merely mentions an
event (for example, a ``Dear SaaStr`` question about event sponsorship), while
removing direct ticket, speaker, attendee, and sponsorship announcements.
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path


PREFIX = "Write in the style of Jason Lemkin about: "

EVENT = re.compile(r"\b(saastr (?:ai )?(?:annual|europa|summit)|saastrdeploy|saastrannual)\b", re.I)
DIRECT_ANNOUNCEMENT = re.compile(
    r"\b(?:"
    r"want leads\? sponsor|last chance|first wave.*speakers?|new .*speakers?|"
    r"speaker submissions?|speaker alert|final speaker|first \d+[+,]? ?speakers?|"
    r"attendees?(?:,|\s+(?:must|guide|coming|registration|opens))|"
    r"early registration|apply to speak|grab \d+% off|buy tickets?|get tickets?|"
    r"one week (?:til|until)|official attendee guide|hoping to hit \d|"
    r"annual .* summit is back|see everyone"
    r")\b",
    re.I,
)
TRAILING_CTA = re.compile(
    r"(?:^|(?<=[.!?])\s+)(?:"
    r"want to (?:meet|join|learn from)|join us|register (?:now|today)|"
    r"get tickets?|buy tickets?|save \d+%|apply to speak|"
    r"(?:become|want to be) a sponsor"
    r").*$",
    re.I | re.S,
)


def _topic(instruction: str) -> str:
    return instruction[len(PREFIX):] if instruction.startswith(PREFIX) else instruction


def _is_direct_event_announcement(row: dict) -> bool:
    title = _topic(str(row.get("instruction") or ""))
    # Questions framed as advice are valuable corpus content, even if they
    # mention event sponsorship.
    if title.lower().startswith("dear saastr:"):
        return False
    return bool(EVENT.search(title) and DIRECT_ANNOUNCEMENT.search(title))


def _strip_trailing_cta(text: str) -> tuple[str, bool]:
    match = TRAILING_CTA.search(text)
    if not match or match.start() < len(text) * 0.55:
        return text, False
    tail = text[match.start():]
    if not EVENT.search(tail) and not re.search(r"\b(ticket|speaker|attendee|sponsor)\b", tail, re.I):
        return text, False
    return text[:match.start()].rstrip(), True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--report", type=Path, default=None)
    args = parser.parse_args()
    out = args.out or args.dataset.with_name(args.dataset.stem + "_nopromo.jsonl")
    report = args.report or out.with_suffix(".promo_filter.json")

    stats: Counter[str] = Counter()
    removed_docs: set[tuple[str, int]] = set()
    examples: list[dict] = []
    with args.dataset.open(encoding="utf-8") as source, out.open("w", encoding="utf-8") as dest:
        for line in source:
            if not line.strip():
                continue
            stats["rows_in"] += 1
            row = json.loads(line)
            key = (str(row.get("source_file") or ""), int(row.get("source_line") or 0))
            if _is_direct_event_announcement(row):
                stats["rows_dropped_direct_event_promo"] += 1
                removed_docs.add(key)
                if len(examples) < 30:
                    examples.append({"action": "dropped_document", "title": _topic(row["instruction"])})
                continue
            clean_output, changed = _strip_trailing_cta(str(row.get("output") or ""))
            if changed:
                stats["rows_trailing_cta_removed"] += 1
                row["output"] = clean_output
            if len(str(row.get("output") or "").split()) < 20:
                stats["rows_dropped_empty_after_cta"] += 1
                continue
            dest.write(json.dumps(row, ensure_ascii=False) + "\n")
            stats["rows_out"] += 1

    stats["source_documents_dropped_direct_event_promo"] = len(removed_docs)
    report.write_text(json.dumps({"stats": dict(stats), "examples": examples}, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(dict(stats), indent=2))
    print(f"report: {report}")


if __name__ == "__main__":
    main()
