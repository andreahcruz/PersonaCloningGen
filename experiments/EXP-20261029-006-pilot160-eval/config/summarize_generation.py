"""Compact generation outcomes. Not a quality score."""

from __future__ import annotations

import json
from pathlib import Path

RAW = Path(__file__).resolve().parents[1] / "raw"


def main() -> None:
    rows = []
    for mode in ("dense", "naive", "mix"):
        payload = json.loads((RAW / f"generation_{mode}.json").read_text(encoding="utf-8"))
        for trace in payload["traces"]:
            answer = trace.get("answer") or ""
            verified = trace.get("verified_claims") or []
            rejected = trace.get("validation_rejected") or []
            rows.append(
                {
                    "mode": mode,
                    "id": trace.get("id"),
                    "medium": trace.get("medium"),
                    "words": len(answer.split()),
                    "verified_claims": len(verified),
                    "validation_rejected": len(rejected),
                    "rejection_reasons": sorted({
                        str(item.get("reason") or item.get("rejection_reason") or item)
                        for item in rejected
                    })[:8] if rejected and isinstance(rejected[0], dict) else [],
                    "numeric_grounding": trace.get("numeric_grounding"),
                    "mentions_anneka": "anneka" in answer.casefold(),
                    "mentions_lemkin": "lemkin" in answer.casefold(),
                    "mentions_5_15": "5.15" in answer,
                    "mentions_webflow": "webflow" in answer.casefold(),
                    "claim_texts": [item.get("claim") for item in verified if isinstance(item, dict)],
                    "answer": answer,
                }
            )
    out = RAW.parent / "metrics" / "generation_inspection.json"
    out.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    for row in rows:
        print(
            f"{row['mode']:6} {row['id']:28} {str(row['medium']):10} "
            f"words={row['words']:3} verified={row['verified_claims']} "
            f"rejected={row['validation_rejected']} anneka={row['mentions_anneka']} "
            f"5.15={row['mentions_5_15']}"
        )
        print(row["answer"][:500].replace("\n", " "))
        print("---")


if __name__ == "__main__":
    main()
