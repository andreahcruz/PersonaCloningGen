"""Coverage table for the retrieval sheet. Not precision or recall."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SHEET = ROOT / "raw" / "retrieval_sheet.json"
FAILED = {
    "gd_5ffdc3a083492246413be6f2",
    "gd_8276b7d28a0837fa1e71181a",
    "gd_8a579b23707dd4f77e169619",
    "gd_a4ad0243c9bcfec4c527a703",
    "gd_aa82ba1eab089122de370f68",
    "gd_de0b05c2882793b64fd76389",
    "gd_f76cf95212bfc67508398f63",
    "gd_f9a237e6ccf11fe4de9924ad",
}


def main() -> None:
    rows = json.loads(SHEET.read_text(encoding="utf-8"))
    summary = []
    for row in rows:
        review = row["review_source_ids"]
        hits = row["hit_source_ids"]
        summary.append(
            {
                "question_id": row["question_id"],
                "mode": row["mode"],
                "backend": row["backend"],
                "designated_in_hits": row["designated_sources_retrieved"],
                "designated_count": len(row["designated_sources_retrieved"]),
                "designated_possible": len(review),
                "rank1_is_designated": bool(hits) and bool(review) and hits[0] in set(review),
                "empty": row["empty"],
                "seconds": row["seconds"],
                "failed_extraction_ids_in_hits": [item for item in hits if item in FAILED],
                "excluded_reasons": row["excluded_reasons"],
                "train_attributed": row["train_attributed"],
                "speaker_on_first_hit": (row["hits"][0].get("quoted_speakers") if row["hits"] else []),
                "attribution_on_first_hit": (
                    row["hits"][0].get("attribution_status") if row["hits"] else None
                ),
            }
        )
    out = ROOT / "metrics" / "retrieval_coverage.json"
    out.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    for item in summary:
        print(
            f"{item['question_id']:28} {item['mode']:8} "
            f"{item['designated_count']}/{item['designated_possible']} "
            f"rank1={item['rank1_is_designated']} empty={item['empty']} "
            f"failed_ids={item['failed_extraction_ids_in_hits']}"
        )


if __name__ == "__main__":
    main()
