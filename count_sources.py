"""One-off helper: count rows per source file and how many would survive
the Spark training-data filter (doc_word_count >= 50, after the same
per-source skip rules used by extract_to_minio).
"""
import json
from pathlib import Path

DATA = Path("data")

FILES = {
    "blog": "jasonlemkin_blog.jsonl",
    "linkedin": "jasonlemkinlinkedin.jsonl",
    "x": "jasonlk_originals.jsonl",
    "youtube_jason": "jasonmlemkinyoutubetranscripts.jsonl",
    "youtube_saastr": "saastryoutubetranscripts.jsonl",
}


def body_for(source: str, row: dict) -> str | None:
    """Mirror the normalizer rules in dags/lemkin_pipeline_dag.py."""
    if source == "blog":
        return (row.get("content") or row.get("text") or "").strip() or None
    if source == "linkedin":
        return (row.get("content") or "").strip() or None
    if source == "x":
        if row.get("is_reply") or row.get("is_repost_or_quote"):
            return None
        t = (row.get("text") or "").strip()
        return t if len(t) >= 25 else None
    if source.startswith("youtube"):
        return (row.get("transcript_text") or "").strip() or None
    return None


def main() -> None:
    print(f"{'source':<16} {'raw_lines':>10} {'after_extract':>14} {'>=50_words':>12}")
    grand_raw = grand_extract = grand_train = 0
    for src, fname in FILES.items():
        path = DATA / fname
        raw = extract = train_eligible = 0
        with path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                raw += 1
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                body = body_for(src, row)
                if body is None:
                    continue
                extract += 1
                if len(body.split()) >= 50:
                    train_eligible += 1
        print(f"{src:<16} {raw:>10} {extract:>14} {train_eligible:>12}")
        grand_raw += raw
        grand_extract += extract
        grand_train += train_eligible
    print(f"{'TOTAL':<16} {grand_raw:>10} {grand_extract:>14} {grand_train:>12}")
    print("\nNote: Spark also dropDuplicates on (source,title,url,text) before the training filter,")
    print("so the actual training row count will be <= the rightmost column.")


if __name__ == "__main__":
    main()
