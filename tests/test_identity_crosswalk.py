"""Crosswalk alignment and coverage. The preview file is read, not rebuilt, here."""
from pathlib import Path

from host_finetune.build_identity_crosswalk import (
    AMBIGUOUS,
    CONTENT_FALLBACK_STATUS,
    EXACT_NATURAL_KEY,
    UNMAPPED,
    align_kept_rows,
    coverage_report,
)
from host_finetune.canonical_ids import experiment_row_id

ROOT = Path(__file__).resolve().parents[1]
PREVIEW = ROOT / "artifacts" / "refactor" / "canonical_identity_crosswalk.preview.jsonl"
COVERAGE = ROOT / "artifacts" / "refactor" / "canonical_identity_crosswalk.coverage.json"
BUILDER = ROOT / "host_finetune" / "build_identity_crosswalk.py"


def test_builder_does_not_call_cleaning_writers():
    text = BUILDER.read_text(encoding="utf-8")
    assert "clean_file(" not in text
    assert "clean_scraped(" not in text
    assert "PersistentClient" not in text
    assert "import chromadb" not in text


def test_count_mismatch_does_not_zip_align():
    result = align_kept_rows(
        [(0, {"url": "a"}, {"content": "a"})],
        [],
        text_field="content",
        locator=lambda row: row.get("url"),
    )
    assert result["ok"] is False
    assert result["reason"] == "count_mismatch"
    assert result["by_line"] == {}


def test_text_mismatch_rejects_the_whole_file():
    kept = [
        (0, {"url": "a"}, {"content": "a"}),
        (1, {"url": "b"}, {"content": "b"}),
    ]
    cleaned = [
        (1, {"url": "a", "content": "DIFFERENT"}),
        (2, {"url": "b", "content": "b"}),
    ]
    result = align_kept_rows(kept, cleaned, text_field="content", locator=lambda row: row["url"])
    assert result["ok"] is False
    assert result["by_line"] == {}


def test_exact_replay_keeps_cleaned_line_numbers():
    kept = [
        (4, {"url": "a"}, {"content": "a"}),
        (9, {"url": "b"}, {"content": "b"}),
    ]
    cleaned = [
        (1, {"url": "a", "content": "a"}),
        (2, {"url": "b", "content": "b"}),
    ]
    result = align_kept_rows(kept, cleaned, text_field="content", locator=lambda row: row["url"])
    assert result["ok"] is True
    assert result["by_line"][1]["raw_index"] == 4
    assert result["by_line"][2]["raw_index"] == 9


def test_coverage_accounts_for_every_row():
    rows = [
        {
            "medium": "blog",
            "mapping_status": EXACT_NATURAL_KEY,
            "mapping_method": "raw_cleaner_replay",
            "model_row_id": "m1",
            "relabel_row_id": "r1",
            "raw_record_id": "raw1",
            "source_id": "s1",
        },
        {
            "medium": "linkedin",
            "mapping_status": CONTENT_FALLBACK_STATUS,
            "mapping_method": "raw_cleaner_replay",
            "model_row_id": "m2",
            "relabel_row_id": "r2",
            "raw_record_id": "raw2",
            "source_id": "s2",
        },
        {
            "medium": "x",
            "mapping_status": AMBIGUOUS,
            "mapping_method": "cleaner_replay_count_mismatch",
            "model_row_id": None,
            "relabel_row_id": None,
            "raw_record_id": None,
            "source_id": None,
        },
        {
            "medium": "youtube_jason",
            "mapping_status": UNMAPPED,
            "mapping_method": "cleaned_line_absent",
            "model_row_id": None,
            "relabel_row_id": None,
            "raw_record_id": None,
            "source_id": None,
        },
    ]
    report = coverage_report(rows, raw_pairs=[], kept_pairs=[])
    assert report["status_sum_equals_rows"] is True
    assert report["totals"]["rows"] == 4
    assert report["totals"]["natural_source_id"] == 1
    assert report["totals"]["fallback_source_identity"] == 1
    assert report["totals"]["exact_raw_replay"] == 2
    assert report["totals"]["ambiguous"] == 1
    assert report["totals"]["unmapped"] == 1
    assert report["by_medium"]["linkedin"]["fallback_source_identity"] == 1


def test_preview_covers_every_relabel_row_without_hiding_status():
    assert PREVIEW.is_file(), "preview crosswalk was not generated"
    assert COVERAGE.is_file()
    import json

    rows = []
    with PREVIEW.open(encoding="utf-8", newline="\n") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    report = json.loads(COVERAGE.read_text(encoding="utf-8"))
    assert report["canonical_production_artifact"] is False
    assert report["total_relabel_rows"] == 26545
    assert len(rows) == 26545
    assert report["status_sum_equals_rows"] is True
    indexes = [row["legacy_exp004_row_index"] for row in rows]
    assert indexes == list(range(26545))
    held = rows[26148]
    assert held["legacy_experiment_row_id"] == experiment_row_id(26148) == "train_26148"
    assert "eligible" not in held
    assert "exclusion_reasons" not in held
    statuses = {row["mapping_status"] for row in rows}
    assert statuses <= {EXACT_NATURAL_KEY, CONTENT_FALLBACK_STATUS, AMBIGUOUS, UNMAPPED}
    mapped = [row for row in rows if row["relabel_row_id"]]
    assert len({row["relabel_row_id"] for row in mapped}) == len(mapped)
    assert len({row["model_row_id"] for row in mapped}) == len(mapped)
    totals = report["totals"]
    assert (
        totals["natural_source_id"]
        + totals["fallback_source_identity"]
        + totals["ambiguous"]
        + totals["unmapped"]
        == 26545
    )
    assert totals["ambiguous"] == 0
    assert totals["unmapped"] == 0
    assert report["by_medium"]["youtube_saastr"]["rows"] == 0
    assert report["raw_records_before_cleaning"]["source_id_with_multiple_raw_records"] == 18
    assert all(row["medium"] == "youtube_jason" for row in report["repeated_raw_captures"])
