"""The development retrieval sheet stays ungraded."""

from __future__ import annotations

import json
from pathlib import Path

from host_finetune.eval_bench.dev_benchmark import GRADE_SCALE, PHENOMENA, build_benchmark
from host_finetune.model_f_context.registry import validate_registry


def test_registry_keeps_descriptions_and_model_e_apart():
    checked = validate_registry()
    assert checked["descriptions_id"] == "F-DESC"
    assert checked["multihop_id"] == "F-HOP"
    assert checked["model_e_id"] == "E-DENSE"
    assert checked["off_id"] == "F-OFF"
    assert "D" not in checked["ids"]


def test_development_benchmark_has_no_grades(tmp_path: Path):
    built = build_benchmark(Path("."))
    rows = built["annotator"]
    assert 30 <= len(rows) <= 50
    assert {row["phenomenon"] for row in rows} == set(PHENOMENA)
    grades = [candidate["grade"] for row in rows for candidate in row["candidates"]]
    assert grades
    assert set(grades) == {None}
    brex = next(row for row in rows if row["question_id"] == "q_brex_price")
    origins = {
        item["origin"]
        for candidate in brex["candidates"]
        for item in candidate["provenance"]
    }
    assert "saved_retrieval" in origins
    assert "author_preselection" in origins
    assert all(item.get("is_grade") is False for candidate in brex["candidates"] for item in candidate["provenance"])
    olympics = next(row for row in rows if row["question_id"] == "q_olympics_ceremony")
    assert olympics["candidates"]
    assert all(candidate["grade"] is None for candidate in olympics["candidates"])
    empty = [row for row in rows if not row["candidates"]]
    assert empty
    assert all(row["missing_relevant_source_ids"] == [] for row in rows)
    for row in rows:
        assert "review_note" not in row
    blob = json.dumps(rows)
    assert "5.15" not in blob
    assert [item["grade"] for item in GRADE_SCALE] == [0, 1, 2, 3]
