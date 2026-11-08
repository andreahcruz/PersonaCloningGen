"""Controlled fixtures for the read-only TASK-002 audit."""
import json
import sqlite3

import pytest

from host_finetune.gold_partition_audit import (
    audit_selection, compare_index, development_questions, digest_json, read_index,
    literal_gold_matches, replay_training_loader, training_exposure,
    read_training_log,
    saved_training_arguments,
)
from host_finetune.split_groups import file_sha256


def fixture_rows():
    dataset = [{"output": text} for text in (
        "Recorded gold family body", "How to sell software to a CIO -- a practical guide https://example.com/x",
        "A sibling chunk without any headline words", "An unrelated complete blog about account handoffs",
        "A validation post", "A test post",
    )]
    assignments = [{"record_type": "meta"}]
    for i in range(len(dataset)):
        assignments.append({
            "row_index": i, "group_id": ("recorded", "headline", "headline", "clean", "valid", "test")[i],
            "split": "train" if i < 4 else ("validation" if i == 4 else "test"),
            "source_platform": "blog", "source_file": "blog.jsonl", "source_line": i + 1,
            "sft_role": "unchanged", "base_title": "Account transitions" if i == 3 else "",
        })
    dispositions = [{"reasons": ["gold_overlap_family"], "source_file": "blog.jsonl", "source_line": 1}]
    gold = [{"id": "gold_001", "topic": "How to sell software to a CIO — a practical guide"}]
    return dataset, assignments, dispositions, gold


def test_reuses_family_and_headline_exclusions_without_mutating_inputs():
    data, assignments, dispositions, gold = fixture_rows()
    before = digest_json([data, assignments, dispositions, gold])
    report, docs, blocked = audit_selection(data, assignments, dispositions, gold)
    assert blocked == {"recorded", "headline"}
    assert [d["id"] for d in docs] == ["train_3"]
    assert report["family_only_selection"]["excluded_gold"] == 1
    assert report["final_selection"]["excluded_gold"] == 3
    assert report["removed_after_headline_check"] == ["train_1", "train_2"]
    assert report["additional_overlap_families"][0]["hits"][0]["gold_id"] == "gold_001"
    assert report["remaining_selected_overlap_families"] == []
    assert digest_json([data, assignments, dispositions, gold]) == before
    assert "Recorded gold family body" not in json.dumps(report)


def test_near_duplicate_jaccard_blocks_entire_family():
    data, assignments, dispositions, _ = fixture_rows()
    data[1]["output"] = "alpha bravo charlie delta echo extra foxtrot golf hotel india juliet"
    gold = [{"id": "gold_001", "gold_reference": "alpha bravo charlie delta echo foxtrot golf hotel india juliet"}]
    report, docs, _ = audit_selection(data, assignments, dispositions, gold)
    assert report["additional_overlap_families"][0]["hits"][0]["jaccard"] >= 0.8
    assert [d["id"] for d in docs] == ["train_3"]


def test_alignment_errors_fail_closed():
    data, assignments, dispositions, gold = fixture_rows()
    assignments[2]["row_index"] = 99
    with pytest.raises(ValueError, match="row_index"):
        audit_selection(data, assignments, dispositions, gold)
    with pytest.raises(ValueError, match="assignment rows"):
        audit_selection(data[:-1], assignments, dispositions, gold)


def test_questions_reject_gold_siblings_heldout_and_duplicate_families():
    data, assignments, dispositions, gold = fixture_rows()
    _, docs, blocked = audit_selection(data, assignments, dispositions, gold)
    questions, rejected = development_questions(data, assignments, docs, blocked, gold, seeds=(
        (0, "A blocked question?"), (2, "A sibling question?"), (4, "A heldout question?"),
        (3, "How should an account handoff work?"), (3, "Another question?"), (99, "Missing?"),
    ))
    assert [q["id"] for q in questions] == ["dev_candidate_001"]
    assert questions[0]["split"] == "train"
    assert questions[0]["status"] == "PROPOSED"
    assert len(rejected) == 5
    assert "An unrelated complete blog" not in json.dumps(questions)
    questions, rejected = development_questions(data, assignments, docs, blocked, gold,
        seeds=((3, gold[0]["topic"]),))
    assert questions == []
    assert rejected[0]["reason"] == "gold lexical overlap"


def test_persisted_comparison_exposes_unexpected_overlap_and_metadata_errors():
    data, assignments, dispositions, gold = fixture_rows()
    _, docs, blocked = audit_selection(data, assignments, dispositions, gold)
    actual = [{**docs[0], "metadata": {**docs[0]["metadata"], "split": "test"}},
              {"id": "train_26148", "text": gold[0]["topic"], "metadata": {"group_id": "headline"}}]
    result = compare_index(docs, actual, blocked, gold)
    assert result["train_26148_present"]
    assert result["unexpected_ids"] == ["train_26148"]
    assert result["content_or_metadata_mismatch_ids"] == ["train_3"]
    assert result["blocked_family_ids"] == ["train_26148"]
    assert result["remaining_lexical_overlaps"][0]["id"] == "train_26148"
    assert compare_index(docs, [], blocked, gold)["missing_ids"] == ["train_3"]


def test_sqlite_reader_scopes_collection_and_preserves_bytes(tmp_path):
    db = tmp_path / "chroma.sqlite3"
    with sqlite3.connect(db) as conn:
        conn.executescript("""
            CREATE TABLE collections (id TEXT, name TEXT, schema_str TEXT);
            CREATE TABLE segments (id TEXT, collection TEXT, scope TEXT);
            CREATE TABLE embeddings (id INTEGER, embedding_id TEXT, segment_id TEXT);
            CREATE TABLE embedding_metadata (id INTEGER, key TEXT, string_value TEXT, int_value INTEGER);
            INSERT INTO collections VALUES ('c', 'lemkin_train_only', '{}'), ('other', 'lemkin_content', '{}');
            INSERT INTO segments VALUES ('s', 'c', 'METADATA'), ('other-s', 'other', 'METADATA');
            INSERT INTO embeddings VALUES (1, 'train_3', 's'), (2, 'train_26148', 'other-s');
            INSERT INTO embedding_metadata VALUES (1, 'chroma:document', 'private body', NULL),
                (1, 'split', 'train', NULL), (1, 'row_id', NULL, 3);
        """)
    before = file_sha256(db)
    docs, identity = read_index(db)
    assert [d["id"] for d in docs] == ["train_3"]
    assert docs[0]["metadata"]["row_id"] == 3
    assert docs[0]["metadata"]["group_id"] is None
    assert identity["metadata_segment_id"] == "s"
    assert file_sha256(db) == before


def test_missing_index_is_not_created(tmp_path):
    db = tmp_path / "missing.sqlite3"
    with pytest.raises(sqlite3.OperationalError):
        read_index(db)
    assert not db.exists()


def test_actual_loader_body_is_reused_without_importing_trainer(tmp_path):
    from host_finetune.gold_partition_audit import ROOT
    source = tmp_path / "input.jsonl"
    source.write_text('\n{"instruction":"a","output":"body"}\nmalformed\n', encoding="utf-8")
    before = file_sha256(source)
    rows = replay_training_loader(source, ROOT / "host_finetune/finetune.py")
    assert rows == [{"instruction": "a", "output": "body"}]
    assert file_sha256(source) == before


def test_literal_text_is_not_confused_with_normalized_or_family_matches():
    data = [{"instruction": "A question?", "output": "a question?"},
            {"instruction": "Something else", "output": "A question? plus context"}]
    records = [{"row_index": 0, "split": "train"}, {"row_index": 1, "split": "test"}]
    hits = literal_gold_matches(data, records, [{"id": "gold_001", "topic": "A question?"}])
    assert [(h["row_id"], h["dataset_field"], h["entire_field_equal"]) for h in hits] == [
        (0, "instruction", True), (1, "output", False)]
    assert all("A question?" not in json.dumps(h) for h in hits)


def test_exposure_keeps_validation_and_test_out_of_train_and_completion_uncertain():
    data, assignments, dispositions, gold = fixture_rows()
    from host_finetune.train_only_index import assignment_rows
    records = assignment_rows(assignments)
    _, _, blocked = audit_selection(data, assignments, dispositions, gold)
    blocked |= {"valid", "test"}
    report = training_exposure(data, records, blocked, gold,
                               {"data": {"processed_dataset_hash": "fixture"}},
                               {"epoch": 0.5, "global_step": 4}, "partial run")
    assert report["counts"]["train"]["gold_family_rows"] == 3
    held = {r["split"]: r for r in report["overlap_rows"] if r["split"] != "train"}
    assert not held["validation"]["included_in_train_dataset"]
    assert held["validation"]["included_in_validation_dataset"]
    assert not held["test"]["included_in_train_dataset"]
    assert not held["test"]["included_in_validation_dataset"]
    assert not report["completion_evidence_consistent"]
    assert "do not infer" in report["consumption_conclusion"]


@pytest.mark.parametrize("encoding", ["utf-16", "utf-8-sig", "utf-8"])
def test_log_decoding_preserves_completion_evidence(tmp_path, encoding):
    log = tmp_path / "train.log"
    text = "Num examples = 21,377\ntrain=21377 eval=2588\nSaving LoRA adapter -> adapter\n"
    log.write_bytes(text.encode(encoding))
    before = file_sha256(log)
    decoded = read_training_log(log)
    data, assignments, dispositions, gold = fixture_rows()
    from host_finetune.train_only_index import assignment_rows
    _, _, blocked = audit_selection(data, assignments, dispositions, gold)
    report = training_exposure(data, assignment_rows(assignments), blocked, gold,
                               {"data": {"processed_dataset_hash": "fixture"}},
                               {"epoch": 2.0, "global_step": 2674, "max_steps": 2674, "num_train_epochs": 2}, decoded)
    assert report["completion_evidence_consistent"]
    assert "strongly support" in report["consumption_conclusion"]
    assert "individual visits" in report["consumption_conclusion"]
    assert file_sha256(log) == before


def test_exact_benchmark_user_message_is_distinct_from_topic_substring():
    data, assignments, dispositions, gold = fixture_rows()
    data[3]["instruction"] = "Write a blog post in the style of Jason Lemkin about: " + gold[0]["topic"]
    from host_finetune.train_only_index import assignment_rows
    _, _, blocked = audit_selection(data, assignments, dispositions, gold)
    manifest = {"data": {"processed_dataset_hash": "fixture"}}
    report = training_exposure(data, assignment_rows(assignments), blocked, gold, manifest, {}, "")
    assert report["counts"]["train"]["exact_retrieval_off_user_message_rows"] == 1
    data[3]["input"] = "Extra context not in the benchmark prompt"
    report = training_exposure(data, assignment_rows(assignments), blocked, gold, manifest, {}, "")
    assert report["counts"]["train"]["literal_topic_in_instruction_rows"] == 1
    assert report["counts"]["train"]["exact_retrieval_off_user_message_rows"] == 0


def test_saved_arguments_are_inspected_without_unpickling(tmp_path, monkeypatch):
    import pickle
    import zipfile
    path = tmp_path / "training_args.bin"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("archive/data.pkl", pickle.dumps({
            "num_train_epochs": 2, "max_steps": -1, "dataloader_drop_last": False,
            "train_sampling_strategy": "random", "packing": False,
            "max_length": 512, "completion_only_loss": None, "unrelated": "ignored",
        }, protocol=2))
    def forbidden(*args, **kwargs):
        raise AssertionError("pickle execution forbidden")
    monkeypatch.setattr(pickle, "loads", forbidden)
    monkeypatch.setattr(pickle, "load", forbidden)
    values = saved_training_arguments(path)
    assert values["num_train_epochs"] == 2
    assert values["max_steps"] == -1
    assert values["train_sampling_strategy"] == "random"
    assert values["dataloader_drop_last"] is False
    assert values["completion_only_loss"] is None
    assert "unrelated" not in values
