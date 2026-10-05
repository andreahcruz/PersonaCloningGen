"""File checks for the provenance sidecars. They do not rebuild the manifests."""
import json
from pathlib import Path

from host_finetune.trace_provenance import trace_relabel_index, traces_for_source

ROOT = Path(__file__).resolve().parents[1]
PROV = ROOT / "artifacts" / "refactor" / "provenance"
BUILDER = ROOT / "host_finetune" / "build_provenance_sidecars.py"


def _rows(name: str) -> list[dict]:
    rows = []
    with (PROV / name).open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _coverage() -> dict:
    return json.loads((PROV / "coverage.json").read_text(encoding="utf-8"))


def test_builder_does_not_call_writers_or_services():
    text = BUILDER.read_text(encoding="utf-8")
    assert "clean_file(" not in text
    assert "clean_scraped(" not in text
    assert "AutoTokenizer" not in text
    assert "PersistentClient" not in text
    assert "import chromadb" not in text


def test_stage_counts_and_explicit_drops():
    coverage = _coverage()
    by_name = {row["transition"]: row for row in coverage["transitions"]}
    assert by_name["RAW → CLEANED"] == {
        **by_name["RAW → CLEANED"],
        "input": 32835,
        "output": 30370,
        "dropped": 2465,
        "expanded": 0,
        "ambiguous": 0,
        "unmapped": 0,
    }
    assert by_name["RELABEL train → RAG"]["input"] == 21377
    assert by_name["RELABEL train → RAG"]["output"] == 21193
    assert coverage["rag"]["gold_family_exclusions"] == 183
    assert coverage["rag"]["headline_exclusions"] == 1
    assert coverage["rag"]["headline_removed_ids"] == ["train_26148"]
    assert coverage["balance_renamed_model_row_ids"] == 0
    assert coverage["split_groups_crossing"] == 0
    assert coverage["replay_failures"] == []
    assert sum(coverage["cleaned_drop_reasons"].values()) == 2465
    sft = coverage["sft_drops"]
    assert (
        sft["cleaned_lines_dropped_short"]
        + sft.get("cleaned_lines_dropped_empty", 0)
        + sft["cleaned_lines_dropped_all_chunks"]
        == by_name["CLEANED → SFT"]["dropped"]
    )


def test_cleaned_rows_keep_raw_identity():
    raw = {(row["source_file"], row["raw_record_index"]): row for row in _rows("raw_manifest.preview.jsonl")}
    cleaned = _rows("cleaned_manifest.preview.jsonl")
    assert len(cleaned) == len(raw) == 32835
    kept = 0
    for row in cleaned:
        parent = raw[(row["source_file"], row["raw_record_index"])]
        assert row["source_id"] == parent["source_id"]
        assert row["raw_record_id"] == parent["raw_record_id"]
        assert row["raw_record_id"]
        if row["status"] == "kept":
            kept += 1
            assert row["cleaned_line"] is not None
            assert row["mapping_status"] == "EXACT_RAW_REPLAY"
        else:
            assert row["drop_reason"]
    assert kept == 30370


def test_balance_does_not_rename_model_rows_and_split_preserves_them():
    fit = {row["fit_row_index"]: row for row in _rows("fit_manifest.preview.jsonl")}
    balance = _rows("balanced_manifest.preview.jsonl")
    selected = []
    for row in balance:
        parent = fit[row["fit_row_index"]]
        assert row["model_row_id"] == parent["model_row_id"]
        assert row["document_id"] == parent["document_id"]
        assert row["source_id"] == parent["source_id"]
        assert row["raw_record_id"] == parent["raw_record_id"]
        if row["selected"]:
            selected.append(row)
    assert len(selected) == 31328
    split_rows = _rows("split_manifest.preview.jsonl")
    assert len(split_rows) == 31328
    groups = {}
    for row, parent in zip(split_rows, selected):
        assert row["model_row_id"] == parent["model_row_id"]
        assert row["raw_record_id"] == parent["raw_record_id"]
        groups.setdefault(row["group_id"], set()).add(row["split"])
    assert all(len(splits) == 1 for splits in groups.values())


def test_relabel_preserves_model_row_and_rag_does_not_rename_it():
    split_rows = _rows("split_manifest.preview.jsonl")
    relabel = _rows("relabel_manifest.preview.jsonl")
    assert len(relabel) == 26545
    priors = [row["legacy_prior_row_index"] for row in relabel]
    assert len(priors) == len(set(priors))
    for row in relabel:
        parent = split_rows[row["legacy_prior_row_index"]]
        assert row["model_row_id"] == parent["model_row_id"]
        assert row["source_id"] == parent["source_id"]
        assert row["raw_record_id"] == parent["raw_record_id"]
        assert row["group_id"] == parent["group_id"]
        assert row["rag_row_id"] == row["relabel_row_id"]
    rag = _rows("rag_manifest.preview.jsonl")
    assert len(rag) == 21377
    eligible = [row for row in rag if row["eligible"]]
    assert len(eligible) == 21193
    assert all(row["split"] == "train" for row in rag)
    assert all(row["raw_record_id"] and row["source_id"] for row in eligible)
    relabel_by_index = {row["legacy_exp004_row_index"]: row for row in relabel}
    for row in rag:
        parent = relabel_by_index[row["legacy_exp004_row_index"]]
        assert row["relabel_row_id"] == parent["relabel_row_id"]
        assert row["model_row_id"] == parent["model_row_id"]
        assert row["raw_record_id"] == parent["raw_record_id"]


def test_train_26148_traces_to_one_capture_and_is_excluded():
    trace = trace_relabel_index(26148, PROV)
    assert trace["legacy_experiment_row_id"] == "train_26148"
    assert trace["raw_record_id"]
    assert trace["source_id"]
    assert trace["document_id"]
    assert trace["model_row_id"]
    assert trace["relabel_row_id"]
    assert trace["rag_eligible"] is False
    assert trace["exclusion_reasons"] == ["headline_overlap"]
    assert "output" not in trace
    reverse = traces_for_source(trace["source_id"], PROV)
    assert trace["raw_record_id"] in {row["raw_record_id"] for row in reverse["captures"]}
    assert 26148 in {row["legacy_exp004_row_index"] for row in reverse["relabel_rows"]}
