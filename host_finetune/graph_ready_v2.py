"""Graph-ready v2 derivative. Reads v1 documents and writes a new directory.

v1 files are not opened for writing. Quote labels change the indexing text
only when a third-party quotation is detected. Same-source residual text stays
out of the index text, because that residual can mix the author's lead with
an embedded quotation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

from host_finetune.graph_ready import (
    DEFAULT_CLEANED,
    POLICY_VERSION,
    _sha256_text,
    _words,
    lead_and_interior_gaps,
    load_jsonl,
)
from host_finetune.quote_attribution import index_text_with_quotes, quote_attribution
from host_finetune.train_only_index import DEFAULT_DATASET

V2_POLICY = "graph-ready-v2"
DEFAULT_V1 = Path("artifacts/lightrag/model_f/graph_ready")
DEFAULT_V2 = Path("artifacts/lightrag/model_f/graph_ready_v2")


def marked_chunk_text(chunks: list[str], interior_gaps: list[int]) -> str | None:
    """Separate sections when the cleaned source has words between them.

    The omitted words are not copied into the index. A zero gap leaves the
    caller on the existing joined text.
    """
    if len(chunks) < 2 or len(interior_gaps) != len(chunks) - 1:
        return None
    if not any(gap > 0 for gap in interior_gaps):
        return None
    parts = []
    for index, chunk in enumerate(chunks):
        if index and interior_gaps[index - 1] > 0:
            parts.append(
                "[Omitted same-source text between these sections: "
                f"{interior_gaps[index - 1]} words.]"
            )
        piece = (chunk or "").strip()
        if piece:
            parts.append(piece)
    return "\n\n".join(parts)


def _v2_id(parent_id: str, index_text: str) -> str:
    material = "\n".join([V2_POLICY, parent_id, index_text])
    return "gd_" + hashlib.sha256(material.encode("utf-8")).hexdigest()[:24]


def annotate_document(document: dict) -> dict:
    """Copy one v1 document and apply quote attribution. Does not drop the row."""
    source_text = document.get("text") or ""
    decision = quote_attribution(
        source_text,
        document.get("attribution_status") or "",
        document.get("author_position") or "",
    )
    index_text = index_text_with_quotes(source_text, decision["quotes"])
    graph_id = document["graph_document_id"]
    if index_text != source_text:
        graph_id = _v2_id(document["graph_document_id"], index_text)
    flags = list(document.get("flags") or [])
    if any(str(flag).startswith("cleaned_residual_words_") or flag == "cleaned_residual_exceeds_limit" for flag in flags):
        if "unindexed_same_source_residual" not in flags:
            flags.append("unindexed_same_source_residual")
    if decision["quotes"] and "contains_third_party_quotation" not in flags:
        flags.append("contains_third_party_quotation")
    derived = dict(document)
    derived.update(
        {
            "graph_document_id": graph_id,
            "parent_graph_document_id": document["graph_document_id"],
            "parent_policy_version": document.get("policy_version") or POLICY_VERSION,
            "policy_version": V2_POLICY,
            "source_text": source_text,
            "source_text_sha256": _sha256_text(source_text),
            "text": index_text,
            "text_sha256": _sha256_text(index_text),
            "attribution_status": decision["attribution_status"],
            "author_position": decision["author_position"],
            "quoted_speakers": decision["quoted_speakers"],
            "quoted_spans": decision["quotes"],
            "quoted_word_share": decision["quoted_word_share"],
            "flags": flags,
            "word_count": len(_words(index_text)),
        }
    )
    return derived


def difference_summary(originals: list[dict], derived: list[dict]) -> dict:
    text_changed = 0
    attribution_changed = 0
    speakers = Counter()
    for before, after in zip(originals, derived):
        if before.get("text") != after.get("source_text"):
            raise ValueError("v2 source_text drifted from v1 text")
        if after["text"] != after["source_text"]:
            text_changed += 1
        if after["attribution_status"] != before.get("attribution_status"):
            attribution_changed += 1
        for speaker in after.get("quoted_speakers") or []:
            speakers[speaker] += 1
    positions = Counter(row["author_position"] for row in derived)
    statuses = Counter(row["attribution_status"] for row in derived)
    return {
        "policy_version": V2_POLICY,
        "documents": len(derived),
        "text_changed": text_changed,
        "attribution_changed": attribution_changed,
        "third_party_quotation": statuses.get("third_party_quotation", 0),
        "documents_with_any_quote": sum(1 for row in derived if row.get("quoted_spans")),
        "author_position": dict(positions),
        "attribution_status": dict(statuses),
        "top_quoted_speakers": speakers.most_common(20),
    }


def _finalize_index_text(document: dict, body: str) -> dict:
    """Re-apply the quote prefix after the body changes, and refresh the id."""
    decision = quote_attribution(
        document["source_text"],
        document.get("parent_attribution_status") or document.get("attribution_status") or "",
        document.get("parent_author_position") or document.get("author_position") or "",
    )
    # parent fields are stored below. Fall back to the current decision when
    # the caller has not set them yet.
    index_text = index_text_with_quotes(body, decision["quotes"])
    parent = document["parent_graph_document_id"]
    document["text"] = index_text
    document["text_sha256"] = _sha256_text(index_text)
    document["graph_document_id"] = (
        parent if index_text == document["source_text"] else _v2_id(parent, index_text)
    )
    document["word_count"] = len(_words(index_text))
    document["attribution_status"] = decision["attribution_status"]
    document["author_position"] = decision["author_position"]
    document["quoted_speakers"] = decision["quoted_speakers"]
    document["quoted_spans"] = decision["quotes"]
    document["quoted_word_share"] = decision["quoted_word_share"]
    return document


def apply_interior_markers(
    originals: list[dict],
    derived: list[dict],
    dataset_path: Path = DEFAULT_DATASET,
    cleaned_root: Path = DEFAULT_CLEANED,
) -> dict:
    """Mark interior gaps. Leaves v1 text in ``source_text``."""
    by_parent = {row["parent_graph_document_id"]: row for row in derived}
    targets = [
        row
        for row in originals
        if row.get("reconstruction_method") == "ordered_eligible_chunks" and row.get("included", True)
    ]
    needed = {}
    for row in targets:
        for row_id in row.get("experiment_row_ids") or []:
            needed[int(str(row_id).split("_", 1)[1])] = row_id
    outputs = {}
    if needed:
        last = max(needed)
        with dataset_path.open(encoding="utf-8") as handle:
            for index, line in enumerate(handle):
                if index in needed and line.strip():
                    outputs[index] = (json.loads(line).get("output") or "").strip()
                if index >= last:
                    break
    by_file: dict[str, set[int]] = {}
    for row in targets:
        by_file.setdefault(row.get("source_file") or "", set()).add(int(row.get("source_line") or 0))
    cleaned = {}
    for name, lines in by_file.items():
        path = cleaned_root / name
        if not name or not path.is_file() or not lines:
            continue
        last = max(lines)
        with path.open(encoding="utf-8") as handle:
            for number, raw in enumerate(handle, start=1):
                if number in lines and raw.strip():
                    record = json.loads(raw)
                    cleaned[(name, number)] = (
                        record.get("content")
                        or record.get("text")
                        or record.get("transcript_text")
                        or ""
                    )
                if number >= last:
                    break
    missing = 0
    marked = 0
    lead_positive = 0
    interior_positive = 0
    max_interior = 0
    max_example = None
    rubrik = []
    for row in targets:
        chunks = []
        for row_id in row.get("experiment_row_ids") or []:
            index = int(str(row_id).split("_", 1)[1])
            chunks.append(outputs.get(index, ""))
        body = cleaned.get((row.get("source_file"), int(row.get("source_line") or 0)))
        measured = lead_and_interior_gaps(body or "", chunks) if body else None
        derived_row = by_parent[row["graph_document_id"]]
        if measured is None:
            missing += 1
            derived_row["join_gap"] = None
            continue
        derived_row["join_gap"] = measured
        if measured["lead_words"] > 0:
            lead_positive += 1
        if any(gap > 0 for gap in measured["interior_gap_words"]):
            interior_positive += 1
            largest = max(measured["interior_gap_words"])
            if largest > max_interior:
                max_interior = largest
                max_example = {
                    "graph_document_id": row["graph_document_id"],
                    "source_file": row.get("source_file"),
                    "source_line": row.get("source_line"),
                    "interior_gap_words": measured["interior_gap_words"],
                    "lead_words": measured["lead_words"],
                }
            marked_body = marked_chunk_text(chunks, measured["interior_gap_words"])
            if marked_body:
                derived_row["parent_attribution_status"] = row.get("attribution_status")
                derived_row["parent_author_position"] = row.get("author_position")
                _finalize_index_text(derived_row, marked_body)
                flags = derived_row.setdefault("flags", [])
                if "interior_section_gap" not in flags:
                    flags.append("interior_section_gap")
                marked += 1
        if row.get("source_line") == 1462 and row.get("source_file") == "jasonlemkin_blog.jsonl":
            rubrik.append(
                {
                    "graph_document_id": row["graph_document_id"],
                    "source_file": row["source_file"],
                    "source_line": row["source_line"],
                    "member_count": row.get("member_count"),
                    **measured,
                    "marker_applied": "interior_section_gap" in (derived_row.get("flags") or []),
                }
            )
    return {
        "ordered_joins": len(targets),
        "gap_measurement_missing": missing,
        "joins_with_positive_lead": lead_positive,
        "joins_with_positive_interior_gap": interior_positive,
        "documents_with_section_marker": marked,
        "max_interior_gap_words": max_interior,
        "largest_interior_gap": max_example,
        "rubrik_line_1462": rubrik,
    }


def write_derivative(v1_dir: Path, v2_dir: Path, mark_gaps: bool = True) -> dict:
    if v1_dir.resolve() == v2_dir.resolve():
        raise SystemExit("refusing to write the v2 derivative onto the v1 directory")
    documents = load_jsonl(v1_dir / "documents.jsonl")
    derived = [annotate_document(row) for row in documents]
    diagnosis = None
    if mark_gaps:
        diagnosis = apply_interior_markers(documents, derived)
    summary = difference_summary(documents, derived)
    if diagnosis is not None:
        summary["join_diagnosis"] = diagnosis
    v2_dir.mkdir(parents=True, exist_ok=True)
    out = v2_dir / "documents.jsonl"
    out.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in derived),
        encoding="utf-8",
    )
    summary["documents_sha256"] = hashlib.sha256(out.read_bytes()).hexdigest()
    summary["v1_directory"] = str(v1_dir)
    summary["v2_directory"] = str(v2_dir)
    (v2_dir / "diff_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v1", type=Path, default=DEFAULT_V1)
    parser.add_argument("--out", type=Path, default=DEFAULT_V2)
    args = parser.parse_args()
    summary = write_derivative(args.v1, args.out)
    printable = {
        key: summary[key]
        for key in (
            "documents",
            "text_changed",
            "attribution_changed",
            "third_party_quotation",
            "documents_with_any_quote",
        )
    }
    printable["join_diagnosis"] = summary.get("join_diagnosis")
    print(json.dumps(printable, indent=2))


if __name__ == "__main__":
    main()
