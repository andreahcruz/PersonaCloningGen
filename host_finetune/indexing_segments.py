"""Split graph-ready text on omission markers.

The marker is the one written by graph-ready v2. Segments do not include the
marker. A document without a marker keeps its original id. A document with a
marker gets new segment ids, so this representation is a different corpus
from ``storage_expand_v1`` and must not be inserted into that graph.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

from host_finetune.lightrag_backend import OMISSION_MARKER

SEGMENT_POLICY = "graph-ready-v2-segments"
_MARKER = re.compile(
    r"\[Omitted same-source text between these sections: (\d+) words\.\]"
)


def split_omission_segments(document: dict) -> list[dict]:
    """Return one or more segment records for a graph-ready document."""
    text = document.get("text") or ""
    parent_id = document.get("graph_document_id") or ""
    if document.get("split") not in {None, "train"}:
        raise ValueError(f"{parent_id} is not train-only")
    if not parent_id:
        raise ValueError("segment source is missing graph_document_id")
    if OMISSION_MARKER not in text:
        return [_segment(document, text, 0, 0, 0, parent_id, changed=False)]
    parts = _MARKER.split(text)
    # split with one group yields [before, words, after, words, after...]
    segments = []
    omitted_before = 0
    cursor = 0
    index = 0
    body = parts[0]
    segments.append(_piece(document, body, index, omitted_before, cursor))
    cursor += len(body)
    index += 1
    group = 1
    while group < len(parts):
        omitted_words = int(parts[group])
        marker = f"[Omitted same-source text between these sections: {omitted_words} words.]"
        cursor += len(marker)
        omitted_before += omitted_words
        body = parts[group + 1] if group + 1 < len(parts) else ""
        segments.append(_piece(document, body, index, omitted_before, cursor))
        cursor += len(body)
        index += 1
        group += 2
    return [item for item in segments if item is not None]


def _piece(document: dict, body: str, index: int, omitted_before: int, start: int) -> dict | None:
    text = body.strip()
    if not text:
        return None
    # The strip can move the span. Search the stripped text in the parent.
    parent = document.get("text") or ""
    found = parent.find(text)
    segment_id = _segment_id(document["graph_document_id"], index, text)
    return _segment(
        document,
        text,
        index,
        omitted_before,
        found if found >= 0 else start,
        segment_id,
        changed=True,
    )


def _segment(
    document: dict,
    text: str,
    index: int,
    omitted_words_before: int,
    char_start: int,
    segment_id: str,
    *,
    changed: bool,
) -> dict:
    parent_id = document["graph_document_id"]
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    if OMISSION_MARKER in text:
        raise ValueError(f"segment {segment_id} still contains an omission marker")
    parent_text = document.get("text") or ""
    if text not in parent_text:
        raise ValueError(f"segment {segment_id} is not a verbatim span of {parent_id}")
    return {
        "graph_document_id": segment_id,
        "segment_index": index,
        "segment_policy": SEGMENT_POLICY,
        "parent_graph_document_id": document.get("parent_graph_document_id") or parent_id,
        "source_graph_document_id": parent_id,
        "representation_changed": changed,
        "text": text,
        "text_sha256": digest,
        "char_start": char_start,
        "omitted_words_before": omitted_words_before,
        "split": document.get("split") or "train",
        "medium": document.get("medium"),
        "attribution_status": document.get("attribution_status"),
        "author_position": document.get("author_position"),
        "source_file": document.get("source_file"),
        "source_line": document.get("source_line"),
        "word_count": len(text.split()),
        "included": True,
    }


def _segment_id(parent_id: str, index: int, text: str) -> str:
    material = "\n".join([SEGMENT_POLICY, parent_id, str(index), text])
    return "gs_" + hashlib.sha256(material.encode("utf-8")).hexdigest()[:24]


def write_segment_corpus(source: Path, destination: Path) -> dict:
    """Write a new JSONL corpus. Refuses to overwrite graph-ready v2."""
    source = Path(source)
    destination = Path(destination)
    if source.resolve() == destination.resolve():
        raise SystemExit("refusing to rewrite the source corpus in place")
    banned = {"graph_ready_v2", "graph_ready", "storage_expand_v1", "storage_pilot160"}
    if destination.name in banned or source.name in banned and destination.resolve() == source.resolve():
        raise SystemExit(f"refusing protected path: {destination}")
    if destination.name in {"graph_ready_v2", "graph_ready", "storage_expand_v1", "storage_pilot160"}:
        raise SystemExit(f"refusing protected path: {destination}")
    destination.mkdir(parents=True, exist_ok=True)
    target = destination / "documents.jsonl"
    if target.exists():
        raise SystemExit(f"refusing to overwrite {target}")
    documents = 0
    segments = 0
    marked_documents = 0
    changed_segments = 0
    with source.open(encoding="utf-8") as incoming, target.open("w", encoding="utf-8") as outgoing:
        for line in incoming:
            if not line.strip():
                continue
            document = json.loads(line)
            rows = split_omission_segments(document)
            documents += 1
            if any(row["representation_changed"] for row in rows):
                marked_documents += 1
            for row in rows:
                if row["representation_changed"]:
                    changed_segments += 1
                segments += 1
                outgoing.write(json.dumps(row, ensure_ascii=False) + "\n")
    (destination / "policy.json").write_text(
        json.dumps(
            {
                "policy": SEGMENT_POLICY,
                "source_documents": documents,
                "documents_kept_whole": documents - marked_documents,
                "marked_documents": marked_documents,
                "segments": segments,
                "segments_from_marked_documents": changed_segments,
                "note": (
                    "Segment ids are not the graph-ready v2 ids when a marker was split. "
                    "Do not insert this file into storage_expand_v1."
                ),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return {
        "policy": SEGMENT_POLICY,
        "source_documents": documents,
        "documents_kept_whole": documents - marked_documents,
        "marked_documents": marked_documents,
        "segments": segments,
        "changed_segments": changed_segments,
        "destination": str(target),
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    args = parser.parse_args(argv)
    print(json.dumps(write_segment_corpus(args.source, args.destination), indent=2))


if __name__ == "__main__":
    main()
