"""Recover deterministically repairable relations from a LightRAG extraction cache.

The tool reads a snapshot. Dry-run writes only the output directory. It does
not call a model, does not mark documents completed, and does not edit the
snapshot cache or graph.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

from host_finetune.graph_relationships import (
    _entity_pattern,
    _entity_tokens,
    _keywords,
    _support,
    _verbatim_clause,
    _windows,
)
from host_finetune.indexing_extraction_audit import (
    POLICY_VERSION,
    normalize_relation_record,
    record_fields,
    split_output_lines,
)

LIVE_STORE_NAME = "storage_expand_v1"


def recovery_id(chunk_id: str, source: str, target: str, description: str) -> str:
    pair = "\t".join(sorted((source.casefold(), target.casefold())))
    payload = f"{POLICY_VERSION}\n{chunk_id}\n{pair}\n{description.strip()}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def classify_evidence(source: str, target: str, keywords: str, description: str, chunk_text: str) -> str:
    """Use the same window rule as source-verified relationships.

    Both names in the chunk are not enough. A candidate is supported only when
    a multi-word keyword, a verbatim description clause, or at least three
    description words occur in one window that contains both endpoints.
    """
    subject = _entity_pattern(source)
    object_ = _entity_pattern(target)
    if subject is None or object_ is None:
        return "rejected_invalid_endpoint"
    text = chunk_text or ""
    if subject.search(text) is None or object_.search(text) is None:
        return "rejected_endpoint_absent"
    spans, _omission = _windows(text, subject, object_)
    if not spans:
        return "needs_review"
    banned = _entity_tokens(source) | _entity_tokens(target)
    words = _keywords(keywords)
    for span in spans:
        keyword_hits, description_hits = _support(span, words, description, banned)
        verbatim = _verbatim_clause(description, span)
        multi = [word for word in keyword_hits if len(word.split()) >= 2]
        if multi or verbatim or len(description_hits) >= 3:
            return "supported"
    return "needs_review"


def _document_id(chunk_id: str) -> str:
    marker = "-chunk-"
    if marker in chunk_id:
        return chunk_id.split(marker, 1)[0]
    return chunk_id


def iter_cache_entries(cache: dict) -> list[dict]:
    rows = []
    for row in cache.values():
        if not isinstance(row, dict) or row.get("cache_type") != "extract":
            continue
        rows.append(row)
    rows.sort(key=lambda row: (str(row.get("chunk_id") or ""), int(row.get("create_time") or 0)))
    return rows


def graph_pairs(graph_path: Path | None) -> set[tuple[str, str]]:
    if graph_path is None or not graph_path.is_file():
        return set()
    from host_finetune.graph_relationships import read_graphml_relationships

    found = set()
    for edge in read_graphml_relationships(graph_path):
        source = str(edge.get("src_id") or "").casefold()
        target = str(edge.get("tgt_id") or "").casefold()
        if source and target:
            found.add(tuple(sorted((source, target))))
    return found


def build_candidates(
    entries: list[dict],
    chunk_text: dict[str, str],
    existing_pairs: set[tuple[str, str]],
) -> list[dict]:
    """One row per repaired identity. Gleaning repeats collapse to the earliest row."""
    chosen: dict[str, dict] = {}
    for entry in entries:
        chunk_id = str(entry.get("chunk_id") or "")
        body = chunk_text.get(chunk_id, "")
        for line in split_output_lines(entry.get("return") or ""):
            fields = record_fields(line)
            if not fields or "relation" not in fields[0].casefold():
                continue
            rewritten, label = normalize_relation_record(line.strip())
            if label not in {"repaired_schema_label", "repaired_comma_packed"}:
                continue
            repaired = record_fields(rewritten)
            source, target = repaired[1], repaired[2]
            keywords, description = repaired[3], repaired[4]
            identity = recovery_id(chunk_id, source, target, description)
            if identity in chosen:
                continue
            evidence = classify_evidence(source, target, keywords, description, body)
            pair = tuple(sorted((source.casefold(), target.casefold())))
            already = pair in existing_pairs
            if evidence == "supported" and already:
                action = "already_in_graph"
            elif evidence == "supported":
                action = "insert_candidate"
            elif evidence.startswith("rejected"):
                action = evidence
            else:
                action = "needs_review"
            chosen[identity] = {
                "recovery_id": identity,
                "policy": POLICY_VERSION,
                "chunk_id": chunk_id,
                "document_id": _document_id(chunk_id),
                "repair": label,
                "src_id": source,
                "tgt_id": target,
                "keywords": keywords,
                "description": description,
                "file_path": chunk_id,
                "evidence": evidence,
                "already_in_graph": already,
                "action": action,
                "changes_document_status": False,
            }
    return [chosen[key] for key in sorted(chosen)]


def summarize_candidates(rows: list[dict]) -> dict:
    counts: dict[str, int] = {}
    for row in rows:
        counts[row["action"]] = counts.get(row["action"], 0) + 1
        counts[row["repair"]] = counts.get(row["repair"], 0) + 1
    return {
        "policy": POLICY_VERSION,
        "candidates": len(rows),
        "insert_candidate": counts.get("insert_candidate", 0),
        "needs_review": counts.get("needs_review", 0),
        "already_in_graph": counts.get("already_in_graph", 0),
        "rejected_endpoint_absent": counts.get("rejected_endpoint_absent", 0),
        "rejected_invalid_endpoint": counts.get("rejected_invalid_endpoint", 0),
        "repaired_schema_label": counts.get("repaired_schema_label", 0),
        "repaired_comma_packed": counts.get("repaired_comma_packed", 0),
        "by_action": {key: value for key, value in sorted(counts.items()) if key in {
            "insert_candidate", "needs_review", "already_in_graph",
            "rejected_endpoint_absent", "rejected_invalid_endpoint",
        }},
    }


def _ledger_path(output: Path) -> Path:
    return output / "recovery_ledger.jsonl"


def write_dry_run(rows: list[dict], output: Path, summary: dict) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    ledger = _ledger_path(output)
    body = "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows)
    previous = ledger.read_text(encoding="utf-8") if ledger.is_file() else None
    ledger.write_text(body, encoding="utf-8")
    summary = dict(summary)
    summary["idempotent_replay"] = previous == body
    summary["ledger_sha256"] = hashlib.sha256(body.encode("utf-8")).hexdigest()
    (output / "recovery_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def assert_safe_paths(snapshot: Path, output: Path, apply: bool) -> None:
    snapshot = snapshot.resolve()
    output = output.resolve()
    if output == snapshot or snapshot in output.parents:
        raise SystemExit("recovery output must be outside the snapshot directory")
    if apply and snapshot.name == LIVE_STORE_NAME:
        raise SystemExit("refusing to apply repairs to storage_expand_v1; use a clone")
    if apply and (snapshot / "indexing.lock").is_file():
        raise SystemExit("refusing to apply while the snapshot holds indexing.lock")


def load_snapshot(snapshot: Path) -> tuple[list[dict], dict[str, str], set[tuple[str, str]]]:
    cache_path = snapshot / "kv_store_llm_response_cache.json"
    chunk_path = snapshot / "kv_store_text_chunks.json"
    if not cache_path.is_file():
        raise SystemExit(f"no extraction cache at {cache_path}")
    cache = json.loads(cache_path.read_text(encoding="utf-8"))
    chunks = json.loads(chunk_path.read_text(encoding="utf-8")) if chunk_path.is_file() else {}
    text = {}
    for key, row in chunks.items():
        if isinstance(row, dict):
            text[key] = row.get("content") or ""
    pairs = graph_pairs(snapshot / "graph_chunk_entity_relation.graphml")
    return iter_cache_entries(cache), text, pairs


def run_recovery(snapshot: Path, output: Path, *, apply: bool, authorize: bool) -> dict:
    assert_safe_paths(snapshot, output, apply)
    if apply:
        if not authorize:
            raise SystemExit(
                "historical insertion is not authorized. Review the dry-run ledger, "
                "snapshot a clone, and pass --authorize. This command will not embed or write the live store."
            )
        raise SystemExit(
            "apply is gated until an embedding pass is separately authorized. "
            "The dry-run ledger is the repair input. Do not point --snapshot at storage_expand_v1."
        )
    before = (snapshot / "kv_store_llm_response_cache.json").read_bytes()
    entries, text, pairs = load_snapshot(snapshot)
    rows = build_candidates(entries, text, pairs)
    summary = summarize_candidates(rows)
    summary["snapshot"] = str(snapshot)
    summary["mode"] = "dry-run"
    written = write_dry_run(rows, output, summary)
    after = (snapshot / "kv_store_llm_response_cache.json").read_bytes()
    if before != after:
        raise SystemExit("recovery changed the snapshot cache; that is a defect")
    written["cache_unchanged"] = True
    return written


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True, help="Read-only graph storage directory.")
    parser.add_argument("--output", type=Path, required=True, help="Directory for the recovery ledger. Not inside the snapshot.")
    parser.add_argument("--dry-run", action="store_true", help="Write the ledger and do not change the snapshot.")
    parser.add_argument("--apply", action="store_true", help="Reserved. Refuses without --authorize and refuses the live store.")
    parser.add_argument("--authorize", action="store_true", help="Required before any future insertion into a clone.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.apply and args.dry_run:
        raise SystemExit("pass either --dry-run or --apply")
    if not args.apply:
        args.dry_run = True
    summary = run_recovery(args.snapshot, args.output, apply=args.apply, authorize=args.authorize)
    json.dump(summary, sys.stdout, indent=2, sort_keys=True)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
