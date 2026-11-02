"""Blog-only scheduling records.

The manifest is a selection policy. It does not change source text, document
ids, or the extraction configuration fingerprint. Indexing status inside the
file is a snapshot; resume still reads the live ledger.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import statistics
from pathlib import Path

POLICY_VERSION = "blog-only-v1"
MANIFEST_NAME = "selection_blog_only_v1.jsonl"
ORIGINAL_MANIFEST_NAME = "selection_manifest.jsonl"


def is_eligible_blog(row: dict) -> bool:
    """A blog the existing graph-ready schedule did not quarantine or skip."""
    if str(row.get("source_medium") or "") != "blog":
        return False
    if str(row.get("status") or "") in {"quarantined", "skipped"}:
        return False
    return True


def scheduled_in_normal_queue(row: dict) -> bool:
    if not row.get("selected"):
        return False
    if row.get("defer_reason"):
        return False
    return str(row.get("status") or "") in {"pending", "failed_retryable", "interrupted"}


def build_blog_manifest(rows: list[dict]) -> list[dict]:
    """Stable blog records ordered by the existing all-media selection rank."""
    blogs = [row for row in rows if is_eligible_blog(row)]
    blogs.sort(
        key=lambda row: (
            row.get("selection_rank") is None,
            row.get("selection_rank") if row.get("selection_rank") is not None else 10**12,
            str(row["document_id"]),
        )
    )
    manifest = []
    seen = set()
    for position, row in enumerate(blogs, start=1):
        doc_id = str(row["document_id"])
        if doc_id in seen:
            raise ValueError(f"duplicate blog document id {doc_id}")
        seen.add(doc_id)
        manifest.append(
            {
                "document_id": doc_id,
                "original_source_id": row.get("original_source_id") or doc_id,
                "source_medium": "blog",
                "source_hash": row.get("source_hash"),
                "corpus_version": row.get("corpus_version_hash"),
                "selection_policy_version": POLICY_VERSION,
                "selection_rank": row.get("selection_rank"),
                "blog_queue_rank": position,
                "estimated_chunks": row.get("estimated_chunks"),
                "indexing_status": row.get("status"),
                "defer_reason": row.get("defer_reason"),
                "selected": bool(row.get("selected")),
                "scheduled_in_normal_queue": scheduled_in_normal_queue(row),
                "has_omission": bool(row.get("has_omission")),
                "source_attribution_status": row.get("source_attribution_status"),
                "graph_description": "mixed_media_seed_with_blog_only_expansion",
            }
        )
    return manifest


def identity_sha256(records: list[dict]) -> str:
    """Hash of the blog set and order, ignoring live indexing status."""
    lines = [
        f"{row['blog_queue_rank']}\t{row['document_id']}\t{row['source_hash']}\t{row['selection_rank']}"
        for row in records
    ]
    payload = ("\n".join(lines) + "\n").encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def write_blog_manifest(records: list[dict], destination: Path) -> dict:
    """Write the versioned manifest. Refuses the original all-media filename."""
    destination = Path(destination)
    if destination.name == ORIGINAL_MANIFEST_NAME:
        raise SystemExit(f"refusing to overwrite the original selection manifest: {destination}")
    if destination.name in {"graph_ready_v2", "documents.jsonl"}:
        raise SystemExit(f"refusing to write the blog manifest over source data: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    body = "".join(json.dumps(row, sort_keys=True) + "\n" for row in records)
    destination.write_text(body, encoding="utf-8")
    digest = hashlib.sha256(body.encode("utf-8")).hexdigest()
    destination.with_suffix(".sha256").write_text(digest + "\n", encoding="utf-8")
    return {
        "path": str(destination),
        "documents": len(records),
        "sha256": digest,
        "identity_sha256": identity_sha256(records),
        "policy": POLICY_VERSION,
    }


def describe_inventory(rows: list[dict]) -> dict:
    """Counts for the status command. Rows are ledger documents."""
    by_medium: dict[str, dict[str, int]] = {}
    for row in rows:
        medium = str(row.get("source_medium") or "unknown")
        status = str(row.get("status") or "unknown")
        bucket = by_medium.setdefault(medium, {})
        bucket[status] = bucket.get(status, 0) + 1
    blogs = [row for row in rows if is_eligible_blog(row)]
    return {
        "description": "mixed_media_seed_with_blog_only_expansion",
        "by_medium_status": by_medium,
        "eligible_blogs": len(blogs),
        "completed_blogs": sum(1 for row in blogs if row.get("status") == "completed"),
        "pending_blogs_normal_queue": sum(1 for row in blogs if scheduled_in_normal_queue(row)),
        "deferred_blogs": sum(
            1
            for row in blogs
            if row.get("defer_reason") and row.get("status") != "completed"
        ),
        "failed_blogs": sum(1 for row in blogs if row.get("status") == "failed_permanent"),
        "completed_nonblog_seed": sum(
            1
            for row in rows
            if row.get("status") == "completed" and row.get("source_medium") != "blog"
        ),
        "pending_nonblog_excluded_from_blog_queue": sum(
            1
            for row in rows
            if row.get("source_medium") != "blog" and scheduled_in_normal_queue(row)
        ),
    }


def blog_time_scenarios(
    completed_seconds: list[float],
    pending: int,
    unsuccessful_attempts: int,
) -> dict:
    """Three rough blog-queue durations. None of them is a completion date.

    The failure-adjusted case charges a 180-second adaptive stall, not the
    historical 420-second cap, for the observed timeout rate. The conservative
    case uses the successful p90 plus a 420-second stall at that same rate.
    """
    if not completed_seconds or pending <= 0:
        return {
            "pending_blogs": pending,
            "optimistic_hours": None,
            "failure_adjusted_hours": None,
            "conservative_hours": None,
            "note": "No completed blog sample or no pending blogs.",
        }
    ordered = sorted(completed_seconds)
    median = statistics.median(ordered)
    mean = statistics.fmean(ordered)
    k = (len(ordered) - 1) * 0.9
    lo = int(k)
    hi = min(lo + 1, len(ordered) - 1)
    p90 = ordered[lo] + (ordered[hi] - ordered[lo]) * (k - lo)
    rate = unsuccessful_attempts / (len(ordered) + unsuccessful_attempts) if unsuccessful_attempts else 0.0
    return {
        "pending_blogs": pending,
        "completed_blog_sample": len(ordered),
        "median_successful_seconds": round(median, 1),
        "p90_successful_seconds": round(p90, 1),
        "timeout_attempts_per_completed_plus_timeout": round(rate, 3),
        "optimistic_hours": round(median * pending / 3600, 1),
        "failure_adjusted_hours": round((mean * pending + pending * rate * 180) / 3600, 1),
        "conservative_hours": round((p90 * pending + pending * rate * 420) / 3600, 1),
        "note": (
            "Hours assume the pending blogs resemble the completed blogs. "
            "Chunk estimates are characters/4, not tokenizer counts. "
            "A few stalls dominate the tail, so the hours are not linear guarantees."
        ),
    }


def export_blog_manifest_readonly(ledger_path: Path, destination: Path) -> dict:
    """Read the ledger without a write lock and write the blog manifest."""
    uri = Path(ledger_path).resolve().as_uri() + "?mode=ro"
    connection = sqlite3.connect(uri, uri=True, timeout=10)
    connection.row_factory = sqlite3.Row
    try:
        rows = [dict(row) for row in connection.execute(
            """
            SELECT document_id, original_source_id, source_medium, source_hash,
                   corpus_version_hash, selection_rank, estimated_chunks, status,
                   defer_reason, selected, has_omission, source_attribution_status
            FROM documents
            """
        )]
    finally:
        connection.close()
    records = build_blog_manifest(rows)
    written = write_blog_manifest(records, destination)
    written["eligible_blogs"] = len(records)
    written["scheduled_in_normal_queue"] = sum(1 for row in records if row["scheduled_in_normal_queue"])
    return written
