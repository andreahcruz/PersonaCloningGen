"""Resumable LightRAG indexing session.

The default queue is bounded by ``--max-documents`` or ``--max-runtime-minutes``.
A session without either limit is refused so a full-corpus job cannot start by
accident. The pilot directory is not a legal ``--workdir``.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import signal
import time
from dataclasses import dataclass
from pathlib import Path

from host_finetune.indexing_ledger import IndexingLedger, utc_now
from host_finetune.indexing_lightrag import (
    IndexConfig,
    LightRAGIndexAdapter,
    configuration_fingerprint,
    configuration_record,
)
from host_finetune.indexing_reports import write_reports
from host_finetune.indexing_schedule import (
    build_schedule,
    file_sha256,
    load_jsonl,
    validation_batch,
)
from host_finetune.indexing_timeout import ADAPTIVE_V1, FIXED, classify_complexity, progress_estimate, select_timeout
from host_finetune.lightrag_backend import (
    DEFAULT_EMBED_MODEL,
    DEFAULT_EXTRACTION_MODEL,
    ROOT,
)

DEFAULT_GRAPH_READY = ROOT / "artifacts" / "lightrag" / "model_f" / "graph_ready_v2"
DEFAULT_WORKDIR = ROOT / "artifacts" / "lightrag" / "model_f" / "storage_expand_v1"
PILOT_DECISION = (
    "The 160-document pilot at artifacts/lightrag/model_f/storage_pilot160 was not "
    "copied. EXP-20261029-004 records that 26 of its documents were extracted under "
    "num_ctx 32768 and the finishing pass used 16384, and the store contains a "
    "duplicate status remnant. This expansion starts empty with one recorded "
    "configuration: project entity guidance, num_ctx 16384, chunk token size 400, "
    "gleaning 1, llm timeout 600 (worker timeout 1200), nomic-embed-text at 768 dimensions."
)


@dataclass
class SessionLimits:
    max_documents: int | None
    max_runtime_seconds: float | None
    max_attempts: int
    max_document_seconds: float | None
    stop_after_current: bool
    grace_seconds: float
    cancel_wait_seconds: float
    merge_wait_seconds: float
    include_deferred: bool
    retry_timeouts: bool
    allow_ids: set[str] | None
    timeout_policy: str = "fixed"
    source_media: tuple[str, ...] | None = None


class StopController:
    """Stop flag shared by Ctrl+C and a STOP file in the workdir."""

    def __init__(self, stop_file: Path | None = None):
        self.stop_file = stop_file
        self.reason = ""
        self._stop = False
        self._cancel = False

    def request(self, reason: str) -> None:
        self._stop = True
        self.reason = self.reason or reason

    def request_cancel(self) -> None:
        self._cancel = True
        self.request("cancel")

    def stop_requested(self) -> bool:
        if self.stop_file is not None and self.stop_file.exists():
            self.request("stop_file")
        return self._stop

    def cancel_requested(self) -> bool:
        return self._cancel


class WorkdirLock:
    def __init__(self, path: Path):
        self.path = path
        self._held = False

    def acquire(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.exists():
            pid = _read_pid(self.path)
            if pid and _pid_alive(pid):
                raise SystemExit(f"indexing lock is held by pid {pid}: {self.path}")
            self.path.unlink(missing_ok=True)
        self.path.write_text(str(os.getpid()), encoding="utf-8")
        self._held = True

    def release(self) -> None:
        if self._held and self.path.exists():
            self.path.unlink(missing_ok=True)
        self._held = False


def assert_workdir_allowed(workdir: Path) -> Path:
    resolved = Path(workdir).resolve()
    banned = [
        ROOT / "artifacts" / "lightrag" / "model_f" / "storage_pilot160",
        ROOT / "artifacts" / "lightrag" / "model_f" / "storage",
        ROOT / "artifacts" / "lightrag" / "model_f" / "graph_ready",
        ROOT / "artifacts" / "lightrag" / "model_f" / "graph_ready_v2",
        ROOT / "host_finetune" / "output" / "chroma_lemkin_train_only",
        ROOT / "host_finetune" / "output" / "chroma_lemkin_train_only_v2",
    ]
    for path in banned:
        if resolved == path.resolve():
            raise SystemExit(f"refusing to write a protected path: {resolved}")
    return resolved


def prepare_ledger(graph_ready: Path, workdir: Path, config: IndexConfig) -> dict:
    documents_path = graph_ready / "documents.jsonl"
    if not documents_path.is_file():
        raise SystemExit(f"graph-ready documents are missing: {documents_path}")
    corpus_hash = file_sha256(documents_path)
    fingerprint = configuration_fingerprint(config)
    documents = load_jsonl(documents_path)
    rows, summary = build_schedule(
        documents,
        corpus_hash=corpus_hash,
        fingerprint=fingerprint,
        indexing_mode=config.store_role,
    )
    ledger = IndexingLedger(workdir / "indexing_ledger.sqlite")
    try:
        synced = ledger.sync_corpus(rows, corpus_hash=corpus_hash, fingerprint=fingerprint)
        ledger.set_meta("configuration_json", json.dumps(configuration_record(config), sort_keys=True))
        ledger.set_meta("corpus_summary_json", json.dumps(summary, sort_keys=True))
        ledger.set_meta("graph_ready", str(graph_ready))
        batch = validation_batch(rows)
        (workdir / "validation_batch_ids.json").write_text(
            json.dumps(batch, indent=2) + "\n", encoding="utf-8"
        )
        manifest_path = workdir / "selection_manifest.jsonl"
        with manifest_path.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(
                    json.dumps(
                        {
                            "document_id": row["document_id"],
                            "selection_rank": row["selection_rank"],
                            "selection_priority": row["selection_priority"],
                            "selected": row["selected"],
                            "status": row["status"],
                            "source_medium": row["source_medium"],
                            "defer_reason": row["defer_reason"],
                            "source_hash": row["source_hash"],
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
        summary = {**summary, **synced, "corpus_sha256": corpus_hash, "fingerprint": fingerprint, "validation_batch": batch}
        return summary
    finally:
        ledger.close()


async def run_indexing_session(
    *,
    workdir: Path,
    graph_ready: Path,
    config: IndexConfig,
    limits: SessionLimits,
    stop: StopController,
    adapter: LightRAGIndexAdapter | None = None,
    documents_by_id: dict[str, dict] | None = None,
    session_id: str | None = None,
) -> dict:
    """Index until a limit or a stop request.

    ``adapter`` and ``documents_by_id`` are injection points for tests. The CLI
    builds a real LightRAG adapter and loads graph-ready v2.
    """
    ledger = IndexingLedger(workdir / "indexing_ledger.sqlite")
    owns_adapter = adapter is None
    if documents_by_id is None:
        documents_by_id = {
            row["graph_document_id"]: row
            for row in load_jsonl(graph_ready / "documents.jsonl")
        }
    if adapter is None:
        adapter = LightRAGIndexAdapter(workdir, config)
    session_id = session_id or time.strftime("sess-%Y%m%dT%H%M%SZ", time.gmtime())
    started_at = utc_now()
    wall = time.perf_counter()
    handled = 0
    last_id = None
    stop_reason = ""
    graph_counts = None
    try:
        await adapter.open()
        await _reconcile(adapter, ledger, documents_by_id)
        while True:
            if stop.stop_requested():
                stop_reason = stop.reason or "stop"
                break
            if limits.stop_after_current and handled >= 1:
                stop_reason = "stop_after_current"
                break
            if limits.max_documents is not None and handled >= limits.max_documents:
                stop_reason = "max_documents"
                break
            if limits.max_runtime_seconds is not None and (time.perf_counter() - wall) >= limits.max_runtime_seconds:
                stop_reason = "max_runtime"
                break
            row = ledger.next_document(
                include_deferred=limits.include_deferred,
                retry_timeouts=limits.retry_timeouts,
                allow_ids=limits.allow_ids,
                source_media=limits.source_media,
            )
            if row is None:
                stop_reason = stop_reason or "queue_empty"
                break
            document = documents_by_id.get(row["document_id"])
            if document is None:
                ledger.record_outcome(
                    row["document_id"],
                    _missing_source_outcome(row),
                )
                handled += 1
                continue
            if (
                last_id == row["document_id"]
                and row["status"] == "failed_retryable"
                and int(row["attempt_number"]) > 0
            ):
                delay = min(30, 2 ** max(0, int(row["attempt_number"]) - 1))
                await asyncio.sleep(delay)
            checkpoint = f"{session_id}-{row['document_id']}-a{int(row['attempt_number']) + 1}"
            active = ledger.mark_processing(row["document_id"], checkpoint)
            policy = ADAPTIVE_V1 if limits.timeout_policy == "adaptive" else FIXED
            document_seconds = select_timeout(
                policy,
                row.get("estimated_chunks"),
                retry=bool(limits.retry_timeouts and row.get("defer_reason") == "expensive_timeout"),
                fixed_seconds=limits.max_document_seconds,
            )
            outcome = await adapter.index_document(
                document,
                attempt=int(active["attempt_number"]),
                max_attempts=limits.max_attempts,
                checkpoint_id=checkpoint,
                max_document_seconds=document_seconds,
                grace_seconds=limits.grace_seconds,
                cancel_wait_seconds=limits.cancel_wait_seconds,
                merge_wait_seconds=limits.merge_wait_seconds,
                stop=stop,
            )
            ledger.record_outcome(row["document_id"], outcome)
            last_id = row["document_id"]
            handled += 1
            print(
                f"{outcome['status']} {row['document_id']} "
                f"seconds={outcome.get('runtime_seconds')} "
                f"category={outcome.get('error_category') or '-'}",
                flush=True,
            )
            if stop.stop_requested():
                stop_reason = stop.reason or "stop"
                break
        graph_counts = await adapter.graph_counts()
    finally:
        if owns_adapter or adapter is not None:
            await adapter.close()
        ledger.close()
    ledger = IndexingLedger(workdir / "indexing_ledger.sqlite")
    try:
        summary = write_reports(
            ledger,
            workdir / "reports",
            graph_counts=graph_counts,
            corpus_summary=_load_meta_json(ledger, "corpus_summary_json"),
            config_record=configuration_record(config),
            pilot_decision=PILOT_DECISION,
        )
        summary["session_id"] = session_id
        summary["started_at"] = started_at
        summary["stop_reason"] = stop_reason
        summary["documents_handled"] = handled
        summary["timeout_policy"] = (ADAPTIVE_V1 if limits.timeout_policy == "adaptive" else FIXED).record()
        summary["selection_policy"] = "blog-only" if limits.source_media == ("blog",) else "all-media"
        summary["extraction_normalization_policy"] = "relation-normalization-v1"
        summary["normalization_counts"] = dict(getattr(adapter, "normalization_counts", {}) or {})
        summary["normalization_scope"] = (
            "Applies to extraction results parsed after this resume. "
            "Documents already completed keep the extraction that was stored. "
            "The configuration fingerprint is unchanged."
        )
        if graph_counts is not None:
            ledger.set_meta("last_graph_counts", json.dumps(graph_counts))
        ledger.record_session(session_id, configuration_record(config), summary, stop_reason)
        return summary
    finally:
        ledger.close()


async def _reconcile(adapter, ledger: IndexingLedger, documents_by_id: dict[str, dict]) -> None:
    for row in ledger.open_statuses():
        if row["status"] not in {"processing", "interrupted"}:
            continue
        document = documents_by_id.get(row["document_id"]) or {}
        inspection = await adapter.inspect(row["document_id"], document.get("text") or "")
        if inspection["verified"]:
            outcome = adapter._success(
                time.perf_counter(),
                row.get("checkpoint_id") or "reconcile",
                row["document_id"],
                inspection,
                note="reconciled from persisted LightRAG state; model was not called",
            )
        elif inspection["lightrag_status"] == "processed":
            outcome = {
                "status": "quarantined",
                "error_stage": "persist",
                "error_category": "persistence_failure",
                "error_message": "processed record failed integrity checks during reconcile",
                "retry_eligible": False,
                "persisted_lightrag_id": row["document_id"],
                "graph_persistence": "not_verified",
                "vector_persistence": "present_unverified",
                "provenance_verified": inspection.get("provenance_ok"),
                "vector_only": False,
                "graph_extracted": False,
                "indexed_chunk_count": inspection.get("chunks_count"),
                "quality_notes": "reconciled",
            }
        elif inspection.get("graph_mutated"):
            outcome = {
                "status": "quarantined",
                "error_stage": "merge",
                "error_category": "graph_store_write",
                "error_message": "interrupted document has graph residue; not deleted",
                "retry_eligible": False,
                "persisted_lightrag_id": row["document_id"],
                "graph_persistence": "residue",
                "vector_persistence": "present_unverified" if inspection.get("chunks_count") else "absent",
                "provenance_verified": inspection.get("provenance_ok"),
                "vector_only": False,
                "graph_extracted": False,
                "indexed_chunk_count": inspection.get("chunks_count"),
                "quality_notes": "reconciled",
            }
        else:
            outcome = {
                "status": "interrupted",
                "error_stage": "interrupt",
                "error_category": "process_interruption",
                "error_message": inspection.get("error_msg") or "stale processing record reconciled for retry",
                "retry_eligible": True,
                "persisted_lightrag_id": row["document_id"],
                "graph_persistence": "not_verified",
                "vector_persistence": "present_unverified" if inspection.get("chunks_count") else "absent",
                "provenance_verified": inspection.get("provenance_ok"),
                "vector_only": False,
                "graph_extracted": False,
                "indexed_chunk_count": inspection.get("chunks_count"),
                "quality_notes": "reconciled; safe to retry because no graph mutation was recorded",
            }
        ledger.reconcile_status(row["document_id"], outcome)


def _missing_source_outcome(row: dict) -> dict:
    return {
        "status": "quarantined",
        "error_stage": "source",
        "error_category": "source_validation",
        "error_message": "document id is in the ledger but not in the loaded corpus",
        "retry_eligible": False,
        "checkpoint_id": row.get("checkpoint_id"),
        "persisted_lightrag_id": None,
        "indexed_chunk_count": None,
        "extraction_metrics": None,
        "embedding_metrics": None,
        "graph_persistence": "not_verified",
        "vector_persistence": "absent",
        "provenance_verified": False,
        "vector_only": False,
        "graph_extracted": False,
        "stage_timings": {},
        "runtime_seconds": 0,
        "quality_notes": "missing source",
        "defer_reason": None,
    }


def _load_meta_json(ledger: IndexingLedger, key: str):
    raw = ledger.get_meta(key)
    if not raw:
        return None
    return json.loads(raw)


def _read_pid(path: Path) -> int | None:
    try:
        return int(path.read_text(encoding="utf-8").strip())
    except (OSError, ValueError):
        return None


def _pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    else:
        return True


def _print_status(workdir: Path, selection_policy: str = "all-media") -> None:
    from host_finetune.indexing_selection import blog_time_scenarios
    from host_finetune.indexing_timeout import progress_estimate

    ledger_path = workdir / "indexing_ledger.sqlite"
    if not ledger_path.is_file():
        raise SystemExit(f"no ledger at {ledger_path}. Run --prepare first.")
    ledger = IndexingLedger(ledger_path)
    try:
        counts = ledger.counts()
        graph = ledger.get_meta("last_graph_counts")
        summary = ledger.selection_summary()
        blog_runs = ledger.runtime_samples_for_medium("blog")
        blog_failures = ledger.unsuccessful_attempt_seconds_for_medium("blog")
        blog_progress = progress_estimate(
            blog_runs,
            blog_failures,
            normal_queued=int(summary["pending_blogs_normal_queue"]),
            deferred_queued=int(summary["deferred_blog_retries"]),
        )
        print(json.dumps({
            "active_queue": selection_policy,
            "graph_description": "mixed_media_seed_with_blog_only_expansion",
            "counts": counts,
            "queues": ledger.queue_sizes(),
            "progress": ledger.progress(),
            "blog_only": summary,
            "blog_progress": blog_progress,
            "blog_time_scenarios": blog_time_scenarios(
                blog_runs,
                int(summary["pending_blogs_normal_queue"]),
                len(blog_failures),
            ),
            "last_graph_counts": json.loads(graph) if graph else None,
        }, indent=2))
    finally:
        ledger.close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph-ready", type=Path, default=DEFAULT_GRAPH_READY)
    parser.add_argument("--workdir", type=Path, default=DEFAULT_WORKDIR)
    parser.add_argument("--prepare", action="store_true", help="Load the corpus into the ledger and write the manifest.")
    parser.add_argument("--resume", action="store_true", help="Index the next eligible documents.")
    parser.add_argument("--dry-run", action="store_true", help="Print the next documents without calling a model.")
    parser.add_argument("--status", action="store_true")
    parser.add_argument(
        "--selection-policy",
        choices=("all-media", "blog-only"),
        default="all-media",
        help="blog-only schedules eligible blogs and leaves other media in the ledger.",
    )
    parser.add_argument(
        "--export-blog-manifest",
        action="store_true",
        help="Write selection_blog_only_v1.jsonl from the ledger without changing it.",
    )
    parser.add_argument("--export-report", type=Path, default=None)
    parser.add_argument("--max-documents", type=int, default=None)
    parser.add_argument("--max-runtime-minutes", type=float, default=None)
    parser.add_argument("--max-attempts", type=int, default=2)
    parser.add_argument("--max-document-seconds", type=float, default=420)
    parser.add_argument("--stop-after-current", action="store_true")
    parser.add_argument("--grace-seconds", type=float, default=30)
    parser.add_argument("--include-deferred", action="store_true")
    parser.add_argument("--retry-timeouts", action="store_true", help="Explicit expensive-document retry queue.")
    parser.add_argument(
        "--timeout-policy",
        choices=("adaptive", "fixed"),
        default="adaptive",
        help="adaptive uses chunk-tier budgets. fixed keeps --max-document-seconds for every document.",
    )
    parser.add_argument("--validation-batch", action="store_true")
    parser.add_argument("--mode", choices=("graph", "vector_only"), default="graph")
    parser.add_argument("--base-url", default="http://localhost:11434")
    parser.add_argument("--extraction-model", default=DEFAULT_EXTRACTION_MODEL)
    parser.add_argument("--embed-model", default=DEFAULT_EMBED_MODEL)
    parser.add_argument("--num-ctx", type=int, default=16384)
    parser.add_argument("--snapshot", type=Path, default=None, help="Copy the workdir after a clean shutdown.")
    parser.add_argument("--audit", action="store_true", help="Write a read-only reconciliation of the workdir.")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    workdir = assert_workdir_allowed(args.workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    config = IndexConfig(
        extraction_model=args.extraction_model,
        embed_model=args.embed_model,
        base_url=args.base_url,
        num_ctx=args.num_ctx,
        store_role="vector_only" if args.mode == "vector_only" else "graph",
    )
    if args.prepare:
        summary = prepare_ledger(args.graph_ready, workdir, config)
        print(json.dumps({key: summary[key] for key in (
            "documents_seen", "eligible", "selected", "deferred_pilot",
            "skipped_duplicate_text", "quarantined", "corpus_sha256", "validation_batch",
        ) if key in summary}, indent=2))
    if args.status and not args.resume:
        _print_status(workdir, args.selection_policy)
    if args.export_blog_manifest and not args.resume:
        from host_finetune.indexing_selection import MANIFEST_NAME, export_blog_manifest_readonly

        destination = workdir / MANIFEST_NAME
        print(json.dumps(export_blog_manifest_readonly(workdir / "indexing_ledger.sqlite", destination), indent=2))
        return
    if args.export_report and not args.resume:
        _export(workdir, args.export_report, config)
    if args.audit and not args.resume:
        from host_finetune.indexing_reconcile import write_audit_bundle

        destination = args.export_report or (workdir / "reports")
        print(json.dumps(write_audit_bundle(workdir, destination), indent=2))
        return
    if args.dry_run:
        _dry_run(workdir, args)
        return
    if args.snapshot and not args.resume:
        _snapshot(workdir, args.snapshot)
        return
    if not args.resume:
        return
    ledger_path = workdir / "indexing_ledger.sqlite"
    if not ledger_path.is_file():
        raise SystemExit(f"no ledger at {ledger_path}. Run --prepare first.")
    stored = IndexingLedger(ledger_path)
    try:
        fingerprint = configuration_fingerprint(config)
        recorded = stored.get_meta("configuration_fingerprint")
        if recorded and recorded != fingerprint:
            raise SystemExit(
                "this workdir was prepared with a different extraction configuration. "
                "Use a new --workdir."
            )
        corpus_hash = file_sha256(args.graph_ready / "documents.jsonl")
        recorded_corpus = stored.get_meta("corpus_version_hash")
        if recorded_corpus and recorded_corpus != corpus_hash:
            raise SystemExit("the graph-ready corpus hash changed. Use a new --workdir.")
    finally:
        stored.close()
    if args.max_documents is None and args.max_runtime_minutes is None and not args.stop_after_current:
        raise SystemExit(
            "refusing an unbounded session. Pass --max-documents, --max-runtime-minutes, or --stop-after-current."
        )
    if args.max_documents is not None and args.max_documents < 1:
        raise SystemExit("--max-documents must be at least 1")
    allow_ids = None
    if args.validation_batch:
        batch_path = workdir / "validation_batch_ids.json"
        if not batch_path.is_file():
            raise SystemExit("validation batch is missing. Run --prepare first.")
        allow_ids = set(json.loads(batch_path.read_text(encoding="utf-8")))
    limits = SessionLimits(
        max_documents=args.max_documents,
        max_runtime_seconds=None if args.max_runtime_minutes is None else args.max_runtime_minutes * 60,
        max_attempts=args.max_attempts,
        max_document_seconds=args.max_document_seconds,
        stop_after_current=args.stop_after_current,
        grace_seconds=args.grace_seconds,
        cancel_wait_seconds=15,
        merge_wait_seconds=180,
        include_deferred=args.include_deferred,
        retry_timeouts=args.retry_timeouts,
        allow_ids=allow_ids,
        timeout_policy=args.timeout_policy,
        source_media=("blog",) if args.selection_policy == "blog-only" else None,
    )
    stop = StopController(workdir / "STOP")
    _install_signal(stop)
    lock = WorkdirLock(workdir / "indexing.lock")
    lock.acquire()
    try:
        summary = asyncio.run(
            run_indexing_session(
                workdir=workdir,
                graph_ready=args.graph_ready,
                config=config,
                limits=limits,
                stop=stop,
            )
        )
    finally:
        lock.release()
    print(json.dumps({
        "stop_reason": summary.get("stop_reason"),
        "documents_handled": summary.get("documents_handled"),
        "completed_graph_indexed": summary.get("completed_graph_indexed"),
        "graph_nodes": summary.get("graph_nodes"),
        "graph_edges": summary.get("graph_edges"),
    }, indent=2))
    if args.export_report:
        _export(workdir, args.export_report, config)
    if args.snapshot:
        _snapshot(workdir, args.snapshot)


def _install_signal(stop: StopController) -> None:
    def handler(signum, frame):
        del signum, frame
        if stop.stop_requested():
            stop.request_cancel()
        else:
            stop.request("sigint")
            print("stop requested; the current document will reach a persistence boundary or be marked interrupted", flush=True)

    signal.signal(signal.SIGINT, handler)


def _dry_run(workdir: Path, args: argparse.Namespace) -> None:
    ledger_path = workdir / "indexing_ledger.sqlite"
    if not ledger_path.is_file():
        raise SystemExit(f"no ledger at {ledger_path}. Run --prepare first.")
    ledger = IndexingLedger(ledger_path)
    try:
        allow = None
        if args.validation_batch:
            allow = set(json.loads((workdir / "validation_batch_ids.json").read_text(encoding="utf-8")))
        limit = args.max_documents or 20
        rows = ledger.peek_documents(
            limit,
            include_deferred=args.include_deferred,
            retry_timeouts=args.retry_timeouts,
            allow_ids=allow,
            source_media=("blog",) if args.selection_policy == "blog-only" else None,
        )
        for row in rows:
            print(
                f"{row.get('selection_rank')} {row['document_id']} {row.get('source_medium')} "
                f"priority={row.get('selection_priority')} status={row['status']}"
            )
    finally:
        ledger.close()


def _export(workdir: Path, destination: Path, config: IndexConfig) -> None:
    ledger = IndexingLedger(workdir / "indexing_ledger.sqlite")
    try:
        graph = ledger.get_meta("last_graph_counts")
        write_reports(
            ledger,
            destination,
            graph_counts=json.loads(graph) if graph else None,
            corpus_summary=_load_meta_json(ledger, "corpus_summary_json"),
            config_record=configuration_record(config),
            pilot_decision=PILOT_DECISION,
        )
    finally:
        ledger.close()
    print(f"wrote reports to {destination}")


def _snapshot(workdir: Path, destination: Path) -> None:
    import shutil

    ledger = IndexingLedger(workdir / "indexing_ledger.sqlite")
    try:
        if any(row["status"] == "processing" for row in ledger.iter_documents()):
            raise SystemExit("refusing snapshot while a document is still processing")
    finally:
        ledger.close()
    status_path = workdir / "kv_store_doc_status.json"
    if status_path.is_file():
        payload = json.loads(status_path.read_text(encoding="utf-8"))
        rows = payload.values() if isinstance(payload, dict) else payload
        for row in rows or []:
            if isinstance(row, dict) and str(row.get("status") or "").lower() in {"processing", "parsing", "analyzing"}:
                raise SystemExit("refusing snapshot while LightRAG still has an in-flight document")
    if destination.exists():
        raise SystemExit(f"snapshot destination already exists: {destination}")
    shutil.copytree(
        workdir,
        destination,
        ignore=shutil.ignore_patterns("STOP", "indexing.lock"),
    )
    print(f"snapshot copied to {destination}")


if __name__ == "__main__":
    main()
