"""SQLite ledger for resumable LightRAG indexing.

The ledger lives beside a LightRAG working directory and is not one of
LightRAG's storage files. Status changes are committed in a single
``BEGIN IMMEDIATE`` transaction. Attempt rows are append-only, so a later
retry does not erase the earlier error.
"""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

from host_finetune.indexing_timeout import classify_complexity, progress_estimate

STATUSES = (
    "pending",
    "processing",
    "completed",
    "failed_retryable",
    "failed_permanent",
    "quarantined",
    "skipped",
    "interrupted",
)

QUEUE_STATUSES = ("pending", "failed_retryable", "interrupted")


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _json(value) -> str | None:
    if value is None:
        return None
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _load(value: str | None):
    if not value:
        return None
    return json.loads(value)


class IndexingLedger:
    """One row per corpus document, plus an append-only attempt history."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.path, isolation_level=None, timeout=30)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._create()

    def close(self) -> None:
        self._conn.close()

    def _create(self) -> None:
        self._conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS documents (
                document_id TEXT PRIMARY KEY,
                original_source_id TEXT,
                source_medium TEXT,
                source_attribution_status TEXT,
                topic_labels TEXT,
                source_path TEXT,
                source_hash TEXT,
                corpus_version_hash TEXT,
                estimated_input_tokens INTEGER,
                estimated_chunks INTEGER,
                word_count INTEGER,
                has_omission INTEGER,
                reconstruction_method TEXT,
                selection_priority REAL,
                selection_rank INTEGER,
                selected INTEGER NOT NULL DEFAULT 0,
                indexing_mode TEXT,
                configuration_fingerprint TEXT,
                status TEXT NOT NULL,
                attempt_number INTEGER NOT NULL DEFAULT 0,
                started_at TEXT,
                finished_at TEXT,
                runtime_seconds REAL,
                error_stage TEXT,
                error_category TEXT,
                error_message TEXT,
                retry_eligible INTEGER,
                checkpoint_id TEXT,
                persisted_lightrag_id TEXT,
                indexed_chunk_count INTEGER,
                extraction_metrics TEXT,
                embedding_metrics TEXT,
                graph_persistence TEXT,
                vector_persistence TEXT,
                provenance_verified INTEGER,
                vector_only INTEGER NOT NULL DEFAULT 0,
                graph_extracted INTEGER NOT NULL DEFAULT 0,
                defer_reason TEXT,
                near_duplicate_of TEXT,
                quality_notes TEXT
            );
            CREATE TABLE IF NOT EXISTS attempts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                document_id TEXT NOT NULL,
                attempt_number INTEGER NOT NULL,
                started_at TEXT,
                finished_at TEXT,
                runtime_seconds REAL,
                status TEXT,
                error_stage TEXT,
                error_category TEXT,
                error_message TEXT,
                checkpoint_id TEXT,
                stage_timings TEXT,
                extraction_metrics TEXT,
                embedding_metrics TEXT,
                persistence_result TEXT,
                FOREIGN KEY (document_id) REFERENCES documents(document_id)
            );
            CREATE TABLE IF NOT EXISTS sessions (
                session_id TEXT PRIMARY KEY,
                started_at TEXT,
                finished_at TEXT,
                stop_reason TEXT,
                config_json TEXT,
                summary_json TEXT
            );
            CREATE TABLE IF NOT EXISTS meta (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_documents_status
                ON documents(status, selected, selection_rank);
            CREATE INDEX IF NOT EXISTS idx_attempts_document
                ON attempts(document_id, attempt_number);
            """
        )

    @contextmanager
    def _tx(self) -> Iterator[sqlite3.Connection]:
        self._conn.execute("BEGIN IMMEDIATE")
        try:
            yield self._conn
        except Exception:
            self._conn.rollback()
            raise
        else:
            self._conn.commit()

    def get_meta(self, key: str) -> str | None:
        row = self._conn.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
        return None if row is None else str(row["value"])

    def set_meta(self, key: str, value: str) -> None:
        with self._tx() as conn:
            conn.execute(
                "INSERT INTO meta(key, value) VALUES(?, ?) "
                "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                (key, value),
            )

    def document(self, document_id: str) -> dict | None:
        row = self._conn.execute(
            "SELECT * FROM documents WHERE document_id = ?", (document_id,)
        ).fetchone()
        return None if row is None else self._row(row)

    def attempts(self, document_id: str) -> list[dict]:
        rows = self._conn.execute(
            "SELECT * FROM attempts WHERE document_id = ? ORDER BY id",
            (document_id,),
        ).fetchall()
        result = []
        for row in rows:
            item = dict(row)
            for key in ("stage_timings", "extraction_metrics", "embedding_metrics", "persistence_result"):
                item[key] = _load(item.get(key))
            result.append(item)
        return result

    def sync_corpus(self, rows: list[dict], *, corpus_hash: str, fingerprint: str) -> dict:
        """Insert new documents and refresh scheduling fields.

        A completed document whose source hash changed is refused. Pending
        documents with a new hash are quarantined instead of being indexed
        under the old identity. Status, attempts, and errors of an unchanged
        document are left in place.
        """
        stored_hash = self.get_meta("corpus_version_hash")
        if stored_hash and stored_hash != corpus_hash:
            raise SystemExit(
                "corpus hash changed for this ledger "
                f"({stored_hash} -> {corpus_hash}). Use a new --workdir."
            )
        stored_fp = self.get_meta("configuration_fingerprint")
        if stored_fp and stored_fp != fingerprint:
            raise SystemExit(
                "indexing configuration changed for this ledger. "
                "Use a new --workdir so embeddings and extraction stay comparable."
            )
        inserted = 0
        updated = 0
        quarantined = 0
        with self._tx() as conn:
            conn.execute(
                "INSERT INTO meta(key, value) VALUES('corpus_version_hash', ?) "
                "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                (corpus_hash,),
            )
            conn.execute(
                "INSERT INTO meta(key, value) VALUES('configuration_fingerprint', ?) "
                "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                (fingerprint,),
            )
            for row in rows:
                current = conn.execute(
                    "SELECT source_hash, status FROM documents WHERE document_id = ?",
                    (row["document_id"],),
                ).fetchone()
                if current is None:
                    self._insert(conn, row)
                    inserted += 1
                    if row["status"] == "quarantined":
                        quarantined += 1
                    continue
                if current["source_hash"] != row["source_hash"]:
                    if current["status"] == "completed":
                        raise SystemExit(
                            f"refusing to change source text for completed document {row['document_id']}"
                        )
                    conn.execute(
                        """
                        UPDATE documents
                        SET source_hash = ?, status = 'quarantined',
                            error_stage = 'source', error_category = 'source_validation',
                            error_message = 'source hash changed before a completed index',
                            retry_eligible = 0, selected = 0
                        WHERE document_id = ?
                        """,
                        (row["source_hash"], row["document_id"]),
                    )
                    quarantined += 1
                    continue
                if current["status"] in {"completed", "processing"}:
                    conn.execute(
                        """
                        UPDATE documents
                        SET selection_priority = ?, selection_rank = ?, selected = ?,
                            defer_reason = COALESCE(defer_reason, ?),
                            near_duplicate_of = ?
                        WHERE document_id = ?
                        """,
                        (
                            row["selection_priority"],
                            row["selection_rank"],
                            row["selected"] if current["status"] != "completed" else 0,
                            row.get("defer_reason"),
                            row.get("near_duplicate_of"),
                            row["document_id"],
                        ),
                    )
                else:
                    conn.execute(
                        """
                        UPDATE documents
                        SET selection_priority = ?, selection_rank = ?, selected = ?,
                            defer_reason = ?, near_duplicate_of = ?, topic_labels = ?,
                            estimated_input_tokens = ?, estimated_chunks = ?,
                            indexing_mode = ?, configuration_fingerprint = ?
                        WHERE document_id = ?
                        """,
                        (
                            row["selection_priority"],
                            row["selection_rank"],
                            row["selected"],
                            row.get("defer_reason"),
                            row.get("near_duplicate_of"),
                            _json(row.get("topic_labels")),
                            row.get("estimated_input_tokens"),
                            row.get("estimated_chunks"),
                            row.get("indexing_mode"),
                            row.get("configuration_fingerprint"),
                            row["document_id"],
                        ),
                    )
                updated += 1
        return {"inserted": inserted, "scheduling_refreshed": updated, "hash_quarantined": quarantined}

    def _insert(self, conn: sqlite3.Connection, row: dict) -> None:
        conn.execute(
            """
            INSERT INTO documents (
                document_id, original_source_id, source_medium, source_attribution_status,
                topic_labels, source_path, source_hash, corpus_version_hash,
                estimated_input_tokens, estimated_chunks, word_count, has_omission,
                reconstruction_method, selection_priority, selection_rank, selected,
                indexing_mode, configuration_fingerprint, status, defer_reason,
                near_duplicate_of, quality_notes, retry_eligible, vector_only, graph_extracted
            ) VALUES (
                ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
            )
            """,
            (
                row["document_id"],
                row.get("original_source_id"),
                row.get("source_medium"),
                row.get("source_attribution_status"),
                _json(row.get("topic_labels") or []),
                row.get("source_path"),
                row.get("source_hash"),
                row.get("corpus_version_hash"),
                row.get("estimated_input_tokens"),
                row.get("estimated_chunks"),
                row.get("word_count"),
                int(bool(row.get("has_omission"))),
                row.get("reconstruction_method"),
                row.get("selection_priority"),
                row.get("selection_rank"),
                int(bool(row.get("selected"))),
                row.get("indexing_mode"),
                row.get("configuration_fingerprint"),
                row["status"],
                row.get("defer_reason"),
                row.get("near_duplicate_of"),
                row.get("quality_notes"),
                0,
                1 if row.get("indexing_mode") == "vector_only" else 0,
                0,
            ),
        )

    def open_statuses(self) -> list[dict]:
        rows = self._conn.execute(
            "SELECT * FROM documents WHERE status IN ('processing', 'interrupted') ORDER BY document_id"
        ).fetchall()
        return [self._row(row) for row in rows]

    def next_document(
        self,
        *,
        include_deferred: bool,
        retry_timeouts: bool,
        allow_ids: set[str] | None = None,
        source_media: tuple[str, ...] | None = None,
    ) -> dict | None:
        clauses = ["status IN ('pending', 'failed_retryable', 'interrupted')", "selected = 1"]
        params: list = []
        if not include_deferred and not retry_timeouts:
            clauses.append("(defer_reason IS NULL OR defer_reason = '')")
        elif not include_deferred and retry_timeouts:
            clauses.append(
                "(defer_reason IS NULL OR defer_reason = '' OR defer_reason = 'expensive_timeout')"
            )
        elif include_deferred and not retry_timeouts:
            clauses.append(
                "(defer_reason IS NULL OR defer_reason = '' OR defer_reason != 'expensive_timeout')"
            )
        if not retry_timeouts:
            clauses.append(
                "(error_category IS NULL OR error_category != 'llm_extraction_timeout' OR status = 'interrupted')"
            )
        if allow_ids is not None:
            if not allow_ids:
                return None
            marks = ",".join("?" for _ in allow_ids)
            clauses.append(f"document_id IN ({marks})")
            params.extend(sorted(allow_ids))
        if source_media:
            marks = ",".join("?" for _ in source_media)
            clauses.append(f"source_medium IN ({marks})")
            params.extend(list(source_media))
        sql = (
            "SELECT * FROM documents WHERE "
            + " AND ".join(clauses)
            + " ORDER BY CASE status WHEN 'interrupted' THEN 0 "
            "WHEN 'failed_retryable' THEN 1 ELSE 2 END, "
            "COALESCE(selection_rank, 1000000000), document_id LIMIT 1"
        )
        row = self._conn.execute(sql, params).fetchone()
        return None if row is None else self._row(row)

    def peek_documents(
        self,
        limit: int,
        *,
        include_deferred: bool,
        retry_timeouts: bool,
        allow_ids: set[str] | None = None,
        source_media: tuple[str, ...] | None = None,
    ) -> list[dict]:
        clauses = ["status IN ('pending', 'failed_retryable', 'interrupted')", "selected = 1"]
        params: list = []
        if not include_deferred and not retry_timeouts:
            clauses.append("(defer_reason IS NULL OR defer_reason = '')")
        elif not include_deferred and retry_timeouts:
            clauses.append(
                "(defer_reason IS NULL OR defer_reason = '' OR defer_reason = 'expensive_timeout')"
            )
        elif include_deferred and not retry_timeouts:
            clauses.append(
                "(defer_reason IS NULL OR defer_reason = '' OR defer_reason != 'expensive_timeout')"
            )
        if not retry_timeouts:
            clauses.append(
                "(error_category IS NULL OR error_category != 'llm_extraction_timeout' OR status = 'interrupted')"
            )
        if allow_ids is not None:
            if not allow_ids:
                return []
            marks = ",".join("?" for _ in allow_ids)
            clauses.append(f"document_id IN ({marks})")
            params.extend(sorted(allow_ids))
        if source_media:
            marks = ",".join("?" for _ in source_media)
            clauses.append(f"source_medium IN ({marks})")
            params.extend(list(source_media))
        params.append(limit)
        sql = (
            "SELECT * FROM documents WHERE "
            + " AND ".join(clauses)
            + " ORDER BY CASE status WHEN 'interrupted' THEN 0 "
            "WHEN 'failed_retryable' THEN 1 ELSE 2 END, "
            "COALESCE(selection_rank, 1000000000), document_id LIMIT ?"
        )
        return [self._row(row) for row in self._conn.execute(sql, params).fetchall()]

    def inventory_by_medium(self) -> list[dict]:
        rows = self._conn.execute(
            """
            SELECT source_medium, status,
                   COUNT(*) AS documents,
                   SUM(CASE WHEN defer_reason IS NOT NULL AND defer_reason != '' THEN 1 ELSE 0 END) AS deferred,
                   SUM(CASE WHEN has_omission = 1 THEN 1 ELSE 0 END) AS omission_markers,
                   SUM(COALESCE(estimated_chunks, 0)) AS estimated_chunks
            FROM documents
            GROUP BY source_medium, status
            ORDER BY source_medium, status
            """
        ).fetchall()
        return [dict(row) for row in rows]

    def selection_summary(self) -> dict:
        row = self._conn.execute(
            """
            SELECT
                SUM(CASE WHEN source_medium = 'blog' AND status NOT IN ('quarantined', 'skipped') THEN 1 ELSE 0 END) AS eligible_blogs,
                SUM(CASE WHEN source_medium = 'blog' AND status = 'completed' THEN 1 ELSE 0 END) AS completed_blogs,
                SUM(CASE WHEN source_medium = 'blog' AND selected = 1
                          AND status IN ('pending', 'failed_retryable', 'interrupted')
                          AND (defer_reason IS NULL OR defer_reason = '') THEN 1 ELSE 0 END) AS pending_blogs_normal_queue,
                SUM(CASE WHEN source_medium = 'blog' AND defer_reason IS NOT NULL AND defer_reason != ''
                          AND status != 'completed' THEN 1 ELSE 0 END) AS deferred_blogs,
                SUM(CASE WHEN source_medium = 'blog' AND defer_reason = 'expensive_timeout'
                          AND status IN ('pending', 'failed_retryable', 'interrupted')
                     THEN 1 ELSE 0 END) AS deferred_blog_retries,
                SUM(CASE WHEN source_medium = 'blog' AND status = 'failed_permanent' THEN 1 ELSE 0 END) AS failed_blogs,
                SUM(CASE WHEN source_medium != 'blog' AND status = 'completed' THEN 1 ELSE 0 END) AS completed_nonblog_seed,
                SUM(CASE WHEN source_medium != 'blog' AND selected = 1
                          AND status IN ('pending', 'failed_retryable', 'interrupted')
                          AND (defer_reason IS NULL OR defer_reason = '') THEN 1 ELSE 0 END) AS pending_nonblog_excluded_from_blog_queue,
                SUM(CASE WHEN source_medium = 'blog' AND has_omission = 1
                          AND status NOT IN ('quarantined', 'skipped') THEN 1 ELSE 0 END) AS blogs_with_omission,
                SUM(CASE WHEN source_medium = 'blog' AND selected = 1
                          AND status IN ('pending', 'failed_retryable', 'interrupted')
                          AND (defer_reason IS NULL OR defer_reason = '')
                         THEN COALESCE(estimated_chunks, 0) ELSE 0 END) AS remaining_blog_estimated_chunks
            FROM documents
            """
        ).fetchone()
        summary = {key: int(row[key] or 0) for key in row.keys()}
        summary["description"] = "mixed_media_seed_with_blog_only_expansion"
        summary["by_medium"] = self.inventory_by_medium()
        return summary

    def mark_processing(self, document_id: str, checkpoint_id: str) -> dict:
        now = utc_now()
        with self._tx() as conn:
            current = conn.execute(
                "SELECT attempt_number, status FROM documents WHERE document_id = ?",
                (document_id,),
            ).fetchone()
            if current is None:
                raise KeyError(document_id)
            if current["status"] == "completed":
                raise RuntimeError(f"{document_id} is already completed")
            attempt = int(current["attempt_number"]) + 1
            conn.execute(
                """
                UPDATE documents
                SET status = 'processing', attempt_number = ?, started_at = ?,
                    finished_at = NULL, runtime_seconds = NULL, checkpoint_id = ?,
                    error_message = NULL
                WHERE document_id = ?
                """,
                (attempt, now, checkpoint_id, document_id),
            )
            conn.execute(
                """
                INSERT INTO attempts (
                    document_id, attempt_number, started_at, status, checkpoint_id
                ) VALUES (?, ?, ?, 'processing', ?)
                """,
                (document_id, attempt, now, checkpoint_id),
            )
        document = self.document(document_id)
        assert document is not None
        return document

    def record_outcome(self, document_id: str, outcome: dict) -> None:
        """Close the open attempt and set the document status.

        Earlier attempt rows are not updated except for the row this call
        opened, which receives the finish fields.
        """
        now = outcome.get("finished_at") or utc_now()
        with self._tx() as conn:
            current = conn.execute(
                "SELECT attempt_number FROM documents WHERE document_id = ?",
                (document_id,),
            ).fetchone()
            if current is None:
                raise KeyError(document_id)
            attempt = int(current["attempt_number"])
            conn.execute(
                """
                UPDATE documents
                SET status = ?, finished_at = ?, runtime_seconds = ?,
                    error_stage = ?, error_category = ?, error_message = ?,
                    retry_eligible = ?, checkpoint_id = ?,
                    persisted_lightrag_id = ?, indexed_chunk_count = ?,
                    extraction_metrics = ?, embedding_metrics = ?,
                    graph_persistence = ?, vector_persistence = ?,
                    provenance_verified = ?, vector_only = ?, graph_extracted = ?,
                    defer_reason = CASE
                        WHEN ? IS NULL THEN defer_reason
                        ELSE ?
                    END,
                    quality_notes = ?
                WHERE document_id = ?
                """,
                (
                    outcome["status"],
                    now,
                    outcome.get("runtime_seconds"),
                    outcome.get("error_stage"),
                    outcome.get("error_category"),
                    outcome.get("error_message"),
                    int(bool(outcome.get("retry_eligible"))),
                    outcome.get("checkpoint_id"),
                    outcome.get("persisted_lightrag_id"),
                    outcome.get("indexed_chunk_count"),
                    _json(outcome.get("extraction_metrics")),
                    _json(outcome.get("embedding_metrics")),
                    outcome.get("graph_persistence"),
                    outcome.get("vector_persistence"),
                    None if outcome.get("provenance_verified") is None else int(bool(outcome["provenance_verified"])),
                    int(bool(outcome.get("vector_only"))),
                    int(bool(outcome.get("graph_extracted"))),
                    outcome.get("defer_reason"),
                    outcome.get("defer_reason"),
                    outcome.get("quality_notes"),
                    document_id,
                ),
            )
            conn.execute(
                """
                UPDATE attempts
                SET finished_at = ?, runtime_seconds = ?, status = ?,
                    error_stage = ?, error_category = ?, error_message = ?,
                    checkpoint_id = ?, stage_timings = ?, extraction_metrics = ?,
                    embedding_metrics = ?, persistence_result = ?
                WHERE document_id = ? AND attempt_number = ?
                """,
                (
                    now,
                    outcome.get("runtime_seconds"),
                    outcome["status"],
                    outcome.get("error_stage"),
                    outcome.get("error_category"),
                    outcome.get("error_message"),
                    outcome.get("checkpoint_id"),
                    _json(outcome.get("stage_timings")),
                    _json(outcome.get("extraction_metrics")),
                    _json(outcome.get("embedding_metrics")),
                    _json(
                        {
                            "graph_persistence": outcome.get("graph_persistence"),
                            "vector_persistence": outcome.get("vector_persistence"),
                            "provenance_verified": outcome.get("provenance_verified"),
                        }
                    ),
                    document_id,
                    attempt,
                ),
            )

    def counts(self) -> dict:
        rows = self._conn.execute(
            "SELECT status, COUNT(*) AS n FROM documents GROUP BY status"
        ).fetchall()
        by_status = {row["status"]: int(row["n"]) for row in rows}
        for status in STATUSES:
            by_status.setdefault(status, 0)
        extra = self._conn.execute(
            """
            SELECT
                COUNT(*) AS eligible,
                SUM(CASE WHEN selected = 1 THEN 1 ELSE 0 END) AS selected,
                SUM(CASE WHEN status = 'completed' AND graph_extracted = 1 THEN 1 ELSE 0 END) AS completed_graph,
                SUM(CASE WHEN status = 'completed' AND vector_only = 1 THEN 1 ELSE 0 END) AS completed_vector_only,
                SUM(CASE WHEN defer_reason IS NOT NULL AND defer_reason != '' THEN 1 ELSE 0 END) AS deferred,
                SUM(COALESCE(indexed_chunk_count, 0)) AS chunks,
                SUM(COALESCE(runtime_seconds, 0)) AS runtime_seconds
            FROM documents
            """
        ).fetchone()
        return {
            "by_status": by_status,
            "eligible": int(extra["eligible"] or 0),
            "selected": int(extra["selected"] or 0),
            "completed_graph": int(extra["completed_graph"] or 0),
            "completed_vector_only": int(extra["completed_vector_only"] or 0),
            "deferred": int(extra["deferred"] or 0),
            "indexed_chunks": int(extra["chunks"] or 0),
            "runtime_seconds": float(extra["runtime_seconds"] or 0),
        }

    def iter_documents(self) -> list[dict]:
        rows = self._conn.execute(
            "SELECT * FROM documents ORDER BY COALESCE(selection_rank, 1000000000), document_id"
        ).fetchall()
        return [self._row(row) for row in rows]

    def iter_attempts(self) -> list[dict]:
        rows = self._conn.execute("SELECT * FROM attempts ORDER BY id").fetchall()
        result = []
        for row in rows:
            item = dict(row)
            for key in ("stage_timings", "extraction_metrics", "embedding_metrics", "persistence_result"):
                item[key] = _load(item.get(key))
            result.append(item)
        return result

    def runtime_samples(self) -> list[float]:
        rows = self._conn.execute(
            """
            SELECT runtime_seconds FROM documents
            WHERE status = 'completed' AND runtime_seconds IS NOT NULL
            """
        ).fetchall()
        return [float(row["runtime_seconds"]) for row in rows]

    def unsuccessful_attempt_seconds(self) -> list[float]:
        rows = self._conn.execute(
            """
            SELECT runtime_seconds FROM attempts
            WHERE status NOT IN ('completed', 'reconciled')
              AND runtime_seconds IS NOT NULL
              AND runtime_seconds > 0
            """
        ).fetchall()
        return [float(row["runtime_seconds"]) for row in rows]

    def runtime_samples_for_medium(self, medium: str) -> list[float]:
        rows = self._conn.execute(
            """
            SELECT runtime_seconds FROM documents
            WHERE status = 'completed' AND source_medium = ? AND runtime_seconds IS NOT NULL
            """,
            (medium,),
        ).fetchall()
        return [float(row["runtime_seconds"]) for row in rows]

    def unsuccessful_attempt_seconds_for_medium(self, medium: str) -> list[float]:
        rows = self._conn.execute(
            """
            SELECT attempts.runtime_seconds
            FROM attempts
            JOIN documents ON documents.document_id = attempts.document_id
            WHERE documents.source_medium = ?
              AND attempts.status NOT IN ('completed', 'reconciled')
              AND attempts.runtime_seconds IS NOT NULL
              AND attempts.runtime_seconds > 0
            """,
            (medium,),
        ).fetchall()
        return [float(row["runtime_seconds"]) for row in rows]

    def queue_sizes(self) -> dict:
        row = self._conn.execute(
            """
            SELECT
                SUM(CASE
                    WHEN selected = 1
                     AND status IN ('pending', 'failed_retryable', 'interrupted')
                     AND (defer_reason IS NULL OR defer_reason = '')
                    THEN 1 ELSE 0 END) AS normal_queue,
                SUM(CASE
                    WHEN defer_reason = 'expensive_timeout'
                     AND status NOT IN ('completed', 'skipped')
                    THEN 1 ELSE 0 END) AS deferred_retry_queue
            FROM documents
            """
        ).fetchone()
        tiers = {"small": 0, "medium": 0, "large": 0}
        for item in self._conn.execute(
            """
            SELECT estimated_chunks FROM documents
            WHERE selected = 1 AND status = 'pending'
              AND (defer_reason IS NULL OR defer_reason = '')
            """
        ):
            tiers[classify_complexity(item["estimated_chunks"])] += 1
        return {
            "normal_queue": int(row["normal_queue"] or 0),
            "deferred_retry_queue": int(row["deferred_retry_queue"] or 0),
            "normal_queue_by_tier": tiers,
        }

    def progress(self) -> dict:
        sizes = self.queue_sizes()
        return progress_estimate(
            self.runtime_samples(),
            self.unsuccessful_attempt_seconds(),
            normal_queued=sizes["normal_queue"],
            deferred_queued=sizes["deferred_retry_queue"],
        )

    def reconcile_status(self, document_id: str, outcome: dict) -> None:
        """Update a stale row after comparing it with LightRAG.

        Inserts a new attempt note and does not rewrite older attempt rows.
        """
        now = utc_now()
        with self._tx() as conn:
            current = conn.execute(
                "SELECT attempt_number FROM documents WHERE document_id = ?",
                (document_id,),
            ).fetchone()
            if current is None:
                raise KeyError(document_id)
            conn.execute(
                """
                UPDATE documents
                SET status = ?, finished_at = ?, error_stage = ?, error_category = ?,
                    error_message = ?, retry_eligible = ?, persisted_lightrag_id = ?,
                    indexed_chunk_count = ?, graph_persistence = ?, vector_persistence = ?,
                    provenance_verified = ?, vector_only = ?, graph_extracted = ?,
                    defer_reason = CASE
                        WHEN ? IS NULL THEN defer_reason
                        ELSE ?
                    END,
                    quality_notes = ?
                WHERE document_id = ?
                """,
                (
                    outcome["status"],
                    now,
                    outcome.get("error_stage"),
                    outcome.get("error_category"),
                    outcome.get("error_message"),
                    int(bool(outcome.get("retry_eligible"))),
                    outcome.get("persisted_lightrag_id"),
                    outcome.get("indexed_chunk_count"),
                    outcome.get("graph_persistence"),
                    outcome.get("vector_persistence"),
                    None
                    if outcome.get("provenance_verified") is None
                    else int(bool(outcome["provenance_verified"])),
                    int(bool(outcome.get("vector_only"))),
                    int(bool(outcome.get("graph_extracted"))),
                    outcome.get("defer_reason"),
                    outcome.get("defer_reason"),
                    outcome.get("quality_notes"),
                    document_id,
                ),
            )
            conn.execute(
                """
                INSERT INTO attempts (
                    document_id, attempt_number, started_at, finished_at, status,
                    error_stage, error_category, error_message, persistence_result
                ) VALUES (?, ?, ?, ?, 'reconciled', ?, ?, ?, ?)
                """,
                (
                    document_id,
                    int(current["attempt_number"]),
                    now,
                    now,
                    outcome.get("error_stage"),
                    outcome.get("error_category"),
                    outcome.get("error_message"),
                    _json({"reconciled_status": outcome["status"]}),
                ),
            )

    def record_session(self, session_id: str, config: dict, summary: dict, stop_reason: str) -> None:
        with self._tx() as conn:
            conn.execute(
                """
                INSERT INTO sessions(session_id, started_at, finished_at, stop_reason, config_json, summary_json)
                VALUES(?, ?, ?, ?, ?, ?)
                ON CONFLICT(session_id) DO UPDATE SET
                    finished_at = excluded.finished_at,
                    stop_reason = excluded.stop_reason,
                    summary_json = excluded.summary_json
                """,
                (
                    session_id,
                    summary.get("started_at") or utc_now(),
                    utc_now(),
                    stop_reason,
                    _json(config),
                    _json(summary),
                ),
            )

    def _row(self, row: sqlite3.Row) -> dict:
        item = dict(row)
        item["topic_labels"] = _load(item.get("topic_labels")) or []
        item["extraction_metrics"] = _load(item.get("extraction_metrics"))
        item["embedding_metrics"] = _load(item.get("embedding_metrics"))
        return item
