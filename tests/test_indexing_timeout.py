"""Adaptive timeouts and cancellation. These tests do not open the expansion store."""

from __future__ import annotations

import asyncio
import time
from pathlib import Path

import pytest

from host_finetune.indexing_ledger import IndexingLedger
from host_finetune.indexing_lightrag import IndexConfig, configuration_fingerprint, configuration_record
from host_finetune.indexing_schedule import build_schedule
from host_finetune.indexing_timeout import (
    ADAPTIVE_V1,
    FIXED,
    abortable,
    classify_complexity,
    progress_estimate,
    projected_timeout_savings,
    select_timeout,
)


def test_complexity_and_timeout_bounds_are_stable():
    assert classify_complexity(None) == "small"
    assert classify_complexity(1) == "small"
    assert classify_complexity(2) == "medium"
    assert classify_complexity(3) == "medium"
    assert classify_complexity(4) == "large"
    assert select_timeout(FIXED, 1, retry=False, fixed_seconds=420) == 420
    assert select_timeout(FIXED, 9, retry=True, fixed_seconds=0.4) == 0.4
    assert select_timeout(ADAPTIVE_V1, 1, retry=False, fixed_seconds=420) == 90
    assert select_timeout(ADAPTIVE_V1, 3, retry=False, fixed_seconds=420) == 180
    assert select_timeout(ADAPTIVE_V1, 4, retry=False, fixed_seconds=420) == 420
    assert select_timeout(ADAPTIVE_V1, 1, retry=True, fixed_seconds=420) == 420
    assert select_timeout(ADAPTIVE_V1, 1, retry=False, fixed_seconds=60) == 60
    first = select_timeout(ADAPTIVE_V1, 2, retry=False, fixed_seconds=420)
    second = select_timeout(ADAPTIVE_V1, 2, retry=False, fixed_seconds=420)
    assert first == second == 180
    assert ADAPTIVE_V1.record()["version"] == "adaptive-v1"


def test_timeout_policy_is_outside_the_graph_fingerprint():
    config = IndexConfig(
        extraction_model="llama3.1:latest",
        embed_model="nomic-embed-text",
        base_url="http://localhost:11434",
    )
    record = configuration_record(config)
    assert "timeout_policy" not in record
    assert configuration_fingerprint(config) == configuration_fingerprint(config)
    assert record["llm_timeout_seconds"] == 600
    assert record["chunk_token_size"] == 400
    assert record["embedding_dim"] == 768


def test_normal_queue_does_not_retry_a_timeout_until_asked(tmp_path: Path):
    document = {
        "graph_document_id": "gd_slow",
        "text": "SaaStr annual showed sales and marketing together.",
        "text_sha256": "abc",
        "medium": "blog",
        "split": "train",
        "attribution_status": "publication_page",
        "word_count": 8,
        "included": True,
    }
    rows, _summary = build_schedule([document], corpus_hash="c", fingerprint="f", indexing_mode="graph")
    rows[0]["status"] = "interrupted"
    rows[0]["defer_reason"] = "expensive_timeout"
    rows[0]["error_category"] = "process_interruption"
    ledger = IndexingLedger(tmp_path / "indexing_ledger.sqlite")
    try:
        ledger.sync_corpus(rows, corpus_hash="c", fingerprint="f")
        assert ledger.next_document(include_deferred=False, retry_timeouts=False) is None
        retried = ledger.next_document(include_deferred=False, retry_timeouts=True)
        assert retried["document_id"] == "gd_slow"
        assert select_timeout(
            ADAPTIVE_V1,
            retried["estimated_chunks"],
            retry=True,
            fixed_seconds=420,
        ) == 420
        sizes = ledger.queue_sizes()
        assert sizes["normal_queue"] == 0
        assert sizes["deferred_retry_queue"] == 1
    finally:
        ledger.close()


def test_eta_includes_unsuccessful_time_and_keeps_the_queues_separate():
    estimate = progress_estimate(
        [10.0, 20.0],
        [436.0],
        normal_queued=4,
        deferred_queued=1,
    )
    assert estimate["effective_seconds_per_completed_document"] == 233.0
    assert estimate["normal_queue_seconds"] == 932.0
    assert estimate["normal_plus_deferred_retry_seconds"] == 1165.0
    assert estimate["confidence"] == "low"
    assert estimate["unsuccessful_attempt_seconds"] == 436.0
    savings = projected_timeout_savings([436.1, 436.1, 15.0], 180)
    assert savings["attempts_over_budget"] == 2
    assert savings["seconds_saved"] == pytest.approx(512.2)
    assert savings["label"] == "counterfactual_not_a_measured_speedup"


def test_abortable_cancels_the_inner_task():
    state = {"finished": False, "cancelled": False}

    async def hang():
        try:
            await asyncio.sleep(30)
            state["finished"] = True
        except asyncio.CancelledError:
            state["cancelled"] = True
            raise

    async def run():
        cancel = asyncio.Event()
        inflight = set()
        job = asyncio.create_task(abortable(hang(), cancel, inflight))
        await asyncio.sleep(0.05)
        assert inflight
        started = time.monotonic()
        cancel.set()
        with pytest.raises(asyncio.CancelledError):
            await job
        return time.monotonic() - started

    elapsed = asyncio.run(run())
    assert elapsed < 1.5
    assert state["cancelled"] is True
    assert state["finished"] is False


def test_priority_queue_does_not_overlap_a_cancelled_extraction():
    """The installed LightRAG 1.5.7 queue keeps running a call after the caller is cancelled.

    abortable cancels that inner call before another document is allowed to start.
    """
    from lightrag.utils import priority_limit_async_func_call

    state = {"running": 0, "max_running": 0, "inner_cancelled": False, "quick_started": None, "hang_finished": None}

    async def hang():
        state["running"] += 1
        state["max_running"] = max(state["max_running"], state["running"])
        try:
            await asyncio.sleep(30)
        except asyncio.CancelledError:
            state["inner_cancelled"] = True
            raise
        finally:
            state["running"] -= 1
            state["hang_finished"] = time.monotonic()

    async def quick():
        state["quick_started"] = time.monotonic()
        return "done"

    @priority_limit_async_func_call(max_size=1, llm_timeout=60, queue_name="indexing-timeout-test")
    async def queued(body):
        return await body()

    async def run():
        cancel = asyncio.Event()
        first = asyncio.create_task(queued(lambda: abortable(hang(), cancel)))
        await asyncio.sleep(0.05)
        assert state["running"] == 1
        started = time.monotonic()
        cancel.set()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(first, timeout=2)
        cut = time.monotonic() - started
        second = await asyncio.wait_for(queued(quick), timeout=2)
        await queued.shutdown(graceful=False, timeout=1)
        return cut, second

    cut, second = asyncio.run(run())
    assert cut < 1.5
    assert second == "done"
    assert state["inner_cancelled"] is True
    assert state["max_running"] == 1
    assert state["running"] == 0
    assert state["quick_started"] >= state["hang_finished"]
