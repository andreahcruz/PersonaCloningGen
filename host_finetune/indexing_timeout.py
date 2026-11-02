"""Document time budgets that do not change the graph configuration fingerprint.

The budgets come from the expansion ledger measured on 2026-10-31.
Seventy-three completed documents had a median of 7.2 seconds and a p95 of
33 seconds. One completed document took 356 seconds, and six interrupted
attempts sat at the 420-second cap for about 436 seconds each. Chunk count
predicts the normal runs. It does not predict those stalls.

``adaptive-v1`` is an operational policy. It is recorded on the session, not
in the source-configuration fingerprint.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass


POLICY_VERSION = "adaptive-v1"


@dataclass(frozen=True)
class TimeoutPolicy:
    name: str
    floor_seconds: float = 60.0
    ceiling_seconds: float = 420.0
    small_seconds: float = 90.0
    medium_seconds: float = 180.0
    large_seconds: float = 420.0
    retry_seconds: float = 420.0

    def record(self) -> dict:
        return {
            "version": self.name,
            "floor_seconds": self.floor_seconds,
            "ceiling_seconds": self.ceiling_seconds,
            "small_seconds": self.small_seconds,
            "medium_seconds": self.medium_seconds,
            "large_seconds": self.large_seconds,
            "retry_seconds": self.retry_seconds,
            "basis": (
                "Completed-document p95 was 33 seconds across 73 documents. "
                "Small and medium budgets sit above that p95. The 420-second "
                "ceiling and the explicit retry budget stay at the previous fixed cap."
            ),
        }


ADAPTIVE_V1 = TimeoutPolicy(name=POLICY_VERSION)
FIXED = TimeoutPolicy(name="fixed")


def classify_complexity(estimated_chunks: int | None) -> str:
    """Tier a document from the pre-ingestion chunk estimate.

    The estimate is characters/4 divided by the chunk size. It is not the
    tokenizer count and it is not the chunk count LightRAG will write.
    """
    try:
        count = int(estimated_chunks or 1)
    except (TypeError, ValueError):
        count = 1
    if count <= 1:
        return "small"
    if count <= 3:
        return "medium"
    return "large"


def select_timeout(
    policy: TimeoutPolicy,
    estimated_chunks: int | None,
    *,
    retry: bool,
    fixed_seconds: float | None,
) -> float | None:
    """Return the document budget.

    ``fixed`` returns ``fixed_seconds`` unchanged, including None when the
    caller disabled the document timer. Adaptive budgets never exceed the
    lower of the policy ceiling and ``fixed_seconds``.
    """
    if policy.name == "fixed":
        return fixed_seconds
    budget = policy.retry_seconds if retry else {
        "small": policy.small_seconds,
        "medium": policy.medium_seconds,
        "large": policy.large_seconds,
    }[classify_complexity(estimated_chunks)]
    ceiling = policy.ceiling_seconds
    if fixed_seconds is not None:
        ceiling = min(ceiling, fixed_seconds)
    floor = min(policy.floor_seconds, ceiling)
    return min(ceiling, max(floor, budget))


def projected_timeout_savings(attempt_seconds: list[float], replacement_seconds: float) -> dict:
    """Counterfactual time saved if each long attempt had stopped at a budget.

    This is not a measured speedup. It only rewrites historical attempt
    durations that already exceeded the replacement budget.
    """
    saved = 0.0
    affected = 0
    for seconds in attempt_seconds:
        if seconds > replacement_seconds:
            saved += seconds - replacement_seconds
            affected += 1
    return {
        "label": "counterfactual_not_a_measured_speedup",
        "attempts_over_budget": affected,
        "replacement_seconds": replacement_seconds,
        "seconds_saved": round(saved, 1),
    }


async def abortable(inner, cancel_event: asyncio.Event, inflight: set | None = None):
    """Run ``inner`` until it finishes or ``cancel_event`` is set.

    Cancelling the LightRAG pipeline task does not cancel the priority-queue
    worker's inner call. This wrapper owns that inner task and cancels it.
    """
    task = asyncio.ensure_future(inner)
    if inflight is not None:
        inflight.add(task)
    waiter = asyncio.create_task(cancel_event.wait())
    try:
        if cancel_event.is_set():
            task.cancel()
            await _wait_cancelled(task)
            raise asyncio.CancelledError()
        done, _pending = await asyncio.wait({task, waiter}, return_when=asyncio.FIRST_COMPLETED)
        if task in done:
            return task.result()
        task.cancel()
        await _wait_cancelled(task)
        raise asyncio.CancelledError()
    finally:
        if inflight is not None:
            inflight.discard(task)
        if not waiter.done():
            waiter.cancel()
            await _wait_cancelled(waiter)


async def _wait_cancelled(task: asyncio.Future) -> None:
    try:
        await task
    except asyncio.CancelledError:
        return
    except Exception:
        return


def progress_estimate(
    completed_seconds: list[float],
    unsuccessful_seconds: list[float],
    *,
    normal_queued: int,
    deferred_queued: int,
) -> dict:
    """Two ETAs. Both include historical unsuccessful attempt time.

    The normal-queue ETA does not assume the deferred documents will be
    retried. The second ETA adds them only when the caller asked for that
    queue. Neither is a completion date.
    """
    sample = len(completed_seconds)
    success_sum = float(sum(completed_seconds))
    lost = float(sum(unsuccessful_seconds))
    if sample < 30:
        confidence = "low"
    elif sample < 100:
        confidence = "limited"
    else:
        confidence = "moderate"
    if sample == 0:
        per_document = None
        normal = None
        combined = None
    else:
        per_document = (success_sum + lost) / sample
        normal = per_document * normal_queued
        combined = per_document * (normal_queued + deferred_queued)
    return {
        "label": "approximate",
        "confidence": confidence,
        "completed_sample": sample,
        "successful_seconds": round(success_sum, 1),
        "unsuccessful_attempt_seconds": round(lost, 1),
        "effective_seconds_per_completed_document": None if per_document is None else round(per_document, 1),
        "normal_queue_remaining": normal_queued,
        "deferred_retry_queue_remaining": deferred_queued,
        "normal_queue_seconds": None if normal is None else round(normal, 1),
        "normal_plus_deferred_retry_seconds": None if combined is None else round(combined, 1),
        "note": (
            "The rate is completed runtime plus unsuccessful attempt runtime, "
            "divided by completed documents. It does not predict which future "
            "documents will stall. The deferred-retry ETA applies only when "
            "that queue is explicitly enabled."
        ),
    }
