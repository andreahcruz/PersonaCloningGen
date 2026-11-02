"""Score saved predictions. This does not retrieve or generate."""

from __future__ import annotations

from host_finetune.eval_bench.matrix import matrix_by_architecture
from host_finetune.eval_bench.metrics import score_efficiency, score_generation, score_retrieval
from host_finetune.eval_bench.schema import RecordError, prediction_key, validate_record
from host_finetune.eval_bench.stats import paired_mean_difference, wilson_interval
from host_finetune.model_f_context.leakage import HELD_OUT_SPLITS


def _mean(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def _held_out(evidence: list) -> list[str]:
    found = []
    for item in evidence:
        if not isinstance(item, dict):
            continue
        if str(item.get("split") or "") in HELD_OUT_SPLITS:
            found.append(str(item.get("id") or item.get("graph_document_id") or "unknown"))
    return found


def score_batch(records: list[dict]) -> dict:
    """Validate, drop duplicates and held-out hits, and score the rest."""
    results = []
    failures = []
    seen = set()
    for raw in records:
        try:
            record = validate_record(raw)
        except RecordError as exc:
            failures.append({
                "question_id": (raw or {}).get("question_id") if isinstance(raw, dict) else None,
                "architecture": (raw or {}).get("architecture") if isinstance(raw, dict) else None,
                "error_class": "schema_error",
                "error": str(exc),
            })
            continue
        key = prediction_key(record)
        if key in seen:
            failures.append({
                "question_id": record["question_id"],
                "architecture": record["architecture"],
                "medium": record["medium"],
                "error_class": "duplicate_prediction",
                "error": "same experiment, architecture, question, and medium was already scored",
            })
            continue
        seen.add(key)
        if record.get("error"):
            failures.append({
                "question_id": record["question_id"],
                "architecture": record["architecture"],
                "error_class": "recorded_error",
                "error": record["error"],
            })
            continue
        contaminated = _held_out(record["evidence"])
        if contaminated:
            failures.append({
                "question_id": record["question_id"],
                "architecture": record["architecture"],
                "error_class": "held_out_evidence",
                "error": "evidence includes validation or test rows: " + ", ".join(contaminated),
            })
            continue
        scopes = set(record.get("scopes") or ()) or {"retrieval", "generation"}
        retrieval = score_retrieval(record) if "retrieval" in scopes else {
            "status": "not_requested",
            "empty": None,
            "precision_at_4": None,
            "recall_at_4": None,
            "mrr": None,
            "ndcg_at_4": None,
            "source_diversity": None,
            "provenance_completeness": None,
        }
        generation = score_generation(record) if "generation" in scopes else {
            "status": "not_requested",
            "faithfulness": {"status": "not_requested", "pass": None},
            "numeric_pass": None,
            "numeric_applicable": False,
            "numeric_value_count": 0,
            "citation_marker_leak": None,
            "word_count": None,
            "unsupported_numeric_count": None,
        }
        efficiency = score_efficiency(record)
        if retrieval.get("empty") is True:
            failures.append({
                "question_id": record["question_id"],
                "architecture": record["architecture"],
                "error_class": "empty_retrieval",
                "error": "no evidence hits; ranked retrieval metrics stay null",
            })
        if record.get("reference_answer") is None:
            generation["reference_answer"] = None
        scored = dict(record)
        scored["retrieval"] = retrieval
        scored["generation"] = generation
        scored["efficiency"] = efficiency
        results.append(scored)
    return {
        "results": results,
        "failures": failures,
        "metrics": aggregate(results),
        "per_question": [
            {
                "question_id": row["question_id"],
                "architecture": row["architecture"],
                "medium": row["medium"],
                "retrieved_source_ids": row["retrieved_source_ids"],
                "retrieval": row["retrieval"],
                "generation": {
                    "faithfulness": (row.get("generation") or {}).get("faithfulness"),
                    "numeric_pass": (row.get("generation") or {}).get("numeric_pass"),
                    "unsupported_numeric_count": (row.get("generation") or {}).get("unsupported_numeric_count"),
                    "citation_marker_leak": (row.get("generation") or {}).get("citation_marker_leak"),
                    "word_count": (row.get("generation") or {}).get("word_count"),
                },
                "efficiency": row["efficiency"],
            }
            for row in results
        ],
    }


def _faithfulness_bit(row: dict) -> float | None:
    faith = (row.get("generation") or {}).get("faithfulness") or {}
    if faith.get("status") == "not_scored" or faith.get("pass") is None:
        return None
    return 1.0 if faith.get("pass") else 0.0


def aggregate(results: list[dict]) -> dict:
    by_architecture: dict[str, list[dict]] = {}
    for row in results:
        by_architecture.setdefault(row["architecture"], []).append(row)
    arms = {}
    for architecture, rows in sorted(by_architecture.items()):
        def values(path) -> list[float]:
            found = []
            for row in rows:
                value = path(row)
                if isinstance(value, (int, float)):
                    found.append(float(value))
            return found

        faith = [bit for bit in (_faithfulness_bit(row) for row in rows) if bit is not None]
        numeric = [
            1.0 if row["generation"]["numeric_pass"] else 0.0
            for row in rows
            if row["generation"]["numeric_applicable"] and row["generation"]["numeric_value_count"]
        ]
        arms[architecture] = {
            "n": len(rows),
            "precision_at_4_mean": _mean(values(lambda row: row["retrieval"]["precision_at_4"])),
            "recall_at_4_mean": _mean(values(lambda row: row["retrieval"]["recall_at_4"])),
            "mrr_mean": _mean(values(lambda row: row["retrieval"]["mrr"])),
            "ndcg_at_4_mean": _mean(values(lambda row: row["retrieval"]["ndcg_at_4"])),
            "source_diversity_mean": _mean(values(lambda row: row["retrieval"]["source_diversity"])),
            "provenance_completeness_mean": _mean(values(lambda row: row["retrieval"]["provenance_completeness"])),
            "faithfulness_rate": _mean(faith),
            "faithfulness_ci95": wilson_interval(int(sum(faith)), len(faith)) if faith else None,
            "numeric_pass_rate": _mean(numeric),
            "citation_leak_rate": _mean([
                1.0 if row["generation"]["citation_marker_leak"] else 0.0
                for row in rows
                if isinstance((row.get("generation") or {}).get("citation_marker_leak"), bool)
            ]),
            "mean_word_count": _mean(values(lambda row: row["generation"]["word_count"])),
            "known_matrix_arm": architecture in matrix_by_architecture(),
        }
    return {
        "composite_score": None,
        "reason": "Retrieval, faithfulness, numeric checks, and latency stay separate.",
        "arms": arms,
    }


def _pair_key(row: dict) -> str:
    """Same question and medium. A second medium is a second pair."""
    medium = str(row.get("medium") or "")
    if not medium:
        return str(row["question_id"])
    return str(row["question_id"]) + "\t" + medium


def paired_faithfulness(results: list[dict], left: str, right: str, *, seed: int = 298) -> dict:
    """Align two architectures on question id and medium, then compare faithfulness."""
    by_question: dict[str, dict[str, float]] = {}
    for row in results:
        bit = _faithfulness_bit(row)
        if bit is None:
            continue
        by_question.setdefault(_pair_key(row), {})[row["architecture"]] = bit
    paired_left = []
    paired_right = []
    for question_id in sorted(by_question):
        sides = by_question[question_id]
        if left in sides and right in sides:
            paired_left.append(sides[left])
            paired_right.append(sides[right])
    if not paired_left:
        return {
            "status": "not_comparable",
            "reason": "no shared scored questions",
            "n": 0,
            "paired_on": "question_id_and_medium",
        }
    difference = paired_mean_difference(paired_left, paired_right, seed=seed)
    difference["left"] = left
    difference["right"] = right
    difference["status"] = "scored"
    difference["paired_on"] = "question_id_and_medium"
    return difference
