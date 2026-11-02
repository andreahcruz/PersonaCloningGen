"""Convert saved Model E and Model F traces into eval-bench records.

EXP-20261031-001 used the letters A–D for a passage ablation. Those letters
are not the five context profiles. This module keeps the original cell name
and refuses to relabel a cell as a profile it did not run.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from host_finetune.compare_adapters import GENERATION_SEED, REPETITION_PENALTY, TEMPERATURE, TOP_P
from host_finetune.eval_bench.annotation import blank_template, pools_from_retrieval
from host_finetune.eval_bench.matrix import profile_alignment
from host_finetune.eval_bench.metrics import designated_source_coverage
from host_finetune.eval_bench.report import refuse_live_store, write_report
from host_finetune.eval_bench.runner import paired_faithfulness, score_batch

SOURCE_EXPERIMENT = "EXP-20261031-001-graph-relationship-conditioning"
CORPUS = {
    "id": "pilot160",
    "processed_documents_recorded": 152,
    "index": "artifacts/lightrag/model_f/storage_pilot160",
    "eligible_for_final_model_f_comparison": False,
    "reason": "Pilot graph. Not the blog expansion and not the active Chroma train index.",
}

CELLS = {
    "A": {
        "architecture": "historical_hybrid_passages",
        "nearest_matrix_architecture": "model_f_source",
        "nearest_profile": "source",
        "profile_match": False,
        "reason": "Hybrid passages under prompt contract factual-reliability-v1. The source profile was not used.",
    },
    "B": {
        "architecture": "historical_hybrid_plus_relationships",
        "nearest_matrix_architecture": "model_f_relationships",
        "nearest_profile": "relationships",
        "profile_match": False,
        "reason": "In-passage relationship block from generation_relationships. Not the relationships profile.",
    },
    "C": {
        "architecture": "historical_matched_dense_preview",
        "nearest_matrix_architecture": "model_e",
        "nearest_profile": "off",
        "profile_match": False,
        "reason": "Matched dense search over pilot texts, truncated to a 400-character preview. Not the active Chroma index.",
    },
    "D_blog": {
        "architecture": "historical_blog_token_pad",
        "nearest_matrix_architecture": None,
        "nearest_profile": None,
        "profile_match": False,
        "reason": "Blog-only token pad. It is not the descriptions profile and it is not multi-hop.",
    },
}

KNOWN_ABLATION = {("A", "B")}


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _load_json(path: Path) -> dict:
    refuse_live_store(path)
    return json.loads(path.read_text(encoding="utf-8"))


def _load_jsonl(path: Path) -> list[dict]:
    refuse_live_store(path)
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def questions_by_query(path: Path) -> dict[str, dict]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    found = {}
    for row in rows:
        query = str(row.get("query") or "")
        if query in found:
            raise ValueError(f"duplicate query in {path}: {row.get('id')}")
        found[query] = row
    return found


def generation_record(trace: dict, *, cell: str, source: str, experiment_id: str) -> dict:
    """One saved draft. Passage text is kept so the factual check can be rerun."""
    spec = CELLS[cell]
    hits = [dict(hit) for hit in (trace.get("retrieval_hits") or []) if isinstance(hit, dict)]
    relations = [item for item in (trace.get("graph_relationships") or []) if isinstance(item, dict)]
    extra = [str(item.get("prompt_fact") or "") for item in relations if item.get("prompt_fact")]
    claims = [dict(item) for item in (trace.get("verified_claims") or []) if isinstance(item, dict)]
    prompt = str(trace.get("prompt") or "")
    return {
        "experiment_id": experiment_id,
        "architecture": spec["architecture"],
        "model_version": "host_finetune/output/lemkin_lora_relabel",
        "question_id": str(trace.get("id") or ""),
        "question": str(trace.get("topic") or trace.get("query") or ""),
        "medium": trace.get("medium"),
        "uses_graph": cell == "B",
        "scopes": ["retrieval", "generation"],
        "retrieval_config": {
            "backend": trace.get("retrieval_backend"),
            "mode": trace.get("retrieval_mode"),
            "evidence_budget": trace.get("retrieval_k"),
            "corpus": CORPUS["id"],
            "historical_cell": cell,
            "context_profile": None,
            "profile_match": False,
        },
        "generation_config": {
            "prompt_contract": trace.get("prompt_contract"),
            "graph_context_contract": None,
            "seed": trace.get("seed"),
            "retry_seed": trace.get("retry_seed"),
            "max_new_tokens": trace.get("max_new_tokens"),
            "fixed_style_enabled": trace.get("fixed_style_enabled"),
            "context_pad": bool(trace.get("context_pad")),
            "temperature_recorded_in_trace": False,
            "temperature_current_code": TEMPERATURE,
            "top_p_current_code": TOP_P,
            "repetition_penalty_current_code": REPETITION_PENALTY,
            "seed_base_current_code": GENERATION_SEED,
            "prompt_sha256": _sha256(prompt) if prompt else None,
        },
        "retrieved_source_ids": [str(hit.get("id") or "") for hit in hits if hit.get("id")],
        "evidence": hits,
        "provenance": [{
            "source_experiment": SOURCE_EXPERIMENT,
            "source_file": source,
            "historical_cell": cell,
            "nearest_matrix_architecture": spec["nearest_matrix_architecture"],
            "nearest_profile": spec["nearest_profile"],
            "profile_match": False,
            "reason": spec["reason"],
            "corpus": CORPUS,
        }],
        "answer": "" if trace.get("answer") is None else str(trace.get("answer")),
        "reference_answer": None,
        "claims": claims,
        "grounding_texts": extra,
        "stored_evidence_status": trace.get("evidence_status"),
        "latency": {
            "generation_seconds": trace.get("latency_seconds"),
            "retrieval_seconds": None,
            "end_to_end_seconds": None,
        },
        "tokens": {
            "prompt": trace.get("prompt_tokens"),
            "output": trace.get("generated_tokens_including_stop"),
            "source": "saved_trace",
        },
        "historical_numeric_pass": (trace.get("numeric_grounding") or {}).get("pass"),
    }


def load_generation_records(directory: Path, experiment_id: str) -> list[dict]:
    refuse_live_store(directory)
    records = []
    for path in sorted((directory / "raw" / "generations").glob("*.json")):
        cell = path.stem.split("__", 1)[1]
        if cell not in CELLS:
            raise ValueError(f"unknown historical cell in {path.name}")
        payload = _load_json(path)
        for trace in payload.get("traces") or []:
            record = generation_record(
                trace,
                cell=cell,
                source=str(path.relative_to(directory)).replace("\\", "/"),
                experiment_id=experiment_id,
            )
            records.append(record)
    return records


def retrieval_record(payload: dict, question: dict, *, source: str, experiment_id: str) -> dict:
    hits = [dict(hit) for hit in (payload.get("hits") or []) if isinstance(hit, dict)]
    mode = str(payload.get("mode") or "")
    architecture = "historical_dense_retrieval" if mode == "dense" else f"historical_lightrag_{mode}"
    return {
        "experiment_id": experiment_id,
        "architecture": architecture,
        "model_version": None,
        "question_id": question["id"],
        "question": question.get("query"),
        "medium": None,
        "uses_graph": mode in {"hybrid", "mix", "local", "global"},
        "scopes": ["retrieval"],
        "retrieval_config": {
            "backend": payload.get("backend"),
            "mode": mode,
            "evidence_budget": payload.get("k"),
            "corpus": CORPUS["id"],
            "context_profile": None,
            "profile_match": False,
        },
        "generation_config": {"prompt_contract": None, "graph_context_contract": None},
        "retrieved_source_ids": [str(hit.get("id") or "") for hit in hits if hit.get("id")],
        "evidence": hits,
        "review_source_ids": list(question.get("review_source_ids") or []),
        "provenance": [{
            "source_experiment": SOURCE_EXPERIMENT,
            "source_file": source,
            "corpus": CORPUS,
            "profile_match": False,
            "reason": "Frozen retrieval payload. No qrels are attached.",
        }],
        "answer": "",
        "reference_answer": None,
        "latency": {
            "retrieval_seconds": payload.get("seconds"),
            "generation_seconds": None,
            "end_to_end_seconds": None,
        },
        "tokens": {},
    }


def load_retrieval_records(directory: Path, questions: Path, experiment_id: str) -> list[dict]:
    by_query = questions_by_query(questions)
    records = []
    for name in ("frozen_pilot_dense.jsonl", "frozen_naive.jsonl", "frozen_hybrid.jsonl", "frozen_mix.jsonl"):
        path = directory / "raw" / name
        for payload in _load_jsonl(path):
            question = by_query.get(str(payload.get("query") or ""))
            if question is None:
                raise ValueError(f"{name} has a query that is not in the development question file")
            records.append(retrieval_record(
                payload,
                question,
                source=f"raw/{name}",
                experiment_id=experiment_id,
            ))
    return records


def _ids(record: dict) -> tuple[str, ...]:
    return tuple(record.get("retrieved_source_ids") or [])


def judge_pair(left: dict, right: dict) -> dict:
    """Decide whether two saved cells can be compared."""
    left_cell = left["provenance"][0]["historical_cell"]
    right_cell = right["provenance"][0]["historical_cell"]
    reasons = []
    status = "historical_ablation"
    if left["question_id"] != right["question_id"]:
        return {"status": "rejected", "reason": "different_question", "comparable": False}
    if left.get("medium") != right.get("medium"):
        return {"status": "rejected", "reason": "different_medium", "comparable": False}
    if left["generation_config"].get("seed") != right["generation_config"].get("seed"):
        reasons.append("seed_differs")
        status = "flagged"
    if left["generation_config"].get("prompt_contract") != right["generation_config"].get("prompt_contract"):
        reasons.append("prompt_contract_differs")
        status = "rejected"
    if left["generation_config"].get("graph_context_contract") or right["generation_config"].get("graph_context_contract"):
        reasons.append("graph_context_contract_present")
        status = "flagged"
    if left["generation_config"].get("prompt_sha256") != right["generation_config"].get("prompt_sha256"):
        reasons.append("prompt_body_differs")
    if _ids(left) != _ids(right):
        reasons.append("passages_not_held_fixed")
        status = "flagged"
    if left["retrieval_config"].get("backend") != right["retrieval_config"].get("backend"):
        reasons.append("retrieval_backend_differs")
        status = "flagged"
    if left["provenance"][0]["corpus"]["id"] != "pilot160" or right["provenance"][0]["corpus"]["id"] != "pilot160":
        reasons.append("corpus_not_recorded")
        status = "flagged"
    else:
        reasons.append("corpus_is_pilot_not_blog_expansion")
    if left_cell == "D_blog" or right_cell == "D_blog":
        reasons.append("token_pad_is_a_length_control")
        status = "flagged"
    if left.get("medium") != "blog" and (left_cell == "D_blog" or right_cell == "D_blog"):
        status = "rejected"
        reasons.append("pad_exists_only_for_blog")
    pair = (left_cell, right_cell)
    if status != "rejected" and pair not in KNOWN_ABLATION and tuple(reversed(pair)) not in KNOWN_ABLATION:
        if "passages_not_held_fixed" in reasons or "token_pad_is_a_length_control" in reasons:
            status = "flagged"
        else:
            status = "flagged"
            reasons.append("not_the_preregistered_ablation")
    if status == "rejected":
        comparable = False
    elif status == "flagged":
        comparable = False
    else:
        comparable = True
    return {
        "status": status,
        "comparable": comparable,
        "interpretable_as_profile_comparison": False,
        "question_id": left["question_id"],
        "medium": left.get("medium"),
        "left_cell": left_cell,
        "right_cell": right_cell,
        "reasons": reasons,
    }


def crosswalk_to_profile(cell: str, profile: str) -> dict:
    spec = CELLS[cell]
    return {
        "status": "rejected",
        "comparable": False,
        "historical_cell": cell,
        "requested_profile": profile,
        "nearest_profile": spec["nearest_profile"],
        "reason": spec["reason"],
    }


def _historical_rows(metrics: dict) -> dict[tuple[str, str, str], dict]:
    found = {}
    for cell, arm in (metrics.get("arms") or {}).items():
        for row in arm.get("rows") or []:
            found[(cell, str(row.get("id")), str(row.get("medium")))] = row
    return found


def reconcile_generation(results: list[dict], metrics: dict) -> dict:
    """Compare a rerun of the same factual function with the saved metric file."""
    saved = _historical_rows(metrics)
    agreements = {"factual_pass": 0, "numeric_pass": 0, "word_count": 0}
    discrepancies = []
    missing = []
    for row in results:
        cell = row["provenance"][0]["historical_cell"]
        key = (cell, row["question_id"], str(row.get("medium")))
        historical = saved.get(key)
        if historical is None:
            missing.append({"cell": cell, "question_id": row["question_id"], "medium": row.get("medium")})
            continue
        replay_pass = (row["generation"].get("faithfulness") or {}).get("pass")
        checks = (
            ("factual_pass", historical.get("factual_pass"), replay_pass),
            ("numeric_pass", historical.get("numeric_pass"), row["generation"].get("numeric_pass")),
            ("word_count", historical.get("words"), row["generation"].get("word_count")),
        )
        for field, old, new in checks:
            if old == new:
                agreements[field] += 1
            else:
                discrepancies.append({
                    "cell": cell,
                    "question_id": row["question_id"],
                    "medium": row.get("medium"),
                    "field": field,
                    "historical": old,
                    "replay": new,
                    "calculation_changed": False,
                    "note": "Both values are reported. The saved metric file was not edited.",
                })
    return {
        "definition": "factual_pass reruns assess_factual on the saved answer, claims, hits, and relationship prompt facts. numeric_pass is recomputed by that same call. The saved numeric_pass is the value stored at generation time.",
        "agreements": agreements,
        "discrepancies": discrepancies,
        "missing_historical_rows": missing,
        "ranked_retrieval_metrics": "null because this experiment published no graded qrels",
    }


def reconcile_retrieval(results: list[dict], summary: dict) -> dict:
    """Designated-source overlap uses the same fraction. Ranked metrics stay null."""
    agreements = 0
    discrepancies = []
    by_question = summary.get("by_question") or {}
    for row in results:
        mode = row["retrieval_config"]["mode"]
        historical = (by_question.get(row["question_id"]) or {}).get(mode) or {}
        saved_ids = [str(item) for item in (historical.get("hit_source_ids") or [])]
        if saved_ids != row["retrieved_source_ids"]:
            discrepancies.append({
                "question_id": row["question_id"],
                "mode": mode,
                "field": "hit_source_ids",
                "historical": saved_ids,
                "replay": row["retrieved_source_ids"],
            })
            continue
        coverage = row["retrieval"]["designated_source_coverage"]
        expected = designated_source_coverage(saved_ids, row.get("review_source_ids") or [])
        ranked_null = all(
            row["retrieval"].get(name) is None
            for name in ("precision_at_4", "recall_at_4", "mrr", "ndcg_at_4")
        )
        if coverage == expected and ranked_null:
            agreements += 1
        else:
            discrepancies.append({
                "question_id": row["question_id"],
                "mode": mode,
                "field": "designated_source_coverage",
                "historical_overlap_count": len(historical.get("designated_sources_retrieved") or []),
                "replay": coverage,
                "expected": expected,
                "ranked_metrics_null": ranked_null,
            })
    return {
        "definition": "Overlap with a preselected id list. The historical notes say this is not precision or recall. Replay precision, recall, MRR, and nDCG stay null.",
        "agreements": agreements,
        "rows": len(results),
        "discrepancies": discrepancies,
    }


def pairing_report(records: list[dict]) -> dict:
    """Pair scored cells. Only A against B on the same question and medium is an ablation."""
    grouped: dict[tuple[str, str], dict[str, dict]] = {}
    for record in records:
        cell = record["provenance"][0]["historical_cell"]
        grouped.setdefault((record["question_id"], str(record.get("medium"))), {})[cell] = record
    decisions = []
    comparable_records = []
    for (_question, _medium), cells in sorted(grouped.items()):
        for left_name, right_name in (("A", "B"), ("A", "C"), ("B", "D_blog")):
            if left_name not in cells or right_name not in cells:
                decisions.append({
                    "status": "rejected",
                    "comparable": False,
                    "question_id": _question,
                    "medium": _medium,
                    "left_cell": left_name,
                    "right_cell": right_name,
                    "reason": "missing_cell",
                })
                continue
            decision = judge_pair(cells[left_name], cells[right_name])
            decisions.append(decision)
            if decision["comparable"]:
                comparable_records.extend((cells[left_name], cells[right_name]))
    scored = paired_faithfulness(
        comparable_records,
        CELLS["A"]["architecture"],
        CELLS["B"]["architecture"],
    )
    scored["interpretable_as_profile_comparison"] = False
    scored["label"] = "historical A minus B on matching question and medium"
    blog_only = paired_faithfulness(
        [row for row in comparable_records if row.get("medium") == "blog"],
        CELLS["A"]["architecture"],
        CELLS["B"]["architecture"],
    )
    blog_only["interpretable_as_profile_comparison"] = False
    blog_only["label"] = "historical A minus B on blog only"
    blog_only["note"] = (
        "The all-media mean can cancel a blog change with a change in another medium. "
        "Instruction-leak flags stay outside this faithfulness bit, matching the saved factual_pass field."
    )
    return {
        "decisions": decisions,
        "ablation_faithfulness": scored,
        "ablation_faithfulness_blog": blog_only,
    }


def replay(directory: Path, questions: Path, out: Path, experiment_id: str) -> dict:
    """Score saved traces. This does not retrieve, generate, or open the live store."""
    refuse_live_store(directory)
    refuse_live_store(questions)
    refuse_live_store(out)
    generation_records = load_generation_records(directory, experiment_id)
    retrieval_records = load_retrieval_records(directory, questions, experiment_id)
    generation = score_batch(generation_records)
    retrieval = score_batch(retrieval_records)
    generation_metrics = _load_json(directory / "metrics" / "generation.json")
    retrieval_summary = _load_json(directory / "metrics" / "retrieval_summary.json")
    report = {
        "generation_reconciliation": reconcile_generation(generation["results"], generation_metrics),
        "retrieval_reconciliation": reconcile_retrieval(retrieval["results"], retrieval_summary),
        "pairing": pairing_report(generation["results"]),
        "profile_crosswalk": [
            crosswalk_to_profile(cell, CELLS[cell]["nearest_profile"] or "none")
            for cell in CELLS
        ],
        "n_generation_records": len(generation_records),
        "n_retrieval_records": len(retrieval_records),
    }
    write_report(
        out / "generation",
        generation,
        experiment_id=experiment_id,
        paired=report["pairing"]["ablation_faithfulness"],
    )
    write_report(out / "retrieval", retrieval, experiment_id=experiment_id)
    question_rows = json.loads(questions.read_text(encoding="utf-8"))
    template = blank_template(
        question_rows,
        pools_from_retrieval(retrieval_records),
        source=str(questions).replace("\\", "/"),
    )
    (out / "annotation_template.json").write_text(
        json.dumps(template, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (out / "arm_alignment.json").write_text(
        json.dumps(profile_alignment(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (out / "reconciliation.json").write_text(
        json.dumps({
            "generation": report["generation_reconciliation"],
            "retrieval": report["retrieval_reconciliation"],
            "profile_crosswalk": report["profile_crosswalk"],
        }, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (out / "pairing.json").write_text(
        json.dumps(report["pairing"], indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report
