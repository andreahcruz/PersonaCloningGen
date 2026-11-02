"""Sequential four-arm runner. Retrieval is frozen once per question.

Generation is a supplied callable. This module does not load a model.
"""

from __future__ import annotations

import json
from pathlib import Path

from host_finetune.model_f_experiment.arms import (
    ADAPTER,
    ARMS,
    DECODING,
    EXPERIMENT_ID,
    PROMPT_FAMILY,
    RETRIEVAL,
    fingerprint,
)
from host_finetune.model_f_experiment.snapshot import SnapshotError, assert_snapshot_path

MEDIA = ("blog", "linkedin", "x", "talk")


def load_questions(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            rows.append({
                "question_id": str(row["question_id"]),
                "query": str(row["query"]),
                "phenomenon": row.get("phenomenon"),
                "abstention_expected": row.get("phenomenon") == "unanswerable",
            })
    if not rows:
        raise ValueError(f"no questions in {path}")
    return rows


def cell_path(directory: Path, question_id: str, arm: str, medium: str) -> Path:
    return directory / "raw" / "generations" / f"{question_id}__{arm}__{medium}.json"


def retrieval_path(directory: Path, question_id: str) -> Path:
    return directory / "raw" / "retrieval" / f"{question_id}.json"


def is_complete(path: Path) -> bool:
    if not path.is_file():
        return False
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return False
    return payload.get("status") == "complete"


def evidence_path(directory: Path, question_id: str) -> Path:
    return directory / "raw" / "evidence" / f"{question_id}.json"


def pending(directory: Path, questions: list[dict]) -> list[dict]:
    work = []
    for question in questions:
        retrieval_done = is_complete(retrieval_path(directory, question["question_id"]))
        for arm in ARMS:
            for medium in MEDIA:
                path = cell_path(directory, question["question_id"], arm, medium)
                if is_complete(path):
                    continue
                work.append({
                    **question,
                    "arm": arm,
                    "medium": medium,
                    "retrieval_frozen": retrieval_done,
                })
    return work


def write_manifest(directory: Path, questions: list[dict], *, snapshot: str) -> dict:
    assert_snapshot_path(Path(snapshot))
    manifest = {
        "experiment_id": EXPERIMENT_ID,
        "fingerprint": fingerprint([row["question_id"] for row in questions]),
        "arms": list(ARMS),
        "retrieval": dict(RETRIEVAL),
        "decoding": dict(DECODING),
        "adapter": ADAPTER,
        "prompt_family": PROMPT_FAMILY,
        "snapshot": snapshot,
        "live_store_queried": False,
        "retrieval_once_per_question": True,
        "questions": len(questions),
        "status": "prepared",
    }
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "experiment_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def freeze_retrieval(directory: Path, question: dict, payload: dict) -> Path:
    path = retrieval_path(directory, question["question_id"])
    if is_complete(path):
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    body = {
        "status": "complete",
        "question_id": question["question_id"],
        "query": question["query"],
        "retrieval": dict(RETRIEVAL),
        "payload": payload,
    }
    path.write_text(json.dumps(body, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path


def write_generation(directory: Path, record: dict) -> Path:
    path = cell_path(directory, record["question_id"], record["arm"], record["medium"])
    if is_complete(path):
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    body = dict(record)
    body["status"] = "complete"
    path.write_text(json.dumps(body, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path


def cell_record(question: dict, arm: str, trace: dict) -> dict:
    """Map one generator trace into the experiment cell schema."""
    context = trace.get("model_f_context") or {}
    budget = context.get("budget") or {}
    return {
        "question_id": question["question_id"],
        "arm": arm,
        "medium": trace.get("medium"),
        "query": question["query"],
        "abstention_expected": bool(question.get("abstention_expected")),
        "answer": trace.get("answer") or "",
        "generation_config": {
            "adapter": ADAPTER,
            "prompt_family": PROMPT_FAMILY,
            "decoding": dict(DECODING),
            "seed": trace.get("seed"),
            "accepted_seed": trace.get("retry_seed") if trace.get("accepted_attempt") == 2 else trace.get("seed"),
            "accepted_attempt": trace.get("accepted_attempt"),
            "retry_seed": trace.get("retry_seed"),
            "max_new_tokens": trace.get("max_new_tokens"),
            "prepare_profile": context.get("profile"),
            "graph_context_contract": trace.get("graph_context_contract"),
            "quantization": "4bit",
            "max_seq_length": 2048,
        },
        "retrieved_source_ids": [
            hit.get("source_id") or hit.get("document_id") or hit.get("id")
            for hit in (trace.get("retrieval_hits") or [])
            if isinstance(hit, dict)
        ],
        "evidence": [
            {
                "id": hit.get("id") or hit.get("source_id") or hit.get("document_id"),
                "text": hit.get("text") or "",
                "source_id": hit.get("source_id") or hit.get("document_id"),
                "split": hit.get("split"),
                "provenance": hit.get("provenance"),
            }
            for hit in (trace.get("retrieval_hits") or [])
            if isinstance(hit, dict)
        ],
        "provenance": list(context.get("passages_retained") or []),
        "selected_evidence_ids": list(context.get("selected_ids") or []),
        "claims": list(trace.get("verified_claims") or []),
        "grounding_texts": list(context.get("grounding_texts") or []),
        "latency": {"generation_seconds": trace.get("latency_seconds")},
        "tokens": {
            "prompt": trace.get("prompt_tokens"),
            "output": trace.get("generated_tokens_including_stop"),
            "context_used": context.get("tokens_used"),
            "context_by_channel": context.get("tokens_by_channel") or {},
            "source": context.get("token_estimate"),
            "budget_total": budget.get("total"),
        },
    }


def frozen_evidence_record(trace: dict) -> dict:
    """The claim record ``generate`` expects. Saving it keeps a resume from extracting again."""
    return {
        "verified": list(trace.get("verified_claims") or []),
        "evidence_status": trace.get("evidence_status") or "verified",
        "proposed": list(trace.get("proposed_atomic_claims") or []),
        "validation_rejected": list(trace.get("validation_rejected") or []),
        "verifier_rejected": list(trace.get("verifier_rejected") or []),
        "evidence_block": trace.get("evidence_block") or "",
        "raw_extraction": trace.get("raw_extraction"),
        "extraction_prompt": trace.get("extraction_prompt"),
        "rendered_extraction_prompt": trace.get("rendered_extraction_prompt"),
        "extraction_seconds": trace.get("extraction_seconds"),
        "verification_seconds": trace.get("verification_seconds"),
        "verification_records": list(trace.get("verification_records") or []),
        "allowed_numeric_values": list(trace.get("allowed_numeric_values") or []),
        "reason": trace.get("evidence_stage_reason"),
        "extraction_attempts": trace.get("extraction_attempts"),
        "extraction_stop_reason": trace.get("extraction_stop_reason"),
    }


def freeze_evidence(directory: Path, question_id: str, trace: dict) -> Path:
    path = evidence_path(directory, question_id)
    if path.is_file():
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(frozen_evidence_record(trace), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return path


def load_evidence(directory: Path, question_id: str) -> dict | None:
    path = evidence_path(directory, question_id)
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def execute_four_arms(directory: Path, questions: list[dict], snapshot: Path) -> dict:
    """Retrieve once per question, then generate the four arms in order.

    A completed cell is skipped. This loads the generator and queries the
    snapshot, so tests do not call it and it refuses the live expansion store.
    """
    snapshot = Path(snapshot)
    assert_snapshot_path(snapshot)
    import asyncio

    from host_finetune.compare_adapters import generate, load_generator
    from host_finetune.lightrag_backend import read_catalog, read_failed_document_ids
    from host_finetune.model_f_experiment.derive import arm_callback
    from host_finetune.model_f_experiment.retrieval import generation_payload, retrieve_raw

    catalog = read_catalog(snapshot / "catalog.jsonl")
    failed_ids = read_failed_document_ids(snapshot)
    if not catalog:
        raise SnapshotError(f"snapshot catalog is empty: {snapshot}")
    from host_finetune.model_f_experiment.context_window import assert_preflight

    assert_preflight(questions)
    model, tokenizer, load_report = load_generator(Path(ADAPTER), "4bit")
    manifest_path = directory / "experiment_manifest.json"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["model_load"] = load_report
        manifest["quantization"] = "4bit"
        manifest["max_seq_length"] = 2048
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    def count_tokens(text: str) -> int:
        return len(tokenizer.encode(text, add_special_tokens=False))

    evidence_cache: dict = {}
    generated = 0
    skipped = 0
    try:
        for index, question in enumerate(questions):
            stored = retrieval_path(directory, question["question_id"])
            if not is_complete(stored):
                raw = asyncio.run(retrieve_raw(question["query"], snapshot))
                freeze_retrieval(directory, question, raw)
            else:
                raw = json.loads(stored.read_text(encoding="utf-8"))["payload"]
            payload = generation_payload(raw, catalog, failed_document_ids=failed_ids)

            def retriever(topic: str, comparison_medium: str, payload=payload) -> dict:
                return payload

            gold = [{"id": question["question_id"], "topic": question["query"]}]
            saved_evidence = load_evidence(directory, question["question_id"])
            if saved_evidence is not None:
                evidence_cache[question["question_id"]] = saved_evidence
            for arm in ARMS:
                missing = [
                    medium
                    for medium in MEDIA
                    if not is_complete(cell_path(directory, question["question_id"], arm, medium))
                ]
                skipped += len(MEDIA) - len(missing)
                if not missing:
                    continue

                def on_trace(trace: dict, arm=arm, question=question) -> None:
                    freeze_evidence(directory, question["question_id"], trace)
                    write_generation(directory, cell_record(question, arm, trace))

                try:
                    generate(
                        Path(ADAPTER),
                        gold,
                        retriever=retriever,
                        seed_base=DECODING["seed_base"],
                        topic_rows=[(index, gold[0])],
                        loaded=(model, tokenizer, load_report),
                        unload=False,
                        evidence_cache=evidence_cache,
                        media_allow=set(missing),
                        graph_context_for=arm_callback(
                            arm,
                            catalog=catalog,
                            failed_document_ids=failed_ids,
                            count_tokens=count_tokens,
                        ),
                        on_trace=on_trace,
                    )
                except Exception as exc:
                    failure = directory / "raw" / "failures.jsonl"
                    failure.parent.mkdir(parents=True, exist_ok=True)
                    with failure.open("a", encoding="utf-8") as handle:
                        handle.write(json.dumps({
                            "question_id": question["question_id"],
                            "arm": arm,
                            "media": missing,
                            "error": f"{type(exc).__name__}: {exc}",
                        }) + "\n")
                    continue
                generated += len(missing)
    finally:
        del model
        del tokenizer
    return {"generated": generated, "skipped_complete": skipped}


def run_pending(directory: Path, questions: list[dict], generate_cell) -> dict:
    """Run missing cells one at a time. ``generate_cell`` must not retrieve again."""
    done = 0
    skipped = 0
    for question in questions:
        retrieval = retrieval_path(directory, question["question_id"])
        if not is_complete(retrieval):
            raise SnapshotError(f"frozen retrieval is missing for {question['question_id']}")
        payload = json.loads(retrieval.read_text(encoding="utf-8"))["payload"]
        for arm in ARMS:
            for medium in MEDIA:
                path = cell_path(directory, question["question_id"], arm, medium)
                if is_complete(path):
                    skipped += 1
                    continue
                record = generate_cell(question, arm, medium, payload)
                record["question_id"] = question["question_id"]
                record["arm"] = arm
                record["medium"] = medium
                write_generation(directory, record)
                done += 1
    return {"generated": done, "skipped_complete": skipped}
