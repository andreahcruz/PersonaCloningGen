"""Paired fixed-style ablation on frozen Chroma v2 evidence.

Arm off and arm on share one extracted claim set per topic. Talk has no
style exemplars, so the style-on talk cell is recorded as unsupported and
is not generated. Completed cells are not overwritten.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import argparse
import hashlib
import json
import subprocess
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

from host_finetune.compare_adapters import (
    GENERATION_SEED,
    ROOT,
    apply_active_rag,
    generate,
    git_state,
    load_generator,
    load_gold,
    load_style_for,
    selected_topic_rows,
    write_frozen_retrieval,
)
from host_finetune.compare_adapters import load_frozen_retriever
from host_finetune.rag_runtime import resolve_rag
from host_finetune.train_only_index import (
    DEFAULT_ASSIGNMENTS,
    DEFAULT_DATASET,
    DEFAULT_GOLD_DISPOSITIONS,
    DEFAULT_STYLE_PATH,
    assignment_rows,
    blocked_group_ids,
    load_jsonl,
)

EXP = Path(__file__).resolve().parents[1]
RAW = EXP / "raw"
CELLS = RAW / "cells"
EVIDENCE = RAW / "evidence"
ADAPTER = ROOT / "host_finetune" / "output" / "lemkin_lora_relabel"
SCHEMA = 1
ARMS = ("off", "on")
STYLE_MEDIA = ("blog", "linkedin", "x")
BASELINE_MEDIA = ("blog", "linkedin", "x", "talk")
MAX_ATTEMPTS = 2


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _stop_ollama() -> None:
    for name in ("qwen2.5:14b", "llama3.1:latest", "nomic-embed-text:latest", "nomic-embed-text"):
        subprocess.run(["ollama", "stop", name], check=False)


def cell_path(topic_id: str, medium: str, arm: str) -> Path:
    return CELLS / f"{topic_id}__{medium}__{arm}.json"


def _load_cell(path: Path) -> dict | None:
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        broken = CELLS / "invalid" / f"{path.stem}-{int(time.time())}.json"
        broken.parent.mkdir(parents=True, exist_ok=True)
        path.replace(broken)
        return None
    if payload.get("schema_version") != SCHEMA:
        return None
    return payload


def cell_done(payload: dict | None) -> bool:
    if not payload:
        return False
    if payload.get("status") == "unsupported":
        return True
    if payload.get("status") == "complete":
        trace = payload.get("trace") or {}
        return bool((trace.get("answer") or "").strip()) and trace.get("seed") is not None
    if payload.get("status") == "error" and int(payload.get("attempts") or 0) >= MAX_ATTEMPTS:
        return True
    return False


def exemplar_manifest() -> dict:
    payload = json.loads(DEFAULT_STYLE_PATH.read_text(encoding="utf-8"))
    assignments = assignment_rows(load_jsonl(DEFAULT_ASSIGNMENTS))
    by_index = {int(row["row_index"]): row for row in assignments}
    dataset = load_jsonl(DEFAULT_DATASET)
    blocked = blocked_group_ids(assignments, load_jsonl(DEFAULT_GOLD_DISPOSITIONS))
    checked = []
    for platform, items in (payload.get("exemplars") or {}).items():
        for item in items:
            row_index = int(item["row_id"])
            assignment = by_index[row_index]
            source = dataset[row_index]
            text = item.get("text") or ""
            checked.append(
                {
                    "platform": platform,
                    "row_id": row_index,
                    "group_id": item.get("group_id"),
                    "split": assignment.get("split"),
                    "sft_role": assignment.get("sft_role"),
                    "source_file": assignment.get("source_file"),
                    "source_line": assignment.get("source_line"),
                    "gold_blocked": assignment.get("group_id") in blocked,
                    "text_matches_dataset_output": text.strip() == (source.get("output") or "").strip(),
                    "sha256": _sha256_text(text),
                    "words": len(text.split()),
                    "use": item.get("use"),
                }
            )
    return {
        "path": str(DEFAULT_STYLE_PATH.relative_to(ROOT)),
        "sha256": _sha256_file(DEFAULT_STYLE_PATH),
        "empty_platforms": payload.get("empty_platforms"),
        "selection": payload.get("selection"),
        "exemplars": checked,
        "talk_supported": False,
        "talk_reason": "youtube_jason has no train-owned unchanged style exemplars",
    }


def _namespace():
    import argparse as ap

    selected = resolve_rag("v2")
    return ap.Namespace(
        rag_version="v2",
        collection=selected["collection"],
        chroma_path=selected["path"],
        k=4,
        ollama_base="http://127.0.0.1:11434",
        embed_model="nomic-embed-text",
        retrieval_backend="dense",
    )


def ensure_frozen(topics: list[str]) -> Path:
    path = RAW / "frozen_chroma_v2.jsonl"
    if path.is_file():
        return path
    _stop_ollama()
    from host_finetune.compare_adapters import build_runtime_retriever

    args = _namespace()
    apply_active_rag(args)
    retriever = build_runtime_retriever(args)
    write_frozen_retrieval(path, topics, retriever)
    _stop_ollama()
    return path


def _gpu() -> dict:
    try:
        import torch

        if not torch.cuda.is_available():
            return {"available": False}
        return {
            "available": True,
            "name": torch.cuda.get_device_name(0),
            "allocated_mb": round(torch.cuda.memory_allocated() / (1024 * 1024), 1),
            "reserved_mb": round(torch.cuda.memory_reserved() / (1024 * 1024), 1),
            "max_allocated_mb": round(torch.cuda.max_memory_allocated() / (1024 * 1024), 1),
        }
    except Exception as exc:  # pragma: no cover - hardware probe
        return {"available": False, "error": str(exc)}


def _save_trace(topic_id: str, medium: str, arm: str, trace: dict, exemplars: dict) -> None:
    path = cell_path(topic_id, medium, arm)
    if path.is_file() and cell_done(_load_cell(path)):
        raise RuntimeError(f"refusing to overwrite {path.name}")
    prompt = trace.get("prompt") or ""
    style_in_prompt = "Style examples from train-owned source text." in prompt
    if arm == "off" and style_in_prompt:
        raise RuntimeError("style block appeared in the style-off prompt")
    if arm == "on" and medium in STYLE_MEDIA and not style_in_prompt:
        raise RuntimeError("style block missing from the style-on prompt")
    claims = trace.get("verified_claims") or []
    evidence_sha = _sha256_text(json.dumps(claims, ensure_ascii=False, sort_keys=True))
    _write_json(
        path,
        {
            "schema_version": SCHEMA,
            "status": "complete",
            "topic_id": topic_id,
            "medium": medium,
            "arm": arm,
            "seed": trace.get("seed"),
            "attempts": 1,
            "evidence_sha256": evidence_sha,
            "retrieval_hit_ids": [hit.get("id") for hit in (trace.get("retrieval_hits") or [])],
            "retrieval_query": trace.get("retrieval_query"),
            "retrieval_filter": trace.get("retrieval_filter"),
            "style_applied": bool(trace.get("style_applied")),
            "style_in_prompt": style_in_prompt,
            "prompt_tokens": trace.get("prompt_tokens"),
            "generated_tokens_including_stop": trace.get("generated_tokens_including_stop"),
            "latency_seconds": trace.get("latency_seconds"),
            "exemplar_ids": [
                item["row_id"] for item in exemplars["exemplars"] if item["platform"] == medium
            ] if arm == "on" else [],
            "saved_at": _now(),
            "trace": trace,
        },
    )


def _save_error(topic_id: str, medium: str, arm: str, attempt: int, exc: BaseException) -> None:
    path = cell_path(topic_id, medium, arm)
    prior = _load_cell(path) or {}
    errors = list(prior.get("errors") or [])
    errors.append({"attempt": attempt, "type": type(exc).__name__, "message": str(exc), "trace": traceback.format_exc()[-4000:]})
    _write_json(
        path,
        {
            "schema_version": SCHEMA,
            "status": "error",
            "topic_id": topic_id,
            "medium": medium,
            "arm": arm,
            "attempts": attempt,
            "errors": errors,
            "saved_at": _now(),
        },
    )


def _save_unsupported(topic_id: str, topic: str) -> None:
    path = cell_path(topic_id, "talk", "on")
    if cell_done(_load_cell(path)):
        return
    _write_json(
        path,
        {
            "schema_version": SCHEMA,
            "status": "unsupported",
            "topic_id": topic_id,
            "topic": topic,
            "medium": "talk",
            "arm": "on",
            "reason": "no_valid_talk_style_exemplars",
            "answer": "",
            "saved_at": _now(),
        },
    )


def _needed(topic_id: str, arm: str) -> list[str]:
    media = STYLE_MEDIA if arm == "on" else BASELINE_MEDIA
    pending = []
    for medium in media:
        if not cell_done(_load_cell(cell_path(topic_id, medium, arm))):
            pending.append(medium)
    return pending


def _persist_evidence(cache: dict) -> None:
    for key, record in cache.items():
        path = EVIDENCE / f"{key}.json"
        if path.is_file():
            continue
        _write_json(path, record)


def _load_evidence() -> dict:
    cache = {}
    if not EVIDENCE.is_dir():
        return cache
    for path in EVIDENCE.glob("*.json"):
        cache[path.stem] = json.loads(path.read_text(encoding="utf-8"))
    return cache


def run(budget_seconds: float, max_topics: int | None, auto_continue: bool) -> None:
    started = time.perf_counter()
    RAW.mkdir(parents=True, exist_ok=True)
    CELLS.mkdir(parents=True, exist_ok=True)
    EVIDENCE.mkdir(parents=True, exist_ok=True)
    (EXP / "metrics").mkdir(exist_ok=True)
    (EXP / "config").mkdir(exist_ok=True)
    exemplars = exemplar_manifest()
    _write_json(EXP / "config" / "style_exemplars.json", exemplars)
    bad = [
        item for item in exemplars["exemplars"]
        if item["split"] != "train" or item["sft_role"] != "unchanged" or item["gold_blocked"]
        or not item["text_matches_dataset_output"]
    ]
    if bad:
        raise SystemExit(f"style exemplars failed ownership checks: {bad}")
    gold = load_gold()
    plan = selected_topic_rows(gold, None, max_topics)
    frozen_path = ensure_frozen([row["topic"] for _index, row in plan])
    retriever = load_frozen_retriever(frozen_path)
    _stop_ollama()
    cache = _load_evidence()
    model, tokenizer, load_report = load_generator(ADAPTER, "4bit")
    _write_json(EXP / "config" / "quantization_load.json", load_report)
    smoke_ok = None
    completed_topics = 0
    topic_limit = len(plan)
    generation_started = time.perf_counter()
    try:
        for offset, (index, row) in enumerate(plan):
            if offset >= topic_limit:
                break
            if time.perf_counter() - started >= budget_seconds:
                print("budget reached before next topic", flush=True)
                break
            topic_id = row["id"]
            _save_unsupported(topic_id, row["topic"])
            halt = False
            for arm in ARMS:
                pending = _needed(topic_id, arm)
                style_for = load_style_for(DEFAULT_STYLE_PATH) if arm == "on" else None
                for medium in pending:
                    if time.perf_counter() - started >= budget_seconds:
                        print("budget reached", flush=True)
                        halt = True
                        break
                    prior = _load_cell(cell_path(topic_id, medium, arm)) or {}
                    attempt = int(prior.get("attempts") or 0) + 1
                    if attempt > MAX_ATTEMPTS:
                        continue
                    print(f"generate {topic_id} {medium} {arm} attempt={attempt}", flush=True)

                    def _on_trace(trace, topic_id=topic_id, medium=medium, arm=arm):
                        if trace.get("medium") != medium:
                            return
                        _save_trace(topic_id, medium, arm, trace, exemplars)

                    try:
                        generate(
                            ADAPTER,
                            [row],
                            retriever=retriever,
                            style_for=style_for,
                            seed_base=GENERATION_SEED,
                            topic_rows=[(index, row)],
                            loaded=(model, tokenizer, load_report),
                            unload=False,
                            evidence_cache=cache,
                            on_trace=_on_trace,
                            media_allow={medium},
                        )
                    except Exception as exc:
                        print(f"failed {topic_id} {medium} {arm}: {exc}", flush=True)
                        _save_error(topic_id, medium, arm, attempt, exc)
                    else:
                        if not cell_done(_load_cell(cell_path(topic_id, medium, arm))):
                            _save_error(
                                topic_id,
                                medium,
                                arm,
                                attempt,
                                RuntimeError("generation returned without a completed cell"),
                            )
                    _persist_evidence(cache)
                    _write_json(EXP / "raw" / "gpu_latest.json", _gpu())
                if halt:
                    break
            if halt:
                break
            if offset == 0:
                missing = [medium + ":" + arm for arm in ARMS for medium in _needed(topic_id, arm)]
                smoke_ok = not missing
                elapsed = round(time.perf_counter() - generation_started, 1)
                print(f"smoke_ok={smoke_ok} generation_s={elapsed} gpu={json.dumps(_gpu())}", flush=True)
                _write_json(
                    EXP / "raw" / "smoke.json",
                    {"ok": smoke_ok, "generation_seconds": elapsed, "gpu": _gpu(), "topic_id": topic_id},
                )
                if not smoke_ok or not auto_continue:
                    break
                remaining = len(plan) - 1
                per_topic = max(elapsed, 1.0)
                left = budget_seconds - (time.perf_counter() - started)
                affordable = remaining if per_topic * remaining <= left else max(0, int(left / per_topic))
                topic_limit = 1 + affordable
                print(
                    f"continuing through topic_limit={topic_limit} "
                    f"(per_topic={per_topic:.0f}s left={left:.0f}s)",
                    flush=True,
                )
            if not _needed(topic_id, "off") and not _needed(topic_id, "on"):
                completed_topics += 1
    finally:
        del model
        import torch

        torch.cuda.empty_cache()
        _stop_ollama()
    _write_json(
        EXP / "raw" / "run_status.json",
        {
            "finished_at": _now(),
            "elapsed_seconds": round(time.perf_counter() - started, 3),
            "smoke_ok": smoke_ok,
            "completed_topics": completed_topics,
            "gpu": _gpu(),
            "git": git_state(),
            "budget_seconds": budget_seconds,
            "frozen_retrieval": str(frozen_path.relative_to(ROOT)),
            "adapter": str(ADAPTER.relative_to(ROOT)),
        },
    )
    print("run finished", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--budget-seconds", type=float, default=18000)
    parser.add_argument("--max-topics", type=int, default=None)
    parser.add_argument("--no-auto-continue", action="store_true")
    args = parser.parse_args()
    run(args.budget_seconds, args.max_topics, not args.no_auto_continue)


if __name__ == "__main__":
    main()
