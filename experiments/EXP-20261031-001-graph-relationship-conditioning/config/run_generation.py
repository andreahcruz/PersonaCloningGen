"""Passage-controlled generation ablation.

Arm A uses hybrid LightRAG passages and validated claims.
Arm B uses those same passages and claims plus in-passage source-verified
relationships. Arm C uses matched pilot dense passages. Arm D is a blog-only
token-budget pad with no relationship facts. The Relabel adapter is not trained.
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parents[1]
RAW = OUT / "raw"
CELLS = RAW / "generations"
QUESTIONS = ROOT / "host_finetune" / "graph_relationship_questions.json"
ADAPTER = ROOT / "host_finetune" / "output" / "lemkin_lora_relabel"

sys.path.insert(0, str(ROOT))

from host_finetune.compare_adapters import GENERATION_SEED, generate, load_frozen_retriever, load_generator
from host_finetune.graph_relationships import context_pad_block, generation_relationships, neutral_pad_text


def _stop_ollama() -> None:
    for name in ("llama3.1:latest", "nomic-embed-text:latest", "qwen2.5:14b"):
        subprocess.run(["ollama", "stop", name], check=False)


def _read(path: Path) -> dict | None:
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _cache_from(path: Path, topic_id: str) -> dict:
    payload = _read(path)
    if not payload:
        return {}
    traces = [row for row in payload.get("traces") or [] if row.get("medium")]
    if not traces:
        return {}
    trace = traces[0]
    return {
        topic_id: {
            "verified": trace.get("verified_claims") or [],
            "evidence_status": trace.get("evidence_status") or "no_accepted_claims",
            "validation_rejected": trace.get("validation_rejected") or [],
            "verifier_rejected": trace.get("verifier_rejected") or [],
            "proposed": trace.get("proposed_atomic_claims") or [],
            "evidence_block": trace.get("evidence_block") or "",
            "reason": trace.get("evidence_stage_reason"),
            "extraction_seconds": trace.get("extraction_seconds"),
        }
    }


def _fit_pad(tokenizer, target_tokens: int) -> str:
    if target_tokens <= 0:
        return ""
    best = neutral_pad_text(1)
    best_gap = abs(target_tokens)
    for words in range(1, target_tokens + 40):
        text = neutral_pad_text(words)
        count = len(tokenizer.encode("\n\n" + context_pad_block(text), add_special_tokens=False))
        gap = abs(count - target_tokens)
        if gap < best_gap:
            best = text
            best_gap = gap
        if count >= target_tokens and gap <= 2:
            break
    return best


def _run_arm(model, tokenizer, topics, retriever, cache, relationship_for, context_pad_for, media_allow, destination_for):
    for index, topic in topics:
        destination = destination_for(topic["id"])
        if destination.is_file():
            print(f"keep {destination.name}", flush=True)
            continue
        started = time.perf_counter()
        traces, _report = generate(
            ADAPTER,
            [topic],
            retriever=retriever,
            seed_base=GENERATION_SEED,
            topic_rows=[(index, topic)],
            quantization="4bit",
            loaded=(model, tokenizer, {"reused": True}),
            unload=False,
            evidence_cache=cache,
            relationship_for=relationship_for,
            context_pad_for=context_pad_for,
            media_allow=media_allow,
        )
        _write(destination, {
            "topic_id": topic["id"],
            "wall_seconds": round(time.perf_counter() - started, 3),
            "traces": traces,
        })
        print(f"wrote {destination.name} traces={len(traces)}", flush=True)


def main() -> None:
    if not ADAPTER.is_dir():
        raise SystemExit(f"Relabel adapter is missing: {ADAPTER}")
    questions = [row for row in json.loads(QUESTIONS.read_text(encoding="utf-8")) if row.get("generation")]
    topics = [
        (index, {"id": row["id"], "topic": row["query"]})
        for index, row in enumerate(questions)
    ]
    _stop_ollama()
    time.sleep(2)
    model, tokenizer, load_report = load_generator(ADAPTER, "4bit")
    (RAW / "load_report.json").write_text(json.dumps(load_report, indent=2, default=str), encoding="utf-8")
    hybrid = load_frozen_retriever(RAW / "frozen_hybrid.jsonl")
    dense = load_frozen_retriever(RAW / "frozen_pilot_dense.jsonl")
    try:
        hybrid_cache = {}
        for index, topic in topics:
            existing = _cache_from(CELLS / f"{topic['id']}__A.json", topic["id"])
            hybrid_cache.update(existing)
        _run_arm(
            model, tokenizer, topics, hybrid, hybrid_cache,
            None, None, None, lambda topic_id: CELLS / f"{topic_id}__A.json",
        )
        _run_arm(
            model, tokenizer, topics, hybrid, hybrid_cache,
            lambda row, payload: generation_relationships(payload),
            None, None, lambda topic_id: CELLS / f"{topic_id}__B.json",
        )
        dense_cache = {}
        for _index, topic in topics:
            dense_cache.update(_cache_from(CELLS / f"{topic['id']}__C.json", topic["id"]))
        _run_arm(
            model, tokenizer, topics, dense, dense_cache,
            None, None, None, lambda topic_id: CELLS / f"{topic_id}__C.json",
        )
        for index, topic in topics:
            destination = CELLS / f"{topic['id']}__D_blog.json"
            if destination.is_file():
                print(f"keep {destination.name}", flush=True)
                continue
            arm_a = _read(CELLS / f"{topic['id']}__A.json")
            arm_b = _read(CELLS / f"{topic['id']}__B.json")
            if not arm_a or not arm_b:
                print(f"skip pad {topic['id']}: A or B missing", flush=True)
                continue
            blog_a = next(row for row in arm_a["traces"] if row.get("medium") == "blog")
            blog_b = next(row for row in arm_b["traces"] if row.get("medium") == "blog")
            delta = int(blog_b.get("prompt_tokens") or 0) - int(blog_a.get("prompt_tokens") or 0)
            pad = _fit_pad(tokenizer, delta)
            if not pad:
                _write(destination, {
                    "topic_id": topic["id"],
                    "skipped": "no_relationship_token_delta",
                    "token_delta": delta,
                    "traces": [],
                })
                continue

            def _pad_for(row, payload, text=pad):
                del row, payload
                return text

            started = time.perf_counter()
            traces, _report = generate(
                ADAPTER,
                [topic],
                retriever=hybrid,
                seed_base=GENERATION_SEED,
                topic_rows=[(index, topic)],
                quantization="4bit",
                loaded=(model, tokenizer, {"reused": True}),
                unload=False,
                evidence_cache=hybrid_cache,
                context_pad_for=_pad_for,
                media_allow={"blog"},
            )
            _write(destination, {
                "topic_id": topic["id"],
                "wall_seconds": round(time.perf_counter() - started, 3),
                "target_token_delta": delta,
                "traces": traces,
            })
            print(f"wrote {destination.name}", flush=True)
    finally:
        import torch

        del model
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
