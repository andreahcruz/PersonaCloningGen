"""Score one saved trace file with the current evidence evaluator.

Does not train, generate, or edit the trace file.

    .venv-test\\Scripts\\python.exe -m host_finetune.score_saved_experiment ^
        --traces experiments/EXP-20261005-003-relabel-rag-final/raw/relabel.jsonl ^
        --out-dir experiments/EXP-20261005-003-relabel-rag-final
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from host_finetune.claim_judge import (
    EvidenceIndex,
    claim_prompt,
    evidence_documents,
    evidence_report,
    parse_claims,
    rag_grounding_report,
    stance_report,
)
from host_finetune.eval_gold_rag import load_format_specs
from host_finetune.judge_client import DEFAULT_JUDGE_MODEL, judge_family
from host_finetune.rescore_evidence import (
    calibration_samples,
    judge_json,
    load_cache,
    passages_for,
    stance_control,
    writing_record,
)
from host_finetune.rescore_unified import base_controls, load_traces, sha256
from host_finetune.train_only_index import (
    DEFAULT_ASSIGNMENTS,
    DEFAULT_DATASET,
    DEFAULT_GOLD_DISPOSITIONS,
    load_jsonl,
)
from host_finetune.unified_eval import (
    SCHEMA,
    UTILITY_WEIGHTS,
    CopyIndex,
    apply_judge_overlay,
    build_reference_pack,
    evaluate_row,
    summarize_model,
)
from host_finetune.voice_distance import iter_train_texts
from host_finetune.writing_quality import writing_prompt, writing_quality_gate

ROOT = Path(__file__).resolve().parents[1]
SPECS = ROOT / "formats" / "format_specs.yaml"
EXPECTED_GENERATION_SHA = "266507aca9effe008596ca64dc199d2e0de32bafbd7a9765d738adb1d34ec550"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--traces", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--model-name", default="relabel")
    parser.add_argument("--source-experiment", default="EXP-20261005-003-relabel-rag-final")
    parser.add_argument("--judge-model", default=DEFAULT_JUDGE_MODEL)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    traces_path = args.traces if args.traces.is_absolute() else ROOT / args.traces
    out = args.out_dir if args.out_dir.is_absolute() else ROOT / args.out_dir
    if not traces_path.is_file():
        raise SystemExit(f"missing traces: {traces_path}")
    generation_sha = sha256(traces_path)
    if generation_sha != EXPECTED_GENERATION_SHA:
        raise SystemExit(
            f"refusing to score: {traces_path.name} sha256 {generation_sha} "
            f"does not match the frozen generation file"
        )
    model = args.judge_model
    family = judge_family(model)
    base_url = os.environ.get("OLLAMA_BASE", "http://127.0.0.1:11434")
    for name in ("config", "metrics", "raw"):
        (out / name).mkdir(parents=True, exist_ok=True)
    cache_path = out / "raw" / "judge_cache.jsonl"
    cache = load_cache(cache_path)
    print(f"judge={model} family={family} cached={len(cache)} generation_sha={generation_sha}", flush=True)
    dataset = load_jsonl(DEFAULT_DATASET)
    assignments = load_jsonl(DEFAULT_ASSIGNMENTS)
    dispositions = load_jsonl(DEFAULT_GOLD_DISPOSITIONS)
    documents = evidence_documents(dataset, assignments, dispositions)
    index = EvidenceIndex(documents)
    print(f"held-out evidence docs={len(documents)}", flush=True)
    samples = calibration_samples(documents, model, base_url, cache, cache_path)
    writing_gate = writing_quality_gate(samples)
    writing_gate["judge_model"] = model
    writing_gate["judge_family"] = family
    writing_gate["candidate_family"] = "llama"
    (out / "metrics" / "writing_calibration.json").write_text(
        json.dumps(writing_gate, indent=2) + "\n", encoding="utf-8"
    )
    print("writing calibration", writing_gate["role"], writing_gate["reason"], flush=True)
    stance_gate = stance_control(index, model, base_url, cache, cache_path)
    (out / "metrics" / "stance_calibration.json").write_text(
        json.dumps(stance_gate, indent=2) + "\n", encoding="utf-8"
    )
    print("stance calibration", stance_gate["role"], stance_gate["reason"], flush=True)
    pack = build_reference_pack(dataset, assignments, dispositions, base_controls())
    pack["writing_quality_calibration"] = {
        "calibrated": writing_gate["calibrated"],
        "role": writing_gate["role"],
    }
    print("building copy index", flush=True)
    copy_index = CopyIndex(iter_train_texts(dataset, assignments))
    specs = load_format_specs(SPECS)
    traces = load_traces(traces_path)
    rows_path = out / "metrics" / "rows.jsonl"
    finished = {}
    if rows_path.exists():
        kept = []
        for line in rows_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                previous = json.loads(line)
            except json.JSONDecodeError:
                continue
            finished[(previous.get("id"), previous.get("medium"))] = previous
            kept.append(line)
        rows_path.write_text("\n".join(kept) + ("\n" if kept else ""), encoding="utf-8")
        print(f"resuming {len(finished)} scored cells", flush=True)
    scored_rows = []
    with rows_path.open("a", encoding="utf-8") as handle:
        for done, trace in enumerate(traces, start=1):
            if "query" not in trace:
                trace = {**trace, "query": trace.get("topic", "")}
            key = (trace.get("id"), trace.get("medium"))
            prior = finished.get(key)
            if prior is not None:
                scored_rows.append(prior)
                continue
            base = evaluate_row(trace, pack, copy_index, specs)
            draft = (trace.get("answer") or "")[:1800]
            topic = trace.get("topic") or ""
            medium = trace.get("medium") or ""
            writing_parsed = judge_json(
                writing_prompt(draft, topic, medium),
                model, base_url, 220, cache, cache_path, "writing",
            )
            retrieved = passages_for(trace)
            evidence_hits = index.search(f"{topic}\n{draft[:900]}", medium, k=4)
            evidence_text = [hit["text"][:650] for hit in evidence_hits]
            claim_parsed = judge_json(
                claim_prompt(draft, topic, retrieved, evidence_text),
                model, base_url, 520, cache, cache_path, "claims",
            )
            claims = parse_claims(claim_parsed, retrieved is not None)
            writing = writing_record(writing_parsed, writing_gate["role"], writing_gate["calibrated"], model)
            row = apply_judge_overlay(
                base,
                writing,
                rag_grounding_report(claims, retrieved is not None),
                evidence_report(claims, len(evidence_text)),
                stance_report(claims),
                writing_role=writing_gate["role"],
                writing_calibrated=writing_gate["calibrated"],
                stance_role=stance_gate["role"],
            )
            row["model"] = args.model_name
            row["source_experiment"] = args.source_experiment
            row["generation_sha256"] = generation_sha
            scored_rows.append(row)
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            handle.flush()
            print(f"scored {done}/{len(traces)} {key[0]} {key[1]}", flush=True)
    if sha256(traces_path) != generation_sha:
        raise SystemExit("generation file changed while scoring")
    summary = summarize_model(scored_rows, f"{args.source_experiment}/{args.model_name}", args.source_experiment)
    payload = {
        "schema": SCHEMA,
        "utility_weights": UTILITY_WEIGHTS,
        "judge_model": model,
        "judge_family": family,
        "candidate_family": "llama",
        "generation_sha256": generation_sha,
        "n_traces": len(traces),
        "writing_quality_calibration": {
            "calibrated": writing_gate["calibrated"],
            "role": writing_gate["role"],
            "reason": writing_gate["reason"],
        },
        "stance_calibration": stance_gate,
        "evidence_documents": len(documents),
        "model": summary,
    }
    (out / "metrics" / "summary.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=False)
    dirty = subprocess.run(["git", "status", "--porcelain"], cwd=ROOT, capture_output=True, text=True, check=False)
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "command": " ".join(sys.argv),
        "git": {"commit": commit.stdout.strip(), "dirty": bool(dirty.stdout.strip())},
        "judge_model": model,
        "judge_config": {"family": family, "temperature": 0, "blinded": True, "candidate_family": "llama"},
        "generation_sha256": generation_sha,
        "python": platform.python_version(),
        "schema": SCHEMA,
        "utility_weights": UTILITY_WEIGHTS,
    }
    (out / "config" / "evaluation_run.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print("evaluation complete", summary["n"], "eligible", summary["n_eligible"], flush=True)


if __name__ == "__main__":
    main()
