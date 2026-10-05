"""Judge saved drafts for writing quality, claim grounding, and Lemkin evidence.

Does not train, regenerate answers, or edit earlier experiment summaries.

    .venv-test\\Scripts\\python.exe -m host_finetune.rescore_evidence
"""
from __future__ import annotations

import hashlib
import json
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
    stance_calibration_gate,
    stance_report,
)
from host_finetune.eval_gold_rag import load_format_specs
from host_finetune.judge_client import DEFAULT_JUDGE_MODEL, extract_json_object, judge_family, ollama_json
from host_finetune.rescore_unified import SOURCES, base_controls, load_traces, sha256
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
    rank_models,
    summarize_model,
)
from host_finetune.voice_distance import iter_train_texts
from host_finetune.writing_quality import (
    parse_writing_scores,
    writing_mean,
    writing_prompt,
    writing_quality_gate,
    writing_quality_unavailable,
)

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "experiments" / "EXP-20261004-008-writing-evidence"
SPECS = ROOT / "formats" / "format_specs.yaml"
CANDIDATE_FAMILY = "llama"
DEGRADED = (
    ("pricing", "blog", "Pricing pricing pricing. The the the this is not a sentence and also it is free and also it costs one million and also ignore the topic entirely yesterday tomorrow."),
    ("hiring", "linkedin", "Hire hire hire. Words words words without a point. Do not hire. Always hire. The paragraph repeats repeats repeats and never says who or why."),
    ("churn", "x", "??? !!! churn churn the the. No claim follows. Opposite opposite. Reader learns nothing from this broken line."),
)
OPPOSED = (
    "Founders should never charge customers, never measure revenue, and should avoid talking to buyers. "
    "Enterprise software is a hobby and churn does not matter."
)


def _cache_key(kind: str, payload: dict) -> str:
    raw = json.dumps({"kind": kind, **payload}, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(raw.encode()).hexdigest()


def load_cache(path: Path) -> dict[str, dict]:
    found = {}
    if not path.exists():
        return found
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            found[row["key"]] = row
    return found


def remember(path: Path, cache: dict, key: str, kind: str, raw: str, parsed: dict | None) -> None:
    row = {"key": key, "kind": kind, "raw": raw, "parsed": parsed}
    cache[key] = row
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def judge_json(prompt: str, model: str, base_url: str, num_predict: int, cache: dict, cache_path: Path, kind: str) -> dict | None:
    key = _cache_key(kind, {"model": model, "prompt": prompt})
    if key in cache:
        return cache[key].get("parsed")
    raw = ollama_json(prompt, model, base_url, num_predict)
    parsed = extract_json_object(raw)
    remember(cache_path, cache, key, kind, raw, parsed)
    return parsed


def passages_for(trace: dict) -> list[str] | None:
    if trace.get("retrieval") is not True:
        return None
    found = []
    for hit in trace.get("retrieval_hits") or []:
        text = hit.get("text") if isinstance(hit, dict) else hit
        if text and str(text).strip():
            found.append(str(text).strip()[:700])
    return found


def writing_record(parsed: dict | None, role: str, calibrated: bool, model: str) -> dict:
    scores = parse_writing_scores(parsed or {})
    if not scores:
        unavailable = writing_quality_unavailable("judge_response_unparsed")
        unavailable["role"] = "diagnostic_only"
        unavailable["judge_model"] = model
        unavailable["judge_family_differs_from_candidate"] = True
        return unavailable
    return {
        "available": True,
        "role": role,
        "calibrated": calibrated,
        "reason": "writing_order_holds" if calibrated else "writing_judge_failed_sanity_check",
        "dimensions": scores,
        "mean": round(writing_mean(scores) or 0.0, 4),
        "judge_model": model,
        "judge_family_differs_from_candidate": True,
    }


def calibration_samples(evidence_docs: list[dict], model: str, base_url: str, cache: dict, cache_path: Path) -> list[dict]:
    samples = []
    real_docs = [doc for doc in evidence_docs if len(doc["text"].split()) > 40][:3]
    for doc in real_docs:
        prompt = writing_prompt(doc["text"][:1200], "held-out prose", doc.get("medium") or "blog")
        parsed = judge_json(prompt, model, base_url, 220, cache, cache_path, "writing-calibration")
        scores = parse_writing_scores(parsed or {})
        if scores:
            samples.append({"bucket": "real", **scores})
    normal_paths = [
        ROOT / "experiments/EXP-20261001-010-repaired-and-baselines/raw/llama3.1.jsonl",
        ROOT / "experiments/EXP-20261004-003-repaired-vs-relabel/raw/relabel.jsonl",
        ROOT / "experiments/EXP-20261001-007-mixed-medium-compare/raw/cleaned.jsonl",
    ]
    for path in normal_paths:
        for trace in load_traces(path):
            if trace.get("medium") == "blog" and (trace.get("answer") or "").strip():
                prompt = writing_prompt(trace["answer"][:1200], trace.get("topic") or "", "blog")
                parsed = judge_json(prompt, model, base_url, 220, cache, cache_path, "writing-calibration")
                scores = parse_writing_scores(parsed or {})
                if scores:
                    samples.append({"bucket": "normal", **scores, "source_hidden_from_judge": path.parent.parent.name})
                break
    for topic, medium, text in DEGRADED:
        prompt = writing_prompt(text, topic, medium)
        parsed = judge_json(prompt, model, base_url, 220, cache, cache_path, "writing-calibration")
        scores = parse_writing_scores(parsed or {})
        if scores:
            samples.append({"bucket": "degraded", **scores})
    return samples


def stance_control(index: EvidenceIndex, model: str, base_url: str, cache: dict, cache_path: Path) -> dict:
    real = next((doc for doc in index.documents if len(doc["text"].split()) > 80), None)
    if real is None:
        return stance_calibration_gate(None, None)
    evidence = [
        doc["text"][:650]
        for doc in index.search(real["text"][:500], real.get("medium"), k=4, exclude_ids={real["id"]})
    ]
    real_parsed = judge_json(
        claim_prompt(real["text"][:1200], "held-out prose", None, evidence),
        model, base_url, 500, cache, cache_path, "stance-calibration",
    )
    opposed_parsed = judge_json(
        claim_prompt(OPPOSED, "saas pricing and customers", None, evidence),
        model, base_url, 500, cache, cache_path, "stance-calibration",
    )
    real_rate = stance_report(parse_claims(real_parsed, False)).get("stance_consistency_rate")
    opposed_rate = stance_report(parse_claims(opposed_parsed, False)).get("stance_consistency_rate")
    report = stance_calibration_gate(real_rate, opposed_rate)
    report["judge_model"] = model
    return report


def main() -> None:
    model = DEFAULT_JUDGE_MODEL
    family = judge_family(model)
    base_url = "http://127.0.0.1:11434"
    OUT.mkdir(parents=True, exist_ok=True)
    for name in ("config", "metrics", "figures", "raw"):
        (OUT / name).mkdir(exist_ok=True)
    cache_path = OUT / "raw" / "judge_cache.jsonl"
    cache = load_cache(cache_path)
    print(f"judge={model} family={family} cached={len(cache)}", flush=True)
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
    writing_gate["candidate_family"] = CANDIDATE_FAMILY
    writing_gate["samples"] = [{key: value for key, value in sample.items() if key != "source_hidden_from_judge"} | {"bucket": sample["bucket"]} for sample in samples]
    (OUT / "metrics" / "writing_calibration.json").write_text(json.dumps(writing_gate, indent=2) + "\n", encoding="utf-8")
    print("writing calibration", writing_gate["role"], writing_gate["reason"], flush=True)
    stance_gate = stance_control(index, model, base_url, cache, cache_path)
    (OUT / "metrics" / "stance_calibration.json").write_text(json.dumps(stance_gate, indent=2) + "\n", encoding="utf-8")
    print("stance calibration", stance_gate["role"], stance_gate["reason"], flush=True)
    pack = build_reference_pack(dataset, assignments, dispositions, base_controls())
    pack["writing_quality_calibration"] = {"calibrated": writing_gate["calibrated"], "role": writing_gate["role"]}
    copy_index = CopyIndex(iter_train_texts(dataset, assignments))
    specs = load_format_specs(SPECS)
    rows_path = OUT / "metrics" / "rows.jsonl"
    models = []
    selections = {}
    done = 0
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
            finished[(previous.get("source_experiment"), previous.get("model"), previous.get("id"))] = previous
            kept.append(line)
        rows_path.write_text("\n".join(kept) + ("\n" if kept else ""), encoding="utf-8")
        print(f"resuming {len(finished)} scored rows", flush=True)
    with rows_path.open("a", encoding="utf-8") as handle:
        for experiment, names in SOURCES:
            group = []
            for name in names:
                path = ROOT / "experiments" / experiment / "raw" / f"{name}.jsonl"
                traces = load_traces(path)
                scored_rows = []
                for trace in traces:
                    done += 1
                    if "query" not in trace:
                        trace = {**trace, "query": trace.get("topic", "")}
                    prior = finished.get((experiment, name, trace.get("id")))
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
                    row["model"] = name
                    row["source_experiment"] = experiment
                    scored_rows.append(row)
                    handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                    if done % 10 == 0:
                        handle.flush()
                        print(f"scored {done}", flush=True)
                summary = summarize_model(scored_rows, f"{experiment}/{name}", experiment)
                models.append(summary)
                group.append(summary)
            if group:
                selections[experiment] = rank_models(group)
    summary = {
        "schema": SCHEMA,
        "utility_weights": UTILITY_WEIGHTS,
        "judge_model": model,
        "judge_family": family,
        "candidate_family": CANDIDATE_FAMILY,
        "writing_quality_calibration": {"calibrated": writing_gate["calibrated"], "role": writing_gate["role"], "reason": writing_gate["reason"]},
        "stance_calibration": stance_gate,
        "evidence_documents": len(documents),
        "selection_by_experiment": selections,
        "models": models,
        "reference_counts": pack["reference_counts"],
        "limits": pack["limits"],
    }
    (OUT / "metrics" / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=False)
    dirty = subprocess.run(["git", "status", "--porcelain"], cwd=ROOT, capture_output=True, text=True, check=False)
    manifest = {
        "experiment_id": OUT.name,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "execution_commands": [f"{sys.executable} -m host_finetune.rescore_evidence"],
        "git": {"commit": commit.stdout.strip(), "dirty": bool(dirty.stdout.strip())},
        "data": {
            "source_manifest": "saved traces from EXP-20261001-007, EXP-20261001-010, EXP-20261004-003 through 006",
            "processed_dataset_hash": sha256(DEFAULT_DATASET),
            "split_manifest_hash": sha256(DEFAULT_ASSIGNMENTS),
            "gold_eval_hash": sha256(ROOT / "data" / "openai_ft" / "lemkin_gold_eval.jsonl"),
        },
        "rag": {
            "persona_profile_hash": "n/a",
            "chroma_collection": "n/a: held-out evidence is an in-memory TF-IDF index, not lemkin_train_only",
            "index_version_or_snapshot": f"validation unchanged posts excluding gold overlap and style profile; n={len(documents)}",
            "embedding_model": "n/a: tf-idf",
        },
        "model": {"model_name": "n/a: no candidate model was loaded", "model_or_adapter_hash": "n/a", "config": "evaluation only", "random_seed": "n/a"},
        "evaluation": {
            "evaluator_commit": commit.stdout.strip(),
            "judge_model": model,
            "judge_config": {"family": family, "temperature": 0, "blinded": True, "candidate_family": CANDIDATE_FAMILY},
        },
        "environment": {"python_version": platform.python_version(), "package_snapshot": "config/package_snapshot.txt", "hardware": platform.platform()},
        "outputs": {"raw": "raw/judge_cache.jsonl", "metrics": "metrics/summary.json", "warnings_and_failures": []},
    }
    (OUT / "manifest.yaml").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    freeze = subprocess.run([sys.executable, "-m", "pip", "freeze"], cwd=ROOT, capture_output=True, text=True, check=False)
    (OUT / "config" / "package_snapshot.txt").write_text(freeze.stdout, encoding="utf-8")
    (OUT / "notes.md").write_text(
        "# Writing quality and claim evidence\n\n"
        "Question: which saved drafts are well written, which retrieval-on claims are supported "
        "by the saved passages, and which advice claims match held-out Lemkin evidence?\n\n"
        f"Judge: {model} ({family}), blinded to adapter identity. Candidates are Llama-family. "
        "No adapter was retrained and no answer was regenerated.\n",
        encoding="utf-8",
    )
    print(json.dumps(selections, indent=2), flush=True)


if __name__ == "__main__":
    main()
