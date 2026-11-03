"""Score completed fixed-style pairs. The writing judge is optional and separate."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import json
import statistics
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

from eval_tools.bootstrap_ci import summarize_paired
from eval_tools.voice_fixed import real_post_reference, score_drafts
from host_finetune.compare_adapters import ROOT, SPECS, _evaluation_resources, score
from host_finetune.eval_gold_rag import load_format_specs
from host_finetune.evidence_capsule import repeats_topic
from host_finetune.factual_reliability import assess_factual
from host_finetune.judge_client import DEFAULT_JUDGE_MODEL, extract_json_object, ollama_json
from host_finetune.train_only_index import longest_shared_word_span
from host_finetune.unified_eval import evaluate_row
from host_finetune.writing_quality import judge_prompt, parse_writing_scores, writing_mean

EXP = Path(__file__).resolve().parents[1]
CELLS = EXP / "raw" / "cells"
METRICS = EXP / "metrics"
STYLE = json.loads((EXP / "config" / "style_exemplars.json").read_text(encoding="utf-8"))
EXEMPLAR_TEXT = json.loads(
    (ROOT / "host_finetune" / "exemplars" / "fixed_style_candidates.json").read_text(encoding="utf-8")
)


def _cells() -> list[dict]:
    rows = []
    for path in sorted(CELLS.glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["path"] = path.name
        rows.append(payload)
    return rows


def _trace_row(cell: dict) -> dict:
    trace = dict(cell.get("trace") or {})
    trace["arm"] = cell.get("arm")
    trace["cell_status"] = cell.get("status")
    return trace


def _mean(values: list[float]) -> float | None:
    return round(statistics.fmean(values), 4) if values else None


def _style_copy(answer: str, medium: str) -> dict:
    items = (EXEMPLAR_TEXT.get("exemplars") or {}).get(medium) or []
    spans = [
        {
            "row_id": item.get("row_id"),
            "shared_word_span": longest_shared_word_span(answer, item.get("text") or ""),
        }
        for item in items
    ]
    longest = max((item["shared_word_span"] for item in spans), default=0)
    return {"longest_shared_word_span": longest, "by_exemplar": spans}


def _pair_metrics(off: dict, on: dict) -> dict:
    off_trace = off.get("trace") or {}
    on_trace = on.get("trace") or {}
    return {
        "topic_id": off.get("topic_id"),
        "medium": off.get("medium"),
        "same_evidence": off.get("evidence_sha256") == on.get("evidence_sha256"),
        "same_hit_ids": off.get("retrieval_hit_ids") == on.get("retrieval_hit_ids"),
        "same_seed": off.get("seed") == on.get("seed"),
        "retrieval_filter_off": off.get("retrieval_filter"),
        "retrieval_filter_on": on.get("retrieval_filter"),
        "prompt_tokens_off": off.get("prompt_tokens"),
        "prompt_tokens_on": on.get("prompt_tokens"),
        "style_token_delta": None
        if off.get("prompt_tokens") is None or on.get("prompt_tokens") is None
        else int(on["prompt_tokens"]) - int(off["prompt_tokens"]),
        "latency_off": off.get("latency_seconds"),
        "latency_on": on.get("latency_seconds"),
        "words_off": len((off_trace.get("answer") or "").split()),
        "words_on": len((on_trace.get("answer") or "").split()),
    }


def _factual(trace: dict) -> dict:
    return assess_factual(
        trace.get("answer") or "",
        trace.get("topic") or "",
        trace.get("verified_claims") or [],
        evidence_status=trace.get("evidence_status") or "verified",
        hits=trace.get("retrieval_hits") or [],
    )


def _judge(traces: list[dict]) -> list[dict]:
    cache_path = METRICS / "judge_cache.jsonl"
    cached = {}
    if cache_path.is_file():
        for line in cache_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            cached[(row.get("id"), row.get("medium"), row.get("arm"))] = row
    rows = []
    ordered = sorted(traces, key=lambda row: (str(row.get("id")), str(row.get("medium")), str(row.get("arm"))))
    with cache_path.open("a", encoding="utf-8") as handle:
        for trace in ordered:
            answer = (trace.get("answer") or "").strip()
            key = {"id": trace.get("id"), "medium": trace.get("medium"), "arm": trace.get("arm")}
            found = cached.get((key["id"], key["medium"], key["arm"]))
            if found is not None:
                rows.append(found)
                continue
            if not answer:
                item = {**key, "writing_mean": None, "reason": "empty_draft"}
            else:
                raw = ollama_json(
                    judge_prompt(answer, trace.get("topic") or "", trace.get("medium") or ""),
                    DEFAULT_JUDGE_MODEL,
                )
                parsed = parse_writing_scores(extract_json_object(raw) or {})
                item = {
                    **key,
                    "writing_quality": parsed,
                    "writing_mean": None if parsed is None else round(writing_mean(parsed), 4),
                    "judge_model": DEFAULT_JUDGE_MODEL,
                }
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")
            handle.flush()
            rows.append(item)
            print(key["id"], key["medium"], key["arm"], item.get("writing_mean"), flush=True)
    return rows


def main() -> None:
    do_judge = "--judge" in sys.argv
    cells = _cells()
    complete = [cell for cell in cells if cell.get("status") == "complete"]
    by_key = {(cell["topic_id"], cell["medium"], cell["arm"]): cell for cell in complete}
    media = ["blog", "linkedin", "x", "talk"]
    topics = sorted({cell["topic_id"] for cell in complete})
    pairs = []
    for topic in topics:
        for medium in media:
            off = by_key.get((topic, medium, "off"))
            on = by_key.get((topic, medium, "on"))
            if off and on:
                pairs.append(_pair_metrics(off, on))
    specs = load_format_specs(SPECS)
    pack, copy_index = _evaluation_resources()
    summaries = {}
    unified_rows = []
    factual_rows = []
    copy_rows = []
    for arm in ("off", "on"):
        traces = [_trace_row(cell) for cell in complete if cell.get("arm") == arm and cell.get("medium")]
        summaries[arm] = score(traces, specs, pack=pack, copy_index=copy_index)["summary"]
        for cell in complete:
            if cell.get("arm") != arm or not cell.get("medium"):
                continue
            trace = _trace_row(cell)
            unified = evaluate_row(trace, pack, copy_index, specs)
            unified_rows.append({"arm": arm, "id": trace.get("id"), "medium": trace.get("medium"), "unified": unified})
            factual = _factual(trace)
            factual_rows.append({
                "arm": arm,
                "id": trace.get("id"),
                "medium": trace.get("medium"),
                "pass": factual.get("pass"),
                "reasons": factual.get("reasons"),
                "unsupported_numbers": [item.get("value") for item in (factual.get("numeric") or {}).get("unsupported") or []],
                "topic_echo": repeats_topic(trace.get("answer") or "", trace.get("topic") or ""),
                "evidence_status": trace.get("evidence_status"),
                "accepted_claims": len(trace.get("verified_claims") or []),
            })
            training = (unified.get("gates") or {}).get("copying_memorization") or {}
            style_copy = _style_copy(trace.get("answer") or "", trace.get("medium") or "")
            copy_rows.append({
                "arm": arm,
                "id": trace.get("id"),
                "medium": trace.get("medium"),
                "training_copy_span": training.get("longest_shared_word_span"),
                "training_copy_pass": training.get("pass"),
                "retrieval_copy_span": (trace.get("rag_source_copying") or {}).get("max_contiguous_words"),
                "retrieval_copy_pass": (trace.get("rag_source_copying") or {}).get("pass"),
                "style_exemplar_copy": style_copy,
            })
    judge_rows = []
    if do_judge:
        subprocess.run(["ollama", "stop", "llama3.1:latest"], check=False)
        judged = []
        for cell in complete:
            if not cell.get("medium"):
                continue
            trace = _trace_row(cell)
            judged.append(trace)
        judge_rows = _judge(judged)
        subprocess.run(["ollama", "stop", DEFAULT_JUDGE_MODEL], check=False)
    voice = {}
    try:
        from eval_tools.length_control import load_kevin_distance

        for medium, words in (("blog", 100), ("linkedin", 100), ("blog", 150)):
            label = f"{medium}_{words}"
            distance, real_texts = load_kevin_distance(ROOT, "blog" if medium == "blog" else medium)
            real = real_post_reference(real_texts, words, distance)
            arm_scores = {}
            paired = []
            clusters = []
            for arm in ("off", "on"):
                rows = [
                    {"id": cell["topic_id"], "medium": medium, "answer": (cell.get("trace") or {}).get("answer") or ""}
                    for cell in complete
                    if cell.get("arm") == arm and cell.get("medium") == medium
                ]
                scored = score_drafts(rows, {medium: distance}, {medium: words})
                arm_scores[arm] = scored
            by_arm = {
                arm: {item["id"]: item for item in items if item.get("style_distance_fixed") is not None}
                for arm, items in arm_scores.items()
            }
            for topic in sorted(set(by_arm["off"]) & set(by_arm["on"])):
                paired.append(by_arm["on"][topic]["style_distance_fixed"] - by_arm["off"][topic]["style_distance_fixed"])
                clusters.append(topic)
            voice[label] = {
                "real_posts": real,
                "mean_off": _mean([item["style_distance_fixed"] for item in by_arm["off"].values()]),
                "mean_on": _mean([item["style_distance_fixed"] for item in by_arm["on"].values()]),
                "paired_on_minus_off": summarize_paired(paired, clusters, n_boot=2000, n_perm=2000) if paired else None,
                "n_off": len(arm_scores["off"]),
                "n_on": len(arm_scores["on"]),
                "scored_off": len(by_arm["off"]),
                "scored_on": len(by_arm["on"]),
            }
    except Exception as exc:
        voice["error"] = f"{type(exc).__name__}: {exc}"
    writing_summary = None
    if judge_rows:
        paired_w = []
        clusters_w = []
        by_judge = defaultdict(dict)
        for row in judge_rows:
            if row.get("writing_mean") is None:
                continue
            by_judge[(row["id"], row["medium"])][row["arm"]] = row["writing_mean"]
        for key, arms in by_judge.items():
            if "off" in arms and "on" in arms:
                paired_w.append(arms["on"] - arms["off"])
                clusters_w.append(key[0])
        writing_summary = {
            "judge_model": DEFAULT_JUDGE_MODEL,
            "mean_off": _mean([row["writing_mean"] for row in judge_rows if row.get("arm") == "off" and row.get("writing_mean") is not None]),
            "mean_on": _mean([row["writing_mean"] for row in judge_rows if row.get("arm") == "on" and row.get("writing_mean") is not None]),
            "paired_on_minus_off": summarize_paired(paired_w, clusters_w, n_boot=2000, n_perm=2000) if paired_w else None,
            "rows": judge_rows,
        }
    report = {
        "cells": {
            "files": len(cells),
            "complete": len(complete),
            "error": sum(cell.get("status") == "error" for cell in cells),
            "unsupported": sum(cell.get("status") == "unsupported" for cell in cells),
        },
        "pairs": {
            "n": len(pairs),
            "same_evidence": sum(item["same_evidence"] for item in pairs),
            "same_hit_ids": sum(item["same_hit_ids"] for item in pairs),
            "same_seed": sum(item["same_seed"] for item in pairs),
            "mean_style_token_delta": _mean([item["style_token_delta"] for item in pairs if item["style_token_delta"] is not None]),
            "rows": pairs,
        },
        "summaries": summaries,
        "factual_rows": factual_rows,
        "copy_rows": copy_rows,
        "voice_fixed": voice,
        "writing": writing_summary,
        "unified_rows": unified_rows,
        "controls": {
            "retrieval_filters_seen": sorted({str(item["retrieval_filter_off"]) for item in pairs}),
            "style_block_is_not_a_factual_claim_list": True,
        },
    }
    METRICS.mkdir(exist_ok=True)
    (METRICS / "comparison.json").write_text(json.dumps(report, indent=2, default=str) + "\n", encoding="utf-8")
    print(json.dumps({
        "cells": report["cells"],
        "pairs": {key: report["pairs"][key] for key in ("n", "same_evidence", "same_hit_ids", "mean_style_token_delta")},
        "voice_fixed": {key: {k: value[k] for k in value if k != "real_posts"} if isinstance(value, dict) else value for key, value in voice.items()},
    }, indent=2, default=str), flush=True)


if __name__ == "__main__":
    main()
