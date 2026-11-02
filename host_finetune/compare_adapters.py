"""Generate gold topics in each trained medium and score the answers.

Prompts use the same medium lines as training. Expected facts stay on the
row for scoring and are not placed in the prompt.

Run from the repo root with the fine-tune venv::

    host_finetune\\.venv\\Scripts\\python.exe -m host_finetune.compare_adapters
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from host_finetune.eval_gold_rag import aggregate, load_format_specs, score_row
from host_finetune.factual_reliability import assess_factual, grounding_regression
from host_finetune.evidence_capsule import (
    DISTILL_BACKEND,
    DISTILL_DO_SAMPLE,
    DISTILL_SEED,
    EXTRACT_MAX_NEW_TOKENS,
    EvidenceParseError,
    EvidenceStageError,
    corrective_addendum,
    corrective_retry_improves,
    draft_retry_reason,
    evidence_provenance,
    extraction_user_prompt,
    freeze_hits,
    media_jobs,
    parse_atomic_evidence,
    partition_evidence,
    require_adapter_toggle,
    verified_claim_block,
)
from host_finetune.generation_diagnostics import stop_metadata, summarize_stops
from host_finetune.grounding_repair import (
    REPAIR_DO_SAMPLE,
    REPAIR_MAX_NEW_TOKENS,
    repair_under_disabled_adapter,
    repair_user_prompt,
    unsupported_surfaces,
)
from host_finetune.numeric_grounding import allowed_numbers, evaluate_numeric_grounding
from host_finetune.rag_source_copying import (
    RAG_SOURCE_COPY_THRESHOLD,
    RETRY_SEED_OFFSET,
    evaluate_rag_source_copying,
    retry_seed,
)
from host_finetune.relabel_continuations import sentence_final
from host_finetune.retrieval_contract import LIGHTRAG_MODES, require_backend
from host_finetune.train_only_index import (
    DEFAULT_STYLE_PATH,
    evidence_review_rows,
    format_factual_excerpts,
    format_voice_exemplars,
    longest_shared_word_span,
    medium_where,
    query_excerpts,
)

ROOT = Path(__file__).resolve().parents[1]
GOLD = ROOT / "data" / "openai_ft" / "lemkin_gold_eval.jsonl"
SPECS = ROOT / "formats" / "format_specs.yaml"
DEFAULT_OUT = ROOT / "experiments" / "EXP-20261001-007-mixed-medium-compare"
ADAPTERS = (
    ("cleaned", ROOT / "host_finetune" / "output" / "lemkin_lora_cleaned"),
    ("balanced", ROOT / "host_finetune" / "output" / "lemkin_lora_balanced"),
    ("r32", ROOT / "host_finetune" / "output" / "lemkin_lora_r32"),
)
RELABEL_PAIR = (
    ("repaired", ROOT / "host_finetune" / "output" / "lemkin_lora_repaired"),
    ("relabel", ROOT / "host_finetune" / "output" / "lemkin_lora_relabel"),
)

# max_new_tokens follows the format length targets: short for X, longer for
# blog and talk. The model is loaded at 2048 so those longer budgets fit.
MEDIUMS = (
    {
        "medium": "blog",
        "format": "blog_draft",
        "instruction": "Write a blog post in the style of Jason Lemkin about: {topic}",
        "max_new_tokens": 768,
    },
    {
        "medium": "linkedin",
        "format": "linkedin_post",
        "instruction": "Write a LinkedIn post in the style of Jason Lemkin about: {topic}",
        "max_new_tokens": 320,
    },
    {
        "medium": "x",
        "format": "x_thread",
        "instruction": "Write an X post in the style of Jason Lemkin about: {topic}",
        "max_new_tokens": 160,
    },
    {
        "medium": "talk",
        "format": "youtube_script",
        "instruction": "Write a talk in the style of Jason Lemkin about: {topic}",
        "max_new_tokens": 768,
    },
)
MAX_SEQ_LENGTH = 2048


def require_context_room(prompt_tokens: int, max_new_tokens: int) -> None:
    """Refuse a sample that would pass the configured window. Do not truncate."""
    if prompt_tokens + max_new_tokens > MAX_SEQ_LENGTH:
        raise RuntimeError(
            f"refusing to generate: prompt tokens {prompt_tokens} + max_new_tokens "
            f"{max_new_tokens} exceed {MAX_SEQ_LENGTH}. Evidence was not truncated."
        )
TEMPERATURE = 0.7
TOP_P = 0.9
REPETITION_PENALTY = 1.15
GENERATION_SEED = 42


def _lcs(a: list[str], b: list[str]) -> int:
    prev = [0] * (len(b) + 1)
    for tok in a:
        cur = [0]
        for j, other in enumerate(b, 1):
            cur.append(prev[j - 1] + 1 if tok == other else max(cur[-1], prev[j]))
        prev = cur
    return prev[-1]


def rouge_l(candidate: str, reference: str) -> float:
    cand = candidate.lower().split()
    ref = reference.lower().split()
    if not cand or not ref:
        return 0.0
    match = _lcs(cand, ref)
    prec = match / len(cand)
    rec = match / len(ref)
    if prec + rec == 0:
        return 0.0
    return 2 * prec * rec / (prec + rec)


def bleu(candidate: str, reference: str) -> float:
    from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu

    return float(sentence_bleu(
        [reference.lower().split()],
        candidate.lower().split(),
        smoothing_function=SmoothingFunction().method1,
    ))


def load_gold() -> list[dict]:
    rows = []
    with GOLD.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def prompt_for(medium: dict, topic: str) -> str:
    """Training-style instruction. Expected facts are not included."""
    return medium["instruction"].format(topic=topic)


_TASK_NAME = {
    "blog": "blog post",
    "linkedin": "LinkedIn post",
    "x": "X post",
    "talk": "talk",
}


def generation_closer(medium: dict, topic: str) -> str:
    """Restate the writing task after excerpts so the user turn does not end on evidence."""
    name = _TASK_NAME[medium["medium"]]
    return (
        f"Now write the requested {name} about: {topic}\n"
        "Use the excerpts only to ground factual claims.\n"
        "Rewrite the evidence in your own words.\n"
        "Do not reproduce 8 or more consecutive words from any excerpt.\n"
        "Do not mention the excerpts, retrieval process, excerpt labels, or these instructions."
    )


def comparison_instruction(
    medium: dict,
    topic: str,
    excerpts: list[str] | None = None,
    style_exemplars: list[str] | None = None,
) -> str:
    """Retrieval-off prompts stay identical to ``prompt_for``.

    A non-empty excerpt list appends a factual block, then a short restatement
    of the writing task. A non-empty style list appends a voice block. Expected
    facts are never inserted by this function. Empty lists leave the one-line
    instruction unchanged.
    """
    instruction = prompt_for(medium, topic)
    factual = format_factual_excerpts(excerpts)
    blocks = [
        block
        for block in (format_voice_exemplars(style_exemplars), factual)
        if block
    ]
    if not blocks:
        return instruction
    text = instruction + "\n\n" + "\n\n".join(blocks)
    if factual:
        text += "\n\n" + generation_closer(medium, topic)
    return text


def instruction_for_request(medium: dict, topic: str, retriever, style_for):
    """Build the prompt. Retrieval and style stay off when their callables are absent."""
    payload = None
    excerpts = None
    if retriever is not None:
        payload = retriever(topic, medium["medium"])
        excerpts = [hit["text"] for hit in payload.get("hits") or [] if hit.get("text")]
    style = style_for(medium["medium"]) if style_for is not None else None
    instruction = comparison_instruction(medium, topic, excerpts, style)
    return instruction, payload


def retrieval_trace_fields(payload: dict | None, answer: str, enabled: bool) -> dict:
    """Evidence log. Retrieval-off traces keep only the boolean."""
    if not enabled or payload is None:
        return {"retrieval": False}
    hits = payload.get("hits") or []
    fields = {
        "retrieval": True,
        "retrieval_query": payload.get("query"),
        "retrieval_filter": payload.get("filter"),
        "retrieval_k": payload.get("k"),
        "retrieval_hits": hits,
        "copying": [
            {
                "id": hit.get("id"),
                "shared_word_span": longest_shared_word_span(answer, hit.get("text") or ""),
            }
            for hit in hits
        ],
    }
    if payload.get("backend"):
        fields["retrieval_backend"] = payload["backend"]
    if payload.get("mode"):
        fields["retrieval_mode"] = payload["mode"]
    excluded = payload.get("excluded")
    if excluded is not None:
        reasons: dict[str, int] = {}
        for item in excluded:
            reason = str((item or {}).get("reason") or "unknown")
            reasons[reason] = reasons.get(reason, 0) + 1
        fields["retrieval_excluded_reasons"] = reasons
    return fields


def decision_metrics(traces: list[dict]) -> dict:
    """Stop-behavior numbers. BLEU and ROUGE stay outside this object."""
    observed = [row for row in traces if row.get("stop_reason") in {"eos", "length"}]
    early = sum(bool(row.get("early_eos")) for row in observed)
    mid = sum(not sentence_final(row.get("answer") or "") for row in traces)
    return {
        "n": len(traces),
        "early_eot_rate": (early / len(observed)) if observed else None,
        "early_eot_observed": len(observed),
        "mid_sentence_stop_rate": (mid / len(traces)) if traces else None,
        "use_for_selection": [],
        "gate_not_selector": ["early_eot_rate", "mid_sentence_stop_rate"],
        "logged_not_for_selection": ["paired_bleu", "paired_rouge_l", "gold_rubric.overall"],
        "selector": "host_finetune.unified_eval.rank_models",
    }


def _complete_once(model, tokenizer, instruction: str, seed: int, max_new_tokens: int, hits: list[dict], retrieval_enabled: bool) -> dict:
    """One sample. Decoding settings stay at the module constants."""
    import torch
    from transformers import set_seed

    set_seed(seed)
    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": instruction}],
        tokenize=False,
        add_generation_prompt=True,
    )
    encoded = tokenizer(prompt, return_tensors="pt")
    require_context_room(int(encoded["input_ids"].shape[1]), max_new_tokens)
    input_ids = encoded["input_ids"].to(model.device)
    attention = encoded.get("attention_mask")
    if attention is not None:
        attention = attention.to(model.device)
    with torch.inference_mode():
        output = model.generate(
            input_ids,
            attention_mask=attention,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=TEMPERATURE,
            top_p=TOP_P,
            repetition_penalty=REPETITION_PENALTY,
            pad_token_id=tokenizer.eos_token_id,
        )
    generated_ids = output[0][input_ids.shape[1]:].tolist()
    answer = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
    stopping = stop_metadata(
        generated_ids, model.generation_config.eos_token_id, max_new_tokens
    )
    report = evaluate_rag_source_copying(answer, hits, enabled=retrieval_enabled)
    return {
        "answer": answer,
        "seed": seed,
        "rendered_prompt": prompt,
        "prompt_tokens": input_ids.shape[1],
        "report": report,
        "word_count": len(answer.split()),
        **stopping,
    }


def _greedy_user(model, tokenizer, user: str, max_new_tokens: int) -> dict:
    """One adapter-off completion. The caller holds disable_adapter."""
    import time

    import torch
    from transformers import set_seed

    rendered = tokenizer.apply_chat_template(
        [{"role": "user", "content": user}],
        tokenize=False,
        add_generation_prompt=True,
    )
    encoded = tokenizer(rendered, return_tensors="pt")
    input_ids = encoded["input_ids"].to(model.device)
    attention = encoded.get("attention_mask")
    if attention is not None:
        attention = attention.to(model.device)
    set_seed(DISTILL_SEED)
    started = time.perf_counter()
    with torch.inference_mode():
        output = model.generate(
            input_ids,
            attention_mask=attention,
            max_new_tokens=max_new_tokens,
            do_sample=DISTILL_DO_SAMPLE,
            pad_token_id=tokenizer.eos_token_id,
        )
    seconds = round(time.perf_counter() - started, 3)
    generated_ids = output[0][input_ids.shape[1]:].tolist()
    text = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
    stopping = stop_metadata(
        generated_ids, model.generation_config.eos_token_id, max_new_tokens
    )
    return {
        "text": text,
        "seconds": seconds,
        "prompt": user,
        "rendered": rendered,
        "prompt_tokens": int(input_ids.shape[1]),
        "generated_tokens": len(generated_ids),
        "stop_reason": stopping["stop_reason"],
    }


def _prepare_evidence(model, tokenizer, topic: str, hits: list[dict]) -> dict:
    """Extract claims with the adapter off, then accept them deterministically."""
    require_adapter_toggle(model)
    frozen = freeze_hits(hits)
    user = extraction_user_prompt(topic, frozen)
    record = {
        "proposed": [],
        "raw_extraction": "",
        "validation_rejected": [],
        "verifier_rejected": [],
        "verified": [],
        "evidence_block": "",
        "extraction_prompt": user,
        "rendered_extraction_prompt": "",
        "extraction_seconds": None,
        "verification_seconds": 0.0,
        "verification_records": [],
        "allowed_numeric_values": [],
        "reason": None,
    }
    def _extract(prompt: str) -> dict:
        with model.disable_adapter():
            return _greedy_user(model, tokenizer, prompt, EXTRACT_MAX_NEW_TOKENS)

    extracted = _extract(user)
    attempts = 1
    proposed = None
    try:
        proposed = parse_atomic_evidence(extracted["text"])
    except EvidenceParseError:
        extracted = _extract(user)
        attempts = 2
        try:
            proposed = parse_atomic_evidence(extracted["text"])
        except EvidenceParseError:
            proposed = None
    record["raw_extraction"] = extracted["text"]
    record["rendered_extraction_prompt"] = extracted["rendered"]
    record["extraction_seconds"] = extracted["seconds"]
    record["extraction_stop_reason"] = extracted["stop_reason"]
    record["extraction_attempts"] = attempts
    if freeze_hits(hits) != frozen:
        raise RuntimeError("evidence preparation changed the retrieved passages")
    if proposed is None:
        record["reason"] = "extraction_failed"
        record["evidence_status"] = "extraction_failed"
        record["verified"] = []
        record["evidence_block"] = verified_claim_block([])
        return record
    record["proposed"] = proposed
    accepted, rejected = partition_evidence(proposed, frozen, topic)
    record["validation_rejected"] = rejected
    record["verified"] = accepted
    record["allowed_numeric_values"] = sorted(allowed_numbers(topic, accepted))
    record["evidence_block"] = verified_claim_block(accepted)
    record["evidence_status"] = "verified" if accepted else "no_accepted_claims"
    return record


def _repair_draft(model, tokenizer, topic: str, draft: str, claims: list[dict], hits: list[dict], numeric_report: dict) -> dict:
    """One greedy edit. The Relabel adapter is off for this call only."""
    prompt = repair_user_prompt(topic, draft, claims, unsupported_surfaces(numeric_report))

    def _run():
        return _greedy_user(model, tokenizer, prompt, REPAIR_MAX_NEW_TOKENS)

    generated = repair_under_disabled_adapter(model, _run)
    repaired = {
        "answer": generated["text"],
        "prompt": prompt,
        "rendered_prompt": generated["rendered"],
        "seconds": generated["seconds"],
        "stop_reason": generated["stop_reason"],
        "generated_tokens": generated["generated_tokens"],
        "adapter_enabled": False,
        "do_sample": REPAIR_DO_SAMPLE,
        "max_new_tokens": REPAIR_MAX_NEW_TOKENS,
        "temperature": None,
        "top_p": None,
        "repair_index": 1,
    }
    repaired["report"] = evaluate_rag_source_copying(repaired["answer"], hits, enabled=True)
    repaired["numeric"] = evaluate_numeric_grounding(repaired["answer"], topic, claims, enabled=True)
    return repaired


def _base_diagnostic(model, tokenizer, instruction: str, seed: int, max_new_tokens: int, hits: list[dict], topic: str, claims: list[dict]) -> dict:
    """Same prompt and sampling, adapter off. Not a deployment output."""

    def _run():
        return _complete_once(model, tokenizer, instruction, seed, max_new_tokens, hits, True)

    completion = repair_under_disabled_adapter(model, _run)
    completion = _attach_numeric(completion, topic, claims, True)
    return {
        "adapter_enabled": False,
        "used_as_deployment": False,
        "word_count": len(completion["answer"].split()),
        "unsupported_quantities": unsupported_surfaces(completion["numeric"]),
        "source_copy_overlap": completion["report"].get("max_contiguous_words"),
        "source_copy_pass": completion["report"].get("pass"),
        "numeric_pass": completion["numeric"].get("pass"),
    }


def _attach_numeric(
    completion: dict,
    topic: str,
    claims: list[dict],
    enabled: bool,
    extra_texts: list[str] | None = None,
) -> dict:
    completion["numeric"] = evaluate_numeric_grounding(
        completion["answer"], topic, claims, enabled=enabled, extra_texts=extra_texts
    )
    return completion


def final_response(copy_report: dict) -> str:
    """Numeric grounding is diagnostic. Only a source-copy failure may retry once."""
    if copy_report.get("applicable") and not copy_report.get("pass"):
        return "copy_retry"
    return "keep"


def kept_draft(first: dict, retry: dict | None) -> dict:
    """Keep a draft. The copy retry replaces it only when that retry is clean."""
    if retry is not None and retry["report"].get("pass"):
        chosen = retry
        attempt = 2
    else:
        chosen = first
        attempt = 1
    flagged = bool(chosen["report"].get("applicable") and not chosen["report"].get("pass"))
    return {"chosen": chosen, "attempt": attempt, "source_copy_flag": flagged}


def _append_generation(
    traces: list[dict],
    row: dict,
    medium: dict,
    instruction: str,
    payload: dict | None,
    first: dict,
    second: dict | None,
    retrieval_enabled: bool,
    extra: dict | None = None,
    repair: dict | None = None,
) -> None:
    if repair is not None:
        raise RuntimeError("post-generation repair is not on the deployment path")
    if second is not None and second["seed"] != retry_seed(first["seed"]):
        raise ValueError("retry seed must be the original seed plus the fixed offset")
    kept = kept_draft(first, second)
    chosen = kept["chosen"]
    traces.append({
        **row,
        "format": medium["format"],
        "medium": medium["medium"],
        "prompt": instruction,
        "rendered_prompt": chosen["rendered_prompt"],
        "answer": chosen["answer"],
        "raw_model_output": first["answer"],
        "deployment_output": chosen["answer"],
        "first_attempt_answer": first["answer"],
        "retry_answer": None if second is None else second["answer"],
        "query": row["topic"],
        "max_new_tokens": medium["max_new_tokens"],
        "seed": first["seed"],
        "retry_seed": None if second is None else second["seed"],
        "deployment_accepted": bool((chosen["answer"] or "").strip()),
        "repair_occurred": False,
        "generation_attempt": {
            "seed": first["seed"],
            "answer": first["answer"],
            "numeric_grounding": first["numeric"],
            "rag_source_copying": first["report"],
            "stop_reason": first["stop_reason"],
        },
        "repair_attempt": None,
        "accepted_attempt": kept["attempt"],
        "rag_source_copying": {
            "applicable": chosen["report"]["applicable"],
            "threshold": chosen["report"].get("threshold", RAG_SOURCE_COPY_THRESHOLD),
            "pass": chosen["report"].get("pass"),
            "flagged": kept["source_copy_flag"],
            "max_contiguous_words": chosen["report"].get("max_contiguous_words"),
            "document_id": chosen["report"].get("document_id"),
            "matched_generated_span": chosen["report"].get("matched_generated_span"),
            "first_attempt": first["report"],
            "retry": None if second is None else second["report"],
            "retry_occurred": second is not None,
            "repair_occurred": False,
        },
        "numeric_grounding": {
            "role": "diagnostic_only",
            "applicable": chosen["numeric"]["applicable"],
            "pass": chosen["numeric"].get("pass"),
            "failure": chosen["numeric"].get("failure"),
            "unsupported": chosen["numeric"].get("unsupported") or [],
            "blocks_output": False,
            "first_attempt": first["numeric"],
            "retry": None if second is None else second["numeric"],
        },
        "prompt_tokens": chosen["prompt_tokens"],
        "backend": "unsloth",
        **retrieval_trace_fields(payload, chosen["answer"], retrieval_enabled),
        **(extra or {}),
        "generated_tokens_including_stop": chosen["generated_tokens_including_stop"],
        "last_token_id": chosen["last_token_id"],
        "effective_eos_token_ids": chosen["effective_eos_token_ids"],
        "stop_reason": chosen["stop_reason"],
        "budget_fraction": chosen["budget_fraction"],
        "early_eos": chosen["early_eos"],
        "first_attempt_stop_reason": first["stop_reason"],
        "retry_stop_reason": None if second is None else second["stop_reason"],
    })
    print(
        f"  {row['id']} {medium['medium']} words={len(chosen['answer'].split())} "
        f"copy_pass={chosen['report'].get('pass')} copy_flag={kept['source_copy_flag']} "
        f"numeric_pass={chosen['numeric'].get('pass')} retry={second is not None}",
        flush=True,
    )


def _bnb_flag(model, name: str) -> bool:
    """PEFT wraps the quantized base. The flag can sit on the wrapper or the inner module."""
    candidates = [model, getattr(model, "base_model", None)]
    base = candidates[1]
    if base is not None:
        candidates.append(getattr(base, "model", None))
    return any(getattr(item, name, False) for item in candidates if item is not None)


def load_generator(adapter: Path, quantization: str = "4bit"):
    """Load the frozen adapter for inference. The caller unloads it."""
    from unsloth import FastLanguageModel

    if quantization == "8bit":
        # The adapter config still names the bnb-4bit repo. load_in_4bit=False
        # makes Unsloth resolve the full Instruct weights, then bitsandbytes
        # quantizes those weights to 8-bit. The LoRA file is not retrained.
        load_kwargs = {"load_in_4bit": False, "load_in_8bit": True}
    elif quantization == "4bit":
        load_kwargs = {"load_in_4bit": True}
    else:
        raise ValueError(f"unsupported quantization: {quantization}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=str(adapter),
        max_seq_length=MAX_SEQ_LENGTH,
        dtype=None,
        **load_kwargs,
    )
    FastLanguageModel.for_inference(model)
    load_report = {
        "requested": quantization,
        "load_in_4bit": quantization == "4bit",
        "load_in_8bit": quantization == "8bit",
        "is_loaded_in_4bit": _bnb_flag(model, "is_loaded_in_4bit"),
        "is_loaded_in_8bit": _bnb_flag(model, "is_loaded_in_8bit"),
        "name_or_path": getattr(getattr(model, "config", None), "_name_or_path", None),
    }
    print(f"quantization load {json.dumps(load_report)}", flush=True)
    if quantization == "8bit" and not load_report["is_loaded_in_8bit"]:
        raise RuntimeError("8-bit load was requested but the model is not loaded in 8-bit")
    if quantization == "4bit" and not load_report["is_loaded_in_4bit"]:
        raise RuntimeError("4-bit load was requested but the model is not loaded in 4-bit")
    return model, tokenizer, load_report


def generate(
    adapter: Path,
    gold: list[dict],
    retriever=None,
    style_for=None,
    seed_base: int = GENERATION_SEED,
    topic_rows: list[tuple[int, dict]] | None = None,
    quantization: str = "4bit",
    loaded: tuple | None = None,
    unload: bool = True,
    evidence_cache: dict | None = None,
    on_trace=None,
    should_stop=None,
    media_allow: set[str] | None = None,
    relationship_for=None,
    context_pad_for=None,
    graph_context_for=None,
) -> tuple[list[dict], dict]:
    import time

    import torch

    if relationship_for is not None and context_pad_for is not None:
        raise ValueError("relationship evidence and a context pad cannot be combined")
    if graph_context_for is not None and (relationship_for is not None or context_pad_for is not None):
        raise ValueError("model f graph context cannot be combined with relationship evidence or a context pad")
    own_model = loaded is None
    if own_model:
        model, tokenizer, load_report = load_generator(adapter, quantization)
    else:
        model, tokenizer, load_report = loaded
    if retriever is not None:
        require_adapter_toggle(model)
    traces = []
    retrieval_enabled = retriever is not None
    indexed = list(enumerate(gold)) if topic_rows is None else topic_rows
    stopped = False
    try:
        for topic_index, row in indexed:
            if should_stop is not None and should_stop():
                stopped = True
                break
            if not retrieval_enabled:
                for medium_index, medium in enumerate(MEDIUMS):
                    if media_allow is not None and medium["medium"] not in media_allow:
                        continue
                    if should_stop is not None and should_stop():
                        stopped = True
                        break
                    seed = seed_base + topic_index * len(MEDIUMS) + medium_index
                    instruction, payload = instruction_for_request(
                        medium, row["topic"], None, style_for
                    )
                    medium_started = time.perf_counter()
                    first = _attach_numeric(
                        _complete_once(
                            model, tokenizer, instruction, seed, medium["max_new_tokens"], [], False
                        ),
                        row["topic"],
                        [],
                        False,
                    )
                    _append_generation(
                        traces, row, medium, instruction, payload, first, None, False,
                        {"latency_seconds": round(time.perf_counter() - medium_started, 3)},
                    )
                    if on_trace is not None:
                        on_trace(traces[-1])
                if stopped:
                    break
                continue
            payload = retriever(row["topic"], MEDIUMS[0]["medium"])
            hits = [
                hit for hit in (payload or {}).get("hits") or []
                if (hit.get("text") or "").strip()
            ]
            frozen = freeze_hits(hits)
            payload = {**payload, "hits": frozen}
            cache_key = row.get("id") or row["topic"]
            if evidence_cache is not None and cache_key in evidence_cache:
                evidence_record = evidence_cache[cache_key]
                print(f"  reuse evidence {row['id']}", flush=True)
            else:
                print(f"  extract {row['id']} hits={[hit.get('id') for hit in frozen]}", flush=True)
                try:
                    evidence_record = _prepare_evidence(model, tokenizer, row["topic"], frozen)
                except EvidenceStageError as exc:
                    print(
                        f"evidence stage failed for {row['id']}: {exc.reason}; "
                        "skipping generation for this topic",
                        flush=True,
                    )
                    traces.append({
                        **row,
                        "medium": None,
                        "format": None,
                        "prompt": "",
                        "answer": "",
                        "query": row["topic"],
                        "evidence_rejected": True,
                        "backend": "unsloth",
                        **retrieval_trace_fields(payload, "", True),
                        **evidence_provenance(frozen, exc.record),
                    })
                    if on_trace is not None:
                        on_trace(traces[-1])
                    continue
                if evidence_cache is not None:
                    evidence_cache[cache_key] = evidence_record
            print(
                f"  evidence {row['id']} accepted={len(evidence_record['verified'])} "
                f"rejected={len(evidence_record['validation_rejected'])} "
                f"extract_s={evidence_record['extraction_seconds']}",
                flush=True,
            )
            relation_rows = relationship_for(row, payload) if relationship_for is not None else None
            context_pad = context_pad_for(row, payload) if context_pad_for is not None else None
            graph_plan = graph_context_for(row, payload) if graph_context_for is not None else None
            if graph_plan is not None:
                from host_finetune.model_f_context.pipeline import context_media_jobs

                jobs = context_media_jobs(
                    row["topic"],
                    frozen,
                    evidence_record["verified"],
                    evidence_record.get("evidence_status") or "verified",
                    graph_plan,
                    style_for=style_for,
                    count_tokens=graph_plan.get("count_tokens"),
                    budget=graph_plan.get("experiment_budget"),
                )
                relation_texts = []
            else:
                jobs = media_jobs(
                    row["topic"],
                    frozen,
                    evidence_record["verified"],
                    evidence_record.get("evidence_status") or "verified",
                    style_for=style_for,
                    relationships=relation_rows,
                    context_pad=context_pad,
                )
                relation_texts = []
                for item in relation_rows or []:
                    if item.get("use_in_generation") and item.get("prompt_fact"):
                        relation_texts.append(str(item["prompt_fact"]))
                    relation_texts.extend(str(qualifier) for qualifier in (item.get("qualifiers") or []))
            base_relation_texts = list(relation_texts)
            if len({job["evidence_block"] for job in jobs}) != 1:
                raise RuntimeError("media reused more than one evidence set")
            provenance = evidence_provenance(frozen, evidence_record)
            provenance["prompt_contract"] = jobs[0]["prompt_contract"] if jobs else None
            for medium_index, job in enumerate(jobs):
                if "grounding_texts" in job:
                    relation_texts = list(job["grounding_texts"])
                else:
                    relation_texts = list(base_relation_texts)
                if media_allow is not None and job["medium"]["medium"] not in media_allow:
                    continue
                if should_stop is not None and should_stop():
                    stopped = True
                    break
                seed = seed_base + topic_index * len(MEDIUMS) + medium_index
                instruction = job["instruction"]
                medium_started = time.perf_counter()
                first = _attach_numeric(
                    _complete_once(
                        model,
                        tokenizer,
                        instruction,
                        seed,
                        job["medium"]["max_new_tokens"],
                        job["hits"],
                        True,
                    ),
                    row["topic"],
                    job["claims"],
                    True,
                    relation_texts,
                )
                second = None
                stored_instruction = instruction
                status = job.get("evidence_status") or "verified"

                def _facts(answer: str) -> dict:
                    return assess_factual(
                        answer,
                        row["topic"],
                        job["claims"],
                        evidence_status=status,
                        hits=job["hits"],
                        extra_texts=relation_texts,
                    )

                if final_response(first["report"]) == "copy_retry":
                    print(f"  copy retry {row['id']} {job['medium']['medium']}", flush=True)
                    candidate = _attach_numeric(
                        _complete_once(
                            model,
                            tokenizer,
                            instruction,
                            retry_seed(seed),
                            job["medium"]["max_new_tokens"],
                            job["hits"],
                            True,
                        ),
                        row["topic"],
                        job["claims"],
                        True,
                        relation_texts,
                    )
                    if candidate["report"].get("pass") and not grounding_regression(
                        _facts(first["answer"]), _facts(candidate["answer"])
                    ):
                        second = candidate
                else:
                    reason = draft_retry_reason(
                        first["answer"],
                        row["topic"],
                        job["medium"]["medium"],
                        first["numeric"],
                        job["claims"],
                        status,
                    )
                    if reason:
                        print(f"  {reason} retry {row['id']} {job['medium']['medium']}", flush=True)
                        augmented = instruction + "\n\n" + corrective_addendum(reason, first["numeric"])
                        candidate = _attach_numeric(
                            _complete_once(
                                model,
                                tokenizer,
                                augmented,
                                retry_seed(seed),
                                job["medium"]["max_new_tokens"],
                                job["hits"],
                                True,
                            ),
                            row["topic"],
                            job["claims"],
                            True,
                            relation_texts,
                        )
                        improves = corrective_retry_improves(
                            reason,
                            first["answer"],
                            candidate["answer"],
                            row["topic"],
                            first["numeric"],
                            candidate["numeric"],
                            job["claims"],
                        )
                        if candidate["report"].get("pass") and improves and not grounding_regression(
                            _facts(first["answer"]), _facts(candidate["answer"])
                        ):
                            second = candidate
                            stored_instruction = augmented
                extra = dict(provenance)
                extra["fixed_style_enabled"] = style_for is not None
                extra["style_applied"] = bool(job.get("style_applied"))
                extra["style_exemplar_count"] = int(job.get("style_exemplar_count") or 0)
                if relationship_for is not None:
                    extra["graph_relationship_count"] = len(relation_texts)
                    extra["graph_relationships"] = list(relation_rows or [])
                    extra["graph_relationship_block"] = job.get("relationship_block") or ""
                if context_pad_for is not None:
                    extra["context_pad"] = True
                    extra["context_pad_text"] = context_pad or ""
                if graph_plan is not None:
                    extra["model_f_context"] = job.get("model_f_context")
                    extra["graph_context_contract"] = job.get("graph_context_contract")
                    extra["graph_relationship_count"] = len(relation_texts)
                extra["latency_seconds"] = round(time.perf_counter() - medium_started, 3)
                _append_generation(
                    traces, row, job["medium"], stored_instruction, payload, first, second, True, extra
                )
                if on_trace is not None:
                    on_trace(traces[-1])
            if stopped:
                break
        return traces, load_report
    finally:
        if own_model and unload:
            del model
            torch.cuda.empty_cache()


def _paired(traces: list[dict]) -> dict:
    if not traces:
        return {"paired_bleu": 0.0, "paired_rouge_l": 0.0, "n": 0}
    bleus = [bleu(row["answer"], row.get("gold_reference") or "") for row in traces]
    rouges = [rouge_l(row["answer"], row.get("gold_reference") or "") for row in traces]
    return {
        "paired_bleu": round(statistics.mean(bleus), 4),
        "paired_rouge_l": round(statistics.mean(rouges), 4),
        "n": len(traces),
    }


def score(traces: list[dict], specs: dict, pack: dict | None = None, copy_index=None) -> dict:
    scored = [score_row(row, specs) for row in traces]
    summary = aggregate(scored)
    summary.update(_paired(traces))
    summary['stopping'] = summarize_stops(traces)
    summary['decision'] = decision_metrics(traces)
    summary['selection_warning'] = (
        'Stop rates are a completion gate, not persona fidelity. Gold rubric overall is '
        'brief/task agreement. BLEU, ROUGE-L, and BERTScore are reference-overlap diagnostics. '
        'Ranking uses unified_eval.rank_models after completion, copying, format, and '
        'applicable grounding gates. Gold topics are a development set.'
    )
    by_medium = {}
    for medium in MEDIUMS:
        subset = [row for row in traces if row.get("medium") == medium["medium"]]
        if not subset:
            continue
        medium_summary = aggregate([score_row(row, specs) for row in subset])
        medium_summary.update(_paired(subset))
        medium_summary['stopping'] = summarize_stops(subset)
        medium_summary['decision'] = decision_metrics(subset)
        by_medium[medium["medium"]] = medium_summary
    summary["by_medium"] = by_medium
    unified_rows = []
    if pack is not None:
        from host_finetune.unified_eval import evaluate_row, summarize_model

        unified_rows = [evaluate_row(row, pack, copy_index, specs) for row in traces]
        summary["unified"] = summarize_model(unified_rows, "pending", "this_run")
    return {"summary": summary, "rows": scored, "unified_rows": unified_rows}


def generate_ollama(
    model_name: str, gold: list[dict], base_url: str, retriever=None, style_for=None,
    seed_base: int = GENERATION_SEED,
) -> list[dict]:
    """Same prompts and sampling settings, through Ollama's chat API."""
    import requests

    traces = []
    for topic_index, row in enumerate(gold):
        for medium_index, medium in enumerate(MEDIUMS):
            seed = seed_base + topic_index * len(MEDIUMS) + medium_index
            instruction, retrieved = instruction_for_request(
                medium, row["topic"], retriever, style_for
            )
            response = requests.post(
                f"{base_url.rstrip('/')}/api/chat",
                json={
                    "model": model_name,
                    "messages": [{"role": "user", "content": instruction}],
                    "stream": False,
                    "options": {
                        "temperature": TEMPERATURE,
                        "top_p": TOP_P,
                        "repeat_penalty": REPETITION_PENALTY,
                        "num_predict": medium["max_new_tokens"],
                        "num_ctx": MAX_SEQ_LENGTH,
                        "seed": seed,
                    },
                },
                timeout=600,
            )
            response.raise_for_status()
            payload = response.json()
            answer = payload["message"]["content"].strip()
            traces.append({
                **row,
                "format": medium["format"],
                "medium": medium["medium"],
                "prompt": instruction,
                "rendered_prompt": instruction,
                "answer": answer,
                "query": row["topic"],
                "max_new_tokens": medium["max_new_tokens"],
                "backend": "ollama",
                "ollama_model": model_name,
                "seed": seed,
                "prompt_tokens": payload.get('prompt_eval_count'),
                "generated_tokens_including_stop": payload.get('eval_count'),
                "backend_done_reason": payload.get('done_reason'),
                "stop_reason": 'length' if payload.get('done_reason') == 'length' else 'other_or_unknown',
                **retrieval_trace_fields(retrieved, answer, retriever is not None),
            })
            print(
                f"  {row['id']} {medium['medium']} words={len(answer.split())}",
                flush=True,
            )
    return traces


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=os.environ.get("EXPERIMENT_OUT_DIR", str(DEFAULT_OUT)),
    )
    parser.add_argument(
        "--adapters",
        default="",
        help="Comma-separated name=path entries. Default is the three trained adapters. Use 'none' to skip.",
    )
    parser.add_argument(
        "--ollama",
        default="",
        help="Comma-separated Ollama model names to score after the adapters.",
    )
    parser.add_argument(
        "--ollama-base",
        default=os.environ.get("OLLAMA_BASE", "http://localhost:11434"),
    )
    parser.add_argument(
        "--retrieval",
        action="store_true",
        help="Append factual excerpts from the active train-only index. Default is off.",
    )
    parser.add_argument(
        "--rag-version",
        default=None,
        help="v1 or v2. Default reads host_finetune/rag_active.json.",
    )
    parser.add_argument("--collection", default=None)
    parser.add_argument("--chroma-path", type=Path, default=None)
    parser.add_argument("--k", type=int, default=4)
    parser.add_argument(
        "--retrieval-backend",
        choices=("dense", "lightrag"),
        default="dense",
        help="dense keeps the active Chroma index. lightrag is off unless this flag is set.",
    )
    parser.add_argument(
        "--lightrag-mode",
        choices=LIGHTRAG_MODES,
        default="hybrid",
        help="Used only when --retrieval-backend lightrag. top_k inside LightRAG is not the evidence budget.",
    )
    parser.add_argument(
        "--lightrag-workdir",
        type=Path,
        default=ROOT / "artifacts" / "lightrag" / "model_f" / "storage",
    )
    parser.add_argument(
        "--evidence-budget",
        type=int,
        default=None,
        help="Final source passages passed to evidence extraction. Default is --k.",
    )
    parser.add_argument(
        "--lightrag-max-per-source",
        type=int,
        default=1,
        help="Cap LightRAG chunks from one graph document. Dense retrieval is unchanged.",
    )
    parser.add_argument("--lightrag-model", default="llama3.1:latest")
    parser.add_argument(
        "--graph-relationships",
        action="store_true",
        help=(
            "Add source-verified LightRAG relationships as optional evidence. "
            "Off by default. Dense Model E is unchanged. The frozen Relabel adapter is unchanged."
        ),
    )
    parser.add_argument(
        "--model-f-context",
        choices=("off", "source", "relationships", "descriptions", "multihop"),
        default="off",
        help=(
            "Model F graph-context profile. Off keeps the current prompt. "
            "source, relationships, descriptions, and multihop are evaluation "
            "configurations of this same Model F path. Requires LightRAG retrieval."
        ),
    )
    parser.add_argument(
        "--lightrag-fallback",
        choices=("error", "dense"),
        default="error",
        help="error refuses a missing LightRAG catalog. dense uses the active Chroma index instead.",
    )
    parser.add_argument("--embed-model", default="nomic-embed-text")
    parser.add_argument(
        "--seed-base",
        type=int,
        default=GENERATION_SEED,
        help="Added to topic and medium indexes. Default 42 matches EXP-003.",
    )
    parser.add_argument(
        "--fixed-style",
        action="store_true",
        help="Append the review-candidate voice file. Default is off.",
    )
    parser.add_argument("--style-path", type=Path, default=DEFAULT_STYLE_PATH)
    parser.add_argument(
        "--max-topics",
        type=int,
        default=None,
        help="Use only the first N gold topics. Default is the full gold file.",
    )
    parser.add_argument(
        "--skip-score",
        action="store_true",
        help="Write generation traces and skip unified_eval.",
    )
    parser.add_argument(
        "--topic-ids",
        default=None,
        help="Comma-separated gold ids to generate. Uses each topic's original index for the seed.",
    )
    parser.add_argument(
        "--quantization",
        choices=("4bit", "8bit"),
        default="4bit",
        help="BitsAndBytes width of the frozen base. 8bit quantizes the full Instruct weights on load. The adapter file is unchanged.",
    )
    parser.add_argument(
        "--freeze-retrieval-out",
        type=Path,
        default=None,
        help="Query each selected topic and write the payloads, then exit before loading the generator.",
    )
    parser.add_argument(
        "--frozen-retrieval",
        type=Path,
        default=None,
        help="Reuse a payload file from --freeze-retrieval-out. Does not call the retriever again.",
    )
    return parser.parse_args()


def adapter_list(raw: str) -> tuple[tuple[str, Path], ...]:
    if raw.strip().lower() == "none":
        return ()
    if raw.strip() == "repaired-vs-relabel":
        return RELABEL_PAIR
    if not raw.strip():
        configured = os.environ.get("RELABEL_ADAPTER_PATH", "").strip()
        if configured:
            return (("relabel", Path(configured)),)
        return ADAPTERS
    found = []
    for item in raw.split(","):
        name, path = item.split("=", 1)
        found.append((name.strip(), Path(path.strip())))
    return tuple(found)


def git_state() -> dict:
    """HEAD and dirty flag. A missing git binary leaves both null."""
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip()
        dirty = bool(
            subprocess.check_output(
                ["git", "status", "--porcelain"], cwd=ROOT, text=True
            ).strip()
        )
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "dirty": None}
    return {"commit": commit, "dirty": dirty}


def adapter_base_model(adapter: Path) -> str | None:
    path = adapter / "adapter_config.json"
    if not path.is_file():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload.get("base_model_name_or_path")


def limit_topics(gold: list[dict], max_topics: int | None) -> list[dict]:
    if max_topics is None:
        return gold
    if max_topics < 1:
        raise SystemExit("--max-topics must be at least 1")
    return gold[:max_topics]


def selected_topic_rows(
    gold: list[dict],
    topic_ids: str | None,
    max_topics: int | None,
) -> list[tuple[int, dict]]:
    """Original gold indexes. A targeted id keeps the seed it would have had in the full run."""
    if not topic_ids:
        return list(enumerate(limit_topics(gold, max_topics)))
    wanted = [part.strip() for part in topic_ids.split(",") if part.strip()]
    if not wanted:
        raise SystemExit("--topic-ids did not name any topics")
    positions = {row["id"]: index for index, row in enumerate(gold)}
    missing = [item for item in wanted if item not in positions]
    if missing:
        raise SystemExit(f"unknown topic ids: {', '.join(missing)}")
    ordered = sorted(set(wanted), key=lambda item: positions[item])
    return [(positions[item], gold[positions[item]]) for item in ordered]


def row_provenance(args: argparse.Namespace, adapter: Path, stamp: dict) -> dict:
    """Identity fields stored on every new trace. Scoring can be rerun later."""
    return {
        "experiment_id": args.out_dir.name,
        "adapter_path": str(adapter),
        "base_model": adapter_base_model(adapter),
        "inference_quantization": args.quantization,
        "collection": args.collection if args.retrieval else None,
        "embedding_model": args.embed_model if args.retrieval else None,
        "top_k": args.k if args.retrieval else None,
        "git_sha": stamp["commit"],
        "timestamp": stamp["timestamp"],
    }


def _recorded_collection(
    retrieval: bool,
    collection: str | None,
    chroma_path: str | None,
) -> str | None:
    """Keep an explicit collection. If one was omitted, recover it from a known path.

    The frozen evidence test records a v1 path and no collection name. Production
    calls pass the collection from ``rag_active.json``, and that value wins.
    """
    if not retrieval:
        return None
    if collection:
        return collection
    if not chroma_path:
        return None
    from host_finetune.rag_runtime import V1_COLLECTION, V1_DIR, V2_COLLECTION, V2_DIR

    name = Path(chroma_path).name
    if name == V1_DIR.name:
        return V1_COLLECTION
    if name == V2_DIR.name:
        return V2_COLLECTION
    return None


def write_parameters(
    out: Path,
    retrieval: bool = False,
    fixed_style: bool = False,
    seed_base: int = GENERATION_SEED,
    n_topics: int = 30,
    max_topics: int | None = None,
    topic_ids: str | None = None,
    skip_score: bool = False,
    embed_model: str | None = None,
    retrieval_k: int | None = None,
    chroma_path: str | None = None,
    collection: str | None = None,
    git: dict | None = None,
    quantization: str = "4bit",
    retrieval_backend: str = "dense",
    lightrag_mode: str | None = None,
    lightrag_workdir: str | None = None,
    evidence_budget: int | None = None,
    lightrag_fallback: str = "error",
    frozen_retrieval: str | None = None,
    model_f_context: str = "off",
) -> None:
    payload = {
        "experiment_id": out.name,
        "gold_eval": "data/openai_ft/lemkin_gold_eval.jsonl",
        "n_topics": n_topics,
        "max_topics": max_topics,
        "topic_ids": topic_ids,
        "skip_score": skip_score,
        "git": git,
        "mediums": [
            {
                "medium": medium["medium"],
                "format": medium["format"],
                "instruction": medium["instruction"],
                "max_new_tokens": medium["max_new_tokens"],
            }
            for medium in MEDIUMS
        ],
        "generation": {
            "max_seq_length": MAX_SEQ_LENGTH,
            "temperature": TEMPERATURE,
            "top_p": TOP_P,
            "repetition_penalty": REPETITION_PENALTY,
            "do_sample": True,
            "quantization": quantization,
            "load_in_4bit": quantization == "4bit",
            "load_in_8bit": quantization == "8bit",
            "seed_base": seed_base,
            "seed_policy": "base + topic_index * n_mediums + medium_index; reset per request",
            "rag_source_copy_threshold": RAG_SOURCE_COPY_THRESHOLD,
            "rag_source_copy_retry_seed_offset": RETRY_SEED_OFFSET,
            "rag_source_copy_max_retries": 1,
        },
        "evidence_distillation": {
            "enabled": retrieval,
            "representation": "atomic_claims" if retrieval else None,
            "once_per_topic": True,
            "adapter_enabled": False,
            "backend": DISTILL_BACKEND if retrieval else None,
            "do_sample": DISTILL_DO_SAMPLE if retrieval else None,
            "extraction_max_new_tokens": EXTRACT_MAX_NEW_TOKENS if retrieval else None,
            "seed": DISTILL_SEED if retrieval else None,
            "temperature": None,
            "top_p": None,
            "repetition_penalty": None,
            "second_model": None,
            "llm_verifier_on_critical_path": False,
            "internal_copy_gate": False,
            "support_spans_in_generator_prompt": False,
            "numeric_grounding": True if retrieval else False,
            "numeric_role": "diagnostic_only" if retrieval else None,
            "numeric_blocks_output": False,
            "numeric_failure_response": "record_only" if retrieval else None,
            "numeric_seed_retry": False,
            "repair_on_critical_path": False,
            "style_rewrite_on_critical_path": False,
            "base_generation_on_critical_path": False,
            "lora_scale": 1.0 if retrieval else None,
            "closer": "minimal" if retrieval else None,
        },
        "expected_facts_in_prompt": False,
        "retrieval": {
            "enabled": retrieval,
            "collection": _recorded_collection(retrieval, collection, chroma_path),
            "chroma_path": chroma_path if retrieval else None,
            "embedding_model": embed_model if retrieval else None,
            "top_k": retrieval_k if retrieval else None,
            "medium_filter": False,
            "backend": retrieval_backend if retrieval else None,
            "lightrag_mode": lightrag_mode if retrieval and retrieval_backend == "lightrag" else None,
            "lightrag_workdir": lightrag_workdir if retrieval and retrieval_backend == "lightrag" else None,
            "evidence_budget": (
                evidence_budget if evidence_budget is not None else retrieval_k
            ) if retrieval else None,
            "lightrag_fallback": lightrag_fallback if retrieval and retrieval_backend == "lightrag" else None,
            "frozen_retrieval": frozen_retrieval if retrieval else None,
            "model_f_context": model_f_context if retrieval else "off",
            "note": "Factual retrieval searches all train-owned media. Output medium still selects the generation instruction.",
        },
        "fixed_style": {
            "enabled": fixed_style,
            "note": "Default off. Voice examples are not evidence and are not loaded for retrieval-off runs.",
        },
    }
    (out / "config").mkdir(parents=True, exist_ok=True)
    (out / "config" / "run_parameters.json").write_text(
        json.dumps(payload, indent=2) + "\n",
        encoding="utf-8",
    )


def apply_active_rag(args: argparse.Namespace):
    """Fill a missing path or collection from rag_active.json. Explicit flags win."""
    from host_finetune.rag_runtime import LEGACY_COLLECTION, resolve_rag

    selected = resolve_rag(args.rag_version)
    if args.collection is None:
        args.collection = selected["collection"]
    if args.chroma_path is None:
        args.chroma_path = selected["path"]
    if args.collection == LEGACY_COLLECTION:
        raise SystemExit("refusing lemkin_content")
    args.rag_version_resolved = selected["version"]
    return selected


def build_runtime_retriever(args: argparse.Namespace):
    """CPU retrieval against the selected train-only index. Used only when --retrieval is set."""
    import chromadb
    from chromadb.config import Settings

    from host_finetune.rebuild_chroma import _embed_batch
    from host_finetune.train_only_index import assert_collection_allowed

    assert_collection_allowed(args.collection)
    if args.collection == "lemkin_content":
        raise SystemExit("refusing lemkin_content")
    if not args.chroma_path.exists():
        raise SystemExit(f"train-only index not found at {args.chroma_path}")
    client = chromadb.PersistentClient(
        path=str(args.chroma_path), settings=Settings(anonymized_telemetry=False)
    )
    collection = client.get_collection(args.collection)
    print(
        f"RAG version={args.rag_version_resolved} path={args.chroma_path} "
        f"collection={args.collection} documents={collection.count()}",
        flush=True,
    )

    def retriever(topic: str, comparison_medium: str) -> dict:
        # Output medium selects the generation instruction only. Factual
        # evidence is the same top-k from the whole train-only collection.
        del comparison_medium
        vectors = _embed_batch([topic], args.ollama_base, args.embed_model, cpu=True)
        hits = query_excerpts(collection, vectors[0], args.k, where=None)
        return {
            "query": topic,
            "filter": None,
            "k": args.k,
            "backend": "dense",
            "mode": None,
            "hits": hits,
        }

    return retriever


def write_frozen_retrieval(path: Path, topics: list[str], retriever) -> None:
    """Save one retrieval payload per topic. The generator is not loaded."""
    rows = []
    for topic in topics:
        payload = dict(retriever(topic, "blog"))
        payload["query"] = topic
        rows.append(payload)
        print(
            f"froze {topic[:80]} hits={len(payload.get('hits') or [])}",
            flush=True,
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
        encoding="utf-8",
    )


def load_frozen_retriever(path: Path):
    """Return the saved payloads. This does not embed or call LightRAG."""
    if not path.is_file():
        raise SystemExit(f"frozen retrieval file is missing: {path}")
    by_query = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        query = row.get("query") or row.get("topic")
        if not query:
            raise SystemExit("frozen retrieval row is missing a query")
        if query in by_query:
            raise SystemExit(f"duplicate frozen retrieval for {query}")
        by_query[query] = row

    def retriever(topic: str, comparison_medium: str) -> dict:
        del comparison_medium
        if topic not in by_query:
            raise SystemExit(f"frozen retrieval has no payload for {topic}")
        payload = dict(by_query[topic])
        payload.setdefault("query", topic)
        payload.setdefault("hits", [])
        return payload

    return retriever


def build_selected_retriever(args: argparse.Namespace):
    """Dense Chroma unless LightRAG was explicitly selected."""
    backend = require_backend(args.retrieval_backend)
    if backend == "dense":
        return build_runtime_retriever(args)
    catalog = Path(args.lightrag_workdir) / "catalog.jsonl"
    if not catalog.is_file():
        if args.lightrag_fallback == "dense":
            print(
                f"LightRAG catalog missing at {catalog}; using dense retrieval",
                flush=True,
            )
            return build_runtime_retriever(args)
        raise SystemExit(f"LightRAG catalog missing: {catalog}")
    from host_finetune.lightrag_backend import build_lightrag_retriever

    return build_lightrag_retriever(args)


def load_style_for(path: Path):
    """Return a medium lookup. The file is read only when --fixed-style is set."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    exemplars = payload.get("exemplars") or {}

    def style_for(comparison_medium: str) -> list[str]:
        platform = medium_where(comparison_medium)["medium"]
        items = exemplars.get(platform) or []
        return [item["text"] for item in items if (item.get("text") or "").strip()]

    return style_for


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
        encoding="utf-8",
    )


def _stamp_traces(traces: list[dict], provenance: dict) -> None:
    for row in traces:
        row.update(provenance)


def main() -> None:
    args = parse_args()
    apply_active_rag(args)
    plan = selected_topic_rows(load_gold(), args.topic_ids, args.max_topics)
    gold = [row for _index, row in plan]
    specs = load_format_specs(SPECS)
    out = args.out_dir
    if args.topic_ids and (out / "raw" / "relabel.jsonl").exists():
        raise SystemExit(
            "refusing to overwrite an existing relabel.jsonl during targeted recovery; "
            "choose a new --out-dir"
        )
    out.mkdir(parents=True, exist_ok=True)
    (out / "raw").mkdir(exist_ok=True)
    (out / "metrics").mkdir(exist_ok=True)
    recorded_git = git_state()
    write_parameters(
        out,
        retrieval=args.retrieval,
        fixed_style=args.fixed_style,
        seed_base=args.seed_base,
        n_topics=len(gold),
        max_topics=args.max_topics,
        topic_ids=args.topic_ids,
        skip_score=args.skip_score,
        embed_model=args.embed_model,
        retrieval_k=args.k,
        chroma_path=str(args.chroma_path),
        collection=args.collection,
        git=recorded_git,
        quantization=args.quantization,
        retrieval_backend=args.retrieval_backend,
        lightrag_mode=args.lightrag_mode,
        lightrag_workdir=str(args.lightrag_workdir),
        evidence_budget=args.evidence_budget,
        lightrag_fallback=args.lightrag_fallback,
        frozen_retrieval=str(args.frozen_retrieval) if args.frozen_retrieval else None,
        model_f_context=args.model_f_context,
    )
    if args.freeze_retrieval_out and args.frozen_retrieval:
        raise SystemExit("choose either --freeze-retrieval-out or --frozen-retrieval")
    if (args.freeze_retrieval_out or args.frozen_retrieval) and not args.retrieval:
        raise SystemExit("frozen retrieval requires --retrieval so claim extraction stays on")
    if args.frozen_retrieval:
        retriever = load_frozen_retriever(args.frozen_retrieval)
    else:
        retriever = build_selected_retriever(args) if args.retrieval else None
    if args.freeze_retrieval_out:
        if retriever is None:
            raise SystemExit("--freeze-retrieval-out requires a retriever")
        write_frozen_retrieval(
            args.freeze_retrieval_out,
            [row["topic"] for row in gold],
            retriever,
        )
        print(f"wrote {args.freeze_retrieval_out}", flush=True)
        return
    style_for = load_style_for(args.style_path) if args.fixed_style else None
    combined = {}
    traces_by_name: dict[str, list[dict]] = {}
    load_reports: dict[str, dict] = {}
    pack, copy_index = (None, None)
    if not args.skip_score:
        pack, copy_index = _evaluation_resources()
    run_stamp = {
        "commit": recorded_git["commit"],
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    for name, adapter in adapter_list(args.adapters):
        print(f"=== {name} {adapter}", flush=True)
        from host_finetune.graph_relationships import generation_relationships
        from host_finetune.model_f_context import graph_context_callback

        context_for = graph_context_callback(args.model_f_context)
        if context_for is not None and args.graph_relationships:
            raise SystemExit(
                "Use either --graph-relationships or --model-f-context. "
                "The graph-context profiles already include source-verified relationships."
            )
        if context_for is not None and (not args.retrieval or args.retrieval_backend != "lightrag"):
            raise SystemExit(
                "--model-f-context is a Model F option. "
                "Pass --retrieval --retrieval-backend lightrag. Dense Model E is unchanged."
            )

        def _relationship_for(topic_row, retrieval_payload):
            del topic_row
            return generation_relationships(retrieval_payload)

        traces, load_report = generate(
            adapter,
            gold,
            retriever=retriever,
            style_for=style_for,
            seed_base=args.seed_base,
            topic_rows=plan,
            quantization=args.quantization,
            relationship_for=_relationship_for if args.graph_relationships else None,
            graph_context_for=context_for,
        )
        load_reports[name] = load_report
        (out / "config" / "quantization_load.json").write_text(
            json.dumps(load_reports, indent=2) + "\n",
            encoding="utf-8",
        )
        _stamp_traces(traces, row_provenance(args, adapter, run_stamp))
        traces_by_name[name] = traces
        (out / "raw" / f"{name}.jsonl").write_text(
            "\n".join(json.dumps(row, ensure_ascii=False) for row in traces) + "\n",
            encoding="utf-8",
        )
        if args.skip_score:
            print("scoring skipped", flush=True)
            continue
        result = score(traces, specs, pack=pack, copy_index=copy_index)
        (out / "metrics" / f"{name}.json").write_text(
            json.dumps(result["summary"], indent=2) + "\n",
            encoding="utf-8",
        )
        combined[name] = result["summary"]
        print(json.dumps(result["summary"], indent=2), flush=True)
    for model_name in [item.strip() for item in args.ollama.split(",") if item.strip()]:
        label = model_name.replace(":", "_")
        print(f"=== {label} ollama:{model_name}", flush=True)
        traces = generate_ollama(
            model_name, gold, args.ollama_base, retriever=retriever, style_for=style_for,
            seed_base=args.seed_base,
        )
        _stamp_traces(
            traces,
            {
                "experiment_id": args.out_dir.name,
                "adapter_path": None,
                "base_model": model_name,
                "collection": args.collection if args.retrieval else None,
                "embedding_model": args.embed_model if args.retrieval else None,
                "top_k": args.k if args.retrieval else None,
                "git_sha": run_stamp["commit"],
                "timestamp": run_stamp["timestamp"],
            },
        )
        traces_by_name[label] = traces
        (out / "raw" / f"{label}.jsonl").write_text(
            "\n".join(json.dumps(row, ensure_ascii=False) for row in traces) + "\n",
            encoding="utf-8",
        )
        if args.skip_score:
            print("scoring skipped", flush=True)
            continue
        result = score(traces, specs, pack=pack, copy_index=copy_index)
        (out / "metrics" / f"{label}.json").write_text(
            json.dumps(result["summary"], indent=2) + "\n",
            encoding="utf-8",
        )
        combined[label] = result["summary"]
        print(json.dumps(result["summary"], indent=2), flush=True)
    if args.retrieval:
        review = []
        for name, traces in traces_by_name.items():
            review.extend(evidence_review_rows(traces, name))
        _write_jsonl(out / "raw" / "evidence_review.jsonl", review)
    rejected = any(
        row.get("evidence_rejected")
        for traces in traces_by_name.values()
        for row in traces
    )
    if args.skip_score:
        (out / "metrics" / "scoring_skipped.json").write_text(
            json.dumps({"scoring": "skipped", "reason": "--skip-score"}, indent=2) + "\n",
            encoding="utf-8",
        )
        if rejected:
            raise SystemExit(2)
        return
    if rejected:
        raise SystemExit(2)
    (out / "metrics" / "summary.json").write_text(
        json.dumps(combined, indent=2) + "\n",
        encoding="utf-8",
    )
    _write_unified_ranking(out, combined)


def _evaluation_resources():
    """Held-out style pack and training-text copy index. Generation prompts are unchanged."""
    from host_finetune.train_only_index import (
        DEFAULT_ASSIGNMENTS,
        DEFAULT_DATASET,
        DEFAULT_GOLD_DISPOSITIONS,
        load_jsonl,
    )
    from host_finetune.unified_eval import CopyIndex, build_reference_pack
    from host_finetune.voice_distance import iter_train_texts

    dataset = load_jsonl(DEFAULT_DATASET)
    assignments = load_jsonl(DEFAULT_ASSIGNMENTS)
    dispositions = load_jsonl(DEFAULT_GOLD_DISPOSITIONS)
    pack = build_reference_pack(dataset, assignments, dispositions)
    copy_index = CopyIndex(iter_train_texts(dataset, assignments))
    return pack, copy_index


def _write_unified_ranking(out: Path, combined: dict) -> None:
    from host_finetune.unified_eval import rank_models

    models = []
    for name, summary in combined.items():
        unified = summary.get("unified")
        if not unified:
            continue
        unified = dict(unified)
        unified["model_id"] = name
        models.append(unified)
    if not models:
        return
    ranking = rank_models(models)
    (out / "metrics" / "selection.json").write_text(
        json.dumps(ranking, indent=2) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
