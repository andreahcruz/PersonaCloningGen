"""Read-only deterministic audit of F-REL panel generations.

Writes only under audits/AUDIT-20261102-001-model-f-frel-prejudge/.
Does not call a judge, load a GPU model, or rewrite experiment files.
"""

from __future__ import annotations

import json
import math
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from host_finetune.eval_bench.metrics import score_generation
from host_finetune.evidence_capsule import repeats_topic
from host_finetune.factual_reliability import embeds_topic, instruction_leak
from host_finetune.model_f_context.prompt import marker_leak
from host_finetune.rag_source_copying import rag_copy_tokens
from host_finetune.unified_eval import completion_gate, format_gate, grounding_gate

EXP = ROOT / "experiments" / "EXP-20261102-001-model-f-unified-panel"
OUT = Path(__file__).resolve().parent
MEDIA = ["blog", "linkedin", "x", "talk"]
EXPECTED_IDS = [f"gold_{index:03d}" for index in range(1, 31)]
PROMPT_MIN_WORDS = {"linkedin": 40, "x": 20}
REQUIRED = [
    "question_id",
    "arm",
    "medium",
    "query",
    "answer",
    "prompt",
    "status",
    "evidence",
    "claims",
    "retrieved_source_ids",
    "tokens",
    "latency",
    "generation_config",
    "grounding_texts",
    "window_omitted",
    "extraction_input_truncated",
]
ARCH = {
    "Base": ROOT / "docs/evaluation/examples/data/architecture/kevin_base_llama31_rows.jsonl",
    "PersonaRAG stand-in": ROOT / "docs/evaluation/examples/data/architecture/colab_personarag_seed42_rows.jsonl",
    "Relabel, RAG off": ROOT / "docs/evaluation/examples/data/relabel_rag_off_rows.jsonl",
    "Relabel + RAG (Model E)": ROOT / "docs/evaluation/examples/data/relabel_rag_on_rows.jsonl",
}


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, math.ceil(fraction * len(ordered)) - 1))
    return ordered[index]


def dist(values: list[float]) -> dict:
    if not values:
        return {"n": 0, "mean": None, "median": None, "p95": None, "min": None, "max": None}
    return {
        "n": len(values),
        "mean": round(statistics.fmean(values), 4),
        "median": round(statistics.median(values), 4),
        "p95": round(percentile(values, 0.95), 4),
        "min": round(min(values), 4),
        "max": round(max(values), 4),
    }


def rate(count: int, denominator: int) -> dict:
    return {
        "count": count,
        "denominator": denominator,
        "rate": None if denominator == 0 else round(count / denominator, 4),
    }


def longest_span(left: list[str], right: list[str]) -> int:
    best = 0
    previous = [0] * (len(right) + 1)
    for word in left:
        current = [0]
        for index, other in enumerate(right):
            span = previous[index] + 1 if word == other else 0
            current.append(span)
            if span > best:
                best = span
        previous = current
    return best


def claim_present(answer: str, claim: str) -> dict:
    """Audit diagnostic. A claim is present when 4+ contiguous claim tokens occur in the answer.

    Claims shorter than 4 tokens are present only when the whole token sequence occurs.
    This is not an existing scorer and is not entailment.
    """
    answer_tokens = rag_copy_tokens(answer)
    claim_tokens = rag_copy_tokens(claim)
    if not claim_tokens:
        return {"status": "not_measurable", "present": None, "span": 0, "claim_tokens": 0}
    span = longest_span(claim_tokens, answer_tokens)
    need = len(claim_tokens) if len(claim_tokens) < 4 else 4
    return {
        "status": "measured",
        "present": span >= need,
        "span": span,
        "claim_tokens": len(claim_tokens),
    }


def repeated_sentence_count(answer: str) -> int:
    seen = Counter()
    for line in (answer or "").splitlines():
        folded = " ".join(line.split()).casefold()
        if len(folded.split()) >= 6:
            seen[folded] += 1
    return sum(count - 1 for count in seen.values() if count > 1)


def fourgram_repeat_rate(answer: str) -> float | None:
    tokens = (answer or "").split()
    if len(tokens) < 8:
        return None
    grams = [" ".join(tokens[index:index + 4]).casefold() for index in range(len(tokens) - 3)]
    if not grams:
        return None
    return round(1 - (len(set(grams)) / len(grams)), 4)


def relationship_lines(prompt: str) -> list[str]:
    if "VERIFIED RELATIONSHIPS" not in prompt:
        return []
    body = prompt.split("VERIFIED RELATIONSHIPS", 1)[1]
    body = body.split("FINAL WRITING INSTRUCTION", 1)[0]
    lines = []
    for line in body.splitlines():
        stripped = line.strip()
        if stripped.startswith("[R"):
            lines.append(stripped)
    return lines


def endpoints_mentioned(answer: str, line: str) -> bool | None:
    text = line.split("]", 1)[-1]
    parts = [part.strip() for part in text.split("—")]
    if len(parts) < 2:
        return None
    folded = " ".join((answer or "").split()).casefold()
    ends = []
    for part in (parts[0], parts[-1]):
        name = part.split("relationship_keywords", 1)[0].strip(" :")
        name = " ".join(name.split())
        if len(name) < 3:
            return None
        ends.append(name.casefold())
    return all(name in folded for name in ends)


def hist_summary(path: Path) -> dict:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    by_medium = {}
    for medium in MEDIA:
        subset = [row for row in rows if row.get("medium") == medium]
        words = [((row.get("gates") or {}).get("format") or {}).get("word_count") for row in subset]
        words = [value for value in words if isinstance(value, (int, float))]
        by_medium[medium] = {
            "n": len(subset),
            "completion_pass": rate(
                sum(1 for row in subset if ((row.get("gates") or {}).get("completion") or {}).get("pass") is True),
                len(subset),
            ),
            "format_pass": rate(
                sum(1 for row in subset if ((row.get("gates") or {}).get("format") or {}).get("pass") is True),
                len(subset),
            ),
            "copy_fail": rate(
                sum(1 for row in subset if ((row.get("gates") or {}).get("copying_memorization") or {}).get("pass") is False),
                len(subset),
            ),
            "mid_sentence": rate(
                sum(1 for row in subset if ((row.get("gates") or {}).get("completion") or {}).get("mid_sentence_stop") is True),
                len(subset),
            ),
            "words": dist(words),
        }
    words = [((row.get("gates") or {}).get("format") or {}).get("word_count") for row in rows]
    words = [value for value in words if isinstance(value, (int, float))]
    early_known = [
        row for row in rows if ((row.get("gates") or {}).get("completion") or {}).get("early_eot_available") is True
    ]
    return {
        "n": len(rows),
        "source": str(path.relative_to(ROOT)),
        "definition": "Saved unified_eval.v1 gates.completion, gates.format, gates.copying_memorization. Seed-42 row files named in UNIFIED_120_COMPARISON.md.",
        "completion_pass": rate(
            sum(1 for row in rows if ((row.get("gates") or {}).get("completion") or {}).get("pass") is True),
            len(rows),
        ),
        "format_pass": rate(
            sum(1 for row in rows if ((row.get("gates") or {}).get("format") or {}).get("pass") is True),
            len(rows),
        ),
        "copy_fail": rate(
            sum(1 for row in rows if ((row.get("gates") or {}).get("copying_memorization") or {}).get("pass") is False),
            len(rows),
        ),
        "mid_sentence": rate(
            sum(1 for row in rows if ((row.get("gates") or {}).get("completion") or {}).get("mid_sentence_stop") is True),
            len(rows),
        ),
        "early_eot_available_rows": len(early_known),
        "early_eot": rate(
            sum(1 for row in early_known if ((row.get("gates") or {}).get("completion") or {}).get("early_eot") is True),
            len(early_known),
        ),
        "words": dist(words),
        "by_medium": by_medium,
        "not_compared": [
            "writing_quality",
            "claim_support_rate",
            "persona_style",
            "unsupported_numeric_count",
        ],
    }


def main() -> None:
    generations = sorted((EXP / "raw" / "generations").glob("*.json"))
    evidence = {path.stem: load_json(path) for path in (EXP / "raw" / "evidence").glob("*.json")}
    retrieval = {path.stem: load_json(path) for path in (EXP / "raw" / "retrieval").glob("*.json")}
    omissions = load_json(EXP / "diagnostics" / "descriptive_omissions.json")
    manifest = load_json(EXP / "generation_manifest.json")
    progress = load_json(EXP / "progress.json")
    rows_path = EXP / "unified_eval" / "rows.jsonl"
    saved_rows = []
    if rows_path.is_file():
        saved_rows = [json.loads(line) for line in rows_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    saved_copy = {
        (row.get("id"), row.get("medium")): (row.get("gates") or {}).get("copying_memorization")
        for row in saved_rows
    }

    cells = []
    seen = Counter()
    for path in generations:
        record = load_json(path)
        question_id = record.get("question_id")
        medium = record.get("medium")
        seen[(question_id, medium)] += 1
        answer = record.get("answer") or ""
        topic = record.get("query") or ""
        claims = list(record.get("claims") or [])
        ev = evidence.get(question_id) or {}
        ret = retrieval.get(question_id) or {}
        summary = ret.get("graph_summary") or {}
        tokens = record.get("tokens") or {}
        latency = record.get("latency") or {}
        config = record.get("generation_config") or {}
        missing = [key for key in REQUIRED if key not in record or record.get(key) is None]
        if record.get("answer") is None:
            missing.append("answer")
        scored = score_generation(
            {
                "answer": answer,
                "question": topic,
                "claims": claims,
                "evidence": record.get("evidence") or [],
                "grounding_texts": record.get("grounding_texts") or [],
                "retrieved_source_ids": record.get("retrieved_source_ids") or [],
                "abstention_expected": record.get("abstention_expected"),
                "question_id": question_id,
            }
        )
        completion = completion_gate(record)
        formatted = format_gate(record)
        lexical = grounding_gate(
            {
                "answer": answer,
                "query": topic,
                "retrieval": True,
                "contexts": [
                    item.get("text")
                    for item in (record.get("evidence") or [])
                    if isinstance(item, dict) and item.get("text")
                ],
                "prompt": record.get("prompt") or "",
                "medium": medium,
            }
        )
        echo_repeat = repeats_topic(answer, topic)
        echo_embed = embeds_topic(answer, topic)
        leak = bool(instruction_leak(answer) or marker_leak(answer))
        copy = scored.get("copying") or {}
        claim_rows = [claim_present(answer, item.get("claim") or "") for item in claims]
        measured_claims = [item for item in claim_rows if item["status"] == "measured"]
        present_claims = [item for item in measured_claims if item["present"]]
        relations = relationship_lines(record.get("prompt") or "")
        relation_hits = [endpoints_mentioned(answer, line) for line in relations]
        relation_known = [item for item in relation_hits if item is not None]
        output_tokens = tokens.get("output")
        max_new = config.get("max_new_tokens")
        at_max = (
            isinstance(output_tokens, int)
            and isinstance(max_new, int)
            and output_tokens >= max_new
        )
        prompt_floor = PROMPT_MIN_WORDS.get(medium)
        words = len(answer.split())
        prompt_short = prompt_floor is not None and words < prompt_floor
        gen_seconds = latency.get("generation_seconds")
        tokens_per_second = None
        if isinstance(output_tokens, (int, float)) and isinstance(gen_seconds, (int, float)) and gen_seconds > 0:
            tokens_per_second = output_tokens / gen_seconds
        memorization = saved_copy.get((question_id, medium))
        window_omitted = record.get("window_omitted") or []
        cell = {
            "path": str(path.relative_to(ROOT)).replace("\\", "/"),
            "question_id": question_id,
            "medium": medium,
            "topic": topic,
            "status": record.get("status"),
            "arm": record.get("arm"),
            "empty": not answer.strip(),
            "missing_fields": missing,
            "words": words,
            "chars": len(answer),
            "output_tokens": output_tokens,
            "prompt_tokens": tokens.get("prompt"),
            "context_tokens": tokens.get("context_used"),
            "context_by_channel": tokens.get("context_by_channel"),
            "extraction_prompt_tokens": tokens.get("extraction_prompt"),
            "extraction_input_truncated": record.get("extraction_input_truncated"),
            "max_new_tokens": max_new,
            "at_max_tokens": at_max,
            "accepted_attempt": config.get("accepted_attempt"),
            "retry_seed": config.get("retry_seed"),
            "seed": config.get("seed"),
            "stop_reason_recorded": "stop_reason" in record or "early_eos" in record,
            "completion": completion,
            "format": formatted,
            "prompt_contract_short": prompt_short,
            "lexical_grounding": {
                "pass": lexical.get("pass"),
                "applicable": lexical.get("applicable"),
                "lexical_overlap": lexical.get("lexical_overlap"),
                "context_recall": lexical.get("context_recall"),
                "method": lexical.get("method"),
            },
            "repeats_topic": echo_repeat,
            "embeds_topic": echo_embed,
            "instruction_or_marker_leak": leak,
            "citation_marker_leak": scored.get("citation_marker_leak"),
            "unsupported_numeric_count": scored.get("unsupported_numeric_count"),
            "numeric_pass": scored.get("numeric_pass"),
            "numeric_applicable": scored.get("numeric_applicable"),
            "numeric_value_count": scored.get("numeric_value_count"),
            "factual_reasons": scored.get("factual_reasons"),
            "faithfulness_pass": (scored.get("faithfulness") or {}).get("pass"),
            "faithfulness_status": (scored.get("faithfulness") or {}).get("status"),
            "role_unsupported_count": scored.get("role_unsupported_count"),
            "speaker": scored.get("speaker"),
            "copy_pass": copy.get("pass"),
            "copy_applicable": copy.get("applicable"),
            "copy_span": copy.get("max_contiguous_words"),
            "copy_threshold": copy.get("threshold"),
            "copy_span_text": copy.get("matched_generated_span"),
            "training_copy_gate": memorization,
            "claim_count": len(claims),
            "claims_present": None if not measured_claims else len(present_claims),
            "claims_measured": len(measured_claims),
            "relationship_lines": len(relations),
            "relationships_both_endpoints_mentioned": None if not relation_known else sum(1 for item in relation_known if item),
            "relationships_measurable": len(relation_known),
            "retrieved_passage_count": len(record.get("evidence") or []),
            "retrieved_source_ids": record.get("retrieved_source_ids") or [],
            "unique_retrieved_sources": len(set(record.get("retrieved_source_ids") or [])),
            "window_omitted_count": len(window_omitted) if isinstance(window_omitted, list) else None,
            "evidence_status": ev.get("evidence_status"),
            "accepted_claims_file": len(ev.get("verified") or []),
            "proposed_claims": len(ev.get("proposed") or []),
            "validation_rejected": len(ev.get("validation_rejected") or []),
            "verifier_rejected": len(ev.get("verifier_rejected") or []),
            "extraction_hits_omitted": ev.get("extraction_hits_omitted") or [],
            "extraction_attempts": ev.get("extraction_attempts"),
            "raw_chunk_count": summary.get("raw_chunk_count"),
            "raw_entity_count": summary.get("raw_entity_count"),
            "raw_relationship_count": summary.get("raw_relationship_count"),
            "accepted_relationships_graph": summary.get("accepted_relationships"),
            "rejected_relationships_graph": summary.get("rejected_relationships"),
            "two_hop_path_count": summary.get("two_hop_path_count"),
            "retrieval_status": ret.get("status"),
            "generation_seconds": gen_seconds,
            "retrieval_seconds": latency.get("retrieval_seconds"),
            "extraction_seconds": latency.get("extraction_seconds"),
            "verification_seconds": latency.get("verification_seconds"),
            "tokens_per_second": None if tokens_per_second is None else round(tokens_per_second, 4),
            "repeated_sentences": repeated_sentence_count(answer),
            "fourgram_repeat_rate": fourgram_repeat_rate(answer),
            "gpu_memory_bytes": record.get("gpu_memory_bytes"),
            "excerpt": answer[:700],
            "ending": answer[-240:],
        }
        reasons = list(scored.get("factual_reasons") or [])
        priority = 0
        priority += 100 * int(scored.get("unsupported_numeric_count") or 0)
        if copy.get("pass") is False:
            priority += 80
        if leak or scored.get("citation_marker_leak"):
            priority += 50
        if not claims:
            priority += 40
        if completion.get("mid_sentence_stop"):
            priority += 30
        if at_max and not completion.get("sentence_final"):
            priority += 25
        elif at_max:
            priority += 8
        if formatted.get("pass") is False:
            priority += 15
        if prompt_short:
            priority += 12
        if ev.get("evidence_status") != "verified":
            priority += 20
        if window_omitted:
            priority += 8
        if ev.get("extraction_hits_omitted"):
            priority += 6
        if echo_repeat or echo_embed:
            priority += 20
        if scored.get("role_unsupported_count"):
            priority += 10 * int(scored.get("role_unsupported_count"))
        if cell["repeated_sentences"]:
            priority += 5 * cell["repeated_sentences"]
        cell["priority"] = priority
        cell["factual_reason_list"] = reasons
        cells.append(cell)

    def aggregate(subset: list[dict], label: str) -> dict:
        n = len(subset)

        def nums(key: str) -> list[float]:
            return [float(item[key]) for item in subset if isinstance(item.get(key), (int, float))]

        def flagged(predicate) -> dict:
            return rate(sum(1 for item in subset if predicate(item)), n)

        claim_cells = [item for item in subset if item["claims_measured"]]
        claim_present_total = sum(item["claims_present"] or 0 for item in claim_cells)
        claim_measured_total = sum(item["claims_measured"] for item in claim_cells)
        relation_cells = [item for item in subset if item["relationships_measurable"]]
        copy_known = [item for item in subset if item["copy_pass"] is not None]
        numeric_known = [item for item in subset if item["numeric_applicable"] is True]
        train_known = [item for item in subset if isinstance(item.get("training_copy_gate"), dict) and item["training_copy_gate"].get("available")]
        lexical_known = [item for item in subset if item["lexical_grounding"]["applicable"]]
        return {
            "label": label,
            "cells": n,
            "empty": flagged(lambda item: item["empty"]),
            "status_not_complete": flagged(lambda item: item["status"] != "complete"),
            "missing_any_required_field": flagged(lambda item: bool(item["missing_fields"])),
            "retry_attempt_gt_1": flagged(lambda item: (item["accepted_attempt"] or 1) > 1),
            "retry_seed_set": flagged(lambda item: item["retry_seed"] is not None),
            "zero_accepted_claims": flagged(lambda item: item["claim_count"] == 0),
            "extraction_truncated": flagged(lambda item: item["extraction_input_truncated"] is True),
            "passage_omitted": flagged(lambda item: bool(item["extraction_hits_omitted"])),
            "graph_items_omitted": flagged(lambda item: (item["window_omitted_count"] or 0) > 0),
            "unsupported_numeric_cells": rate(
                sum(1 for item in numeric_known if item["numeric_pass"] is False),
                len(numeric_known),
            ),
            "unsupported_numeric_items": {
                "count": sum(item["unsupported_numeric_count"] or 0 for item in subset),
                "cells_with_any": sum(1 for item in subset if (item["unsupported_numeric_count"] or 0) > 0),
                "denominator_cells": len(numeric_known),
            },
            "repeats_topic": flagged(lambda item: item["repeats_topic"]),
            "embeds_topic": flagged(lambda item: item["embeds_topic"]),
            "instruction_or_marker_leak": flagged(lambda item: item["instruction_or_marker_leak"]),
            "citation_marker_leak": flagged(lambda item: item["citation_marker_leak"]),
            "rag_copy_fail": rate(
                sum(1 for item in copy_known if item["copy_pass"] is False),
                len(copy_known),
            ),
            "rag_copy_span_ge_8": flagged(lambda item: (item["copy_span"] or 0) >= 8),
            "training_copy_fail": rate(
                sum(1 for item in train_known if item["training_copy_gate"].get("pass") is False),
                len(train_known),
            ),
            "completion_pass": flagged(lambda item: item["completion"]["pass"] is True),
            "mid_sentence": flagged(lambda item: item["completion"]["mid_sentence_stop"] is True),
            "format_pass": flagged(lambda item: item["format"]["pass"] is True),
            "prompt_contract_short": flagged(lambda item: item["prompt_contract_short"]),
            "at_max_tokens": flagged(lambda item: item["at_max_tokens"]),
            "stop_reason_recorded": flagged(lambda item: item["stop_reason_recorded"]),
            "faithfulness_fail": rate(
                sum(1 for item in subset if item["faithfulness_pass"] is False),
                sum(1 for item in subset if item["faithfulness_pass"] is not None),
            ),
            "lexical_grounding_pass": rate(
                sum(1 for item in lexical_known if item["lexical_grounding"]["pass"] is True),
                len(lexical_known),
            ),
            "role_unsupported_cells": flagged(lambda item: (item["role_unsupported_count"] or 0) > 0),
            "repeated_sentence_cells": flagged(lambda item: item["repeated_sentences"] > 0),
            "claim_presence": {
                "present_claims": claim_present_total,
                "measured_claims": claim_measured_total,
                "rate": None if claim_measured_total == 0 else round(claim_present_total / claim_measured_total, 4),
                "cells": len(claim_cells),
                "definition": "Audit diagnostic: 4-token contiguous overlap, or the full sequence when the claim has fewer than 4 tokens. Not an existing scorer.",
            },
            "relationship_endpoint_mention": {
                "mentioned": sum(item["relationships_both_endpoints_mentioned"] or 0 for item in relation_cells),
                "measurable_lines": sum(item["relationships_measurable"] for item in relation_cells),
                "rate": None
                if not relation_cells
                else round(
                    sum(item["relationships_both_endpoints_mentioned"] or 0 for item in relation_cells)
                    / sum(item["relationships_measurable"] for item in relation_cells),
                    4,
                ),
                "definition": "Audit diagnostic: both endpoint names from a [R] line occur in the answer. Not proof the relationship was used correctly.",
            },
            "words": dist(nums("words")),
            "output_tokens": dist(nums("output_tokens")),
            "prompt_tokens": dist(nums("prompt_tokens")),
            "context_tokens": dist(nums("context_tokens")),
            "generation_seconds": dist(nums("generation_seconds")),
            "tokens_per_second": dist(nums("tokens_per_second")),
            "copy_span": dist(nums("copy_span")),
            "lexical_overlap": dist(
                [item["lexical_grounding"]["lexical_overlap"] for item in lexical_known if isinstance(item["lexical_grounding"]["lexical_overlap"], (int, float))]
            ),
            "fourgram_repeat_rate": dist([item["fourgram_repeat_rate"] for item in subset if item["fourgram_repeat_rate"] is not None]),
        }

    question_ids = sorted({item["question_id"] for item in cells})
    expected_pairs = {(qid, medium) for qid in EXPECTED_IDS for medium in MEDIA}
    actual_pairs = {(item["question_id"], item["medium"]) for item in cells}
    duplicate_pairs = [pair for pair, count in seen.items() if count > 1]

    per_question = []
    for qid in EXPECTED_IDS:
        ev = evidence.get(qid) or {}
        ret = retrieval.get(qid) or {}
        summary = ret.get("graph_summary") or {}
        qcells = [item for item in cells if item["question_id"] == qid]
        per_question.append(
            {
                "question_id": qid,
                "topic": qcells[0]["topic"] if qcells else None,
                "evidence_status": ev.get("evidence_status"),
                "accepted_claims": len(ev.get("verified") or []),
                "proposed": len(ev.get("proposed") or []),
                "validation_rejected": len(ev.get("validation_rejected") or []),
                "verifier_rejected": len(ev.get("verifier_rejected") or []),
                "extraction_hits_omitted": ev.get("extraction_hits_omitted") or [],
                "extraction_attempts": ev.get("extraction_attempts"),
                "extraction_input_truncated": ev.get("extraction_input_truncated"),
                "extraction_prompt_tokens": ev.get("extraction_prompt_tokens"),
                "extraction_seconds": ev.get("extraction_seconds"),
                "retrieval_status": ret.get("status"),
                "retrieval_seconds": ret.get("retrieval_seconds"),
                "raw_chunk_count": summary.get("raw_chunk_count"),
                "raw_entity_count": summary.get("raw_entity_count"),
                "raw_relationship_count": summary.get("raw_relationship_count"),
                "accepted_relationships": summary.get("accepted_relationships"),
                "rejected_relationships": summary.get("rejected_relationships"),
                "two_hop_path_count": summary.get("two_hop_path_count"),
                "cells": len(qcells),
                "unsupported_numeric_cells": sum(1 for item in qcells if (item["unsupported_numeric_count"] or 0) > 0),
                "copy_fail_cells": sum(1 for item in qcells if item["copy_pass"] is False),
                "mid_sentence_cells": sum(1 for item in qcells if item["completion"]["mid_sentence_stop"]),
                "max_token_cells": sum(1 for item in qcells if item["at_max_tokens"]),
            }
        )

    retrieval_seconds = [item["retrieval_seconds"] for item in per_question if isinstance(item["retrieval_seconds"], (int, float))]
    extraction_seconds = [item["extraction_seconds"] for item in per_question if isinstance(item["extraction_seconds"], (int, float))]
    chunk_counts = [item["raw_chunk_count"] for item in per_question if isinstance(item["raw_chunk_count"], (int, float))]
    accepted_rel = [item["accepted_relationships"] for item in per_question if isinstance(item["accepted_relationships"], (int, float))]
    rejected_rel = [item["rejected_relationships"] for item in per_question if isinstance(item["rejected_relationships"], (int, float))]
    accepted_claims = [item["accepted_claims"] for item in per_question]
    rejected_claims = [item["validation_rejected"] + item["verifier_rejected"] for item in per_question]

    ranked = sorted(cells, key=lambda item: (-item["priority"], item["question_id"], item["medium"]))
    chosen = []
    chosen_keys = set()

    def take(item: dict, reason: str) -> None:
        key = (item["question_id"], item["medium"])
        if key in chosen_keys or len(chosen) >= 12:
            return
        chosen_keys.add(key)
        chosen.append({**item, "selection_reason": reason})

    clean = [
        item
        for item in cells
        if item["priority"] == 0 and item["faithfulness_pass"] is True and item["format"]["pass"] and item["completion"]["pass"]
    ]
    for medium in MEDIA:
        pool = [item for item in clean if item["medium"] == medium] or [item for item in cells if item["medium"] == medium]
        med_words = statistics.median([item["words"] for item in pool])
        pick = min(pool, key=lambda item: (item["priority"], abs(item["words"] - med_words)))
        take(pick, f"Representative {medium}: closest clean cell to the medium median word count.")

    for item in ranked:
        if item["question_id"] == "gold_030":
            continue
        if (item["question_id"], item["medium"]) in chosen_keys:
            continue
        take(item, "Strongest remaining deterministic warning signal.")
        if sum(1 for sample in chosen if sample["selection_reason"].startswith("Strongest")) >= 4:
            break

    for medium in MEDIA:
        match = next(item for item in cells if item["question_id"] == "gold_030" and item["medium"] == medium)
        take(match, "gold_030 has zero accepted claims and evidence_status extraction_failed.")

    for item in ranked:
        take(item, "Next highest-priority cell after the required sample categories.")

    checks_executed = [
        "host_finetune.eval_bench.metrics.score_generation",
        "host_finetune.factual_reliability.assess_factual via score_generation (numeric, role, speaker, retrieval source copying)",
        "host_finetune.evidence_capsule.repeats_topic",
        "host_finetune.factual_reliability.embeds_topic",
        "host_finetune.factual_reliability.instruction_leak",
        "host_finetune.model_f_context.prompt.marker_leak",
        "host_finetune.unified_eval.completion_gate",
        "host_finetune.unified_eval.format_gate",
        "host_finetune.unified_eval.grounding_gate on saved passage text (lexical overlap, threshold 0.2)",
        "saved unified_eval/rows.jsonl gates.copying_memorization (training-span gate; not recomputed)",
    ]
    not_executed = [
        "LLM judge / writing_quality / claim_vs_saved_retrieval_hits",
        "early EOS stop reason (not stored on generation records)",
        "training CopyIndex rebuild",
        "GPU memory (field is null)",
        "BLEU/ROUGE/BERTScore recomputation",
        "style distance and perspective distance",
    ]

    summary = {
        "experiment_id": manifest.get("experiment_id"),
        "profile": manifest.get("profile"),
        "manifest_status": manifest.get("status"),
        "progress": progress,
        "gold_sha256": manifest.get("gold_sha256"),
        "expected_cells": 120,
        "saved_generation_files": len(generations),
        "saved_evidence_files": len(evidence),
        "saved_retrieval_files": len(retrieval),
        "unique_question_medium_pairs": len(actual_pairs),
        "missing_pairs": sorted(f"{qid}__{medium}" for qid, medium in expected_pairs - actual_pairs),
        "unexpected_pairs": sorted(f"{qid}__{medium}" for qid, medium in actual_pairs - expected_pairs),
        "duplicate_pairs": [f"{qid}__{medium}" for qid, medium in duplicate_pairs],
        "question_ids_found": question_ids,
        "questions_found": len(question_ids),
        "overall": aggregate(cells, "all"),
        "by_medium": {medium: aggregate([item for item in cells if item["medium"] == medium], medium) for medium in MEDIA},
        "per_question_retrieval_seconds": dist(retrieval_seconds),
        "per_question_extraction_seconds": dist(extraction_seconds),
        "per_question_raw_chunks": dist(chunk_counts),
        "per_question_accepted_relationships": dist(accepted_rel),
        "per_question_rejected_relationships": dist(rejected_rel),
        "per_question_accepted_claims": dist(accepted_claims),
        "per_question_rejected_claims": dist(rejected_claims),
        "zero_claim_questions": [item["question_id"] for item in per_question if item["accepted_claims"] == 0],
        "passage_omission_questions": [item["question_id"] for item in per_question if item["extraction_hits_omitted"]],
        "descriptive_omissions": omissions,
        "saved_unified_rows": len(saved_rows),
        "saved_rows_with_writing_quality": sum(
            1 for row in saved_rows if ((row.get("writing_quality") or {}).get("available") is True)
        ),
        "judge_cache_lines": sum(
            1
            for line in (EXP / "unified_eval" / "judge_cache.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
        if (EXP / "unified_eval" / "judge_cache.jsonl").is_file()
        else 0,
        "checks_executed": checks_executed,
        "checks_not_executed": not_executed,
        "comparison": {name: hist_summary(path) for name, path in ARCH.items() if path.is_file()},
        "highest_priority": [
            {
                "question_id": item["question_id"],
                "medium": item["medium"],
                "priority": item["priority"],
                "topic": item["topic"],
                "path": item["path"],
                "unsupported_numeric_count": item["unsupported_numeric_count"],
                "copy_pass": item["copy_pass"],
                "copy_span": item["copy_span"],
                "claim_count": item["claim_count"],
                "mid_sentence": item["completion"]["mid_sentence_stop"],
                "at_max_tokens": item["at_max_tokens"],
                "words": item["words"],
                "format_pass": item["format"]["pass"],
                "factual_reasons": item["factual_reason_list"],
                "evidence_status": item["evidence_status"],
            }
            for item in ranked[:15]
        ],
        "samples": [
            {
                "selection_reason": item["selection_reason"],
                "question_id": item["question_id"],
                "medium": item["medium"],
                "topic": item["topic"],
                "path": item["path"],
                "priority": item["priority"],
                "words": item["words"],
                "output_tokens": item["output_tokens"],
                "claim_count": item["claim_count"],
                "claims_present": item["claims_present"],
                "unsupported_numeric_count": item["unsupported_numeric_count"],
                "factual_reasons": item["factual_reason_list"],
                "copy_pass": item["copy_pass"],
                "copy_span": item["copy_span"],
                "copy_span_text": item["copy_span_text"],
                "lexical_overlap": item["lexical_grounding"]["lexical_overlap"],
                "repeats_topic": item["repeats_topic"],
                "embeds_topic": item["embeds_topic"],
                "mid_sentence": item["completion"]["mid_sentence_stop"],
                "sentence_final": item["completion"]["sentence_final"],
                "at_max_tokens": item["at_max_tokens"],
                "format_pass": item["format"]["pass"],
                "prompt_contract_short": item["prompt_contract_short"],
                "evidence_status": item["evidence_status"],
                "accepted_relationships_graph": item["accepted_relationships_graph"],
                "rejected_relationships_graph": item["rejected_relationships_graph"],
                "raw_chunk_count": item["raw_chunk_count"],
                "relationship_lines": item["relationship_lines"],
                "relationships_both_endpoints_mentioned": item["relationships_both_endpoints_mentioned"],
                "window_omitted_count": item["window_omitted_count"],
                "excerpt": item["excerpt"],
                "ending": item["ending"],
            }
            for item in chosen
        ],
    }

    # Drop bulky fields from the cell export.
    export_cells = []
    for item in cells:
        slim = dict(item)
        slim.pop("excerpt", None)
        slim.pop("ending", None)
        slim.pop("copy_span_text", None)
        export_cells.append(slim)

    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    (OUT / "cells.jsonl").write_text(
        "".join(json.dumps(item, ensure_ascii=False) + "\n" for item in export_cells),
        encoding="utf-8",
    )
    (OUT / "questions.json").write_text(json.dumps(per_question, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({
        "cells": len(cells),
        "missing": summary["missing_pairs"],
        "duplicates": summary["duplicate_pairs"],
        "zero_claim": summary["zero_claim_questions"],
        "overall_unsupported_cells": summary["overall"]["unsupported_numeric_cells"],
        "overall_copy_fail": summary["overall"]["rag_copy_fail"],
        "overall_mid": summary["overall"]["mid_sentence"],
        "overall_format": summary["overall"]["format_pass"],
        "overall_completion": summary["overall"]["completion_pass"],
        "overall_words": summary["overall"]["words"],
        "samples": [(item["question_id"], item["medium"], item["selection_reason"][:40]) for item in chosen],
        "top": [(item["question_id"], item["medium"], item["priority"]) for item in ranked[:10]],
    }, indent=2))


if __name__ == "__main__":
    main()
