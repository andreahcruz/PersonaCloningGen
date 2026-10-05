"""One evaluation harness with separate metric families.

Gates decide whether a draft may be ranked. Persona style, perspective,
writing quality, and task adherence stay visible and are not averaged into the
gold rubric. BLEU, ROUGE, BERTScore, and gold-reference overlap stay diagnostics.
"""
from __future__ import annotations

import hashlib
import statistics
from typing import Iterable

from host_finetune.eval_gold_rag import grounding_scores, requirement_coverage, score_row
from host_finetune.relabel_continuations import sentence_final
from host_finetune.voice_distance import (
    DEGRADED_PROSE,
    GENERIC_PROSE,
    NOT_AUTHORSHIP,
    PERSPECTIVE_FEATURE_NAMES,
    STYLE_FEATURE_NAMES,
    held_out_reference_texts,
    iter_train_texts,
    metric_calibrated,
    partition_references,
    reference_profile,
    style_distance,
)
from host_finetune.writing_quality import writing_quality_unavailable
from host_finetune.train_only_index import blocked_group_ids

SCHEMA = "unified_eval.v1"
COPY_SPAN_FAIL = 12
COPY_INDEX_WIDTH = 8
GROUNDING_OVERLAP_PASS = 0.2
LENGTH_BANDS = {
    "blog": (40, 1200),
    "linkedin": (15, 500),
    "x": (3, 280),
    "talk": (40, 1200),
}
UTILITY_WEIGHTS = {
    "persona_style": 0.45,
    "perspective_fidelity": 0.20,
    "writing_quality": 0.25,
    "task_adherence": 0.10,
}
REFUSED_DIAGNOSTICS = (
    "bleu",
    "rouge_l",
    "bertscore",
    "gold_rubric_overall",
    "early_eot_rate",
    "mid_sentence_stop_rate",
)


def _lcs(a: list[str], b: list[str]) -> int:
    previous = [0] * (len(b) + 1)
    for token in a:
        current = [0]
        for index, other in enumerate(b, 1):
            current.append(previous[index - 1] + 1 if token == other else max(current[-1], previous[index]))
        previous = current
    return previous[-1]


def rouge_l(candidate: str, reference: str) -> float | None:
    cand = (candidate or "").lower().split()
    ref = (reference or "").lower().split()
    if not cand or not ref:
        return None
    match = _lcs(cand, ref)
    precision = match / len(cand)
    recall = match / len(ref)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def _ngram_counts(tokens: list[str], n: int) -> dict[tuple[str, ...], int]:
    counts: dict[tuple[str, ...], int] = {}
    for start in range(0, len(tokens) - n + 1):
        gram = tuple(tokens[start:start + n])
        counts[gram] = counts.get(gram, 0) + 1
    return counts


def bleu(candidate: str, reference: str) -> float | None:
    """Smoothed sentence BLEU. Local implementation so rescoring does not require NLTK.

    Zero-overlap precisions use the add-one form of NLTK smoothing method 1.
    This is reference overlap, not persona fidelity.
    """
    cand = (candidate or "").lower().split()
    ref = (reference or "").lower().split()
    if not cand or not ref:
        return None
    import math
    from collections import Counter

    precisions = []
    for n in range(1, 5):
        cand_counts = _ngram_counts(cand, n)
        if not cand_counts:
            precisions.append(1 / (2 ** n))
            continue
        ref_counts = Counter(_ngram_counts(ref, n))
        overlap = 0
        for gram, count in cand_counts.items():
            overlap += min(count, ref_counts[gram])
        if overlap == 0:
            precisions.append(1 / (sum(cand_counts.values()) + 1))
        else:
            precisions.append(overlap / sum(cand_counts.values()))
    brevity = 1.0 if len(cand) > len(ref) else math.exp(1 - len(ref) / len(cand))
    geometric = math.exp(sum(math.log(item) for item in precisions) / 4)
    return brevity * geometric


def completion_gate(row: dict) -> dict:
    """Sentence-final text passes. Early EOS is recorded and does not fail a finished answer."""
    answer = row.get("answer") or ""
    final = bool(answer.strip()) and sentence_final(answer)
    early = row.get("early_eos") if ("early_eos" in row or "stop_reason" in row) else None
    return {
        "pass": final,
        "early_eot": None if early is None else bool(early),
        "early_eot_available": early is not None,
        "mid_sentence_stop": not final,
        "sentence_final": final,
        "gate_rule": (
            "A non-empty sentence-final answer passes. Mid-sentence stops fail. "
            "Early EOS is recorded and does not fail a sentence-final answer."
        ),
    }


def format_gate(row: dict) -> dict:
    """Length band for the requested medium. The legacy thread rubric is not this gate."""
    medium = row.get("medium") or ""
    answer = (row.get("answer") or "").strip()
    words = len(answer.split())
    lo, hi = LENGTH_BANDS.get(medium, (1, 5000))
    prompt = (row.get("prompt") or "").strip()
    echo = bool(prompt) and answer == prompt
    return {
        "pass": (not echo) and lo <= words <= hi,
        "word_count": words,
        "band": [lo, hi],
        "prompt_echo": echo,
        "rule": "requested-medium length band, not the legacy x_thread heading rubric",
    }


def grounding_gate(row: dict) -> dict:
    scores = grounding_scores(row)
    if not scores["applicable"]:
        return {
            "pass": True,
            "applicable": False,
            "lexical_overlap": None,
            "threshold": GROUNDING_OVERLAP_PASS,
            "passage_source": scores["passage_source"],
            "method": "not_applicable",
            "reason": "retrieval_off_or_no_passages",
        }
    overlap = scores["lexical_overlap"] if scores["lexical_overlap"] is not None else 0.0
    return {
        "pass": overlap >= GROUNDING_OVERLAP_PASS,
        "applicable": True,
        "lexical_overlap": overlap,
        "context_recall": scores.get("context_recall"),
        "threshold": GROUNDING_OVERLAP_PASS,
        "passage_source": scores["passage_source"],
        "method": scores["method"],
        "n_passages": scores["n_passages"],
        "reason": "lexical_overlap_with_retrieved_passages",
    }


def _gram_key(words: tuple[str, ...]) -> bytes:
    return hashlib.sha1("\0".join(words).encode()).digest()[:8]


class CopyIndex:
    """8-word index over training targets. Spans shorter than 8 are below the gate."""

    def __init__(self, texts: Iterable[str]):
        self.texts: list[str] = []
        self._docs: dict[bytes, list[int]] = {}
        for text in texts:
            cleaned = (text or "").strip()
            if not cleaned:
                continue
            doc_id = len(self.texts)
            self.texts.append(cleaned)
            words = tuple(cleaned.lower().split())
            seen: set[bytes] = set()
            width = COPY_INDEX_WIDTH
            for start in range(0, max(len(words) - width + 1, 0)):
                key = _gram_key(words[start:start + width])
                if key in seen:
                    continue
                seen.add(key)
                bucket = self._docs.setdefault(key, [])
                if len(bucket) < 12:
                    bucket.append(doc_id)

    def longest_span(self, answer: str, prompt: str) -> int | None:
        words = (answer or "").lower().split()
        prompt_l = (prompt or "").lower()
        if len(words) < COPY_INDEX_WIDTH:
            return 0
        prompt_keys = set()
        prompt_words = (prompt or "").lower().split()
        for start in range(0, max(len(prompt_words) - COPY_INDEX_WIDTH + 1, 0)):
            prompt_keys.add(_gram_key(tuple(prompt_words[start:start + COPY_INDEX_WIDTH])))
        candidates: set[int] = set()
        for start in range(0, len(words) - COPY_INDEX_WIDTH + 1):
            gram = tuple(words[start:start + COPY_INDEX_WIDTH])
            key = _gram_key(gram)
            if key in prompt_keys:
                continue
            candidates.update(self._docs.get(key, ()))
            if len(candidates) >= 40:
                break
        if not candidates:
            return 0
        best = 0
        for doc_id in list(candidates)[:40]:
            best = max(best, _span_outside_prompt(words, self.texts[doc_id].lower().split(), prompt_l))
        return best


def _span_outside_prompt(left: list[str], right: list[str], prompt_l: str) -> int:
    if not left or not right:
        return 0
    best = 0
    previous = [0] * (len(right) + 1)
    for i, word in enumerate(left):
        current = [0]
        for index, other in enumerate(right):
            if word == other:
                span = previous[index] + 1
                current.append(span)
                if span > best and span >= COPY_INDEX_WIDTH:
                    phrase = " ".join(left[i - span + 1:i + 1])
                    if phrase not in prompt_l:
                        best = span
            else:
                current.append(0)
        previous = current
    return best


def copying_gate(row: dict, copy_index: CopyIndex | None) -> dict:
    if copy_index is None:
        return {
            "pass": False,
            "available": False,
            "longest_shared_word_span": None,
            "threshold": COPY_SPAN_FAIL,
            "reason": "copy_index_missing",
        }
    span = copy_index.longest_span(row.get("answer") or "", row.get("prompt") or "")
    return {
        "pass": span is not None and span < COPY_SPAN_FAIL,
        "available": True,
        "longest_shared_word_span": span,
        "threshold": COPY_SPAN_FAIL,
        "excluded": "word spans that also occur in the prompt",
        "reason": "longest_shared_training_span",
    }


def persona_utility_score(
    style_distance: float | None,
    perspective_distance: float | None,
    writing_mean: float | None,
    task_score: float | None,
    *,
    eligible: bool,
    style_role: str,
    perspective_role: str,
    writing_role: str,
    perspective_score: float | None = None,
) -> float | None:
    """Project weights. None when a gate failed or a component is not selection-valid.

    ``perspective_score`` is a 0–1 stance-consistency rate. When it is absent,
    the framing distance is converted with 1 / (1 + distance).
    """
    if not eligible:
        return None
    if style_role != "selector" or perspective_role != "selector" or writing_role != "selector":
        return None
    if perspective_score is None:
        if perspective_distance is None:
            return None
        perspective_score = 1 / (1 + perspective_distance)
    if None in (style_distance, writing_mean, task_score):
        return None
    style_score = 1 / (1 + style_distance)
    writing_score = writing_mean / 5
    value = (
        UTILITY_WEIGHTS["persona_style"] * style_score
        + UTILITY_WEIGHTS["perspective_fidelity"] * perspective_score
        + UTILITY_WEIGHTS["writing_quality"] * writing_score
        + UTILITY_WEIGHTS["task_adherence"] * task_score
    )
    return round(value, 4)


def _distances(texts: list[str], profile: dict) -> list[float]:
    found = []
    for text in texts:
        distance = style_distance(text, profile)
        if distance is not None:
            found.append(distance)
    return found


def build_reference_pack(
    dataset_rows: list[dict],
    assignments: list[dict],
    dispositions: list[dict],
    base_by_medium: dict[str, list[str]] | None = None,
) -> dict:
    """Held-out profiles plus calibration. Train text is not a style reference."""
    blocked = blocked_group_ids(assignments, dispositions)
    held = held_out_reference_texts(dataset_rows, assignments, blocked)
    media = {}
    for medium, texts in held.items():
        profile_texts, real_texts = partition_references(texts)
        style_profile = reference_profile(profile_texts, STYLE_FEATURE_NAMES)
        perspective_profile = reference_profile(profile_texts, PERSPECTIVE_FEATURE_NAMES)
        base_texts = (base_by_medium or {}).get(medium) or []
        style_calibration = metric_calibrated(
            _distances(real_texts, style_profile),
            _distances(list(GENERIC_PROSE), style_profile),
            base_distances=_distances(base_texts, style_profile) if base_texts else None,
            degraded_distances=_distances(list(DEGRADED_PROSE), style_profile),
        )
        perspective_calibration = metric_calibrated(
            _distances(real_texts, perspective_profile),
            _distances(list(GENERIC_PROSE), perspective_profile),
            base_distances=_distances(base_texts, perspective_profile) if base_texts else None,
            degraded_distances=_distances(list(DEGRADED_PROSE), perspective_profile),
        )
        media[medium] = {
            "n_profile": len(profile_texts),
            "n_real_controls": len(real_texts),
            "n_base_controls": len(base_texts),
            "style_profile": style_profile,
            "perspective_profile": perspective_profile,
            "style_calibration": style_calibration,
            "perspective_calibration": perspective_calibration,
            "style_role": "selector" if style_calibration["calibrated"] else "diagnostic_only",
            "perspective_role": "selector" if perspective_calibration["calibrated"] else "diagnostic_only",
        }
    return {
        "limits": NOT_AUTHORSHIP,
        "blocked_group_count": len(blocked),
        "reference_counts": {medium: len(texts) for medium, texts in held.items()},
        "media": media,
        "writing_quality_calibration": {
            "calibrated": False,
            "role": "diagnostic_only",
            "reason": "judge_not_run",
        },
    }


def _axis(distance: float | None, role: str, medium_matched: bool) -> dict:
    return {
        "distance": None if distance is None else round(distance, 4),
        "role": role,
        "medium_matched": medium_matched,
        "limits": NOT_AUTHORSHIP,
    }


def evaluate_row(row: dict, pack: dict, copy_index: CopyIndex | None, specs: dict) -> dict:
    completion = completion_gate(row)
    copying = copying_gate(row, copy_index)
    formatting = format_gate(row)
    grounding = grounding_gate(row)
    eligible = bool(completion["pass"] and copying["pass"] and formatting["pass"] and grounding["pass"])
    medium = row.get("medium") or ""
    medium_pack = (pack.get("media") or {}).get(medium) or {}
    style_profile = medium_pack.get("style_profile") or {}
    perspective_profile = medium_pack.get("perspective_profile") or {}
    answer = row.get("answer") or ""
    style = _axis(
        style_distance(answer, style_profile),
        medium_pack.get("style_role", "diagnostic_only"),
        bool(style_profile),
    )
    perspective = _axis(
        style_distance(answer, perspective_profile),
        medium_pack.get("perspective_role", "diagnostic_only"),
        bool(perspective_profile),
    )
    writing = writing_quality_unavailable()
    writing_calibration = pack.get("writing_quality_calibration") or {}
    writing["calibrated"] = bool(writing_calibration.get("calibrated"))
    writing["role"] = writing_calibration.get("role", "diagnostic_only")
    requirements = requirement_coverage(row)
    task_score = round((requirements["brief_alignment"] + (1.0 if formatting["pass"] else 0.0)) / 2, 4)
    gold = score_row(row, specs)
    reference = row.get("gold_reference") or ""
    bleu_value = bleu(answer, reference)
    rouge_value = rouge_l(answer, reference)
    utility = persona_utility_score(
        style["distance"],
        perspective["distance"],
        writing["mean"],
        task_score,
        eligible=eligible,
        style_role=style["role"],
        perspective_role=perspective["role"],
        writing_role=writing["role"],
    )
    return {
        "schema": SCHEMA,
        "id": row.get("id"),
        "medium": medium,
        "gates": {
            "completion": completion,
            "copying_memorization": copying,
            "format": formatting,
            "grounding": grounding,
        },
        "eligible_for_ranking": eligible,
        "persona_style": style,
        "perspective_fidelity": perspective,
        "writing_quality": writing,
        "task_adherence": {
            "score": task_score,
            "brief_alignment": requirements["brief_alignment"],
            "format_pass": formatting["pass"],
            "role": "quality_dimension",
        },
        "grounding": grounding,
        "diagnostics": {
            "reference_similarity": {
                "bertscore": None,
                "bertscore_reason": "not_computed_reference_similarity_only",
                "bleu": None if bleu_value is None else round(bleu_value, 4),
                "rouge_l": None if rouge_value is None else round(rouge_value, 4),
                "role": "diagnostic_not_persona_fidelity",
            },
            "brief_overlap": {
                "gold_rubric_overall": gold["gold_rubric"]["overall"],
                "role": "brief_task_agreement_not_persona_quality",
                "components": {
                    "format_score": gold["gold_rubric"]["score"],
                    "brief_alignment": gold["gold_rubric"]["brief_alignment"],
                    "expected_fact_coverage": gold["gold_rubric"]["expected_fact_coverage"],
                    "gold_reference_coverage": gold["gold_rubric"]["gold_reference_coverage"],
                    "rubric_must_have_coverage": gold["gold_rubric"]["rubric_must_have_coverage"],
                },
            },
            "latency_seconds": row.get("latency_seconds"),
        },
        "persona_utility": utility,
        "utility_weights": None if utility is None else dict(UTILITY_WEIGHTS),
    }


def apply_judge_overlay(
    scored: dict,
    writing: dict,
    rag: dict,
    evidence: dict,
    stance: dict,
    *,
    writing_role: str,
    writing_calibrated: bool,
    stance_role: str,
) -> dict:
    """Attach judge results and recompute eligibility and persona utility."""
    lexical = (scored.get("grounding") or {}).get("lexical_overlap")
    row = dict(scored)
    diagnostics = dict(row.get("diagnostics") or {})
    diagnostics["retrieved_lexical_overlap"] = lexical
    row["diagnostics"] = diagnostics
    grounding = {
        "applicable": rag.get("applicable", False),
        "method": rag.get("method", "not_applicable"),
        "rag_grounding": rag.get("rag_grounding", rag.get("claim_support_rate") if rag.get("applicable") else None),
        "claim_support_rate": rag.get("claim_support_rate"),
        "contradiction_rate": rag.get("contradiction_rate"),
        "counts": rag.get("counts"),
        "confidence": rag.get("confidence"),
        "n_claims": rag.get("n_claims"),
        "pass": bool(rag.get("pass", not rag.get("applicable"))),
        "reason": rag.get("reason"),
        "lexical_overlap": None,
    }
    if not grounding["applicable"]:
        grounding["rag_grounding"] = None
        grounding["reason"] = "retrieval_off"
        grounding["pass"] = True
    row["grounding"] = grounding
    gates = dict(row.get("gates") or {})
    gates["grounding"] = {
        "pass": grounding["pass"],
        "applicable": grounding["applicable"],
        "claim_support_rate": grounding["claim_support_rate"],
        "contradiction_rate": grounding["contradiction_rate"],
        "reason": grounding.get("reason") or grounding.get("method"),
    }
    row["gates"] = gates
    completion = gates.get("completion") or {}
    copying = gates.get("copying_memorization") or {}
    formatting = gates.get("format") or {}
    eligible = bool(completion.get("pass") and copying.get("pass") and formatting.get("pass") and grounding["pass"])
    row["eligible_for_ranking"] = eligible
    writing_out = dict(writing)
    writing_out["role"] = writing_role
    writing_out["calibrated"] = writing_calibrated
    row["writing_quality"] = writing_out
    perspective = dict(row.get("perspective_fidelity") or {})
    perspective["framing_distance"] = perspective.get("distance")
    perspective["framing_role"] = "diagnostic_only"
    perspective["stance_consistency_rate"] = stance.get("stance_consistency_rate")
    perspective["stance_counts"] = stance.get("counts")
    perspective["method"] = stance.get("method")
    perspective["role"] = stance_role
    perspective["limits"] = stance.get("limits")
    row["perspective_fidelity"] = perspective
    row["lemkin_evidence_support"] = evidence
    row["persona_utility"] = persona_utility_score(
        (row.get("persona_style") or {}).get("distance"),
        None,
        writing_out.get("mean"),
        (row.get("task_adherence") or {}).get("score"),
        eligible=eligible,
        style_role=(row.get("persona_style") or {}).get("role", "diagnostic_only"),
        perspective_role=stance_role,
        writing_role=writing_role,
        perspective_score=stance.get("stance_consistency_rate"),
    )
    row["utility_weights"] = None if row["persona_utility"] is None else dict(UTILITY_WEIGHTS)
    return row


def _mean(values: list[float]) -> float | None:
    if not values:
        return None
    return round(statistics.fmean(values), 4)


def summarize_model(rows: list[dict], model_id: str, source_experiment: str) -> dict:
    eligible = [row for row in rows if row["eligible_for_ranking"]]
    def collect(path_row, only_eligible: bool = False) -> list[float]:
        chosen = eligible if only_eligible else rows
        values = []
        for row in chosen:
            if path_row(row) is not None:
                values.append(path_row(row))
        return values

    style_rows = [row for row in eligible if row["persona_style"]["role"] == "selector"]
    perspective_rows = [row for row in eligible if row["perspective_fidelity"]["role"] == "selector"]
    style_role = "selector" if style_rows else "diagnostic_only"
    perspective_role = "selector" if perspective_rows else "diagnostic_only"
    writing = rows[0]["writing_quality"] if rows else writing_quality_unavailable()
    utilities = [row["persona_utility"] for row in rows if row["persona_utility"] is not None]
    grounding_applicable = [row for row in rows if row["grounding"]["applicable"]]
    return {
        "schema": SCHEMA,
        "model_id": model_id,
        "source_experiment": source_experiment,
        "n": len(rows),
        "n_eligible": len(eligible),
        "eligible_rate": round(len(eligible) / len(rows), 4) if rows else 0.0,
        "gates": {
            "completion_pass_rate": _mean([1.0 if row["gates"]["completion"]["pass"] else 0.0 for row in rows]),
            "copying_pass_rate": _mean([1.0 if row["gates"]["copying_memorization"]["pass"] else 0.0 for row in rows]),
            "format_pass_rate": _mean([1.0 if row["gates"]["format"]["pass"] else 0.0 for row in rows]),
            "grounding_applicable_rows": len(grounding_applicable),
            "grounding_pass_rate": _mean(
                [1.0 if row["gates"]["grounding"]["pass"] else 0.0 for row in grounding_applicable]
            ),
        },
        "persona_style": {
            "role": style_role,
            "mediums_in_mean": sorted({row["medium"] for row in style_rows}),
            "mean_distance_eligible": _mean(
                [row["persona_style"]["distance"] for row in style_rows if row["persona_style"]["distance"] is not None]
            ),
            "limits": NOT_AUTHORSHIP,
        },
        "perspective_fidelity": {
            "role": perspective_role,
            "mediums_in_mean": sorted({row["medium"] for row in perspective_rows}),
            "mean_distance_eligible": _mean(
                [row["perspective_fidelity"].get("framing_distance", row["perspective_fidelity"].get("distance"))
                 for row in rows if row["perspective_fidelity"].get("framing_distance", row["perspective_fidelity"].get("distance")) is not None]
            ),
            "mean_stance_consistency": _mean(
                [row["perspective_fidelity"].get("stance_consistency_rate") for row in eligible
                 if row["perspective_fidelity"].get("stance_consistency_rate") is not None]
            ),
            "method": perspective_rows[0]["perspective_fidelity"].get("method", "framing_rate_distance_vs_held_out") if perspective_rows else "framing_rate_distance_vs_held_out",
            "framing_role": "diagnostic_only",
            "limits": "Stance consistency is the perspective score. Framing distance is diagnostic.",
        },
        "writing_quality": {
            "available": writing.get("available", False),
            "role": writing.get("role", "diagnostic_only"),
            "calibrated": writing.get("calibrated", False),
            "reason": writing.get("reason"),
            "mean_eligible": _mean(collect(lambda row: row["writing_quality"]["mean"], True)),
        },
        "task_adherence": {
            "mean_eligible": _mean(collect(lambda row: row["task_adherence"]["score"], True)),
            "role": "quality_dimension",
        },
        "grounding": {
            "applicable": bool(grounding_applicable),
            "mean_claim_support_rate": _mean(
                [row["grounding"].get("claim_support_rate") for row in grounding_applicable
                 if row["grounding"].get("claim_support_rate") is not None]
            ),
            "mean_contradiction_rate": _mean(
                [row["grounding"].get("contradiction_rate") for row in grounding_applicable
                 if row["grounding"].get("contradiction_rate") is not None]
            ),
            "method": "claim_vs_saved_retrieval_hits" if grounding_applicable else "not_applicable",
            "rag_grounding": "scored" if grounding_applicable else None,
        },
        "lemkin_evidence_support": {
            "mean_claim_support_rate": _mean(
                [(row.get("lemkin_evidence_support") or {}).get("claim_support_rate") for row in rows
                 if (row.get("lemkin_evidence_support") or {}).get("claim_support_rate") is not None]
            ),
            "not_rag_grounding": True,
        },
        "persona_utility": _mean(utilities),
        "utility_weights": dict(UTILITY_WEIGHTS),
        "diagnostics": {
            "reference_similarity": {
                "bertscore": None,
                "bleu": _mean(collect(lambda row: row["diagnostics"]["reference_similarity"]["bleu"])),
                "rouge_l": _mean(collect(lambda row: row["diagnostics"]["reference_similarity"]["rouge_l"])),
                "role": "diagnostic_not_persona_fidelity",
            },
            "brief_overlap": {
                "gold_rubric_overall": _mean(
                    collect(lambda row: row["diagnostics"]["brief_overlap"]["gold_rubric_overall"])
                ),
                "role": "brief_task_agreement_not_persona_quality",
            },
            "latency_seconds": _mean(collect(lambda row: row["diagnostics"]["latency_seconds"])),
        },
    }


def rank_models(models: list[dict]) -> dict:
    """Gates first. Diagnostic overlap and stop rates cannot select a model."""
    refused = list(REFUSED_DIAGNOSTICS)
    eligible = [model for model in models if model.get("n_eligible", 0) > 0]
    if not eligible:
        return {
            "selected": None,
            "reason": "no_model_passed_gates",
            "used_for_selection": [],
            "refused_diagnostics": refused,
            "ranking": [],
        }
    selectable = [
        model for model in eligible
        if model["persona_style"]["role"] == "selector"
        and model["persona_style"].get("mean_distance_eligible") is not None
    ]
    if not selectable:
        return {
            "selected": None,
            "reason": "persona_style_not_calibrated",
            "used_for_selection": [],
            "refused_diagnostics": refused,
            "ranking": [model["model_id"] for model in eligible],
        }
    used = ["persona_style"]
    if any(model["perspective_fidelity"]["role"] == "selector" for model in selectable):
        used.append("perspective_fidelity")
    if any(model["writing_quality"]["role"] == "selector" for model in selectable):
        used.append("writing_quality")
    used.append("task_adherence")
    if any(model.get("persona_utility") is not None for model in selectable):
        used.append("persona_utility")

    def _key(model: dict) -> tuple:
        perspective = model["perspective_fidelity"]
        writing = model["writing_quality"]
        if perspective["role"] == "selector" and perspective.get("mean_stance_consistency") is not None:
            perspective_key = -perspective["mean_stance_consistency"]
        elif perspective["role"] == "selector" and perspective.get("mean_distance_eligible") is not None:
            perspective_key = perspective["mean_distance_eligible"]
        else:
            perspective_key = 0
        writing_key = (
            -(writing["mean_eligible"] or 0)
            if writing["role"] == "selector"
            else 0
        )
        task = model["task_adherence"]["mean_eligible"]
        utility = model.get("persona_utility")
        return (
            model["persona_style"]["mean_distance_eligible"],
            perspective_key,
            writing_key,
            -(task if task is not None else 0),
            -(utility if utility is not None else -1),
        )

    ordered = sorted(selectable, key=_key)
    return {
        "selected": ordered[0]["model_id"],
        "reason": "passed_gates_then_ranked_persona_style_then_later_axes",
        "used_for_selection": used,
        "refused_diagnostics": refused,
        "ranking": [model["model_id"] for model in ordered],
        "weights_are_project_decision": True,
        "utility_weights": dict(UTILITY_WEIGHTS),
    }
