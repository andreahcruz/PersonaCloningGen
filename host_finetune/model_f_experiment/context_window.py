"""Check the configured generation window before any draft is sampled.

The generator is loaded at 2048 tokens. The tokenizer file still records a
training ``model_max_length`` of 512, and Llama 3.1 can address a much longer
context. Neither of those is the generation window. This check uses 2048.
"""

from __future__ import annotations

from host_finetune.compare_adapters import MAX_SEQ_LENGTH, MEDIUMS
from host_finetune.model_f_experiment.arms import ADAPTER, ARMS, experiment_budget
from host_finetune.model_f_experiment.runner import MEDIA

# Longest fixed corrective addendum is 38 tokens. One spare token stays above it.
RETRY_MARGIN = 40
# "[E15] " is 4 tokens. Twelve claim labels, plus one path-status line.
TAG_ALLOWANCE = 48
GRAPH_HEADING = 11


class ContextWindowError(RuntimeError):
    """A planned cell cannot be generated without dropping evidence."""


def load_tokenizer():
    """Load the Relabel tokenizer from disk. This does not load model weights."""
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(ADAPTER, local_files_only=True)
    tokenizer.model_max_length = 131072
    return tokenizer


def prompt_tokens(tokenizer, instruction: str) -> int:
    """Match ``_complete_once``: chat template, then the default tokenizer call."""
    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": instruction}],
        tokenize=False,
        add_generation_prompt=True,
    )
    return int(tokenizer(prompt, return_tensors="pt").input_ids.shape[1])


def _medium(name: str) -> dict:
    for item in MEDIUMS:
        if item["medium"] == name:
            return item
    raise KeyError(name)


def cell_requirement(tokenizer, question: dict, arm: str, medium: str) -> dict:
    from host_finetune.model_f_context.prompt import render_context_prompt

    rendered = render_context_prompt(
        {"selected": {name: [] for name in (
            "primary_source",
            "source_supported_relationship",
            "multi_hop_path",
            "derived_description",
        )}},
        _medium(medium),
        question["query"],
        [],
        "verified",
    )
    fixed = prompt_tokens(tokenizer, rendered["instruction"])
    budget = experiment_budget(medium)
    uses_graph = arm != "F-SOURCE"
    evidence = budget.primary + (budget.relationships if uses_graph else 0)
    overhead = TAG_ALLOWANCE + (GRAPH_HEADING if uses_graph else 0)
    max_new = _medium(medium)["max_new_tokens"]
    total = fixed + evidence + overhead + RETRY_MARGIN + max_new
    return {
        "question_id": question["question_id"],
        "arm": arm,
        "medium": medium,
        "fixed_prompt_tokens": fixed,
        "evidence_allowance": evidence,
        "label_overhead": overhead,
        "retry_margin": RETRY_MARGIN,
        "max_new_tokens": max_new,
        "total_required": total,
        "context_window": MAX_SEQ_LENGTH,
        "fits": total <= MAX_SEQ_LENGTH,
    }


def preflight(questions: list[dict], tokenizer=None) -> dict:
    """One row per planned cell. An overflow is reported and nothing is truncated."""
    chosen = tokenizer or load_tokenizer()
    rows = [
        cell_requirement(chosen, question, arm, medium)
        for question in questions
        for arm in ARMS
        for medium in MEDIA
    ]
    overflowing = [row for row in rows if not row["fits"]]
    return {
        "cells": len(rows),
        "overflowing": overflowing,
        "context_window": MAX_SEQ_LENGTH,
        "max_required": max(row["total_required"] for row in rows) if rows else 0,
    }


def assert_preflight(questions: list[dict], tokenizer=None) -> dict:
    report = preflight(questions, tokenizer)
    if report["overflowing"]:
        sample = report["overflowing"][0]
        raise ContextWindowError(
            f"{len(report['overflowing'])} cells exceed {MAX_SEQ_LENGTH} tokens; "
            f"first is {sample['question_id']} {sample['arm']} {sample['medium']} "
            f"requiring {sample['total_required']}"
        )
    return report
