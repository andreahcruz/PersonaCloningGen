"""Read saved unified_eval rows and pull metric values out of them.

A row is one scored draft (``schema: unified_eval.v1``). Rows are identified by
``(id, medium)``: ``id`` is the gold topic, ``medium`` is blog, linkedin, x or talk.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Iterable

# Metric name -> dotted path inside a row. Names are what the reports print.
METRIC_PATHS: dict[str, str] = {
    "writing_quality": "writing_quality.mean",
    "task_adherence": "task_adherence.score",
    "stance_consistency": "perspective_fidelity.stance_consistency_rate",
    "style_distance": "persona_style.distance",
    "framing_distance": "perspective_fidelity.framing_distance",
    "persona_utility": "persona_utility",
    "claim_support": "grounding.claim_support_rate",
    "evidence_support": "lemkin_evidence_support.claim_support_rate",
    "word_count": "gates.format.word_count",
    "latency_seconds": "diagnostics.latency_seconds",
    # Filled from a side file (see ``attach_extra``), for example the grammar tool.
    "grammar_rate": "extra.grammar_rate",
    "grammar_deviation": "extra.grammar_deviation",
}

# For these, a lower number is better, so "B minus A" below zero means B is better.
LOWER_IS_BETTER = {"style_distance", "framing_distance", "latency_seconds", "grammar_rate", "grammar_deviation"}

# Neither direction is better (a longer draft is not a better draft). Reports say
# "B higher" or "B lower" for these instead of "B better" or "B worse".
NEUTRAL_METRICS = {"word_count"}

# Gates that every system faces. Grounding is left out on purpose: only retrieval
# systems have it, so using it would compare different subsets of drafts.
COMMON_GATES = ("completion", "copying_memorization", "format")

Key = tuple[str, str]


def get_path(row: dict, path: str) -> float | None:
    """Return a number at a dotted path, or None if it is missing, null or not a number."""
    node: Any = row
    for part in path.split("."):
        if not isinstance(node, dict) or part not in node:
            return None
        node = node[part]
    if isinstance(node, bool) or not isinstance(node, (int, float)):
        return None
    if isinstance(node, float) and math.isnan(node):
        return None
    return float(node)


def metric_value(row: dict, metric: str) -> float | None:
    return get_path(row, METRIC_PATHS[metric])


def _model_name(row: dict) -> str:
    model = row.get("model")
    if isinstance(model, dict):
        model = model.get("model_id") or model.get("name")
    return str(model or "")


def load_rows(
    path: str | Path,
    source_experiment: str | None = None,
    model: str | None = None,
) -> list[dict]:
    """Load a rows.jsonl file. Optional filters pick one experiment and one model.

    ``model`` matches the whole model name or its last path part, because some files
    store ``EXP-.../relabel`` and others just ``relabel``.
    """
    rows: list[dict] = []
    with open(path, encoding="utf-8-sig") as handle:
        for line_no, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as err:
                raise ValueError(f"{path} line {line_no} is not valid JSON: {err}") from err
            if not isinstance(row, dict):
                continue
            if source_experiment and row.get("source_experiment") != source_experiment:
                continue
            if model:
                name = _model_name(row)
                if name != model and name.rsplit("/", 1)[-1] != model:
                    continue
            rows.append(row)
    return rows


def row_key(row: dict) -> Key:
    return (str(row.get("id")), str(row.get("medium")))


def index_rows(rows: Iterable[dict]) -> dict[Key, dict]:
    """Map (id, medium) to a row. Two rows with one key mean the filters were too loose."""
    out: dict[Key, dict] = {}
    for row in rows:
        key = row_key(row)
        if key in out:
            raise ValueError(
                f"Two rows share the key {key}. Narrow the file with --*-experiment and --*-model."
            )
        out[key] = row
    return out


def common_gates_pass(row: dict) -> bool:
    """True when the draft passes completion, copying and format (not grounding)."""
    gates = row.get("gates") or {}
    return all(isinstance(gates.get(name), dict) and gates[name].get("pass") is True for name in COMMON_GATES)


def attach_extra(rows: dict[Key, dict], extra_path: str | Path) -> int:
    """Merge a side file of per draft numbers into the rows under ``row["extra"]``.

    The side file is JSONL with ``id``, ``medium`` and any numeric fields, for example
    the output of ``python -m eval_tools.grammar``. Returns how many rows were matched.
    """
    matched = 0
    with open(extra_path, encoding="utf-8-sig") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            key = row_key(item)
            if key in rows:
                rows[key].setdefault("extra", {}).update(
                    {k: v for k, v in item.items() if k not in ("id", "medium")}
                )
                matched += 1
    return matched
