"""Surface style distance against train-owned posts.

This is a diagnostic. It does not decide which draft sounds like the author.
"""
from __future__ import annotations

import json
import math
import re
import statistics
from pathlib import Path

from host_finetune.train_only_index import (
    DEFAULT_ASSIGNMENTS,
    DEFAULT_DATASET,
    DEFAULT_GOLD_DISPOSITIONS,
    assignment_rows,
    blocked_group_ids,
    load_jsonl,
)

WORD_RE = re.compile(r"[A-Za-z0-9']+")
SENTENCE_RE = re.compile(r"[.!?…]+")
CONTRACTION_RE = re.compile(r"\b\w+'\w+\b")
FIRST_PERSON_RE = re.compile(r"\b(?:I|we|my|our|I'm|we're|I've|we've)\b")
FEATURE_NAMES = (
    "words_per_sentence",
    "questions_per_100w",
    "exclamations_per_100w",
    "contractions_per_100w",
    "numbers_per_100w",
    "first_person_per_100w",
)
PLATFORM_TO_MEDIUM = {
    "blog": "blog",
    "linkedin": "linkedin",
    "x": "x",
}


def style_features(text: str) -> dict[str, float]:
    """Counts that can be computed from the draft alone."""
    raw = text or ""
    words = WORD_RE.findall(raw)
    sentences = [part for part in SENTENCE_RE.split(raw) if part.strip()]
    n_words = max(len(words), 1)
    n_sentences = max(len(sentences), 1)
    return {
        "words_per_sentence": len(words) / n_sentences,
        "questions_per_100w": 100 * raw.count("?") / n_words,
        "exclamations_per_100w": 100 * raw.count("!") / n_words,
        "contractions_per_100w": 100 * len(CONTRACTION_RE.findall(raw)) / n_words,
        "numbers_per_100w": 100 * len(re.findall(r"\d", raw)) / n_words,
        "first_person_per_100w": 100 * len(FIRST_PERSON_RE.findall(raw)) / n_words,
    }


def reference_profile(texts: list[str]) -> dict[str, dict[str, float]]:
    """Mean and sample standard deviation for each feature. Empty input is empty."""
    if not texts:
        return {}
    columns = {name: [] for name in FEATURE_NAMES}
    for text in texts:
        features = style_features(text)
        for name in FEATURE_NAMES:
            columns[name].append(features[name])
    profile = {}
    for name, values in columns.items():
        mean = statistics.fmean(values)
        stdev = statistics.stdev(values) if len(values) > 1 else 0.0
        profile[name] = {"mean": mean, "stdev": stdev}
    return profile


def style_distance(text: str, profile: dict[str, dict[str, float]]) -> float | None:
    """Mean absolute z-score. Lower is closer to the reference posts."""
    if not profile:
        return None
    features = style_features(text)
    scores = []
    for name, stats in profile.items():
        stdev = stats["stdev"]
        if stdev <= 0:
            continue
        scores.append(abs(features[name] - stats["mean"]) / stdev)
    if not scores:
        return None
    return statistics.fmean(scores)


def closer_side(distance_a: float | None, distance_b: float | None, margin: float = 0.05) -> str:
    """Return a, b, tie, or unavailable. A small gap stays a tie."""
    if distance_a is None or distance_b is None or math.isnan(distance_a) or math.isnan(distance_b):
        return "unavailable"
    if abs(distance_a - distance_b) <= margin:
        return "tie"
    return "a" if distance_a < distance_b else "b"


def reference_texts(
    dataset_rows: list[dict],
    assignments: list[dict],
    blocked_groups: set[str],
    per_medium: int = 40,
) -> dict[str, list[str]]:
    """Earliest unchanged train posts, grouped by comparison medium."""
    records = assignment_rows(assignments)
    if len(records) != len(dataset_rows):
        raise ValueError(
            f"assignment rows ({len(records)}) != dataset rows ({len(dataset_rows)})"
        )
    chosen = {"blog": [], "linkedin": [], "x": [], "talk": []}
    for row, rec in zip(dataset_rows, records):
        medium = PLATFORM_TO_MEDIUM.get(rec.get("source_platform"))
        if medium is None or len(chosen[medium]) >= per_medium:
            continue
        if rec.get("split") != "train" or rec.get("sft_role") != "unchanged":
            continue
        if rec.get("group_id") in blocked_groups:
            continue
        text = (row.get("output") or "").strip()
        if text:
            chosen[medium].append(text)
    return chosen


def load_reference_texts(per_medium: int = 40) -> dict[str, list[str]]:
    dataset = load_jsonl(DEFAULT_DATASET)
    assignments = load_jsonl(DEFAULT_ASSIGNMENTS)
    dispositions = load_jsonl(DEFAULT_GOLD_DISPOSITIONS)
    return reference_texts(
        dataset, assignments, blocked_group_ids(assignments, dispositions), per_medium
    )


def score_sheet(sheet: list[dict], references: dict[str, list[str]]) -> dict:
    """Blind distances for each pair. Model names are not an input."""
    profiles = {medium: reference_profile(texts) for medium, texts in references.items()}
    rows = []
    for pair in sheet:
        medium = pair.get("medium")
        profile = profiles.get(medium) or {}
        distance_a = style_distance(pair.get("answer_a") or "", profile)
        distance_b = style_distance(pair.get("answer_b") or "", profile)
        rows.append({
            "pair_id": pair.get("pair_id"),
            "medium": medium,
            "distance_a": None if distance_a is None else round(distance_a, 4),
            "distance_b": None if distance_b is None else round(distance_b, 4),
            "closer": closer_side(distance_a, distance_b),
        })
    return {
        "role": "diagnostic_not_a_voice_decision",
        "method": "mean absolute z-score of surface counts versus unchanged train posts",
        "not_used": "TF-IDF authorship classifier",
        "limits": (
            "Closer surface counts are not authorship accuracy. "
            "Talk has no unchanged train references, so those pairs stay unavailable."
        ),
        "reference_counts": {medium: len(texts) for medium, texts in references.items()},
        "pairs": rows,
    }


def attribute_closer(scored: dict, key_rows: list[dict]) -> dict:
    """Join the blind distances to adapter names. Keep this file out of the reviewer."""
    key = {row["pair_id"]: row for row in key_rows}
    wins = {"repaired": 0, "relabel": 0, "tie": 0, "unavailable": 0}
    by_medium: dict[str, dict[str, int]] = {}
    for pair in scored["pairs"]:
        medium = pair.get("medium") or "unknown"
        bucket = by_medium.setdefault(medium, {"repaired": 0, "relabel": 0, "tie": 0, "unavailable": 0})
        closer = pair["closer"]
        if closer in {"tie", "unavailable"}:
            wins[closer] += 1
            bucket[closer] += 1
            continue
        model = key[pair["pair_id"]][f"answer_{closer}_model"]
        wins[model] = wins.get(model, 0) + 1
        bucket[model] = bucket.get(model, 0) + 1
    return {
        "role": scored["role"],
        "method": scored["method"],
        "limits": scored["limits"],
        "reference_counts": scored["reference_counts"],
        "closer_counts": wins,
        "by_medium": by_medium,
    }


def write_scores(sheet_path: Path, key_path: Path, out_dir: Path) -> dict:
    """Write a blind file and a separate attributed summary."""
    sheet = [json.loads(line) for line in sheet_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    key_rows = [json.loads(line) for line in key_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    scored = score_sheet(sheet, load_reference_texts())
    attributed = attribute_closer(scored, key_rows)
    out_dir.mkdir(parents=True, exist_ok=True)
    blind_path = out_dir / "style_distance_blind.json"
    summary_path = out_dir / "style_distance.json"
    blind_path.write_text(json.dumps(scored, indent=2) + "\n", encoding="utf-8")
    summary_path.write_text(json.dumps(attributed, indent=2) + "\n", encoding="utf-8")
    return attributed
