#!/usr/bin/env python3
"""Offline gold-rubric and retrieval-grounding metrics.

This script is meant to cover the first two evaluation buckets even when an
LLM judge or RAGAS backend is unavailable:

1. Gold / rubric evaluation
   - format adherence
   - brief alignment
   - optional expected-fact coverage

2. Retrieval-grounded evaluation
   - answer relevance
   - faithfulness (sentence support ratio)
   - context precision
   - context recall

It can score either:
  - a JSONL of trace rows with ``query``, ``answer``, and ``contexts`` fields, or
  - the OpenAI-style message JSONL used in ``data/openai_ft/*``.
"""
from __future__ import annotations

import argparse
import json
import re
import statistics
from pathlib import Path

import yaml

STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "but", "by", "for", "from", "had",
    "has", "have", "he", "her", "his", "i", "if", "in", "into", "is", "it", "its",
    "more", "no", "not", "of", "on", "or", "our", "that", "the", "their", "them",
    "there", "they", "this", "to", "too", "was", "we", "were", "what", "when",
    "which", "who", "why", "will", "with", "you", "your",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--input-jsonl",
        required=True,
        help="Trace JSONL or OpenAI-style messages JSONL.",
    )
    p.add_argument(
        "--input-format",
        choices=["trace", "messages"],
        required=True,
        help="How to parse --input-jsonl.",
    )
    p.add_argument(
        "--format-specs",
        default=str(Path(__file__).resolve().parents[1] / "formats" / "format_specs.yaml"),
        help="Path to format_specs.yaml.",
    )
    p.add_argument(
        "--gold-jsonl",
        default=None,
        help="Optional hand-authored gold eval JSONL with expected_facts / gold_reference.",
    )
    p.add_argument(
        "--out",
        default=str(Path(__file__).resolve().parent / "output" / "eval" / "gold_rag_summary.json"),
        help="Where to write the JSON summary.",
    )
    return p.parse_args()


def load_format_specs(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def tokenize(text: str) -> list[str]:
    return re.findall(r"[A-Za-z0-9$%]+", text.lower())


def content_tokens(text: str) -> set[str]:
    return {t for t in tokenize(text) if t not in STOPWORDS and len(t) > 2}


def split_sentences(text: str) -> list[str]:
    chunks = re.split(r"(?<=[.!?])\s+|\n{2,}", text.strip())
    return [c.strip() for c in chunks if len(c.strip()) >= 20]


def overlap_f1(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    inter = len(a & b)
    if inter == 0:
        return 0.0
    precision = inter / len(a)
    recall = inter / len(b)
    return 2 * precision * recall / (precision + recall)


def token_coverage(required: set[str], observed: set[str]) -> float:
    if not required:
        return 0.0
    return len(required & observed) / len(required)


def normalize_score(value: float) -> float:
    return round(max(0.0, min(1.0, value)), 4)


def extract_between(text: str, start: str, end: str | None = None) -> str:
    if start not in text:
        return ""
    s = text.split(start, 1)[1]
    if end and end in s:
        s = s.split(end, 1)[0]
    return s.strip()


def parse_messages_row(row: dict, idx: int) -> dict:
    messages = row.get("messages", [])
    user_msg = next((m.get("content", "") for m in messages if m.get("role") == "user"), "")
    answer = next((m.get("content", "") for m in reversed(messages) if m.get("role") == "assistant"), "")

    format_name = ""
    topic = ""
    audience = ""
    goal = ""
    cta = ""
    m = re.search(r"Target length:\s*(\d+)\D+(\d+)\s*words", user_msg)
    content_brief = extract_between(user_msg, "[CONTENT BRIEF]", "[CONTEXT")
    for line in content_brief.splitlines():
        line = line.strip()
        if line.startswith("Topic:"):
            topic = line.split(":", 1)[1].strip()
        elif line.startswith("Audience:"):
            audience = line.split(":", 1)[1].strip()
        elif line.startswith("Goal:"):
            goal = line.split(":", 1)[1].strip()
        elif line.startswith("Call to action:"):
            cta = line.split(":", 1)[1].strip()

    # Infer format from the structure block.
    format_rules = extract_between(user_msg, "[FORMAT RULES]", "[CONTENT BRIEF]")
    if "Post 1" in format_rules or "1/" in format_rules:
        format_name = "x_thread"
    elif "## Hook" in format_rules or "Working video title" in format_rules:
        format_name = "youtube_script"
    elif "3–5 H2 sections" in format_rules:
        format_name = "blog_draft"
    else:
        format_name = "linkedin_post"

    contexts = re.findall(
        r"\[Source excerpt \d+\]\n(.*?)(?=\n\[Source excerpt \d+\]|\n\[INSTRUCTIONS\]|\Z)",
        user_msg,
        flags=re.S,
    )

    expected_facts: list[str] = []
    for ctx in contexts:
        first = " ".join(ctx.strip().split())[:220]
        if first:
            expected_facts.append(first)

    return {
        "id": f"messages_{idx:03d}",
        "format": format_name,
        "topic": topic,
        "audience": audience,
        "goal": goal,
        "cta": cta,
        "query": f"{topic}. {goal}. Audience: {audience}.",
        "answer": answer,
        "contexts": contexts,
        "expected_facts": expected_facts[:3],
    }


def load_rows(path: Path, input_format: str) -> list[dict]:
    rows: list[dict] = []
    with open(path, encoding="utf-8") as f:
        for idx, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if input_format == "messages":
                rows.append(parse_messages_row(row, idx))
            else:
                rows.append(row)
    return rows


def load_gold_rows(path: Path) -> list[dict]:
    rows: list[dict] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def merge_gold(rows: list[dict], gold_rows: list[dict]) -> list[dict]:
    by_topic = {g.get("topic", "").strip(): g for g in gold_rows}
    merged: list[dict] = []
    for idx, row in enumerate(rows):
        row = dict(row)
        gold = by_topic.get(row.get("topic", "").strip())
        if gold is None and idx < len(gold_rows):
            gold = gold_rows[idx]
        if gold:
            row["id"] = gold.get("id", row.get("id"))
            row["format"] = gold.get("format", row.get("format"))
            row["cta"] = gold.get("cta", row.get("cta"))
            row["expected_facts"] = gold.get("expected_facts", row.get("expected_facts", []))
            row["rubric_must_haves"] = gold.get("rubric_must_haves", [])
            row["gold_reference"] = gold.get("gold_reference", "")
        merged.append(row)
    return merged


def length_score(answer: str, spec: dict) -> float:
    words = len(answer.split())
    lo, hi = spec.get("length_target_words", [0, 10**9])
    if lo <= words <= hi:
        return 1.0
    if words < lo:
        return max(0.0, words / max(lo, 1))
    return max(0.0, hi / max(words, 1))


def format_score(answer: str, format_name: str, spec: dict) -> dict:
    lines = [ln.rstrip() for ln in answer.splitlines()]
    nonempty = [ln for ln in lines if ln.strip()]
    heading_count = sum(1 for ln in nonempty if ln.startswith("## "))
    bullet_count = sum(1 for ln in nonempty if re.match(r"^[-*]\s+|^\d+\.\s+", ln))
    question_end = bool(nonempty and nonempty[-1].strip().endswith("?"))

    checks: list[float] = [length_score(answer, spec)]
    if format_name == "blog_draft":
        checks.append(1.0 if nonempty else 0.0)
        checks.append(1.0 if 3 <= heading_count <= 5 else max(0.0, heading_count / 3.0))
        checks.append(1.0 if 3 <= bullet_count <= 5 else 0.0)
    elif format_name == "linkedin_post":
        checks.append(1.0 if 3 <= bullet_count <= 5 else 0.0)
        checks.append(1.0 if question_end else 0.0)
    elif format_name == "x_thread":
        post_markers = sum(
            1 for ln in nonempty if re.match(r"^(Post\s+\d+|[1-9]\d*/)\b", ln)
        )
        checks.append(1.0 if 3 <= post_markers <= 7 else 0.0)
    elif format_name == "youtube_script":
        checks.append(1.0 if any(ln.startswith("## Hook") for ln in nonempty) else 0.0)
        checks.append(1.0 if heading_count >= 2 else 0.0)

    return {
        "word_count": len(answer.split()),
        "h2_sections": heading_count,
        "list_items": bullet_count,
        "score": normalize_score(statistics.mean(checks) if checks else 0.0),
    }


def requirement_coverage(row: dict) -> dict:
    answer = row.get("answer", "")
    answer_tokens = content_tokens(answer)
    query_tokens = content_tokens(
        " ".join([row.get("topic", ""), row.get("goal", ""), row.get("audience", "")])
    )
    brief_alignment = token_coverage(query_tokens, answer_tokens)

    expected = row.get("expected_facts") or []
    covered = 0
    for fact in expected:
        fact_tokens = content_tokens(fact)
        if len(fact_tokens & answer_tokens) >= min(3, max(1, len(fact_tokens) // 3)):
            covered += 1
    expected_cov = covered / len(expected) if expected else 0.0
    gold_ref = row.get("gold_reference", "")
    gold_ref_cov = token_coverage(content_tokens(gold_ref), answer_tokens) if gold_ref else 0.0
    must_haves = row.get("rubric_must_haves") or []
    must_have_hits = 0
    for item in must_haves:
        item_tokens = content_tokens(item)
        if token_coverage(item_tokens, answer_tokens) >= 0.35:
            must_have_hits += 1
    must_have_cov = must_have_hits / len(must_haves) if must_haves else 0.0
    return {
        "brief_alignment": normalize_score(brief_alignment),
        "expected_fact_coverage": normalize_score(expected_cov),
        "gold_reference_coverage": normalize_score(gold_ref_cov),
        "rubric_must_have_coverage": normalize_score(must_have_cov),
    }


def best_context_support(sentence: str, contexts: list[str]) -> tuple[float, int]:
    s_tokens = content_tokens(sentence)
    if not s_tokens:
        return 0.0, -1
    best_score = 0.0
    best_idx = -1
    for idx, ctx in enumerate(contexts):
        c_tokens = content_tokens(ctx)
        inter = s_tokens & c_tokens
        score = overlap_f1(s_tokens, c_tokens)
        if re.search(r"\d", sentence) and re.search(r"\d", ctx) and inter:
            score += 0.1
        if len(inter) >= 3 and score > best_score:
            best_score = score
            best_idx = idx
    return best_score, best_idx


def grounding_scores(row: dict) -> dict:
    answer = row.get("answer", "")
    contexts = row.get("contexts", []) or []
    query = row.get("query", "")
    sentences = split_sentences(answer)
    supported = 0
    used_contexts: set[int] = set()
    for sent in sentences:
        score, idx = best_context_support(sent, contexts)
        if score >= 0.2:
            supported += 1
            if idx >= 0:
                used_contexts.add(idx)

    answer_tokens = content_tokens(answer)
    query_tokens = content_tokens(query)
    answer_relevance = token_coverage(query_tokens, answer_tokens)
    precision = supported / len(sentences) if sentences else 0.0
    recall = len(used_contexts) / len(contexts) if contexts else 0.0
    return {
        "answer_relevance": normalize_score(answer_relevance),
        "faithfulness": normalize_score(precision),
        "context_precision": normalize_score(precision),
        "context_recall": normalize_score(recall),
        "n_sentences": len(sentences),
        "n_supported_sentences": supported,
    }


def score_row(row: dict, format_specs: dict) -> dict:
    format_name = row.get("format", "linkedin_post")
    spec = format_specs.get(format_name, {})
    fmt = format_score(row.get("answer", ""), format_name, spec)
    req = requirement_coverage(row)
    grd = grounding_scores(row)
    gold_overall = statistics.mean(
        [
            fmt["score"],
            req["brief_alignment"],
            req["expected_fact_coverage"],
            req["gold_reference_coverage"],
            req["rubric_must_have_coverage"],
        ]
    )
    row_out = {
        "id": row.get("id"),
        "format": format_name,
        "gold_rubric": {
            **fmt,
            **req,
            "overall": normalize_score(gold_overall),
        },
        "retrieval_grounding": grd,
    }
    return row_out


def aggregate(scores: list[dict]) -> dict:
    def avg(path: str) -> float:
        vals = []
        for row in scores:
            cur = row
            for key in path.split("."):
                cur = cur[key]
            vals.append(float(cur))
        return normalize_score(statistics.mean(vals) if vals else 0.0)

    return {
        "n_rows": len(scores),
        "gold_rubric": {
            "overall": avg("gold_rubric.overall"),
            "format_score": avg("gold_rubric.score"),
            "brief_alignment": avg("gold_rubric.brief_alignment"),
            "expected_fact_coverage": avg("gold_rubric.expected_fact_coverage"),
            "gold_reference_coverage": avg("gold_rubric.gold_reference_coverage"),
            "rubric_must_have_coverage": avg("gold_rubric.rubric_must_have_coverage"),
        },
        "retrieval_grounding": {
            "answer_relevance": avg("retrieval_grounding.answer_relevance"),
            "faithfulness": avg("retrieval_grounding.faithfulness"),
            "context_precision": avg("retrieval_grounding.context_precision"),
            "context_recall": avg("retrieval_grounding.context_recall"),
        },
    }


def main() -> None:
    args = parse_args()
    rows = load_rows(Path(args.input_jsonl), args.input_format)
    if args.gold_jsonl:
        rows = merge_gold(rows, load_gold_rows(Path(args.gold_jsonl)))
    format_specs = load_format_specs(Path(args.format_specs))
    per_row = [score_row(row, format_specs) for row in rows]
    summary = aggregate(per_row)
    out = {"summary": summary, "rows": per_row}

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"\n[eval_gold_rag] wrote {out_path}")


if __name__ == "__main__":
    main()
