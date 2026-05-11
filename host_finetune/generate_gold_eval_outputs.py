#!/usr/bin/env python3
"""Generate outputs for the hand-authored gold eval set.

This uses title-matched blog corpus entries as the grounding context and writes
trace-style JSONL rows compatible with ``eval_gold_rag.py``.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import requests
import yaml


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--gold-jsonl", default=str(root / "data" / "openai_ft" / "lemkin_gold_eval.jsonl"))
    p.add_argument("--blog-jsonl", default=str(root / "data" / "jasonlemkin_blog.jsonl"))
    p.add_argument("--persona", default=str(root / "data" / "persona_profile.json"))
    p.add_argument("--format-specs", default=str(root / "formats" / "format_specs.yaml"))
    p.add_argument("--model", default="llama3.2:latest")
    p.add_argument("--ollama-base", default="http://127.0.0.1:11434")
    p.add_argument("--out", default=str(root / "host_finetune" / "output" / "eval" / "gold_eval_traces.jsonl"))
    p.add_argument("--max-context-chars", type=int, default=2600)
    p.add_argument("--num-predict", type=int, default=550)
    p.add_argument("--offset", type=int, default=0)
    p.add_argument("--limit", type=int, default=0, help="0 means no limit.")
    p.add_argument("--append", action="store_true", help="Append to output instead of overwrite.")
    return p.parse_args()


def normalize(text: str) -> str:
    text = text.lower()
    text = text.replace("’", "'").replace("“", '"').replace("”", '"').replace("—", "-")
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return " ".join(text.split())


def title_tokens(text: str) -> set[str]:
    toks = set(normalize(text).split())
    return {t for t in toks if len(t) > 2}


def load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_persona_summary(path: Path) -> str:
    profile = json.loads(path.read_text(encoding="utf-8"))
    vocab = ", ".join(profile.get("vocabulary_tendencies", {}).get("top_50_content_words", [])[:15])
    starters = ", ".join(profile.get("structure_patterns", {}).get("common_sentence_starters_bigrams", [])[:8])
    framing = ", ".join(profile.get("favorite_framing", {}).get("favorite_framing_phrases", [])[:10])
    themes = profile.get("favorite_themes", {}).get("top_content_words_and_bigrams", [])[:10]
    theme_text = ", ".join(themes)
    return "\n".join(
        [
            "You are writing in the persona of Jason Lemkin — blog, LinkedIn, X posts, SaaStr / channel YouTube transcripts.",
            "Emulate: direct, experienced B2B / SaaS operator tone; concrete metrics and examples.",
            f"Vocabulary tendencies: {vocab}.",
            f"Sentence openers to echo sparingly: {starters}.",
            f"Framing phrases: {framing}.",
            f"Themes: {theme_text}.",
            "Favor clarity and specificity over generic hype.",
        ]
    )


def load_format_block(path: Path, format_name: str) -> str:
    specs = yaml.safe_load(path.read_text(encoding="utf-8"))
    spec = specs[format_name]
    lines = ["[FORMAT RULES]", f"Output type: {spec.get('output_type', 'markdown')}."]
    lo, hi = spec.get("length_target_words", [0, 0])
    if lo and hi:
        lines.append(f"Target length: {lo}-{hi} words.")
    if "structure" in spec:
        lines.append("Structure:")
        for item in spec["structure"]:
            if isinstance(item, dict):
                for k, v in item.items():
                    lines.append(f"  - {k}: {v}")
            else:
                lines.append(f"  - {item}")
    if "style" in spec:
        lines.append("Style: " + "; ".join(spec["style"]))
    if "banned" in spec:
        lines.append("Avoid: " + "; ".join(spec["banned"]))
    return "\n".join(lines)


def choose_blog_row(topic: str, blog_rows: list[dict]) -> dict:
    topic_norm = normalize(topic)
    topic_tok = title_tokens(topic)
    exact = next((r for r in blog_rows if normalize(r.get("title", "")) == topic_norm), None)
    if exact:
        return exact

    best_row = None
    best_score = -1.0
    for row in blog_rows:
        title = row.get("title", "")
        toks = title_tokens(title)
        if not toks:
            continue
        overlap = len(topic_tok & toks)
        score = overlap / max(len(topic_tok), 1)
        if topic_norm in normalize(title) or normalize(title) in topic_norm:
            score += 0.25
        if score > best_score:
            best_score = score
            best_row = row
    if best_row is None:
        raise RuntimeError(f"No matching blog row found for topic: {topic}")
    return best_row


def build_prompt(row: dict, persona_summary: str, format_block: str, context_text: str) -> str:
    return "\n\n".join(
        [
            "[PERSONA]\n\n" + persona_summary,
            format_block,
            "\n".join(
                [
                    "[CONTENT BRIEF]",
                    f"Topic: {row['topic']}",
                    f"Audience: {row['audience']}",
                    f"Goal: {row['goal']}",
                    f"Call to action: {row.get('cta', 'none')}",
                ]
            ),
            "[CONTEXT — EXCERPTS FROM JASON LEMKIN CORPUS]\n[Source excerpt 1]\n" + context_text,
            "[INSTRUCTIONS]\n\nWrite a concise draft that matches the format rules and content brief as closely as possible. "
            "Use the context for tone and ideas; do not copy long verbatim passages. "
            "Be concrete, use short sections, and output only markdown.",
        ]
    )


def ollama_generate(prompt: str, base_url: str, model: str, num_predict: int) -> str:
    resp = requests.post(
        f"{base_url.rstrip('/')}/api/generate",
        json={
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.2,
                "num_predict": num_predict,
            },
        },
        timeout=1800,
    )
    resp.raise_for_status()
    return resp.json().get("response", "").strip()


def main() -> None:
    args = parse_args()
    gold_rows = load_jsonl(Path(args.gold_jsonl))
    if args.limit > 0:
        gold_rows = gold_rows[args.offset : args.offset + args.limit]
    else:
        gold_rows = gold_rows[args.offset :]
    blog_rows = load_jsonl(Path(args.blog_jsonl))
    persona_summary = load_persona_summary(Path(args.persona))

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    mode = "a" if args.append else "w"
    with out_path.open(mode, encoding="utf-8") as out:
        for i, row in enumerate(gold_rows, 1):
            blog_row = choose_blog_row(row["topic"], blog_rows)
            context_text = (blog_row.get("content", "") or "").strip()
            context_text = context_text[: args.max_context_chars]
            format_block = load_format_block(Path(args.format_specs), row["format"])
            prompt = build_prompt(row, persona_summary, format_block, context_text)
            print(
                f"[{i}/{len(gold_rows)}] generating {row['id']} with source "
                f"'{blog_row.get('title', '')[:80]}'",
                flush=True,
            )
            answer = ollama_generate(prompt, args.ollama_base, args.model, args.num_predict)
            trace = {
                "id": row["id"],
                "format": row["format"],
                "topic": row["topic"],
                "audience": row["audience"],
                "goal": row["goal"],
                "cta": row.get("cta", "none"),
                "query": f"{row['topic']}. {row['goal']}. Audience: {row['audience']}.",
                "contexts": [context_text],
                "answer": answer,
                "matched_source_title": blog_row.get("title"),
                "matched_source_url": blog_row.get("url"),
            }
            out.write(json.dumps(trace, ensure_ascii=False) + "\n")
            out.flush()

    print(f"\nWrote traces -> {out_path}")


if __name__ == "__main__":
    main()
