#!/usr/bin/env python3
"""
Prepare a small, high-quality OpenAI supervised fine-tuning dataset for the
Jason Lemkin repo.

This first-pass exporter intentionally uses blog posts only. The blog corpus is
the cleanest long-form source in this repo and maps best to the existing
`blog_draft` prompt structure used by the root RAG generator.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from generate import build_prompt, load_format_spec, load_persona, summarize_persona


DEFAULT_AUDIENCE = "B2B SaaS founders and operators"
DEFAULT_GOAL = "Teach one concrete operator lesson with specific examples and conviction."
DEFAULT_CTA = "none"
DEFAULT_FORMAT = "blog_draft"
DEFAULT_EPOCHS = 3
DEFAULT_TRAINING_RATE_PER_MILLION = 1.50


@dataclass
class Example:
    title: str
    date: str
    url: str
    text: str
    messages: list[dict]
    estimated_tokens: int


def _normalize_paragraphs(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace("\u00a0", " ")
    text = re.sub(r"(?i)\n?related posts.*$", "", text, flags=re.DOTALL)
    text = re.sub(r"https?://pic\.twitter\.com/\S+", "", text)
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    lines = [line.strip() for line in text.split("\n")]
    text = "\n".join(lines).strip()
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text


def _word_count(text: str) -> int:
    return len(text.split())


def _paragraphs(text: str) -> list[str]:
    parts = [p.strip() for p in re.split(r"\n{2,}", text) if p.strip()]
    if parts:
        return parts
    return [text.strip()] if text.strip() else []


def _context_from_text(text: str, max_paragraphs: int = 3, max_chars_each: int = 450) -> str:
    paras = _paragraphs(text)
    chosen = paras[:max_paragraphs]
    lines = ["[CONTEXT — EXCERPTS FROM JASON LEMKIN CORPUS]"]
    for idx, para in enumerate(chosen, start=1):
        snippet = para[:max_chars_each].strip()
        if len(para) > max_chars_each:
            snippet += "..."
        lines.append(f"[Source excerpt {idx}]")
        lines.append(snippet)
    return "\n".join(lines)


def _estimate_tokens_from_text(text: str) -> int:
    try:
        import tiktoken
        encoding = tiktoken.get_encoding("cl100k_base")
        return len(encoding.encode(text))
    except Exception:
        return max(1, int(len(text) / 4))


def _estimate_tokens_for_messages(messages: list[dict]) -> int:
    serialized = json.dumps({"messages": messages}, ensure_ascii=False, separators=(",", ":"))
    return _estimate_tokens_from_text(serialized)


def _build_messages(
    *,
    title: str,
    body: str,
    persona_summary: str,
    format_spec: dict,
) -> list[dict]:
    prompt = build_prompt(
        format_spec=format_spec,
        persona_summary=persona_summary,
        topic=title,
        audience=DEFAULT_AUDIENCE,
        goal=DEFAULT_GOAL,
        cta=DEFAULT_CTA,
        context_block=_context_from_text(body),
    )
    system = (
        "Follow the user prompt exactly. Use the provided persona and context to write a grounded "
        "draft in Jason Lemkin's operator voice. Do not invent facts outside the provided material."
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": body},
    ]


def _load_blog_examples(
    path: Path,
    *,
    persona_summary: str,
    format_spec: dict,
    min_words: int,
    max_words: int,
) -> list[Example]:
    examples: list[Example] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            title = (row.get("title") or "").strip()
            body = _normalize_paragraphs((row.get("content") or row.get("text") or "").strip())
            if not title or not body:
                continue
            wc = _word_count(body)
            if wc < min_words or wc > max_words:
                continue
            if body.count("Related Posts") > 0:
                continue
            messages = _build_messages(
                title=title,
                body=body,
                persona_summary=persona_summary,
                format_spec=format_spec,
            )
            examples.append(
                Example(
                    title=title,
                    date=str(row.get("date") or ""),
                    url=str(row.get("url") or ""),
                    text=body,
                    messages=messages,
                    estimated_tokens=_estimate_tokens_for_messages(messages),
                )
            )
    return examples


def _write_jsonl(path: Path, examples: list[Example]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps({"messages": ex.messages}, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare a small OpenAI nano fine-tuning dataset.")
    parser.add_argument("--input", default="data/jasonlemkin_blog.jsonl", help="Input blog JSONL file.")
    parser.add_argument("--persona", default="data/persona_profile.json", help="Persona profile JSON.")
    parser.add_argument("--format-specs", default="formats/format_specs.yaml", help="Format specs YAML.")
    parser.add_argument("--format-name", default=DEFAULT_FORMAT, help="Format spec to mirror in the training prompts.")
    parser.add_argument("--out-dir", default="data/openai_ft", help="Output directory.")
    parser.add_argument("--max-examples", type=int, default=100, help="Maximum examples to export.")
    parser.add_argument("--eval-fraction", type=float, default=0.1, help="Validation split fraction.")
    parser.add_argument("--min-words", type=int, default=550, help="Minimum blog length.")
    parser.add_argument("--max-words", type=int, default=1100, help="Maximum blog length.")
    parser.add_argument("--seed", type=int, default=42, help="Shuffle seed.")
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS, help="Epochs for cost estimate only.")
    parser.add_argument(
        "--training-rate-per-million",
        type=float,
        default=DEFAULT_TRAINING_RATE_PER_MILLION,
        help="Estimated training price per 1M tokens for cost reporting.",
    )
    args = parser.parse_args()

    input_path = REPO_ROOT / args.input
    persona_path = REPO_ROOT / args.persona
    format_specs_path = REPO_ROOT / args.format_specs
    out_dir = REPO_ROOT / args.out_dir

    if not input_path.exists():
        raise SystemExit(f"Input file not found: {input_path}")
    if not persona_path.exists():
        raise SystemExit(f"Persona profile not found: {persona_path}")
    if not format_specs_path.exists():
        raise SystemExit(f"Format specs not found: {format_specs_path}")

    profile = load_persona(persona_path)
    persona_summary = summarize_persona(profile)
    format_spec = load_format_spec(format_specs_path, args.format_name)

    examples = _load_blog_examples(
        input_path,
        persona_summary=persona_summary,
        format_spec=format_spec,
        min_words=args.min_words,
        max_words=args.max_words,
    )
    if len(examples) < 20:
        raise SystemExit(f"Only found {len(examples)} candidate examples after filtering; widen the filters.")

    rng = random.Random(args.seed)
    rng.shuffle(examples)
    selected = examples[: min(args.max_examples, len(examples))]

    eval_count = max(1, int(round(len(selected) * args.eval_fraction)))
    train = selected[:-eval_count]
    eval_set = selected[-eval_count:]

    train_path = out_dir / "lemkin_blog_nano_train.jsonl"
    eval_path = out_dir / "lemkin_blog_nano_eval.jsonl"
    _write_jsonl(train_path, train)
    _write_jsonl(eval_path, eval_set)

    train_tokens = sum(ex.estimated_tokens for ex in train)
    eval_tokens = sum(ex.estimated_tokens for ex in eval_set)
    est_cost = (train_tokens * args.epochs / 1_000_000) * args.training_rate_per_million

    manifest = {
        "input": str(input_path),
        "format_name": args.format_name,
        "selected_examples": len(selected),
        "train_examples": len(train),
        "eval_examples": len(eval_set),
        "train_estimated_tokens": train_tokens,
        "eval_estimated_tokens": eval_tokens,
        "epochs_for_estimate": args.epochs,
        "training_rate_per_million": args.training_rate_per_million,
        "estimated_training_cost": round(est_cost, 4),
        "seed": args.seed,
        "filters": {
            "min_words": args.min_words,
            "max_words": args.max_words,
            "source": "blog-only",
        },
        "sample_titles": [ex.title for ex in train[:5]],
    }
    manifest_path = out_dir / "lemkin_blog_nano_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Wrote train dataset -> {train_path}")
    print(f"Wrote eval dataset  -> {eval_path}")
    print(f"Wrote manifest      -> {manifest_path}")
    print(
        f"Train examples={len(train)} eval examples={len(eval_set)} "
        f"train_tokens~{train_tokens} eval_tokens~{eval_tokens}"
    )
    print(
        f"Estimated training cost at ${args.training_rate_per_million}/1M over {args.epochs} epoch(s): "
        f"${est_cost:.2f}"
    )


if __name__ == "__main__":
    main()
