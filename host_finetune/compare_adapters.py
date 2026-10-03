"""Generate gold topics in each trained medium and score the answers.

Prompts use the same medium lines as training. Expected facts stay on the
row for scoring and are not placed in the prompt.

Run from the repo root with the fine-tune venv::

    host_finetune\\.venv\\Scripts\\python.exe -m host_finetune.compare_adapters
"""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

from host_finetune.eval_gold_rag import aggregate, load_format_specs, score_row

ROOT = Path(__file__).resolve().parents[1]
GOLD = ROOT / "data" / "openai_ft" / "lemkin_gold_eval.jsonl"
SPECS = ROOT / "formats" / "format_specs.yaml"
DEFAULT_OUT = ROOT / "experiments" / "EXP-20261001-007-mixed-medium-compare"
ADAPTERS = (
    ("cleaned", ROOT / "host_finetune" / "output" / "lemkin_lora_cleaned"),
    ("balanced", ROOT / "host_finetune" / "output" / "lemkin_lora_balanced"),
    ("r32", ROOT / "host_finetune" / "output" / "lemkin_lora_r32"),
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
TEMPERATURE = 0.7
TOP_P = 0.9
REPETITION_PENALTY = 1.15


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


def generate(adapter: Path, gold: list[dict]) -> list[dict]:
    import torch
    from unsloth import FastLanguageModel

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=str(adapter),
        max_seq_length=MAX_SEQ_LENGTH,
        dtype=None,
        load_in_4bit=True,
    )
    FastLanguageModel.for_inference(model)
    traces = []
    for row in gold:
        for medium in MEDIUMS:
            instruction = prompt_for(medium, row["topic"])
            prompt = tokenizer.apply_chat_template(
                [{"role": "user", "content": instruction}],
                tokenize=False,
                add_generation_prompt=True,
            )
            encoded = tokenizer(prompt, return_tensors="pt")
            input_ids = encoded["input_ids"].to(model.device)
            attention = encoded.get("attention_mask")
            if attention is not None:
                attention = attention.to(model.device)
            with torch.inference_mode():
                output = model.generate(
                    input_ids,
                    attention_mask=attention,
                    max_new_tokens=medium["max_new_tokens"],
                    do_sample=True,
                    temperature=TEMPERATURE,
                    top_p=TOP_P,
                    repetition_penalty=REPETITION_PENALTY,
                    pad_token_id=tokenizer.eos_token_id,
                )
            answer = tokenizer.decode(
                output[0][input_ids.shape[1]:], skip_special_tokens=True
            ).strip()
            traces.append({
                **row,
                "format": medium["format"],
                "medium": medium["medium"],
                "prompt": instruction,
                "answer": answer,
                "query": row["topic"],
                "max_new_tokens": medium["max_new_tokens"],
            })
            print(
                f"  {row['id']} {medium['medium']} words={len(answer.split())}",
                flush=True,
            )
    del model
    torch.cuda.empty_cache()
    return traces


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


def score(traces: list[dict], specs: dict) -> dict:
    scored = [score_row(row, specs) for row in traces]
    summary = aggregate(scored)
    summary.update(_paired(traces))
    by_medium = {}
    for medium in MEDIUMS:
        subset = [row for row in traces if row.get("medium") == medium["medium"]]
        if not subset:
            continue
        medium_summary = aggregate([score_row(row, specs) for row in subset])
        medium_summary.update(_paired(subset))
        by_medium[medium["medium"]] = medium_summary
    summary["by_medium"] = by_medium
    return {"summary": summary, "rows": scored}


def generate_ollama(model_name: str, gold: list[dict], base_url: str) -> list[dict]:
    """Same prompts and sampling settings, through Ollama's chat API."""
    import requests

    traces = []
    for row in gold:
        for medium in MEDIUMS:
            instruction = prompt_for(medium, row["topic"])
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
                    },
                },
                timeout=600,
            )
            response.raise_for_status()
            answer = response.json()["message"]["content"].strip()
            traces.append({
                **row,
                "format": medium["format"],
                "medium": medium["medium"],
                "prompt": instruction,
                "answer": answer,
                "query": row["topic"],
                "max_new_tokens": medium["max_new_tokens"],
                "backend": "ollama",
                "ollama_model": model_name,
            })
            print(
                f"  {row['id']} {medium['medium']} words={len(answer.split())}",
                flush=True,
            )
    return traces


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
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
    parser.add_argument("--ollama-base", default="http://localhost:11434")
    return parser.parse_args()


def adapter_list(raw: str) -> tuple[tuple[str, Path], ...]:
    if raw.strip().lower() == "none":
        return ()
    if not raw.strip():
        return ADAPTERS
    found = []
    for item in raw.split(","):
        name, path = item.split("=", 1)
        found.append((name.strip(), Path(path.strip())))
    return tuple(found)


def write_parameters(out: Path) -> None:
    payload = {
        "experiment_id": out.name,
        "gold_eval": "data/openai_ft/lemkin_gold_eval.jsonl",
        "n_topics": 30,
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
            "load_in_4bit": True,
        },
        "expected_facts_in_prompt": False,
    }
    (out / "config").mkdir(parents=True, exist_ok=True)
    (out / "config" / "run_parameters.json").write_text(
        json.dumps(payload, indent=2) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    gold = load_gold()
    specs = load_format_specs(SPECS)
    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    (out / "raw").mkdir(exist_ok=True)
    (out / "metrics").mkdir(exist_ok=True)
    write_parameters(out)
    combined = {}
    for name, adapter in adapter_list(args.adapters):
        print(f"=== {name} {adapter}", flush=True)
        traces = generate(adapter, gold)
        (out / "raw" / f"{name}.jsonl").write_text(
            "\n".join(json.dumps(row, ensure_ascii=False) for row in traces) + "\n",
            encoding="utf-8",
        )
        result = score(traces, specs)
        (out / "metrics" / f"{name}.json").write_text(
            json.dumps(result["summary"], indent=2) + "\n",
            encoding="utf-8",
        )
        combined[name] = result["summary"]
        print(json.dumps(result["summary"], indent=2), flush=True)
    for model_name in [item.strip() for item in args.ollama.split(",") if item.strip()]:
        label = model_name.replace(":", "_")
        print(f"=== {label} ollama:{model_name}", flush=True)
        traces = generate_ollama(model_name, gold, args.ollama_base)
        (out / "raw" / f"{label}.jsonl").write_text(
            "\n".join(json.dumps(row, ensure_ascii=False) for row in traces) + "\n",
            encoding="utf-8",
        )
        result = score(traces, specs)
        (out / "metrics" / f"{label}.json").write_text(
            json.dumps(result["summary"], indent=2) + "\n",
            encoding="utf-8",
        )
        combined[label] = result["summary"]
        print(json.dumps(result["summary"], indent=2), flush=True)
    (out / "metrics" / "summary.json").write_text(
        json.dumps(combined, indent=2) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
