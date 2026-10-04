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
from host_finetune.generation_diagnostics import stop_metadata, summarize_stops
from host_finetune.relabel_continuations import sentence_final
from host_finetune.train_only_index import (
    DEFAULT_PERSIST_DIR,
    DEFAULT_STYLE_PATH,
    TRAIN_ONLY_COLLECTION,
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


def comparison_instruction(
    medium: dict,
    topic: str,
    excerpts: list[str] | None = None,
    style_exemplars: list[str] | None = None,
) -> str:
    """Retrieval-off prompts stay identical to ``prompt_for``.

    A non-empty excerpt list appends a factual block. A non-empty style list
    appends a voice block. Expected facts are never inserted by this function.
    Empty lists leave the one-line instruction unchanged.
    """
    instruction = prompt_for(medium, topic)
    blocks = [
        block
        for block in (
            format_voice_exemplars(style_exemplars),
            format_factual_excerpts(excerpts),
        )
        if block
    ]
    if not blocks:
        return instruction
    return instruction + "\n\n" + "\n\n".join(blocks)


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
    return {
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
        "use_for_selection": ["early_eot_rate", "mid_sentence_stop_rate"],
        "logged_not_for_selection": ["paired_bleu", "paired_rouge_l"],
    }


def generate(adapter: Path, gold: list[dict], retriever=None, style_for=None, seed_base: int = GENERATION_SEED) -> list[dict]:
    import torch
    from unsloth import FastLanguageModel
    from transformers import set_seed

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=str(adapter),
        max_seq_length=MAX_SEQ_LENGTH,
        dtype=None,
        load_in_4bit=True,
    )
    FastLanguageModel.for_inference(model)
    traces = []
    for topic_index, row in enumerate(gold):
        for medium_index, medium in enumerate(MEDIUMS):
            seed = seed_base + topic_index * len(MEDIUMS) + medium_index
            set_seed(seed)
            instruction, payload = instruction_for_request(
                medium, row["topic"], retriever, style_for
            )
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
            generated_ids = output[0][input_ids.shape[1]:].tolist()
            answer = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
            stopping = stop_metadata(generated_ids, model.generation_config.eos_token_id,
                                     medium['max_new_tokens'])
            traces.append({
                **row,
                "format": medium["format"],
                "medium": medium["medium"],
                "prompt": instruction,
                "answer": answer,
                "query": row["topic"],
                "max_new_tokens": medium["max_new_tokens"],
                "seed": seed,
                "prompt_tokens": input_ids.shape[1],
                "backend": "unsloth",
                **retrieval_trace_fields(payload, answer, retriever is not None),
                **stopping,
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
    summary['stopping'] = summarize_stops(traces)
    summary['decision'] = decision_metrics(traces)
    summary['selection_warning'] = (
        'Legacy lexical/format scores are not validated voice metrics. X is prompted as a '
        'single post but legacy format scoring expects a thread; other format constraints '
        'are not all requested in these prompts. Do not select an adapter by overall score. '
        'Early EOS alone is not an incomplete answer. Gold topics are a development set.'
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
    return {"summary": summary, "rows": scored}


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
    parser.add_argument(
        "--retrieval",
        action="store_true",
        help="Append factual excerpts from lemkin_train_only. Default is off.",
    )
    parser.add_argument("--collection", default=TRAIN_ONLY_COLLECTION)
    parser.add_argument("--chroma-path", type=Path, default=DEFAULT_PERSIST_DIR)
    parser.add_argument("--k", type=int, default=4)
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
    return parser.parse_args()


def adapter_list(raw: str) -> tuple[tuple[str, Path], ...]:
    if raw.strip().lower() == "none":
        return ()
    if raw.strip() == "repaired-vs-relabel":
        return RELABEL_PAIR
    if not raw.strip():
        return ADAPTERS
    found = []
    for item in raw.split(","):
        name, path = item.split("=", 1)
        found.append((name.strip(), Path(path.strip())))
    return tuple(found)


def write_parameters(out: Path, retrieval: bool = False, fixed_style: bool = False, seed_base: int = GENERATION_SEED) -> None:
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
            "seed_base": seed_base,
            "seed_policy": "base + topic_index * n_mediums + medium_index; reset per request",
        },
        "expected_facts_in_prompt": False,
        "retrieval": {
            "enabled": retrieval,
            "collection": TRAIN_ONLY_COLLECTION if retrieval else None,
            "medium_filter": retrieval,
            "note": "Default off. The repaired versus relabel run uses the EXP-010 prompts and budgets.",
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


def build_runtime_retriever(args: argparse.Namespace):
    """CPU retrieval against lemkin_train_only. Used only when --retrieval is set."""
    import chromadb
    from chromadb.config import Settings

    from host_finetune.rebuild_chroma import _embed_batch
    from host_finetune.train_only_index import assert_collection_allowed

    assert_collection_allowed(args.collection)
    if not args.chroma_path.exists():
        raise SystemExit(
            f"train-only index not found at {args.chroma_path}. "
            "Build it with: python -m host_finetune.rebuild_chroma --train-only"
        )
    client = chromadb.PersistentClient(
        path=str(args.chroma_path), settings=Settings(anonymized_telemetry=False)
    )
    collection = client.get_collection(args.collection)

    def retriever(topic: str, comparison_medium: str) -> dict:
        where = medium_where(comparison_medium)
        vectors = _embed_batch([topic], args.ollama_base, args.embed_model, cpu=True)
        hits = query_excerpts(collection, vectors[0], args.k, where=where)
        return {"query": topic, "filter": where, "k": args.k, "hits": hits}

    return retriever


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


def main() -> None:
    args = parse_args()
    gold = load_gold()
    specs = load_format_specs(SPECS)
    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    (out / "raw").mkdir(exist_ok=True)
    (out / "metrics").mkdir(exist_ok=True)
    write_parameters(
        out, retrieval=args.retrieval, fixed_style=args.fixed_style, seed_base=args.seed_base
    )
    retriever = build_runtime_retriever(args) if args.retrieval else None
    style_for = load_style_for(args.style_path) if args.fixed_style else None
    combined = {}
    traces_by_name: dict[str, list[dict]] = {}
    for name, adapter in adapter_list(args.adapters):
        print(f"=== {name} {adapter}", flush=True)
        traces = generate(
            adapter, gold, retriever=retriever, style_for=style_for, seed_base=args.seed_base
        )
        traces_by_name[name] = traces
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
        traces = generate_ollama(
            model_name, gold, args.ollama_base, retriever=retriever, style_for=style_for,
            seed_base=args.seed_base,
        )
        traces_by_name[label] = traces
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
    if args.retrieval:
        review = []
        for name, traces in traces_by_name.items():
            review.extend(evidence_review_rows(traces, name))
        _write_jsonl(out / "raw" / "evidence_review.jsonl", review)
    (out / "metrics" / "summary.json").write_text(
        json.dumps(combined, indent=2) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
