"""Match-training smoke test against Ollama (Lemkin SFT GGUF).

Training now uses **Llama 3 Instruct** ``apply_chat_template`` (see ``finetune.py``).
The bundled ``Modelfile`` wraps ``{{ .Prompt }}`` in the same control tokens.

Mode ``modelfile`` sends the **user** message string only (instruction, or
instruction + ``\\n\\n`` + input when the row has Input) — Ollama fills the template.

Mode ``full_prefix`` bypasses the Modelfile and POSTs the exact **generation-prefix**
string from ``tokenizer.apply_chat_template(..., add_generation_prompt=True)``,
for debugging template drift.

Examples:
    python -m host_finetune.smoke_lemkin_infer --model lemkin-lora-v3
    python -m host_finetune.smoke_lemkin_infer --model lemkin-lora-v3 --mode full_prefix
    python -m host_finetune.smoke_lemkin_infer --model lemkin-clone --row 42
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from host_finetune.config import DATASET_LOCAL
from host_finetune.llama_chat_format import (
    inference_prompt_from_user_text,
    load_tokenizer_for_prompt_build,
    user_message_from_alpaca_row,
)


def _load_instruction_row(dataset_path: Path, row_index: int) -> dict:
    if not dataset_path.is_file():
        raise FileNotFoundError(f"No dataset at {dataset_path}")
    idx = 0
    with dataset_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            out = (obj.get("output") or "").strip()
            if not out:
                continue
            if idx == row_index:
                return obj
            idx += 1
    raise IndexError(f"Row index {row_index} not found (non-empty outputs only)")


def _build_prompts(example: dict) -> tuple[str, str]:
    """(modelfile_prompt, full_prefix_prompt)."""
    inst = example.get("instruction") or ""
    inp = example.get("input") or ""
    user_msg = user_message_from_alpaca_row(str(inst), str(inp))
    modelfile_p = user_msg
    tok = load_tokenizer_for_prompt_build()
    full_prefix = inference_prompt_from_user_text(tok, user_msg)
    return modelfile_p, full_prefix


def _ollama_generate(
    base: str,
    model: str,
    prompt: str,
    *,
    temperature: float,
    top_p: float,
    num_predict: int,
) -> str:
    url = base.rstrip("/") + "/api/generate"
    body = json.dumps(
        {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
                "top_p": top_p,
                "repeat_penalty": 1.1,
                "num_predict": num_predict,
            },
        }
    ).encode("utf-8")
    req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    return (data.get("response") or "").strip()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default=os.environ.get("OLLAMA_MODEL_NAME", "lemkin-clone"))
    p.add_argument(
        "--mode",
        choices=("modelfile", "full_prefix"),
        default="modelfile",
        help="modelfile = user message only (Ollama TEMPLATE). "
        "full_prefix = full Llama-3 generation prefix from HF tokenizer (no Modelfile).",
    )
    p.add_argument("--row", type=int, default=0, help="N-th nonempty dataset row")
    p.add_argument("--dataset", type=Path, default=DATASET_LOCAL)
    p.add_argument(
        "--instruction",
        default=None,
        help='Override dataset row entirely (still uses mode to choose wrapping). Ignores --row.',
    )
    p.add_argument("--ollama-base", default=os.environ.get("OLLAMA_BASE", "http://127.0.0.1:11434"))
    p.add_argument("--temperature", type=float, default=0.35)
    p.add_argument("--top-p", type=float, dest="top_p", default=0.9)
    p.add_argument("--max-tokens", type=int, dest="num_predict", default=320)
    args = p.parse_args()

    if args.instruction is not None:
        example = {
            "instruction": args.instruction,
            "input": "",
            "output": "",
        }
    else:
        example = _load_instruction_row(args.dataset, args.row)

    mf_p, fp_p = _build_prompts(example)
    prompt = mf_p if args.mode == "modelfile" else fp_p

    print(f"Dataset: {args.dataset}")
    print(f"Model: {args.model} @ {args.ollama_base}")
    print(f"Mode: {args.mode}")
    print(f"Temperature={args.temperature} top_p={args.top_p} num_predict={args.num_predict}")
    print()
    print("=== Prompt sent to /api/generate ===")
    print(prompt[:2500] + ("…" if len(prompt) > 2500 else ""))
    print("=== End prompt ===")
    print()

    try:
        text = _ollama_generate(
            args.ollama_base,
            args.model,
            prompt,
            temperature=args.temperature,
            top_p=args.top_p,
            num_predict=args.num_predict,
        )
    except urllib.error.URLError as e:
        print(f"Ollama request failed: {e}", file=sys.stderr)
        sys.exit(1)

    print("=== Model output ===")
    print(text)
    print("===================")


if __name__ == "__main__":
    main()
