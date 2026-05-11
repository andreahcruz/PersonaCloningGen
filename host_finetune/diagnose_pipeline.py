"""Pipeline diagnosis for the Lemkin FT collapse.

Runs (any subset of) three tests and prints results so we can localize where the
collapse is introduced:

    A. Base unsloth bnb-4bit -> generate via Transformers (no LoRA, no merge)
       Sanity: is the bnb-4bit base itself producing junk on this Blackwell+cu128 stack?

    B. Merged HF (16-bit) -> generate via Transformers (output/lemkin-clone)
       Was the merge step broken, independent of GGUF / Ollama?

    C. Q4_K_M GGUF -> generate via llama-cli.exe (output/lemkin-clone_gguf/...)
       Is the GGUF itself broken, independent of Ollama / Modelfile?

Decision matrix:
    A garbage              -> Unsloth bnb-4bit -> generate is broken on this stack
    A clean, B garbage     -> merge step (dequant -> +LoRA -> save_pretrained) is broken
    B clean, C garbage     -> llama.cpp GGUF conversion is broken
    B clean, C clean       -> Ollama Modelfile / template wrap is the bug

Run:
    python -m host_finetune.diagnose_pipeline --tests C       # cheapest first
    python -m host_finetune.diagnose_pipeline --tests C,B,A
"""
from __future__ import annotations

import argparse
import io
import os
import subprocess
import sys
import textwrap
import time
from pathlib import Path

# Force UTF-8 on stdout/stderr so Windows cp1252 doesn't choke on the rare-token
# Unicode shards we're trying to diagnose.
for _stream_name in ("stdout", "stderr"):
    _stream = getattr(sys, _stream_name, None)
    if _stream is not None and hasattr(_stream, "reconfigure"):
        try:
            _stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
    elif _stream is not None and hasattr(_stream, "buffer"):
        try:
            setattr(
                sys,
                _stream_name,
                io.TextIOWrapper(_stream.buffer, encoding="utf-8", errors="replace"),
            )
        except Exception:
            pass

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

MERGED_HF_DIR = REPO_ROOT / "host_finetune" / "output" / "lemkin-clone"
GGUF_PATH = Path(
    os.environ.get(
        "GGUF_PATH",
        str(
            REPO_ROOT
            / "host_finetune"
            / "output"
            / "lemkin-clone_gguf"
            / "Meta-Llama-3.1-8B-Instruct.Q4_K_M.gguf"
        ),
    )
)
LLAMA_CLI = Path(
    os.environ.get(
        "LLAMA_CLI",
        r"C:\Users\okevi\.unsloth\llama.cpp\build\bin\Release\llama-cli.exe",
    )
)

PROMPTS = [
    "Give one short SaaS growth tip.",
    "Write three sentences about why SaaS margins matter.",
]

BANNER = "=" * 72


def _hdr(label: str) -> None:
    print()
    print(BANNER)
    print(f"  {label}")
    print(BANNER, flush=True)


def _build_chat_prefix(user_text: str) -> str:
    """Llama 3.1 Instruct chat template with default system block."""
    return (
        "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
        "Cutting Knowledge Date: December 2023\nToday Date: 26 Jul 2024\n\n"
        "<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n"
        f"{user_text}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
    )


def test_c_llama_cli() -> None:
    """C: run llama-cli.exe directly on the Q4_K_M GGUF (no Ollama / Modelfile)."""
    _hdr("TEST C: llama-cli.exe -> Q4_K_M GGUF (no Ollama, no Modelfile)")
    if not LLAMA_CLI.is_file():
        print(f"SKIP: llama-cli.exe not at {LLAMA_CLI}")
        return
    if not GGUF_PATH.is_file():
        print(f"SKIP: GGUF not at {GGUF_PATH}")
        return
    print(f"llama-cli: {LLAMA_CLI}")
    print(f"GGUF:      {GGUF_PATH}  ({GGUF_PATH.stat().st_size / 1e9:.2f} GB)")

    for prompt in PROMPTS:
        prefix = _build_chat_prefix(prompt)
        print(f"\n--- prompt: {prompt!r} ---", flush=True)
        cmd = [
            str(LLAMA_CLI),
            "--model", str(GGUF_PATH),
            "-ngl", "99",
            "--temp", "0.35",
            "--top-p", "0.9",
            "-n", "200",
            "--no-display-prompt",
            "--single-turn",
            "-no-cnv",
            "-p", prefix,
        ]
        t0 = time.time()
        try:
            proc = subprocess.run(
                cmd,
                check=False,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=300,
            )
        except subprocess.TimeoutExpired:
            print("  TIMEOUT after 300s")
            continue
        dt = time.time() - t0
        out = (proc.stdout or "").strip()
        print(f"  [elapsed {dt:.1f}s, exit={proc.returncode}]")
        if not out:
            print("  (no stdout) stderr tail:")
            print(textwrap.indent((proc.stderr or "")[-1500:], "  | "))
        else:
            print(textwrap.indent(out[:1500], "  > "))
            if len(out) > 1500:
                print(f"  ... [truncated {len(out) - 1500} more chars]")


def test_b_hf_generate() -> None:
    """B: load merged HF (bf16 + CPU offload) and run model.generate.

    bnb-4bit on this stack is broken ("Skipping import of cpp extensions due to
    incompatible torch version"), so we load in plain bf16 with device_map='auto',
    which spills the tail of the model to CPU (slower but reliable on 16 GB VRAM).
    """
    _hdr("TEST B: Transformers -> merged HF model (bf16 + CPU offload, pre-GGUF)")
    if not MERGED_HF_DIR.is_dir():
        print(f"SKIP: merged HF dir missing at {MERGED_HF_DIR}")
        return
    print(f"HF dir: {MERGED_HF_DIR}")

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(str(MERGED_HF_DIR))
    offload_dir = REPO_ROOT / "host_finetune" / "_offload_tmp"
    offload_dir.mkdir(parents=True, exist_ok=True)
    max_memory = {0: "13GiB", "cpu": "48GiB"}
    print(
        "Loading merged model in bf16 with explicit CPU offload "
        f"(max_memory={max_memory}, offload_folder={offload_dir})...",
        flush=True,
    )
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        str(MERGED_HF_DIR),
        torch_dtype=torch.bfloat16,
        device_map="auto",
        max_memory=max_memory,
        offload_folder=str(offload_dir),
        low_cpu_mem_usage=True,
    )
    model.eval()
    print(f"  loaded in {time.time() - t0:.1f}s")
    print(f"  eos_token_id={tok.eos_token_id} bos_token_id={tok.bos_token_id}")
    try:
        from collections import Counter
        dev_count = Counter(str(p.device) for p in model.parameters())
        print(f"  device map: {dict(dev_count)}")
    except Exception:
        pass

    for prompt in PROMPTS:
        print(f"\n--- prompt: {prompt!r} ---", flush=True)
        msgs = [{"role": "user", "content": prompt}]
        # Render to string then tokenize separately (apply_chat_template returns
        # BatchEncoding in newer transformers; .shape on it raises AttributeError).
        prompt_str = tok.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True
        )
        enc = tok(prompt_str, return_tensors="pt")
        input_ids = enc["input_ids"].to(model.device)
        attn = enc.get("attention_mask")
        if attn is not None:
            attn = attn.to(model.device)
        t0 = time.time()
        with torch.inference_mode():
            out = model.generate(
                input_ids=input_ids,
                attention_mask=attn,
                max_new_tokens=60,
                do_sample=True,
                temperature=0.35,
                top_p=0.9,
                pad_token_id=tok.eos_token_id,
            )
        dt = time.time() - t0
        new = tok.decode(out[0][input_ids.shape[1]:], skip_special_tokens=True).strip()
        print(f"  [elapsed {dt:.1f}s]")
        print(textwrap.indent(new[:1500], "  > "))

    del model
    torch.cuda.empty_cache()


def test_a_base_only() -> None:
    """A: base unsloth bnb-4bit (no LoRA, no merge) -> generate (sanity)."""
    _hdr("TEST A: base unsloth bnb-4bit, NO LoRA -> generate (sanity)")
    from unsloth import FastLanguageModel

    base = os.environ.get(
        "HF_MODEL_NAME", "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit"
    )
    print(f"Loading base: {base}", flush=True)
    t0 = time.time()
    model, tok = FastLanguageModel.from_pretrained(
        model_name=base,
        max_seq_length=2048,
        dtype=None,
        load_in_4bit=True,
    )
    FastLanguageModel.for_inference(model)
    print(f"  loaded in {time.time() - t0:.1f}s")

    import torch

    for prompt in PROMPTS:
        print(f"\n--- prompt: {prompt!r} ---", flush=True)
        msgs = [{"role": "user", "content": prompt}]
        ids = tok.apply_chat_template(
            msgs, return_tensors="pt", add_generation_prompt=True
        ).to(model.device)
        t0 = time.time()
        with torch.inference_mode():
            out = model.generate(
                ids,
                max_new_tokens=200,
                do_sample=True,
                temperature=0.35,
                top_p=0.9,
                pad_token_id=tok.eos_token_id,
            )
        dt = time.time() - t0
        new = tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True).strip()
        print(f"  [elapsed {dt:.1f}s]")
        print(textwrap.indent(new[:1500], "  > "))

    del model
    torch.cuda.empty_cache()


def test_d_base_plus_lora_inference() -> None:
    """D: load base bnb-4bit + LoRA adapter (no merge) and generate.

    Distinguishes 'LoRA application is the bug' from 'merge-save-to-disk is the bug'.
    """
    _hdr("TEST D: base bnb-4bit + LoRA adapter, NO MERGE -> generate")
    from unsloth import FastLanguageModel

    adapter_dir = REPO_ROOT / "host_finetune" / "output" / "lemkin_lora_stash"
    if not adapter_dir.is_dir():
        adapter_dir = REPO_ROOT / "host_finetune" / "output" / "lemkin_lora"
    print(f"Adapter dir: {adapter_dir}")
    if not (adapter_dir / "adapter_config.json").is_file():
        print(f"SKIP: no adapter_config.json under {adapter_dir}")
        return

    print("Loading base + adapter via Unsloth (load_in_4bit=True)...", flush=True)
    t0 = time.time()
    model, tok = FastLanguageModel.from_pretrained(
        model_name=str(adapter_dir),
        max_seq_length=2048,
        dtype=None,
        load_in_4bit=True,
    )
    FastLanguageModel.for_inference(model)
    print(f"  loaded in {time.time() - t0:.1f}s")

    import torch

    for prompt in PROMPTS:
        print(f"\n--- prompt: {prompt!r} ---", flush=True)
        msgs = [{"role": "user", "content": prompt}]
        prompt_str = tok.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True
        )
        enc = tok(prompt_str, return_tensors="pt")
        input_ids = enc["input_ids"].to(model.device)
        attn = enc.get("attention_mask")
        if attn is not None:
            attn = attn.to(model.device)
        t0 = time.time()
        with torch.inference_mode():
            out = model.generate(
                input_ids=input_ids,
                attention_mask=attn,
                max_new_tokens=80,
                do_sample=True,
                temperature=0.35,
                top_p=0.9,
                pad_token_id=tok.eos_token_id,
            )
        dt = time.time() - t0
        new = tok.decode(out[0][input_ids.shape[1]:], skip_special_tokens=True).strip()
        print(f"  [elapsed {dt:.1f}s]")
        print(textwrap.indent(new[:1500], "  > "))

    del model
    torch.cuda.empty_cache()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--tests",
        default="C",
        help="Comma list from A,B,C,D. Default 'C' (llama-cli on GGUF, cheapest).",
    )
    args = p.parse_args()
    requested = [t.strip().upper() for t in args.tests.split(",") if t.strip()]
    seen: set[str] = set()
    ordered = []
    for t in requested:
        if t in {"A", "B", "C", "D"} and t not in seen:
            seen.add(t)
            ordered.append(t)

    for t in ordered:
        if t == "C":
            test_c_llama_cli()
        elif t == "B":
            test_b_hf_generate()
        elif t == "A":
            test_a_base_only()
        elif t == "D":
            test_d_base_plus_lora_inference()

    _hdr("DONE")


if __name__ == "__main__":
    main()
