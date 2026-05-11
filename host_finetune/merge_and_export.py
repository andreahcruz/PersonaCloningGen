"""Merge the LoRA adapter into the base model and export GGUF for Ollama.

Loads ``host_finetune/output/lemkin_lora/`` on top of ``MERGE_BASE_MODEL`` (the
**un-quantized** Llama 3.1 8B Instruct by default) and writes:

    host_finetune/output/lemkin-clone/                 # merged bf16 HF model
    host_finetune/output/lemkin-clone_gguf/*.gguf      # Q4_K_M GGUF (default)

**Why not Unsloth's ``save_pretrained_gguf``?**

On the user's stack (torch 2.10.0+cu128 + Blackwell RTX 5070 Ti +
``bitsandbytes`` whose C++ extensions can't load — you'll see
``Skipping import of cpp extensions due to incompatible torch version``),
Unsloth's GGUF export calls ``bitsandbytes.functional.dequantize_4bit`` to
materialize bf16 weights from the bnb-4bit base. Without the CUDA kernel, bnb
falls back to a Python path that produces **numerically wrong but
statistically plausible** floats. The merged model on disk looks healthy
(no NaN/Inf, sensible weight stats) but generates pure garbage when used
(``vibe vibe vibe ...``). Loss curves during training look normal because
training uses a different bnb code path (CUDA matmul kernel) that works.

To prove this, ``host_finetune/diagnose_pipeline.py`` runs four tests and shows:
  • Base bnb-4bit (no LoRA) generates clean text          (sanity)
  • Base + LoRA adapter, no merge, generates clean text   (LoRA itself is fine)
  • Merged HF → GGUF (Unsloth wrapper) → garbage          (merge is the bug)
  • Merged HF → GGUF (llama.cpp directly) → still garbage (the saved HF is bad)

Workaround implemented here: load the **un-quantized** base in bf16 (no bnb on
the merge path at all), apply the LoRA via ``peft.PeftModel``, call
``merge_and_unload()``, save bf16 safetensors, then shell out to llama.cpp's
own ``convert_hf_to_gguf.py`` + ``llama-quantize.exe`` (which we proved produce
correct GGUFs when the source HF model is healthy).

Run:
    python -m host_finetune.merge_and_export
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from host_finetune.config import (
    GGUF_DIR,
    GGUF_QUANT,
    GGUF_SIDECAR_DIR,
    LLAMA_CPP_DIR,
    LORA_ADAPTER_PATH,
    MERGE_BASE_MODEL,
    MERGE_DEVICE,
)


def _ensure_llama_cpp_tools() -> tuple[Path, Path]:
    convert_py = LLAMA_CPP_DIR / "convert_hf_to_gguf.py"
    quantize_exe = LLAMA_CPP_DIR / "build" / "bin" / "Release" / "llama-quantize.exe"
    if not convert_py.is_file():
        raise FileNotFoundError(
            f"llama.cpp converter not found at {convert_py}.\n"
            "Either build llama.cpp under LLAMA_CPP_DIR (default: ~/.unsloth/llama.cpp) "
            "or set LLAMA_CPP_DIR=<your-llama.cpp-checkout>."
        )
    if not quantize_exe.is_file():
        # Try platform-agnostic fallback (Linux/macOS).
        alt = LLAMA_CPP_DIR / "build" / "bin" / "llama-quantize"
        if alt.is_file():
            quantize_exe = alt
        else:
            raise FileNotFoundError(
                f"llama-quantize not found at {quantize_exe} (or {alt}).\n"
                "Build llama.cpp first (the same build Unsloth produced under "
                "~/.unsloth/llama.cpp/build is fine)."
            )
    return convert_py, quantize_exe


def _resolve_resolved_adapter() -> Path:
    if not LORA_ADAPTER_PATH.is_dir() or not (LORA_ADAPTER_PATH / "adapter_config.json").is_file():
        raise FileNotFoundError(
            f"No adapter_config.json at {LORA_ADAPTER_PATH}. "
            "Run `python -m host_finetune.finetune`, or set LORA_ADAPTER_PATH to a "
            "checkpoint-* folder (e.g. host_finetune\\output\\lemkin_lora\\checkpoint-800)."
        )
    return LORA_ADAPTER_PATH


def _merge_lora_into_base() -> Path:
    """Load un-quantized base + LoRA, merge, save bf16 HF model. Returns merged dir."""
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    adapter_dir = _resolve_resolved_adapter()

    print(f"[merge] base:    {MERGE_BASE_MODEL}")
    print(f"[merge] adapter: {adapter_dir}")
    print(f"[merge] device:  {MERGE_DEVICE}")
    print(f"[merge] output:  {GGUF_DIR}")

    print(
        "[merge] loading un-quantized base in bf16 "
        f"(device_map={MERGE_DEVICE}, low_cpu_mem_usage=True)...",
        flush=True,
    )
    base = AutoModelForCausalLM.from_pretrained(
        MERGE_BASE_MODEL,
        torch_dtype=torch.bfloat16,
        device_map=MERGE_DEVICE,
        low_cpu_mem_usage=True,
    )
    print("[merge] loading adapter on top of base...", flush=True)
    model = PeftModel.from_pretrained(base, str(adapter_dir))
    print("[merge] merge_and_unload()...", flush=True)
    merged = model.merge_and_unload()

    if GGUF_DIR.exists():
        for child in GGUF_DIR.iterdir():
            try:
                if child.is_dir():
                    shutil.rmtree(child)
                else:
                    child.unlink()
            except OSError as e:
                print(f"  [warn] could not remove {child}: {e}")
    GGUF_DIR.mkdir(parents=True, exist_ok=True)

    print(f"[merge] save_pretrained -> {GGUF_DIR}", flush=True)
    merged.save_pretrained(str(GGUF_DIR), safe_serialization=True)

    # Save tokenizer from the adapter dir (carries any FT-side template tweaks);
    # fall back to the merge base if the adapter dir has no tokenizer files.
    tok_src = adapter_dir if (adapter_dir / "tokenizer.json").is_file() else MERGE_BASE_MODEL
    print(f"[merge] tokenizer source: {tok_src}", flush=True)
    tok = AutoTokenizer.from_pretrained(str(tok_src))
    tok.save_pretrained(str(GGUF_DIR))

    return GGUF_DIR


def _convert_to_gguf(merged_dir: Path, quant: str = GGUF_QUANT) -> Path:
    """Convert merged HF -> bf16 GGUF -> requested quant (via llama.cpp directly)."""
    convert_py, quantize_exe = _ensure_llama_cpp_tools()
    GGUF_SIDECAR_DIR.mkdir(parents=True, exist_ok=True)

    bf16_gguf = GGUF_SIDECAR_DIR / "merged.BF16.gguf"
    quant_upper = quant.upper()
    final_name = f"Meta-Llama-3.1-8B-Instruct.{quant_upper}.gguf"
    final_gguf = GGUF_SIDECAR_DIR / final_name

    for p in (bf16_gguf, final_gguf):
        if p.exists():
            try:
                p.unlink()
            except OSError as e:
                print(f"  [warn] could not remove existing {p}: {e}")

    print(f"[gguf] convert HF -> bf16 GGUF: {bf16_gguf}", flush=True)
    subprocess.run(
        [
            sys.executable,
            str(convert_py),
            str(merged_dir),
            "--outfile", str(bf16_gguf),
            "--outtype", "bf16",
        ],
        check=True,
    )
    print(f"[gguf] quantize -> {quant_upper}: {final_gguf}", flush=True)
    subprocess.run(
        [str(quantize_exe), str(bf16_gguf), str(final_gguf), quant_upper],
        check=True,
    )

    # Drop the giant intermediate bf16 GGUF (16 GB) unless explicitly kept.
    if os.environ.get("KEEP_BF16_GGUF", "").lower() not in ("1", "true", "yes"):
        try:
            bf16_gguf.unlink()
        except OSError:
            pass

    return final_gguf


def main() -> None:
    merged_dir = _merge_lora_into_base()
    final_gguf = _convert_to_gguf(merged_dir)
    size_gb = final_gguf.stat().st_size / 1e9
    print(f"\nWrote: {final_gguf}  ({size_gb:.2f} GB)")
    print("\nNext: python -m host_finetune.register_ollama")


if __name__ == "__main__":
    main()
