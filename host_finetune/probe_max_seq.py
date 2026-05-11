"""VRAM probe + **training sweet spot** for chunking.

**Two separate ideas (this script ties them together):**

1. **VRAM ceiling** — Largest ``seq_len`` where **one** synthetic forward+backward
   stays under the CUDA memory **budget** (total VRAM − headroom, or ``--budget-gb``).
   That is *not* “how long your rows are”; it is “how long a padded batch can be.”

2. **Chunk size** — ``SFT_CHUNK_OUTPUT_CHARS`` must match the **``MAX_SEQ_LENGTH`` you
   actually train with**: roughly ``(MAX_SEQ_LENGTH − instruction/template reserve) ×
   chars per token``. If you already chunked for 1024, your rows fit **1024**; this
   script answers whether you can **raise** training length (e.g. 1536) without OOM,
   and what **new** ``SFT_CHUNK_OUTPUT_CHARS`` should be.

**Sweet spot** — Binary search yields ``ceiling``. If peak CUDA usage at that length is
already **≥ ~92%** of the memory budget (``PROBE_TIGHT_BUDGET_FRAC``), we set
``train_max = ceiling`` — no token subtraction (you are already VRAM-limited).
Otherwise we set ``train_max = ceiling − PROBE_TRAIN_MARGIN_TOKENS`` to leave slack
for ``SFTTrainer`` / eval / Windows above the one-step probe.

**Important:** One micro-step only; tune ``PROBE_VRAM_HEADROOM_GB`` / ``--budget-gb``
if dedicated VRAM still looks tight during real runs.

Usage:
    python -m host_finetune.probe_max_seq
    python -m host_finetune.probe_max_seq --budget-gb 12 --margin-tokens 384

Environment (same as finetune): ``HF_MODEL_NAME``, ``LORA_R``, ``LORA_ALPHA``, etc.
``PROBE_VRAM_HEADROOM_GB`` (default ``3``): subtract from total VRAM when
``--budget-gb`` omitted. ``PROBE_TRAIN_MARGIN_TOKENS`` (default ``512``): subtract from ceiling for the printed
training recommendation **only when** the probe is **not** already using most of the
budget (see ``PROBE_TIGHT_BUDGET_FRAC``). If peak memory at ``ceiling`` is **tight**
against ``budget``, we recommend ``train_max == ceiling`` — no extra subtraction.
"""
from __future__ import annotations

import argparse
import gc
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

if os.environ.get("UNSLOTH_CE_LOSS_TARGET_GB") is None:
    os.environ["UNSLOTH_CE_LOSS_TARGET_GB"] = "2"
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
from transformers import BitsAndBytesConfig  # noqa: E402
from unsloth import FastLanguageModel  # noqa: E402

from host_finetune.config import (
    HF_MODEL_NAME,
    LORA_ALPHA,
    LORA_DROPOUT,
    LORA_R,
    LORA_TARGET_MODULES,
    USE_EXPLICIT_BNB_CONFIG,
)


def _bytes_gb(b: int) -> float:
    return b / (1024**3)


def _quantize_kw() -> dict:
    if not USE_EXPLICIT_BNB_CONFIG:
        return {"load_in_4bit": True}
    dt = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    return {
        "quantization_config": BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=dt,
        )
    }


def _fit_one_step(model, seq_len: int, device: torch.device, vocab_size: int) -> int:
    """Single train-like forward+backward; returns peak allocated bytes after step."""
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.empty_cache()
    gc.collect()

    input_ids = torch.randint(
        low=0,
        high=min(vocab_size - 1, 32000),
        size=(1, seq_len),
        dtype=torch.long,
        device=device,
    )
    labels = input_ids.clone()

    model.train()
    out = model(input_ids=input_ids, labels=labels)
    loss = out.loss
    loss.backward()

    peak = torch.cuda.max_memory_allocated(device)
    model.zero_grad(set_to_none=True)
    del out, loss, input_ids, labels
    torch.cuda.empty_cache()
    gc.collect()
    return peak


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--budget-gb",
        type=float,
        default=None,
        help="Max allowed peak memory for one probe step (GiB). Overrides headroom default.",
    )
    p.add_argument(
        "--high",
        type=int,
        default=2048,
        help="Upper bound sequence length to probe (default 2048).",
    )
    p.add_argument(
        "--low",
        type=int,
        default=256,
        help="Lower bound when binary searching (default 256).",
    )
    p.add_argument(
        "--margin-tokens",
        type=int,
        default=None,
        help="When probe peak is *below* tight budget threshold, subtract this many "
        "tokens from ceiling for train_max (default: env PROBE_TRAIN_MARGIN_TOKENS or 512). "
        "Use 0 to always recommend ceiling (same as tight fit).",
    )
    args = p.parse_args()

    if not torch.cuda.is_available():
        print("CUDA not available.")
        sys.exit(1)

    device = torch.device("cuda:0")
    total_b = torch.cuda.get_device_properties(0).total_memory
    total_gb = _bytes_gb(total_b)
    headroom_gb = float(os.environ.get("PROBE_VRAM_HEADROOM_GB", "3.0"))
    budget_b = (
        int(args.budget_gb * (1024**3))
        if args.budget_gb is not None
        else int(total_b - headroom_gb * (1024**3))
    )
    print(
        f"GPU total VRAM ≈ {total_gb:.2f} GiB | probe peak budget ≈ {_bytes_gb(budget_b):.2f} GiB "
        f"(total − {headroom_gb:g} GiB; set PROBE_VRAM_HEADROOM_GB or --budget-gb to tune)"
    )
    print(
        "One-synthetic-step estimate only — SFTTrainer + eval + Windows often need a lower "
        "MAX_SEQ_LENGTH than reported if shared GPU memory is still high.\n"
    )
    print(
        f"Loading {HF_MODEL_NAME} (4-bit dq={USE_EXPLICIT_BNB_CONFIG}) + LoRA "
        f"targets={len(LORA_TARGET_MODULES)} modules r={LORA_R} alpha={LORA_ALPHA} ..."
    )

    lw = _quantize_kw()
    try:
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=HF_MODEL_NAME,
            max_seq_length=args.high,
            dtype=None,
            **lw,
        )
    except Exception as exc:
        if USE_EXPLICIT_BNB_CONFIG:
            print(f"[probe_max_seq] BNB retry after {exc!r}")
            model, tokenizer = FastLanguageModel.from_pretrained(
                model_name=HF_MODEL_NAME,
                max_seq_length=args.high,
                dtype=None,
                load_in_4bit=True,
            )
        else:
            raise
    model = FastLanguageModel.get_peft_model(
        model,
        r=LORA_R,
        target_modules=list(LORA_TARGET_MODULES),
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=42,
    )
    vocab_size = tokenizer.vocab_size

    lo, hi = args.low, args.high
    ok_at = {}

    # Raise hi if smallest length already fails (OOM)
    try:
        peak0 = _fit_one_step(model, lo, device, vocab_size)
        ok_at[lo] = peak0
        print(f"seq_len={lo}: peak {_bytes_gb(peak0):.2f} GiB")
    except RuntimeError as e:
        print(f"Even seq_len={lo} failed: {e}")
        sys.exit(2)

    if peak0 > budget_b:
        print(
            f"\nEven MIN sequence length {lo} exceeds budget ({_bytes_gb(peak0):.2f} > {_bytes_gb(budget_b):.2f} GiB). "
            "Try smaller HF_MODEL_NAME, lower LORA_R, or --budget-gb closer to total VRAM."
        )
        sys.exit(3)

    # Binary search for largest L with peak <= budget_b
    best = lo
    while lo <= hi:
        mid = (lo + hi) // 2
        try:
            peak = _fit_one_step(model, mid, device, vocab_size)
            ok_at[mid] = peak
            gb = _bytes_gb(peak)
            print(f"seq_len={mid}: peak {gb:.2f} GiB", end="")
            if peak <= budget_b:
                print("  OK")
                best = mid
                lo = mid + 1
            else:
                print("  OVER budget")
                hi = mid - 1
        except RuntimeError as e:
            print(f"seq_len={mid}: OOM — {e}")
            hi = mid - 1

    ceiling = best
    peak_ceiling = ok_at.get(ceiling, _fit_one_step(model, ceiling, device, vocab_size))

    margin = (
        args.margin_tokens
        if args.margin_tokens is not None
        else int(os.environ.get("PROBE_TRAIN_MARGIN_TOKENS", "512"))
    )
    tight_frac = float(os.environ.get("PROBE_TIGHT_BUDGET_FRAC", "0.92"))
    usage_frac = peak_ceiling / budget_b if budget_b > 0 else 1.0
    if usage_frac >= tight_frac or margin <= 0:
        train_max = ceiling
        tight_reason = (
            f"peak {_bytes_gb(peak_ceiling):.2f} GiB is ≥ {tight_frac:.0%} of budget "
            f"(tight fit — no token margin applied)"
            if margin > 0
            else "--margin-tokens 0"
        )
    else:
        train_max = max(args.low, ceiling - margin)
        tight_reason = (
            f"peak {_bytes_gb(peak_ceiling):.2f} GiB is below {tight_frac:.0%} of budget "
            f"(room left — subtracted {margin} tokens for trainer overhead)"
        )

    print()
    print(f"=== VRAM ceiling (largest seq_len within budget): {ceiling} ===")
    print(f"    Measured peak at this length: {_bytes_gb(peak_ceiling):.2f} GiB")

    print()
    print(
        f"=== Training sweet spot (for config + dataset chunking): {train_max} ===\n"
        f"    {tight_reason}. "
        "Tune ``PROBE_TIGHT_BUDGET_FRAC`` / ``PROBE_TRAIN_MARGIN_TOKENS`` / "
        "``--margin-tokens`` if this feels too aggressive or too cautious."
    )

    # Reserve tokens for Alpaca headers + instruction (long titles / "part i of n" lines).
    reserve_tokens = int(os.environ.get("PROBE_INSTRUCTION_RESERVE_TOKENS", "380"))
    per_char = float(os.environ.get("PROBE_CHARS_PER_OUTPUT_TOKEN", "3.2"))
    out_token_budget = max(train_max - reserve_tokens, 64)
    suggested_chars = int(out_token_budget * per_char)
    print()
    print(
        "Use these together (single train length → one chunk width):\n"
        f"  MAX_SEQ_LENGTH={train_max}\n"
        f"  SFT_CHUNK_OUTPUT_CHARS={suggested_chars}   "
        f"(≈ {out_token_budget} tokens for ``output``, reserve ≈ {reserve_tokens} for template+instruction)"
    )
    print(
        "\nChunking rule: **pick MAX_SEQ_LENGTH first** (``train_max`` above), then set "
        "``SFT_CHUNK_OUTPUT_CHARS`` so Alpaca instruction + output rarely exceeds that "
        "length in tokens. If those env vars differ from how you built ``dataset.jsonl``, "
        "**rechunk once** to match."
    )
    print()
    print(
        "Tuning VRAM: **raise** ``PROBE_VRAM_HEADROOM_GB`` or **lower** ``--budget-gb`` "
        "for a **lower** ceiling; **lower** headroom / **higher** budget for a **higher** "
        "ceiling. If shared GPU memory is still high at ``train_max``, reduce "
        "``MAX_SEQ_LENGTH`` or increase margin."
    )


if __name__ == "__main__":
    main()
