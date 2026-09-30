"""QLoRA fine-tune of Llama 3.1 8B on the Lemkin SFT dataset (Unsloth + TRL).

Reads `host_finetune/data/dataset.jsonl` (one Alpaca-style record per line:
``{"instruction": ..., "input": "", "output": ...}``) and writes the LoRA
adapter to `host_finetune/output/lemkin_lora/`.

Rows are converted with the model tokenizer’s **Llama 3 Instruct** chat template
(``apply_chat_template``). Supervision is **assistant-only** via
``unsloth.chat_templates.train_on_responses_only`` (not raw Alpaca ``###`` loss).

Quick sanity run (smoke / “vibe-check”): set ``MAX_STEPS`` (e.g. ``20``) and optionally
``FINETUNE_TRAIN_HEAD_N`` to slice the train split; then merge, register Ollama, and run
``python -m host_finetune.smoke_lemkin_infer``.

Run directly:
    python -m host_finetune.finetune

After ``python -m host_finetune.clean_dataset_v2`` (no repunct), your rows are
already chunked; set ``SKIP_DATASET_PREP=1`` before training to skip the
download-prep path entirely. If you leave prep enabled, long rows expand with
sentence-aware splits and ``clean_dataset`` may still scrub tags and dedupe.
"""
from __future__ import annotations

import inspect
import json
import os
import random
import sys
from pathlib import Path

# Unsloth chunks the LM head + CE loss to fit within this many GB of peak VRAM.
# Llama-3 has a 128k vocab so the materialized logits tensor is huge; 0.5 GB is
# the sweet spot for ~16 GB GPUs (1.0 GB peaks above what your activation
# budget can spare). Override via env if you have more headroom.
if os.environ.get("UNSLOTH_CE_LOSS_TARGET_GB") is None:
    os.environ["UNSLOTH_CE_LOSS_TARGET_GB"] = "0.5"

# Aggressive allocator settings for a small-VRAM box: expandable segments
# avoids permanent fragmentation, garbage_collection_threshold:0.6 forces
# torch to release more aggressively before OOM, and the smaller max-split
# size reduces "200 MB free but no contiguous block" stalls.
os.environ.setdefault(
    "PYTORCH_CUDA_ALLOC_CONF",
    "expandable_segments:True,garbage_collection_threshold:0.6,max_split_size_mb:128",
)

# Unsloth must be imported before transformers/peft for patches.
from unsloth import FastLanguageModel  # noqa: E402

import torch
from datasets import Dataset
from transformers import BitsAndBytesConfig
from trl import SFTConfig, SFTTrainer
from unsloth.chat_templates import train_on_responses_only

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from host_finetune.llama_chat_format import training_text_from_row, unsloth_response_markers
from host_finetune.sft_chunk_utils import extract_base_topic
from host_finetune.config import (
    ADAPTER_DIR,
    DATASET_LOCAL,
    EVAL_STEPS,
    GRAD_ACCUM,
    HF_MODEL_NAME,
    LEARNING_RATE,
    LOAD_BEST_MODEL_AT_END,
    LOG_VRAM_USAGE,
    LORA_ALPHA,
    LORA_DROPOUT,
    LORA_R,
    LORA_TARGET_MODULES,
    MAX_SEQ_LENGTH,
    MAX_VRAM_FRACTION,
    NUM_EPOCHS,
    OUTPUT_DIR,
    PACKING,
    PER_DEVICE_BATCH,
    RESUME_FROM_CHECKPOINT,
    SAVE_STEPS,
    SAVE_TOTAL_LIMIT,
    SKIP_EVAL,
    TRAIN_DATALOADER_NUM_WORKERS,
    TRAIN_OPTIM,
    USE_EXPLICIT_BNB_CONFIG,
    USE_LIGER_KERNEL,
    MAX_STEPS,
    FINETUNE_TRAIN_HEAD_N,
)

def load_dataset(path: Path) -> Dataset:
    if not path.is_file():
        raise FileNotFoundError(
            f"Training file not found at {path}. Run `python -m host_finetune.watcher` "
            "first (which downloads dataset.jsonl from MinIO)."
        )
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    if not rows:
        raise RuntimeError(f"No usable rows in {path}.")
    print(f"Loaded {len(rows)} training rows from {path}")
    return Dataset.from_list(rows)


def split_by_source_document(raw_ds: Dataset, test_fraction: float = 0.1, seed: int = 42) -> tuple[Dataset, Dataset]:
    """Split whole source documents before chunk formatting.

    Chunk-level splitting leaks parts of the same original post into both
    training and evaluation.  New prepared datasets carry ``source_file`` and
    ``source_line``; older datasets fall back to the base instruction title.
    The fallback is conservative: identically titled rows stay together.
    """
    groups: dict[tuple[str, str], list[int]] = {}
    for idx, row in enumerate(raw_ds):
        source_file = str(row.get("source_file") or "").strip()
        source_line = str(row.get("source_line") or "").strip()
        if source_file and source_line:
            key = (source_file, source_line)
        else:
            key = ("instruction_title", extract_base_topic(str(row.get("instruction") or "")))
        groups.setdefault(key, []).append(idx)

    keys = list(groups)
    random.Random(seed).shuffle(keys)
    target_test_rows = max(1, round(len(raw_ds) * test_fraction))
    test_indices: list[int] = []
    for key in keys:
        # Keep adding whole documents until the desired evaluation size is met.
        # A slight overshoot is preferable to leaking chunks between splits.
        if len(test_indices) >= target_test_rows:
            break
        test_indices.extend(groups[key])
    test_set = set(test_indices)
    train_indices = [i for i in range(len(raw_ds)) if i not in test_set]
    if not train_indices or not test_indices:
        raise RuntimeError("Document-level split produced an empty train or eval set.")
    print(
        f"[finetune] document-level split: {len(groups)} source groups; "
        f"train={len(train_indices)} eval={len(test_indices)}"
    )
    return raw_ds.select(train_indices), raw_ds.select(sorted(test_indices))


def _chat_text_batches(examples: dict, tokenizer) -> dict:
    """One Llama-3 chat-formatted string per row (system defaults come from the tokenizer)."""
    inst_col = examples.get("instruction")
    inp_col = examples.get("input")
    out_col = examples.get("output")
    if not inst_col:
        raise KeyError(
            'dataset rows must contain "instruction"; got columns '
            + f"{sorted(examples.keys())}"
        )
    n = len(inst_col)
    if inp_col is None:
        inp_col = [""] * n
    if out_col is None:
        out_col = [""] * n
    texts: list[str] = []
    for i in range(n):
        texts.append(
            training_text_from_row(
                tokenizer,
                str(inst_col[i] or ""),
                str(inp_col[i] or ""),
                str(out_col[i] or ""),
            )
        )
    return {"text": texts}


def _resolve_eval_steps() -> int | None:
    if SKIP_EVAL:
        return None
    if EVAL_STEPS is not None and str(EVAL_STEPS).strip().isdigit():
        return max(int(str(EVAL_STEPS).strip()), 1)
    if LOAD_BEST_MODEL_AT_END:
        return max(SAVE_STEPS, 1)
    return max(SAVE_STEPS * 5, 500)


def _resolve_resume() -> bool | str | None:
    v = RESUME_FROM_CHECKPOINT
    if not v:
        return None
    low = v.lower()
    if low in ("1", "true", "yes", "last", "latest"):
        return True
    return v


def _filter_sft_config_kwargs(desired: dict) -> dict:
    """Strip keys this install's ``SFTConfig`` / ``TrainingArguments`` does not accept."""
    try:
        sig = inspect.signature(SFTConfig.__init__)
    except (TypeError, ValueError):
        return desired
    params = sig.parameters
    if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()):
        return desired
    names = frozenset(p for p in params if p != "self")
    out = {k: v for k, v in desired.items() if k in names}
    dropped = sorted(set(desired) - set(out))
    if dropped:
        print(f"[finetune] SFTConfig skips unknown fields: {dropped}")
    return out


def _print_vram(label: str, *, trainer=None) -> None:
    """Print compact CUDA / system memory summary. Helps spot Windows sysmem fallback."""
    try:
        free_b, total_b = torch.cuda.mem_get_info()
        alloc_b = torch.cuda.memory_allocated()
        reserved_b = torch.cuda.memory_reserved()
        max_alloc_b = torch.cuda.max_memory_allocated()
        print(
            f"[vram:{label}] alloc={alloc_b / 1e9:.2f}GB "
            f"reserved={reserved_b / 1e9:.2f}GB "
            f"peak_alloc={max_alloc_b / 1e9:.2f}GB "
            f"free={free_b / 1e9:.2f}GB / total={total_b / 1e9:.2f}GB"
        )
        # Warn loudly if peak alloc already exceeds physical VRAM — that means
        # Windows is paging into shared system RAM (huge slowdown).
        if max_alloc_b > total_b:
            print(
                "  [vram:WARNING] peak_alloc > total VRAM — Windows is spilling "
                "into shared memory. Close apps or lower CE_LOSS_TARGET_GB."
            )
    except Exception as e:
        print(f"[vram:{label}] (unavailable: {e!r})")


def _quantize_load_kwargs() -> dict:
    if not USE_EXPLICIT_BNB_CONFIG:
        return {"load_in_4bit": True}
    compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    qcfg = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=compute_dtype,
    )
    # ``BitsAndBytesConfig`` already sets load_in_4bit=True; omit duplicate kwarg for Unsloth.
    return {"quantization_config": qcfg}


def main() -> None:
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        # Cap the caching allocator before any large allocation; this prevents
        # silent paging into Windows shared memory under VRAM pressure.
        try:
            torch.cuda.set_per_process_memory_fraction(MAX_VRAM_FRACTION)
            free_b, total_b = torch.cuda.mem_get_info()
            cap_gb = MAX_VRAM_FRACTION * total_b / 1e9
            print(
                f"[finetune] VRAM cap: {MAX_VRAM_FRACTION:.0%} of "
                f"{total_b / 1e9:.2f} GB = {cap_gb:.2f} GB usable "
                f"(currently {free_b / 1e9:.2f} GB free)"
            )
            if free_b < 8 * 1024 * 1024 * 1024:
                print(
                    "[finetune] WARNING: < 8 GB free VRAM before training starts. "
                    "Close GPU-accelerated apps (browsers, Discord, Wallpaper Engine, "
                    "Cursor renderers) or training will OOM at peak. nvidia-smi -q -d "
                    "MEMORY shows current holders."
                )
        except Exception as e:
            print(f"[finetune] could not set VRAM cap: {e!r}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    ADAPTER_DIR.mkdir(parents=True, exist_ok=True)

    st = None
    if os.environ.get("SKIP_DATASET_PREP", "").lower() not in ("1", "true", "yes"):
        from host_finetune.dataset_prepare import prepare_dataset_file

        st = prepare_dataset_file(DATASET_LOCAL)
        if st.get("chunk_expand"):
            ce = st["chunk_expand"]
            print(
                f"[finetune] chunk expand: rows {ce.get('rows_in')} -> {ce.get('rows_out')}, "
                f"long_docs_split={ce.get('split_from_long')}"
            )
        if st.get("clean"):
            cl = st["clean"]
            print(
                f"[finetune] dataset clean: rows {cl.get('rows_in')} -> {cl.get('rows_out')}, "
                f"dropped_junk={cl.get('dropped_junk')} truncated={cl.get('truncated_outputs')} "
                f"scrubbed_tags={cl.get('rows_scrubbed_transcript_tags', 0)} "
                f"dup_out={cl.get('dropped_duplicate_output', 0)}"
            )
    else:
        print("[finetune] SKIP_DATASET_PREP: skipped chunk + clean (use with care).")

    raw_ds = load_dataset(DATASET_LOCAL)

    print(
        f"Loading {HF_MODEL_NAME} via Unsloth (4-bit, "
        f"explicit_nf4dq={USE_EXPLICIT_BNB_CONFIG})..."
    )
    load_kw = _quantize_load_kwargs()
    try:
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=HF_MODEL_NAME,
            max_seq_length=MAX_SEQ_LENGTH,
            dtype=None,
            **load_kw,
        )
    except Exception as e:
        if USE_EXPLICIT_BNB_CONFIG:
            print(
                f"[finetune] explicit BitsAndBytesConfig failed ({e!r}); "
                "retrying load_in_4bit=True (Unsloth default)."
            )
            model, tokenizer = FastLanguageModel.from_pretrained(
                model_name=HF_MODEL_NAME,
                max_seq_length=MAX_SEQ_LENGTH,
                dtype=None,
                load_in_4bit=True,
            )
        else:
            raise

    def _map_chat_rows(examples: dict) -> dict:
        return _chat_text_batches(examples, tokenizer)

    train_raw, eval_raw = split_by_source_document(raw_ds, test_fraction=0.1, seed=42)

    train_ds = train_raw.map(
        _map_chat_rows,
        batched=True,
        remove_columns=list(train_raw.column_names),
        desc="format train Llama-3 chat (apply_chat_template)",
    )
    eval_ds = eval_raw.map(
        _map_chat_rows,
        batched=True,
        remove_columns=list(eval_raw.column_names),
        desc="format eval Llama-3 chat (apply_chat_template)",
    )
    if FINETUNE_TRAIN_HEAD_N is not None:
        n_keep = min(FINETUNE_TRAIN_HEAD_N, len(train_ds))
        train_ds = train_ds.select(range(n_keep))
        print(
            f"[finetune] FINETUNE_TRAIN_HEAD_N={FINETUNE_TRAIN_HEAD_N}: "
            f"using {n_keep} train rows (subset)"
        )
    print(f"train={len(train_ds)} eval={len(eval_ds)}")

    ip_dbg, rp_dbg = unsloth_response_markers(tokenizer)
    print(
        "[finetune] Llama-3 chat SFT | "
        f"train_on_responses_only markers len(user_header)={len(ip_dbg)} "
        f"len(assistant_header)={len(rp_dbg)}"
    )

    print(f"[finetune] LoRA targets={LORA_TARGET_MODULES} r={LORA_R} alpha={LORA_ALPHA}")
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

    tokenizer.model_max_length = MAX_SEQ_LENGTH

    eval_steps = _resolve_eval_steps()
    eval_ds_arg = None if SKIP_EVAL else eval_ds

    packing_effective = bool(PACKING)
    if packing_effective and MAX_SEQ_LENGTH < 1024:
        packing_effective = False
        print(
            "[finetune] packing disabled: MAX_SEQ_LENGTH < 1024 (packed batches "
            "can exceed the cap and break Unsloth fused CE on Windows)."
        )

    bf16_ok = torch.cuda.is_bf16_supported()
    cfg_fields = {
        "output_dir": str(ADAPTER_DIR),
        "num_train_epochs": NUM_EPOCHS,
        "per_device_train_batch_size": PER_DEVICE_BATCH,
        "gradient_accumulation_steps": GRAD_ACCUM,
        "learning_rate": LEARNING_RATE,
        "bf16": bf16_ok,
        "fp16": not bf16_ok,
        "logging_steps": 10,
        "save_strategy": "steps",
        "save_steps": SAVE_STEPS,
        "save_total_limit": SAVE_TOTAL_LIMIT,
        "warmup_ratio": 0.03,
        "lr_scheduler_type": "cosine",
        "max_grad_norm": 0.3,
        # Unsloth's SFTConfig does not accept ``group_by_length`` (HF TrainingArguments field).
        "dataloader_num_workers": TRAIN_DATALOADER_NUM_WORKERS,
        "dataloader_pin_memory": torch.cuda.is_available(),
        "dataloader_persistent_workers": TRAIN_DATALOADER_NUM_WORKERS > 0,
        "report_to": "none",
        "optim": TRAIN_OPTIM,
        "use_liger_kernel": USE_LIGER_KERNEL,
        "seed": 42,
        "max_length": MAX_SEQ_LENGTH,
        "packing": packing_effective,
        "dataset_text_field": "text",
        # Unsloth enables checkpointing on the wrapped model; avoid double-enable from TRL.
        "gradient_checkpointing": False,
        "dataset_num_proc": None,
        "dataset_kwargs": {},
    }
    if SKIP_EVAL:
        cfg_fields["eval_strategy"] = "no"
    else:
        cfg_fields["eval_strategy"] = "steps"
        cfg_fields["eval_steps"] = eval_steps
        if LOAD_BEST_MODEL_AT_END:
            cfg_fields["load_best_model_at_end"] = True
            cfg_fields["metric_for_best_model"] = "eval_loss"
            cfg_fields["greater_is_better"] = False

    # Unsloth's Trainer.train() wrapper does not accept ``max_steps=``; use TrainingArguments.
    if MAX_STEPS is not None:
        cfg_fields["max_steps"] = MAX_STEPS

    sft_args = SFTConfig(**_filter_sft_config_kwargs(cfg_fields))

    resume = _resolve_resume()
    trainer = SFTTrainer(
        model=model,
        args=sft_args,
        processing_class=tokenizer,
        train_dataset=train_ds,
        eval_dataset=eval_ds_arg,
    )
    instr_part, resp_part = unsloth_response_markers(tokenizer)
    trainer = train_on_responses_only(
        trainer,
        instruction_part=instr_part,
        response_part=resp_part,
    )
    if resume:
        print(f"Resuming from checkpoint ({resume!r})...")
    msg = (
        f"Training: Llama-3 chat + assistant-only loss | save every {SAVE_STEPS} (keep {SAVE_TOTAL_LIMIT}), "
        f"eval={'off' if SKIP_EVAL else f'every {eval_steps}'}, "
        f"packing={packing_effective}, workers={TRAIN_DATALOADER_NUM_WORKERS}, "
        f"max_length={MAX_SEQ_LENGTH}, optim={TRAIN_OPTIM}, "
        f"liger={'on' if USE_LIGER_KERNEL else 'off'}, "
        f"ce_target_gb={os.environ.get('UNSLOTH_CE_LOSS_TARGET_GB')}"
    )
    if MAX_STEPS is not None:
        msg += f", max_steps={MAX_STEPS} (caps run length)"
    print(msg)
    if LOG_VRAM_USAGE and torch.cuda.is_available():
        _print_vram("pre-train", trainer=trainer)
    if MAX_STEPS is not None and SAVE_STEPS > MAX_STEPS:
        print(
            f"[finetune] hint: SAVE_STEPS ({SAVE_STEPS}) > MAX_STEPS ({MAX_STEPS}); "
            "no mid-run checkpoint unless you lower SAVE_STEPS."
        )
    train_kw: dict = {}
    if resume:
        train_kw["resume_from_checkpoint"] = resume
    trainer.train(**train_kw)
    if LOG_VRAM_USAGE and torch.cuda.is_available():
        _print_vram("post-train", trainer=trainer)

    print(f"Saving LoRA adapter -> {ADAPTER_DIR}")
    trainer.model.save_pretrained(str(ADAPTER_DIR))
    tokenizer.save_pretrained(str(ADAPTER_DIR))

    if trainer.state.log_history:
        print("\n=== Loss curve ===")
        for entry in trainer.state.log_history:
            if "loss" in entry:
                print(f"  step {entry.get('step', '?')}: loss={entry['loss']:.4f}")
            elif "eval_loss" in entry:
                print(f"  eval @ step {entry.get('step', '?')}: eval_loss={entry['eval_loss']:.4f}")


if __name__ == "__main__":
    main()
