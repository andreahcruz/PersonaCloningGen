"""QLoRA fine-tune of Llama 3.1 8B on the Lemkin SFT dataset (Unsloth).

Reads `host_finetune/data/dataset.jsonl` (one Alpaca-style record per line:
``{"instruction": ..., "input": "", "output": ...}``) and writes the LoRA
adapter to `host_finetune/output/lemkin_lora/`.

Run directly:
    python -m host_finetune.finetune

Tuned for a 16 GB Blackwell GPU (RTX 5070 Ti). Drop to Llama 3.2 3B by setting
``HF_MODEL_NAME=unsloth/Llama-3.2-3B-Instruct-bnb-4bit`` if you OOM.
"""
from __future__ import annotations

import inspect
import json
import os
import sys
from pathlib import Path

# Fused cross-entropy in unsloth_zoo reads VRAM via torch.cuda.mem_get_info().
# On Windows + WDDM, "free" VRAM often reads ~0 GiB while weights are resident,
# which raises RuntimeError("No or negligible GPU memory available for fused
# cross entropy."). Pin a scratch budget (GiB) BEFORE importing unsloth — the
# env is read at import time in cross_entropy_loss.py. Increase if you still
# OOM during the CE backward (try "3"), decrease only if CE spikes VRAM.
if os.environ.get("UNSLOTH_CE_LOSS_TARGET_GB") is None:
    os.environ["UNSLOTH_CE_LOSS_TARGET_GB"] = "2"

# Set before CUDA init so the allocator sees it (helps long-run fragmentation).
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

# Unsloth must be imported before transformers/peft for its monkey-patches to
# take effect; otherwise you lose the speed-ups and Blackwell kernels.
from unsloth import FastLanguageModel  # noqa: E402  (import order matters)

import torch
from datasets import Dataset
from transformers import TrainingArguments
from trl import SFTTrainer

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from host_finetune.config import (
    ADAPTER_DIR,
    DATASET_LOCAL,
    EVAL_STEPS,
    GRAD_ACCUM,
    HF_MODEL_NAME,
    LEARNING_RATE,
    LORA_ALPHA,
    LORA_DROPOUT,
    LORA_R,
    MAX_SEQ_LENGTH,
    NUM_EPOCHS,
    OUTPUT_DIR,
    PACKING,
    PER_DEVICE_BATCH,
    RESUME_FROM_CHECKPOINT,
    SAVE_STEPS,
    SAVE_TOTAL_LIMIT,
    SKIP_EVAL,
    TRAIN_DATALOADER_NUM_WORKERS,
)


# Same Alpaca template the dataset was produced for; keep in sync with the
# Spark to_instruction() helper and the Ollama Modelfile prompt template.
ALPACA_TEMPLATE = (
    "### Instruction:\n{instruction}\n\n"
    "### Input:\n{input}\n\n"
    "### Response:\n{output}"
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


def format_example(example: dict) -> dict:
    return {
        "text": ALPACA_TEMPLATE.format(
            instruction=example.get("instruction", ""),
            input=example.get("input", ""),
            output=example.get("output", ""),
        )
    }


def _tokenize_text_batch(examples: dict, tokenizer, max_len: int) -> dict:
    """Hard-truncate at tokenization (Unsloth's text path can skip this on small max_len)."""
    enc = tokenizer(
        examples["text"],
        truncation=True,
        max_length=max_len,
        padding=False,
        add_special_tokens=True,
    )
    enc["labels"] = [list(ids) for ids in enc["input_ids"]]
    return enc


def _resolve_eval_steps() -> int | None:
    if SKIP_EVAL:
        return None
    if EVAL_STEPS is not None and str(EVAL_STEPS).strip().isdigit():
        return max(int(str(EVAL_STEPS).strip()), 1)
    return max(SAVE_STEPS * 5, 500)


def _resolve_resume() -> bool | str | None:
    v = RESUME_FROM_CHECKPOINT
    if not v:
        return None
    low = v.lower()
    if low in ("1", "true", "yes", "last", "latest"):
        return True
    return v


def _filter_training_arguments_kwargs(desired: dict) -> dict:
    """Strip keys this install's ``TrainingArguments`` does not accept (HF API drift)."""
    try:
        sig = inspect.signature(TrainingArguments.__init__)
    except (TypeError, ValueError):
        return desired
    params = sig.parameters
    if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()):
        return desired
    names = frozenset(p for p in params if p != "self")
    out = {k: v for k, v in desired.items() if k in names}
    dropped = sorted(set(desired) - set(out))
    if dropped:
        print(f"[finetune] TrainingArguments skips unknown fields: {dropped}")
    return out


def main() -> None:
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    ADAPTER_DIR.mkdir(parents=True, exist_ok=True)

    ds = load_dataset(DATASET_LOCAL).map(format_example, remove_columns=None)
    split = ds.train_test_split(test_size=0.1, seed=42)
    train_ds, eval_ds = split["train"], split["test"]
    print(f"train={len(train_ds)} eval={len(eval_ds)}")

    print(f"Loading {HF_MODEL_NAME} (4-bit) via Unsloth...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=HF_MODEL_NAME,
        max_seq_length=MAX_SEQ_LENGTH,
        dtype=None,
        load_in_4bit=True,
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=LORA_R,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=42,
    )

    tokenizer.model_max_length = MAX_SEQ_LENGTH

    eval_steps = _resolve_eval_steps()
    eval_ds_arg = None if SKIP_EVAL else eval_ds

    # With packing=True and a small MAX_SEQ_LENGTH (e.g. 512 for VRAM), TRL can
    # still emit batches longer than the model cap. Unsloth then truncates
    # input_ids but fused CE sees logits seq_len != labels seq_len → Dynamo
    # cross_entropy mismatch. Disable packing unless the cap is comfortably large.
    packing_effective = bool(PACKING)
    if packing_effective and MAX_SEQ_LENGTH < 1024:
        packing_effective = False
        print(
            "[finetune] packing disabled: MAX_SEQ_LENGTH < 1024 (packed batches "
            "can exceed the cap and break Unsloth fused CE). Raise MAX_SEQ_LENGTH "
            "to 1024+ to re-enable packing, or keep PACKING=0."
        )

    if packing_effective:
        train_for_trainer = train_ds
        eval_for_trainer = eval_ds_arg
        text_field_kw = {"dataset_text_field": "text"}
    else:
        max_len = MAX_SEQ_LENGTH

        def tok_fn(batch: dict) -> dict:
            return _tokenize_text_batch(batch, tokenizer, max_len)

        ncols = train_ds.column_names
        print(
            f"[finetune] Pre-tokenizing with truncation max_length={max_len} "
            "(avoids Unsloth fused CE length mismatch when cap < doc length)."
        )
        train_for_trainer = train_ds.map(tok_fn, batched=True, remove_columns=ncols)
        eval_for_trainer = (
            eval_ds.map(tok_fn, batched=True, remove_columns=ncols)
            if eval_ds_arg is not None
            else None
        )
        text_field_kw = {}

    arg_fields: dict = {
        "output_dir": str(ADAPTER_DIR),
        "num_train_epochs": NUM_EPOCHS,
        "per_device_train_batch_size": PER_DEVICE_BATCH,
        "gradient_accumulation_steps": GRAD_ACCUM,
        "learning_rate": LEARNING_RATE,
        "bf16": torch.cuda.is_bf16_supported(),
        "fp16": not torch.cuda.is_bf16_supported(),
        "logging_steps": 10,
        "save_strategy": "steps",
        "save_steps": SAVE_STEPS,
        "save_total_limit": SAVE_TOTAL_LIMIT,
        "warmup_ratio": 0.03,
        "lr_scheduler_type": "cosine",
        "max_grad_norm": 0.3,
        # Packing builds fixed-length chunks; sorting by length is only for
        # dynamic padding when packing is off.
        "group_by_length": not packing_effective,
        "dataloader_num_workers": TRAIN_DATALOADER_NUM_WORKERS,
        "dataloader_pin_memory": torch.cuda.is_available(),
        "dataloader_persistent_workers": TRAIN_DATALOADER_NUM_WORKERS > 0,
        "report_to": "none",
        "optim": "adamw_8bit",
        "seed": 42,
    }
    if SKIP_EVAL:
        arg_fields["eval_strategy"] = "no"
    else:
        arg_fields["eval_strategy"] = "steps"
        arg_fields["eval_steps"] = eval_steps

    training_args = TrainingArguments(**_filter_training_arguments_kwargs(arg_fields))

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=train_for_trainer,
        eval_dataset=eval_for_trainer,
        max_seq_length=MAX_SEQ_LENGTH,
        packing=packing_effective,
        **text_field_kw,
    )

    resume = _resolve_resume()
    if resume:
        print(f"Resuming from checkpoint ({resume!r})...")
    print(
        f"Training: save every {SAVE_STEPS} steps (keep {SAVE_TOTAL_LIMIT}), "
        f"eval={'off' if SKIP_EVAL else f'every {eval_steps} steps'}, "
        f"packing={packing_effective}, dataloader_workers={TRAIN_DATALOADER_NUM_WORKERS}"
    )
    trainer.train(resume_from_checkpoint=resume)

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
