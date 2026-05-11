"""QLoRA fine-tune of Llama 3.1 8B on the Lemkin SFT dataset (Unsloth + TRL).

Reads `host_finetune/data/dataset.jsonl` (one Alpaca-style record per line:
``{"instruction": ..., "input": "", "output": ...}``) and writes the LoRA
adapter to `host_finetune/output/lemkin_lora/`.

Uses **completion-only loss**: gradients only on tokens after ``### Response:``.

Run directly:
    python -m host_finetune.finetune
"""
from __future__ import annotations

import inspect
import json
import os
import sys
from pathlib import Path

if os.environ.get("UNSLOTH_CE_LOSS_TARGET_GB") is None:
    os.environ["UNSLOTH_CE_LOSS_TARGET_GB"] = "2"

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

# Unsloth must be imported before transformers/peft for patches.
from unsloth import FastLanguageModel  # noqa: E402

import torch
from datasets import Dataset
from transformers import BitsAndBytesConfig
from trl import SFTConfig, SFTTrainer

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
    LOAD_BEST_MODEL_AT_END,
    LORA_ALPHA,
    LORA_DROPOUT,
    LORA_R,
    LORA_TARGET_MODULES,
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
    USE_EXPLICIT_BNB_CONFIG,
)

# Same Alpaca layout as Spark export + Ollama Modelfile. Only the completion is supervised.
_INSTRUCTION_AND_INPUT_PROMPT = (
    "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:\n"
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


def _prompt_completion_batches(examples: dict) -> dict:
    prompts: list[str] = []
    completions: list[str] = []
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
    for i in range(n):
        inst = inst_col[i] or ""
        inp = inp_col[i] or ""
        out = str(out_col[i] or "").strip()
        prompts.append(
            _INSTRUCTION_AND_INPUT_PROMPT.format(instruction=str(inst), input=str(inp))
        )
        completions.append(out)
    return {"prompt": prompts, "completion": completions}


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
    col_names = list(raw_ds.column_names)
    ds = raw_ds.map(
        _prompt_completion_batches,
        batched=True,
        remove_columns=col_names,
        desc="format prompt+completion",
    )
    split = ds.train_test_split(test_size=0.1, seed=42)
    train_ds, eval_ds = split["train"], split["test"]
    print(f"train={len(train_ds)} eval={len(eval_ds)}")

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
        "optim": "adamw_8bit",
        "seed": 42,
        "max_length": MAX_SEQ_LENGTH,
        "packing": packing_effective,
        "completion_only_loss": True,
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

    sft_args = SFTConfig(**_filter_sft_config_kwargs(cfg_fields))

    resume = _resolve_resume()
    trainer = SFTTrainer(
        model=model,
        args=sft_args,
        processing_class=tokenizer,
        train_dataset=train_ds,
        eval_dataset=eval_ds_arg,
    )
    if resume:
        print(f"Resuming from checkpoint ({resume!r})...")
    print(
        f"Training: completion-only loss | save every {SAVE_STEPS} (keep {SAVE_TOTAL_LIMIT}), "
        f"eval={'off' if SKIP_EVAL else f'every {eval_steps}'}, "
        f"packing={packing_effective}, workers={TRAIN_DATALOADER_NUM_WORKERS}, "
        f"max_length={MAX_SEQ_LENGTH}"
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
