"""
Phase 8 — QLoRA fine-tuning of LLaMA 3.1 8B on Jason Lemkin's writing.

Designed for Google Colab A100 (free tier).  Freezes the base model weights
and trains only small LoRA adapter matrices (~1 % of parameters).

Run: python persona_pipeline/training/finetune_lora.py
"""
from __future__ import annotations

import json
import os
import sys
import warnings
from pathlib import Path

REPO_ROOT = str(Path(__file__).resolve().parents[2])
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model, TaskType
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from trl import SFTTrainer

from persona_pipeline.config.config import LORA_OUTPUT, MODEL_NAME, TRAINING_DIR


def load_training_data(path: str) -> Dataset:
    records: list[dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return Dataset.from_list(records)


def format_instruction(example: dict) -> dict:
    """Convert instruction/input/output triple into a single 'text' field."""
    text = (
        f"### Instruction:\n{example['instruction']}\n\n"
        f"### Input:\n{example.get('input', '')}\n\n"
        f"### Response:\n{example['output']}"
    )
    return {"text": text}


def main() -> None:
    data_path = str(Path(str(TRAINING_DIR)) / "instruction_format.jsonl")
    if not os.path.exists(data_path):
        warnings.warn(f"Training data not found at {data_path}; run load_duckdb.py first.")
        return

    # 1. Load & split data (90/10 at document level)
    dataset = load_training_data(data_path)
    dataset = dataset.map(format_instruction)
    split = dataset.train_test_split(test_size=0.1, seed=42)
    train_ds = split["train"]
    val_ds = split["test"]
    print(f"Training samples: {len(train_ds)}, Validation samples: {len(val_ds)}")

    # 2. Quantization config (4-bit for QLoRA)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    # 3. Load base model + tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    model.config.use_cache = False

    # 4. LoRA config
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )

    # 5. Training arguments
    os.makedirs(LORA_OUTPUT, exist_ok=True)
    training_args = TrainingArguments(
        output_dir=LORA_OUTPUT,
        num_train_epochs=3,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        fp16=True,
        logging_steps=10,
        save_strategy="epoch",
        evaluation_strategy="epoch",
        report_to="none",
    )

    # 6. SFTTrainer
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tokenizer,
        peft_config=lora_config,
        dataset_text_field="text",
        max_seq_length=1024,
    )

    # 7. Train
    trainer.train()

    # 8. Save adapter
    trainer.model.save_pretrained(LORA_OUTPUT)
    tokenizer.save_pretrained(LORA_OUTPUT)
    print(f"\nLoRA adapter saved -> {LORA_OUTPUT}")

    # Print training loss curve
    if trainer.state.log_history:
        print("\n=== Training Loss Curve ===")
        for entry in trainer.state.log_history:
            if "loss" in entry:
                step = entry.get("step", "?")
                loss = entry["loss"]
                print(f"  step {step}: loss={loss:.4f}")


if __name__ == "__main__":
    main()
