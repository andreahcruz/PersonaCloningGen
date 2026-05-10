"""Shared paths + env config for the host-side fine-tune pipeline."""
from __future__ import annotations

import os
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent

# ── MinIO (the Docker stack's MinIO is published on localhost from host POV) ──
MINIO_ENDPOINT = os.environ.get("MINIO_ENDPOINT_HOST", "http://localhost:9000")
MINIO_ACCESS_KEY = os.environ.get("MINIO_ACCESS_KEY", "minioadmin")
MINIO_SECRET_KEY = os.environ.get("MINIO_SECRET_KEY", "minioadmin")
BUCKET_PROCESSED = os.environ.get("MINIO_BUCKET_PROCESSED", "lemkin-processed")

# Object keys produced by the Airflow DAG (mark_training_ready)
DATASET_KEY = "training/dataset.jsonl"
SENTINEL_KEY = "training/_READY"
ADAPTER_PREFIX = "training/adapters/lemkin_lora/"

# ── Local layout ─────────────────────────────────────────────────────────
DATA_DIR = HERE / "data"
OUTPUT_DIR = HERE / "output"
ADAPTER_DIR = OUTPUT_DIR / "lemkin_lora"
# Merge step only: after a cancelled run, point at a checkpoint dir, e.g.
# host_finetune\output\lemkin_lora\checkpoint-800
LORA_ADAPTER_PATH = Path(
    os.environ.get("LORA_ADAPTER_PATH", str(ADAPTER_DIR))
).expanduser()
GGUF_DIR = OUTPUT_DIR / "lemkin-clone"
LAST_RUN_FILE = HERE / ".last_run"
DATASET_LOCAL = DATA_DIR / "dataset.jsonl"

# ── Training hyperparams (16 GB VRAM on Llama 3.1 8B QLoRA) ──────────────
# Override the model with HF_MODEL_NAME if dropping to Llama 3.2 3B for OOM.
HF_MODEL_NAME = os.environ.get(
    "HF_MODEL_NAME", "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit"
)
MAX_SEQ_LENGTH = int(os.environ.get("MAX_SEQ_LENGTH", "1024"))
LORA_R = int(os.environ.get("LORA_R", "16"))
LORA_ALPHA = int(os.environ.get("LORA_ALPHA", "32"))
# 0 enables Unsloth's fast LoRA patch path; >0 disables it (noticeable slowdown).
LORA_DROPOUT = float(os.environ.get("LORA_DROPOUT", "0"))
NUM_EPOCHS = int(os.environ.get("NUM_EPOCHS", "3"))
PER_DEVICE_BATCH = int(os.environ.get("PER_DEVICE_BATCH", "2"))
GRAD_ACCUM = int(os.environ.get("GRAD_ACCUM", "8"))
LEARNING_RATE = float(os.environ.get("LEARNING_RATE", "2e-4"))

# Checkpoints + eval (long runs: frequent saves, rarer eval avoids slowdown from
# full validation passes every N steps).
SAVE_STEPS = int(os.environ.get("SAVE_STEPS", "200"))
SAVE_TOTAL_LIMIT = int(os.environ.get("SAVE_TOTAL_LIMIT", "4"))
# If unset, finetune.py defaults to max(5 * SAVE_STEPS, 500). Set to 0 with
# SKIP_EVAL=1 to disable validation entirely (fastest).
EVAL_STEPS = os.environ.get("EVAL_STEPS")
SKIP_EVAL = os.environ.get("SKIP_EVAL", "").lower() in ("1", "true", "yes")

# Resume: pass path to a checkpoint folder, or "1"/"true" to pick the latest
# checkpoint under the training output dir.
RESUME_FROM_CHECKPOINT = os.environ.get("RESUME_FROM_CHECKPOINT", "").strip()

# On Windows, dataloader workers > 0 can cause slowdowns or instability; override if needed.
_TRAIN_WORKERS_DEFAULT = "0" if os.name == "nt" else "2"
TRAIN_DATALOADER_NUM_WORKERS = int(
    os.environ.get("TRAIN_DATALOADER_NUM_WORKERS", _TRAIN_WORKERS_DEFAULT)
)

# Packed sequences improve throughput when MAX_SEQ_LENGTH is large enough.
# finetune.py turns packing off automatically when MAX_SEQ_LENGTH < 1024 (packed
# batches can otherwise exceed the cap and break Unsloth fused CE on Windows).
PACKING = os.environ.get("PACKING", "1").lower() not in ("0", "false", "no")

GGUF_QUANT = os.environ.get("GGUF_QUANT", "q4_k_m")

# ── Ollama ───────────────────────────────────────────────────────────────
OLLAMA_MODEL_NAME = os.environ.get("OLLAMA_MODEL_NAME", "lemkin-clone")
MODELFILE_PATH = HERE / "Modelfile"
