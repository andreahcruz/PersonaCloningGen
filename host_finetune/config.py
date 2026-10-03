"""Shared paths + env config for the host-side fine-tune pipeline."""
from __future__ import annotations

import os
from pathlib import Path
HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent


def _parse_csv_list(value: str) -> list[str]:
    parts = value.replace(";", ",").split(",")
    return [p.strip() for p in parts if p.strip()]


def _truthy(env_name: str, default: bool = True) -> bool:
    raw = os.environ.get(env_name)
    if raw is None or not str(raw).strip():
        return default
    return str(raw).strip().lower() in ("1", "true", "yes", "on")

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
ADAPTER_DIR = Path(
    os.environ.get("ADAPTER_DIR", str(OUTPUT_DIR / "lemkin_lora"))
).expanduser()
# Merge step only: after a cancelled run, point at a checkpoint dir, e.g.
# host_finetune\output\lemkin_lora\checkpoint-800
LORA_ADAPTER_PATH = Path(
    os.environ.get("LORA_ADAPTER_PATH", str(ADAPTER_DIR))
).expanduser()
GGUF_DIR = Path(os.environ.get("GGUF_DIR", str(OUTPUT_DIR / "lemkin-clone"))).expanduser()
# Newer Unsloth writes *.gguf under ``{GGUF_DIR.name}_gguf`` next to the HF merge dir,
# not inside GGUF_DIR itself.
GGUF_SIDECAR_DIR = Path(
    os.environ.get("GGUF_SIDECAR_DIR", str(OUTPUT_DIR / f"{GGUF_DIR.name}_gguf"))
).expanduser()
LAST_RUN_FILE = HERE / ".last_run"
DATASET_LOCAL = DATA_DIR / "dataset.jsonl"
# Cleaned-source QLoRA file. finetune.py trains this by default.
# DATASET_LOCAL stays the MinIO download path so the watcher cannot overwrite it.
CLEANED_SFT_DATASET = DATA_DIR / "dataset_from_cleaned_sources_fit512.jsonl"
DEFAULT_SPLIT_MANIFEST = (
    REPO_ROOT
    / "experiments"
    / "EXP-20261001-002-medium-stratified-split"
    / "raw"
    / "split_assignments.jsonl"
)
# Per-label chunk size (chars): keep ~3.125 chars per MAX_SEQ_LENGTH slot (3200/1024).
# At MAX_SEQ_LENGTH=512 → 1600 chars/label. Override via SFT_CHUNK_OUTPUT_CHARS.
SFT_CHUNK_OUTPUT_CHARS = int(os.environ.get("SFT_CHUNK_OUTPUT_CHARS", "1600"))
# clean_dataset per-row cap; should exceed a single chunk (rare long rows / old exports).
DATASET_MAX_OUTPUT_CHARS = int(os.environ.get("DATASET_MAX_OUTPUT_CHARS", "2000"))
# Remove transcript stage directions ([Music], …) and drop exact-duplicate outputs after scrub.
SCRUB_TRANSCRIPT_TAGS = _truthy("SCRUB_TRANSCRIPT_TAGS", True)
DEDUPE_EXACT_OUTPUT = _truthy("DEDUPE_EXACT_OUTPUT", True)
# Explicit NF4 + double quant (BitsAndBytesConfig) in finetune. Set USE_EXPLICIT_BNB_CONFIG=0 to rely on defaults.
USE_EXPLICIT_BNB_CONFIG = _truthy("USE_EXPLICIT_BNB_CONFIG", True)
# Set to 1 to skip ``clean_dataset`` (use raw MinIO export as-is).
SKIP_DATASET_CLEAN = os.environ.get("SKIP_DATASET_CLEAN", "").lower() in (
    "1",
    "true",
    "yes",
)
# After MinIO download / before clean_dataset: split long outputs (matches Spark chunking).
# Set SKIP_CHUNK_DATASET=1 only if Spark already chunked (and you trust no mega-rows).
CHUNK_DATASET_AFTER_DOWNLOAD = os.environ.get("SKIP_CHUNK_DATASET", "").lower() not in (
    "1",
    "true",
    "yes",
)

# ── Training hyperparams (16 GB VRAM on Llama 3.1 8B QLoRA) ──────────────
# Override the model with HF_MODEL_NAME if dropping to Llama 3.2 3B for OOM.
HF_MODEL_NAME = os.environ.get(
    "HF_MODEL_NAME", "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit"
)
# 512 + 1600-char SFT chunks: lighter VRAM footprint (raise via env if probe allows).
MAX_SEQ_LENGTH = int(os.environ.get("MAX_SEQ_LENGTH", "512"))
# LoRA defaults widened slightly (VRAM permitting). Narrow with LORA_TARGET_MODULES / lower LORA_R.
LORA_R = int(os.environ.get("LORA_R", "16"))
LORA_ALPHA = int(os.environ.get("LORA_ALPHA", "32"))
_lora_tgt = os.environ.get("LORA_TARGET_MODULES")
if _lora_tgt:
    LORA_TARGET_MODULES = _parse_csv_list(_lora_tgt)
else:
    LORA_TARGET_MODULES = [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ]
# 0 enables Unsloth's fast LoRA patch path; >0 disables it (noticeable slowdown).
LORA_DROPOUT = float(os.environ.get("LORA_DROPOUT", "0"))
NUM_EPOCHS = int(os.environ.get("NUM_EPOCHS", "1"))
PER_DEVICE_BATCH = int(os.environ.get("PER_DEVICE_BATCH", "2"))
GRAD_ACCUM = int(os.environ.get("GRAD_ACCUM", "8"))
LEARNING_RATE = float(os.environ.get("LEARNING_RATE", "1e-4"))

# ── VRAM-saving knobs (RTX 5070 Ti is 16 GB; Windows apps eat ~1-3 GB at idle) ──
# ``adamw_8bit`` keeps Adam state on GPU; ``paged_adamw_8bit`` lets bnb page
# optimizer state to CPU on pressure (slightly slower but no spillover stalls).
TRAIN_OPTIM = os.environ.get("TRAIN_OPTIM", "paged_adamw_8bit")
# Liger Kernel ships a fused / chunked cross-entropy that drops final-layer
# memory by ~30-50% on Llama-3 (vocab=128k). Off by default for portability;
# turn on if ``pip install liger-kernel`` succeeded for your CUDA build.
USE_LIGER_KERNEL = _truthy("USE_LIGER_KERNEL", False)
# Print VRAM telemetry at every logging step (helps spot spillover into
# Windows shared-memory fallback, which silently slows training).
LOG_VRAM_USAGE = _truthy("LOG_VRAM_USAGE", True)
# Cap PyTorch's CUDA allocator at this fraction of physical VRAM so it can't
# silently overflow into Windows' shared-memory fallback (which paging through
# PCIe makes training 5-10× slower). On a 16 GB card 0.93 ≈ 15.2 GB of usable
# budget; anything above this throws a clean OOM instead of a multi-hour stall.
MAX_VRAM_FRACTION = float(os.environ.get("MAX_VRAM_FRACTION", "0.93"))

# Checkpoints + eval (long runs: frequent saves, rarer eval avoids slowdown from
# full validation passes every N steps).
SAVE_STEPS = int(os.environ.get("SAVE_STEPS", "200"))
SAVE_TOTAL_LIMIT = int(os.environ.get("SAVE_TOTAL_LIMIT", "4"))
# If unset, finetune.py defaults to max(5 * SAVE_STEPS, 500). Set to 0 with
# SKIP_EVAL=1 to disable validation entirely (fastest).
EVAL_STEPS = os.environ.get("EVAL_STEPS")
SKIP_EVAL = os.environ.get("SKIP_EVAL", "").lower() in ("1", "true", "yes")
# Ignored when SKIP_EVAL=1 (no metric to compare).
LOAD_BEST_MODEL_AT_END = os.environ.get("LOAD_BEST_MODEL_AT_END", "1").lower() not in (
    "0",
    "false",
    "no",
)

# Resume: pass path to a checkpoint folder, or "1"/"true" to pick the latest
# checkpoint under the training output dir.
RESUME_FROM_CHECKPOINT = os.environ.get("RESUME_FROM_CHECKPOINT", "").strip()


def _optional_positive_int(env_name: str) -> int | None:
    raw = os.environ.get(env_name, "").strip()
    if not raw or not str(raw).isdigit():
        return None
    v = int(raw)
    return v if v > 0 else None


# Smoke / vibe-check: ``MAX_STEPS`` caps total optimizer steps via ``TrainingArguments``
# (``SFTConfig(max_steps=...)`` — do not pass ``max_steps`` to ``Trainer.train`` under Unsloth).
# ``FINETUNE_TRAIN_HEAD_N`` keeps only the first N rows of the train split after the 90/10 split
# (full eval split unchanged unless you shrink it too).
MAX_STEPS = _optional_positive_int("MAX_STEPS")
FINETUNE_TRAIN_HEAD_N = _optional_positive_int("FINETUNE_TRAIN_HEAD_N")

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

# ── merge_and_export.py ──────────────────────────────────────────────────
# Unsloth's ``save_pretrained_gguf`` calls bitsandbytes' ``dequantize_4bit`` to
# materialize bf16 weights for saving. On torch 2.10.0+cu128 + Blackwell GPUs,
# bnb's C++ extensions can't load and the Python fallback produces numerically
# wrong floats — the resulting GGUF generates garbage tokens. We bypass by
# merging against the **un-quantized** base in plain bf16 (no bnb in merge
# path) and then invoking llama.cpp's converter + quantizer directly.
# See ``host_finetune/diagnose_pipeline.py`` for the evidence chain.
MERGE_BASE_MODEL = os.environ.get(
    "MERGE_BASE_MODEL",
    HF_MODEL_NAME[: -len("-bnb-4bit")] if HF_MODEL_NAME.endswith("-bnb-4bit") else HF_MODEL_NAME,
)
# CPU merge is slow but reliable; set MERGE_DEVICE=cuda only if you have enough
# VRAM for the un-quantized base in bf16 (~16 GB for Llama 3.1 8B).
MERGE_DEVICE = os.environ.get("MERGE_DEVICE", "cpu")
LLAMA_CPP_DIR = Path(
    os.environ.get("LLAMA_CPP_DIR", str(Path.home() / ".unsloth" / "llama.cpp"))
).expanduser()

# ── Ollama ───────────────────────────────────────────────────────────────
OLLAMA_MODEL_NAME = os.environ.get("OLLAMA_MODEL_NAME", "lemkin-clone")
MODELFILE_PATH = HERE / "Modelfile"
