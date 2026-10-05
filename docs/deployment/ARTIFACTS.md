# Local artifacts

A clone of this repository does not contain the model weights or the Chroma index. Copy the directories below, or regenerate them with the commands in this file. This handoff did not retrain, re-embed, or rescore.

Paths are relative to the repository root.

## Relabel adapter

| Field | Value |
|---|---|
| Expected path | `host_finetune/output/lemkin_lora_relabel` |
| Directory size | about 952 MB, including checkpoints and optimizer states |
| Weights Unsloth loads | `adapter_model.safetensors` at the adapter root, 160.1 MB |
| SHA-256 of that file | `850e9aa88d04fa4e7c9490d5d974715c8a1018c47c7ada340d642b4b2afea530` |
| Base model | `unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit` |
| LoRA | rank 16, alpha 32, dropout 0, scale 1.0 at inference |
| Training run | `experiments/EXP-20261003-005-relabel-qlora` |
| Best eval loss | 1.3316 at step 2,674 |

The root `adapter_model.safetensors` is the inference file. Checkpoints under that directory are training history.

Regeneration is a full fine-tune, not a download. The recorded settings are `NUM_EPOCHS=2`, `LORA_R=16`, `LORA_ALPHA=32`, `LEARNING_RATE=1e-4`, `PER_DEVICE_BATCH=1`, `GRAD_ACCUM=16`, and the EXP-004 relabel dataset. From the repository root, with the host venv:

```powershell
$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"
$env:NUM_EPOCHS = "2"
$env:TRAIN_OPTIM = "adamw_8bit"
$env:MAX_VRAM_FRACTION = "0.85"
$env:PER_DEVICE_BATCH = "1"
$env:GRAD_ACCUM = "16"
$env:LORA_R = "16"
$env:LORA_ALPHA = "32"
$env:LEARNING_RATE = "1e-4"
$env:FINETUNE_DATASET = "experiments/EXP-20261003-004-relabel-continuations/raw/dataset.jsonl"
$env:SPLIT_MANIFEST = "experiments/EXP-20261003-004-relabel-continuations/raw/split_assignments.jsonl"
$env:ADAPTER_DIR = "host_finetune/output/lemkin_lora_relabel"
python -m host_finetune.finetune
```

Do not run that command to reproduce this handoff. Copy the existing adapter. A new train will not match the hash above.

## Chroma index

| Field | Value |
|---|---|
| Collection | `lemkin_train_only` |
| Documents | 21,193 |
| Split | `train` on every document |
| Absent id | `train_26148` |
| Embedding model | `nomic-embed-text` |
| Distance | cosine |
| Query | topic text only |
| Top k | 4 |
| Medium filter | off |
| Reranker | none |
| Expected path | `host_finetune/output/chroma_lemkin_train_only` |
| Directory size | about 370 MB |
| SHA-256 of `chroma.sqlite3` | `b81a70e74fe8a2ecb302e797163f8e4bfa005ba38321aa661004ccae8bec69e0` |

The sqlite hash does not cover the HNSW segment files in the same directory. Copy the entire directory.

Recorded corpus check: `experiments/EXP-20261004-009-dense-rag-v1/metrics/corpus.json` and `config/retriever.json` (count 21,193, `train_26148` absent, cosine). The retriever config in that experiment still contains a machine-local persist path from the original run. Use the relative path in this file.

A rebuild deletes and rewrites the collection. It was not run for this handoff, and no experiment manifest records a byte-identical rebuild of the 21,193-document directory. Current code can build a train-only index with the same selection rules:

```powershell
python -m host_finetune.rebuild_chroma --train-only --batch-size 16
```

That command needs Ollama, `nomic-embed-text`, and the EXP-004 dataset, split assignments, and gold-overlap dispositions already in the repo. After it finishes, check that the count is 21,193 and that `train_26148` is absent before using the index for generation. Prefer a directory copy when the frozen index is available.

## Base model

| Field | Value |
|---|---|
| Model id | `unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit` |
| Local cache size on the training machine | about 5.5 GB |
| In Git | no |

Unsloth downloads it on the first `FastLanguageModel.from_pretrained` call. Other local Llama 3.1 snapshots, including full-precision copies and merged `lemkin-relabel` exports, are not this load path.

## Ollama embedding model

| Field | Value |
|---|---|
| Name | `nomic-embed-text` |
| Local size | 274 MB |
| In Git | no |

```powershell
ollama pull nomic-embed-text
```

Generation does not use `llama3.1` or `lemkin-clone`. Those Ollama tags belong to the older stack.

## Judge model

Needed only to rescore. Not needed to generate.

| Field | Value |
|---|---|
| Name | `qwen2.5:14b` |
| Local size | about 9.0 GB |

```powershell
ollama pull qwen2.5:14b
```

## Final experiment

| Field | Value |
|---|---|
| Path | `experiments/EXP-20261005-003-relabel-rag-final` |
| Status | 120/120 drafts |
| Generation SHA-256 | `266507aca9effe008596ca64dc199d2e0de32bafbd7a9765d738adb1d34ec550` |
| File | `raw/relabel.jsonl` |
| Scores | `metrics/summary.json`, `metrics/comparison.json`, `metrics/rows.jsonl` |
| Recovery inputs | `experiments/EXP-20261005-003-relabel-rag-final-recovery` and `experiments/EXP-20261005-003-relabel-rag-final-recovery-rest` |

The recovery directories hold the four topics that failed the first pass. The merged 120-draft file is the scored artifact. Its bytes were not rewritten when local paths were removed from the config JSON.
