# Deployment handoff

Branch: `kevin/298b-integration`.

This note is for the person integrating the frozen generator. The modeling phase is finished. Do not retrain, rebuild the index, or rescore unless you are intentionally replacing a local artifact that is missing.

## Decision

Deploy **Relabel QLoRA + Factual Dense RAG v1 + atomic evidence extraction**.

RAG is in the system because grounded claims are a desired property. On the final paired comparison it does not improve Blog Voice or Writing Quality, and Persona Utility is slightly lower than the same Relabel adapter with retrieval off. Deploy it with those limits visible. The numbers are in the [evaluation summary](#final-evaluation-summary) below and in `experiments/EXP-20261005-003-relabel-rag-final`.

## System components

| Component | What it is | How the final path uses it |
|---|---|---|
| Base model | `unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit` | Loaded in 4-bit by Unsloth. One model stays in memory. |
| Relabel adapter | `host_finetune/output/lemkin_lora_relabel` | On for the final draft, at trained scale 1.0. Off during evidence extraction. |
| Chroma index | Persistent collection `lemkin_train_only`, 21,193 documents | Topic embedding query, cosine distance, top 4. |
| Embedding model | Ollama `nomic-embed-text` | Embeds the topic only. Embed calls set `num_gpu` to 0. |
| Atomic evidence extraction | `host_finetune/evidence_capsule.py` | The base model, adapter off, turns the four passages into JSON claims with exact support spans. |
| Deterministic validation | `partition_evidence` in the same module | Keeps a claim only when its span occurs in the named source. Raw passages are not copied into the Relabel prompt. |
| Final generation | `host_finetune.compare_adapters.generate` | Relabel writes the requested medium from the accepted claim block. |

Numeric checks live in `host_finetune/numeric_grounding.py`. They record unsupported quantities and do not replace the draft. Source-copy checks live in `host_finetune/rag_source_copying.py`. A span of 12 or more words can retry generation once. `host_finetune/grounding_repair.py` is imported so the repair helper can be tested, and the deployment path refuses to emit a repaired draft.

`generate.py` and Streamlit are the older Compose application. They are not wired to this adapter or this collection.

## Local artifacts required

None of these are in Git. Sizes were measured on the machine that produced EXP-20261005-003. Hashes and rebuild notes are in [ARTIFACTS.md](ARTIFACTS.md).

| Artifact | Expected path | Approximate size | Purpose | Regenerable |
|---|---|---:|---|---|
| Relabel adapter | `host_finetune/output/lemkin_lora_relabel` | 952 MB for the directory; 160 MB for the root `adapter_model.safetensors` | LoRA weights Unsloth loads | Yes, by rerunning EXP-20261003-005. Copy the directory for this handoff. |
| Chroma index | `host_finetune/output/chroma_lemkin_train_only` | 370 MB | Frozen 21,193-document train-only index | A rebuild command exists. Copying the directory is the way to keep this index. |
| 4-bit base model | Hugging Face cache for `unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit` | 5.5 GB | Frozen base under the adapter | Yes. First Unsloth load downloads it. |
| Ollama `nomic-embed-text` | Ollama model store | 274 MB | Query embeddings | Yes. `ollama pull nomic-embed-text` |

The adapter directory also contains training checkpoints and optimizer states. Inference reads the adapter root. Copy the whole directory so the root weights and tokenizer stay together.

The Chroma directory contains `chroma.sqlite3` and HNSW segment files. Copy the whole directory. Do not copy the sqlite file alone.

Other local folders under `host_finetune/output/` (merged GGUF exports, older adapters, `lemkin-relabel`) are not this deployment candidate.

## Setup

Python 3.11, a CUDA torch install, then the host requirements:

```powershell
py -3.11 -m venv host_finetune\.venv
host_finetune\.venv\Scripts\activate
python -m pip install --upgrade pip
pip install "torch>=2.10.0,<2.11.0" "torchvision>=0.25.0,<0.26.0" --index-url https://download.pytorch.org/whl/cu128
pip install -r host_finetune\requirements.txt
ollama pull nomic-embed-text
```

Place the adapter and the Chroma directory at the paths above, or set:

```powershell
$env:RELABEL_ADAPTER_PATH = "host_finetune/output/lemkin_lora_relabel"
$env:CHROMA_PERSIST_DIR = "host_finetune/output/chroma_lemkin_train_only"
$env:OLLAMA_BASE = "http://localhost:11434"
```

`.env.example` lists the same names with placeholders. Do not commit a real `.env`.

`CHROMA_COLLECTION_NAME` is the older Docker collection knob. This inference path does not read it. The collection argument defaults to `lemkin_train_only` and the loader refuses `lemkin_content`.

## Run the system

Start Ollama, activate the venv, and generate from the repository root:

```powershell
python -m host_finetune.compare_adapters --adapters relabel=host_finetune/output/lemkin_lora_relabel --retrieval --skip-score --out-dir experiments/EXP-YYYYMMDD-NNN-name
```

What that command does:

1. Embeds each gold topic with `nomic-embed-text`.
2. Queries `lemkin_train_only` for four passages.
3. Extracts and validates atomic claims with the adapter disabled.
4. Writes four drafts (blog, LinkedIn, X, talk) with Relabel enabled.
5. Writes `raw/relabel.jsonl` and does not call the judge.

There is no service process to start. The first integration task is to wrap this function for one topic and one or more media without changing retrieval, validation, or the Relabel prompt.

## Evaluate a saved run

```powershell
python -m host_finetune.score_saved_experiment --traces experiments/EXP-20261005-003-relabel-rag-final/raw/relabel.jsonl --out-dir experiments/EXP-20261005-003-relabel-rag-final
```

This needs `qwen2.5:14b` in Ollama. It checks the generation SHA-256 before scoring. The published metrics were already produced this way. A second run calls the judge again.

## Final evaluation summary

Source: `experiments/EXP-20261005-003-relabel-rag-final`.

- 120/120 drafts in `raw/relabel.jsonl`
- SHA-256 `266507aca9effe008596ca64dc199d2e0de32bafbd7a9765d738adb1d34ec550`
- Original successful 104 drafts were kept during recovery
- Eligible: 72/120 with RAG, 77/120 without RAG
- Eligible Blog Voice: 0.7765 with RAG, 0.7119 without (lower is closer)
- Writing Quality: 3.0602 with RAG, 3.1537 without
- Stance: 0.9906 with RAG, 0.9583 without
- Task: 0.7157 with RAG, 0.7160 without
- Persona Utility: 0.6878 on 19 eligible blogs with RAG, 0.705 on 23 without
- Grounding: 495 supported / 498 verifiable claims (600 examined; pooled support rate 0.994)
- Diagnostics: 60/120 unsupported-number rows, 2/120 source-copy flags, 103 EOS, 17 length stops

## Known limitations

- The model can still emit numbers that are not in the accepted evidence. Those rows are flagged and still shipped as drafts.
- Voice and writing metrics moved the wrong way relative to the retrieval-off Relabel baseline. Grounding is why RAG stays on.
- Two drafts copy a retrieved passage past the 12-word gate.
- Seventeen drafts ended because they used the token budget, not because they emitted EOS.
- Fourteen drafts fail the RAG grounding gate. Twelve of those have no verifiable claim. Two contain a contradicted claim (`gold_008` X and `gold_028` X).

## Recommended first deployment task

Add a single-topic entry point that calls the same retrieval, atomic-evidence, and Relabel generation path. Keep `lemkin_train_only`, k=4, `nomic-embed-text`, and scale 1.0. Do not route that entry point through Streamlit's `lemkin_content` collection or through an Ollama chat model.

Copy the adapter and the Chroma directory onto the integration machine before that work. Confirm the collection count is 21,193 and that `train_26148` is absent before generating.
