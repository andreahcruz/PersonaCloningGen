# Jason Lemkin persona generation (DATA 298B)

This repository generates Jason Lemkin / SaaStr writing in four media: blog, LinkedIn, X, and talk.

The deployment candidate is **Relabel QLoRA + Factual Dense RAG v1 + atomic evidence extraction**. RAG is deployed because it grounds verifiable claims. It does not improve Blog Voice or Writing Quality, and it slightly lowers Persona Utility. Those limits are part of the decision. See [deployment handoff](docs/deployment/DEPLOYMENT_HANDOFF.md) and [artifact locations](docs/deployment/ARTIFACTS.md).

`docs/project_state.md` is a 2026-10-04 audit. It is not the deployment description.

## Final architecture

```
user topic
  -> nomic-embed-text (Ollama, CPU)
  -> Chroma collection lemkin_train_only (cosine, top 4)
  -> atomic evidence extraction (same 4-bit Llama, Relabel adapter off)
  -> deterministic evidence validation
  -> Relabel QLoRA at trained scale 1.0
  -> generated blog, LinkedIn, X, or talk
```

Raw retrieved prose is not inserted into the final generation prompt. The generator sees accepted claim text only. Support spans stay in provenance.

Retrieval uses the topic string only. There is no medium filter and no reranker. One topic is retrieved once and reused across the four media. Expected facts from the gold file are never placed in the prompt.

Numeric mismatches are recorded and do not block the draft. A source-copy span of 12 or more words may trigger one retry. Post-generation repair is not on this path.

## Frozen RAG

| Setting | Value |
|---|---|
| Collection | `lemkin_train_only` |
| Documents | 21,193 |
| Split | every document is `split=train` |
| Excluded leakage row | `train_26148` is absent |
| Embedding model | `nomic-embed-text` |
| Distance | cosine |
| Query | topic text only |
| `retrieval_k` | 4 |
| Medium filter | off |
| Reranker | none |
| Persist directory | `host_finetune/output/chroma_lemkin_train_only` |

This is not the older 21,194-document build, and it is not the Compose collection `lemkin_content`.

## Working model

Relabel QLoRA, loaded at trained scale 1.0.

- Base: `unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit`
- Adapter directory: `host_finetune/output/lemkin_lora_relabel`
- LoRA rank 16, alpha 32
- Generation length budget: blog 768, LinkedIn 320, X 160, talk 768
- Sampling: temperature 0.7, top-p 0.9, repetition penalty 1.15, seed base 42

The inference entry point is `host_finetune.compare_adapters`. Streamlit and `generate.py` still target the older Compose collection and an Ollama generator. They are not this deployment path.

## Local services and dependencies

Inference needs:

- Python 3.11 and the host fine-tune environment in `host_finetune/requirements.txt`
- NVIDIA GPU, 4-bit Llama 3.1 8B (developed on an RTX 5070 Ti, 16 GB)
- [Ollama](https://ollama.com) running locally, with `nomic-embed-text` pulled
- The Relabel adapter directory and the Chroma persist directory on disk

Those model and index files are not in Git. Paths, sizes, and copy-or-rebuild notes are in [ARTIFACTS.md](docs/deployment/ARTIFACTS.md).

Evaluation of a saved run also needs Ollama model `qwen2.5:14b`. That judge is not the generator.

## Setup

```powershell
py -3.11 -m venv host_finetune\.venv
host_finetune\.venv\Scripts\activate
python -m pip install --upgrade pip
pip install "torch>=2.10.0,<2.11.0" "torchvision>=0.25.0,<0.26.0" --index-url https://download.pytorch.org/whl/cu128
pip install -r host_finetune\requirements.txt
ollama pull nomic-embed-text
```

Optional environment file, with placeholders only:

```powershell
copy .env.example .env
```

Inference reads these variables when they are set. Unset, the batch command below is unchanged.

| Variable | Role |
|---|---|
| `RELABEL_ADAPTER_PATH` | Relabel adapter directory. Used when `--adapters` is omitted. |
| `CHROMA_PERSIST_DIR` | Train-only Chroma directory. |
| `OLLAMA_BASE` | Ollama host. Default `http://localhost:11434`. |
| `OLLAMA_EMBED_MODEL` | Embedding model for index builds. Default `nomic-embed-text`. |
| `EXPERIMENT_OUT_DIR` | Default output directory for `compare_adapters`. |

Do not set `CHROMA_COLLECTION_NAME` to steer this path. That variable belongs to the older `lemkin_content` rebuild. This path defaults to `lemkin_train_only` and refuses `lemkin_content`.

## Run inference

From the repository root, with the host venv active and Ollama running:

```powershell
python -m host_finetune.compare_adapters --adapters relabel=host_finetune/output/lemkin_lora_relabel --retrieval --skip-score --out-dir experiments/EXP-YYYYMMDD-NNN-name
```

`--retrieval` turns on the frozen factual path. `--skip-score` writes drafts and does not call the judge. The gold topic file is `data/openai_ft/lemkin_gold_eval.jsonl` (30 topics, 120 drafts when all four media are generated).

`--topic-ids gold_009,gold_015` regenerates named topics and keeps each topic's original seed index. The command refuses to overwrite an existing `raw/relabel.jsonl` during a targeted run. Use a new `--out-dir`.

This command loads the GPU model. It is the generation entry point, not a long-running API.

## Evaluate an existing experiment

Scoring does not train or regenerate. It requires `qwen2.5:14b` in Ollama. The saved final run is already scored. Run this only to rescore that frozen file:

```powershell
python -m host_finetune.score_saved_experiment --traces experiments/EXP-20261005-003-relabel-rag-final/raw/relabel.jsonl --out-dir experiments/EXP-20261005-003-relabel-rag-final
```

The scorer refuses a trace file whose SHA-256 is not `266507aca9effe008596ca64dc199d2e0de32bafbd7a9765d738adb1d34ec550`.

## Where artifacts live

| Artifact | Path | In Git |
|---|---|---|
| Relabel adapter | `host_finetune/output/lemkin_lora_relabel` | no |
| Train-only Chroma | `host_finetune/output/chroma_lemkin_train_only` | no |
| 4-bit base model | Hugging Face cache for `unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit` | no |
| `nomic-embed-text` | Ollama local model store | no |
| Final 120 drafts and scores | `experiments/EXP-20261005-003-relabel-rag-final` | yes |

## Known limitations

- Unsupported numeric details can still be generated. On the final run, 60 of 120 drafts have an unsupported-number diagnostic. That diagnostic does not reject the draft.
- RAG improves grounding of verifiable claims. It does not improve Blog Voice or Writing Quality, and Persona Utility is slightly lower than retrieval-off Relabel.
- Source copying is monitored. Two of 120 final drafts are flagged. The gate can retry once.
- Some outputs stop because they hit the generation budget. The final run has 103 EOS stops and 17 length stops.
- Eligible-draft counts are lower with RAG (72/120) than without it (77/120). Fourteen RAG drafts fail the grounding gate.

## Final evaluation

Registered run: `experiments/EXP-20261005-003-relabel-rag-final`.

Generation file: `raw/relabel.jsonl`, 120 drafts, 30 topics, 30 of each medium. SHA-256 `266507aca9effe008596ca64dc199d2e0de32bafbd7a9765d738adb1d34ec550`. The first pass saved 104 drafts. Four topics were recovered without regenerating those 104. The merged file is the scored file.

Judge: blinded `qwen2.5:14b`, temperature 0. Schema `unified_eval.v1`. Blog Voice is a style distance. Lower is closer to the held-out Lemkin blog profile. It is not an authorship proof.

| | Relabel, no RAG | Relabel + RAG |
|---|---:|---:|
| Eligible drafts | 77/120 | 72/120 |
| Eligible Blog Voice | 0.7119 | 0.7765 |
| Writing Quality | 3.1537 | 3.0602 |
| Stance | 0.9583 | 0.9906 |
| Task | 0.7160 | 0.7157 |
| Persona Utility | 0.705 on 23 eligible blogs | 0.6878 on 19 eligible blogs |

RAG grounding on the retrieved-evidence run: 600 claims examined, 498 verifiable, 495 supported, 1 unsupported, 2 contradicted. Pooled support rate 0.994.

The selector ranks retrieval-off first because Blog Voice is the first ranking key. Relabel + RAG is still the deployment candidate because grounded claims are a required system property.

## Tests

Unit tests for the deployment path do not load weights:

```powershell
python -m pytest tests/test_compare_prompts.py tests/test_evidence_capsule.py tests/test_numeric_grounding.py tests/test_rag_source_copying.py tests/test_train_only_index.py tests/test_grounding_repair.py tests/test_generation_diagnostics.py -q
```

## Older Compose application

`docker-compose.yml` still defines Airflow, Spark, MinIO, a Chroma server, and Streamlit. That stack loads collection `lemkin_content` and generates with an Ollama model. It is an earlier application. It is not the Relabel adapter, not `lemkin_train_only`, and not the atomic-evidence prompt.

Do not treat a Streamlit reply from that stack as output from this deployment candidate.
