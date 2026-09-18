# Model Improvements and Design Changes

Linear: **PUK-11** · Project: Lemkin persona-cloning pipeline (DATA 298B, Group 2)

This document records what changed in the project since the first ETL commit
(2026-03-24), why it changed, and what is still open. Everything below is taken
from the git history, code comments, and the log files committed under
`host_finetune/`. Commit hashes are given so each claim can be checked.

**Scope note:** the repo contains the evaluation harness (`host_finetune/eval_rag.py`)
but no evaluation *results*. This document therefore describes what was changed and
why, not measured quality gains. See [Open items](#8-open-items).

---

## 1. Timeline at a glance

| Date | Commit | Change |
|---|---|---|
| 2026-03-24 | `efb9ca3`, `f933091` | Dockerized ETL pipeline: Airflow, Spark, MinIO, Chroma, Compose; README |
| 2026-03-26 | `bc57596`, `af0c83d` | Persona pipeline project imported and merged |
| 2026-03-30 | `dcbe155` | Persona switched from patio11 to Jason Lemkin (SaaStr) |
| 2026-04-06 | `79bd2f7` → `0aaee52` | Stack stabilised: Spark run works, Chroma fixed, Streamlit first run ("BASIC PIPELINE DONE") |
| 2026-04-08 | `657ee7e` | Data notebook v1 (rubric evidence, pipeline demo) |
| 2026-05-10 | `a60785a` → `7a26d4e` | Fine-tune added (attempt 1), then attempt 2 (produced gibberish), then FTv2 |
| 2026-05-11 | `40567e3` | Root cause of gibberish found and worked around ("Working non-gibberish qloraFT") |
| 2026-05-11 | `7219f57` | Evaluation harness added |
| 2026-05-12 | `0990c93` | Data cleaning v2, transcript repunctuation, footer scrub, RAG rebuild from cleaned corpus |

---

## 2. Persona and data-source changes

**patio11 → Jason Lemkin (`dcbe155`).**
- The Hacker News and Reddit collectors were removed.
- YouTube transcript collectors were added (`collect_lemkin_yt`, `collect_saastr_yt`, built on `yt-dlp` with retry, resume, and JSONL streaming).
- The blog collector was rewritten for paginated SaaStr scraping.
- Topics, formats, and inference prompts were retargeted to Lemkin.

**Current corpus (`data/*.jsonl`, raw line counts):**

| Source | File | Rows |
|---|---|---|
| X posts | `jasonlk_originals.jsonl` | 25,997 |
| Blog | `jasonlemkin_blog.jsonl` | 3,502 |
| LinkedIn | `jasonlemkinlinkedin.jsonl` | 2,675 |
| YouTube (Jason) | `jasonmlemkinyoutubetranscripts.jsonl` | 462 |
| YouTube (SaaStr) | `saastryoutubetranscripts.jsonl` | 199 |
| **Total** | | **32,835** |

**Ingestion filters (`dags/lemkin_pipeline_dag.py`).** X replies, reposts/quotes,
and posts shorter than 25 characters are skipped. Rows with no body text
(for example YouTube rows with `transcript_text: null`) are skipped.

---

## 3. Pipeline and infrastructure changes

The architecture is unchanged in shape:
`JSONL → Airflow → MinIO (raw) → Spark clean/chunk/embed → MinIO (processed) → Chroma → generate.py / Streamlit`.
The changes below are the fixes and additions that made it run reliably.

| Problem | Fix | Where |
|---|---|---|
| Chroma "Could not connect to tenant default_tenant" (client/server version and old volume mismatch) | Pin Chroma server image and pip `chromadb` to the same version, **1.5.5** | `docker-compose.yml`, `Dockerfile.airflow` (`0aaee52`) |
| `load_to_chroma` crashed with `TypeError` in `posthog.types` | Airflow image on **Python 3.10** and `posthog<3` | `Dockerfile.airflow` |
| Spark `InvalidClassException` (pip PySpark driver vs Bitnami worker JAR mismatch) | Default Spark connection is **`local[*]`** inside the Airflow container | `AIRFLOW_CONN_SPARK_DEFAULT` |
| `S3AFileSystem` not found | `SparkSubmitOperator` passes `hadoop-aws` and AWS bundle via `--packages` | DAG |
| Airflow log streaming returned 403 (JWT signature) | Same `WEBSERVER__SECRET_KEY` on webserver and scheduler | `docker-compose.yml` |
| `airflow-init` raced the webserver and scheduler | `depends_on: airflow-init: service_completed_successfully`; init uses `python -m airflow` | `docker-compose.yml` |
| Ollama timeouts and empty embeddings under load | Retry with exponential backoff and jitter; text capped at 12,000 chars; `SPARK_EMBED_PARTITIONS` limits concurrent `/api/embed` calls | `spark_jobs/clean_and_embed.py` |
| Streamlit and Airflow needed conflicting `email-validator` pins | Streamlit image installs only `requirements-streamlit.txt` | `Dockerfile` |

**New handoff to fine-tuning (`a60785a`).** The DAG gained a `mark_training_ready`
task. Spark writes an SFT dataset to `training/dataset_jsonl/`; the task promotes
it to a stable `training/dataset.jsonl` and writes a `training/_READY` sentinel.
A host-side watcher (`host_finetune/watcher.py`) polls for the sentinel and runs
fine-tune → merge/export → register-in-Ollama. This keeps GPU work on the host,
outside Docker.

Current DAG shape: `extract_to_minio → trigger_spark_clean → [load_to_chroma, mark_training_ready]`,
plus `extract_to_minio → extract_persona`.

---

## 4. Training-data design changes

**SFT export from Spark.** Documents (not 500-word RAG chunks) are exported as
Alpaca rows: `instruction = "Write in the style of Jason Lemkin about: {title}"`,
`output = body`. Rules in `spark_jobs/clean_and_embed.py`:
- Documents under 50 words are dropped; outputs under 80 characters are dropped.
- Rows containing the Unicode replacement character (`U+FFFD`) are dropped.
- Known junk titles are dropped (for example "test post with ai chat").
- Untitled sources (LinkedIn, X) get a short fallback topic per source so every row has a usable instruction.
- Long bodies are split at **1,600 characters** per label, preferring paragraph breaks, and titled "(part k of N)". 1,600 characters was chosen to pair with `MAX_SEQ_LENGTH = 512` (same 3200:1024 ratio as the earlier setting).
- The same splitting logic lives in `host_finetune/sft_chunk_utils.py`; the two copies must be kept in sync.

**Footer scrub (`spark_jobs/corpus_footer_scrub.py`, `0990c93`).** Removes SaaStr
"Related Posts / Read more / You may also like" tails (only when they start in the
last 38% of the text, so mid-article mentions survive) and the "— Jason …" X
attribution tail. It is used by Spark, the host cleaners, and RAG context formatting.
*Why:* Linear issue PUK-7 records that early training exposed ads and event
promotions in the data.

**Cleaning v2 (`host_finetune/clean_dataset_v2.py`, `0990c93`).** Run on 2026-05-11
against the committed `clean_v2_output.txt`:

| Step | Result |
|---|---|
| Rows loaded | 21,151 |
| "Echo-bug" rows dropped (instruction title ≈ literal prefix of the output) | 1,939 |
| Podcast intros / sponsor reads dropped | 41 |
| Non-Lemkin guest openers dropped | 5 |
| Cyrillic/CJK-heavy rows dropped | 6 |
| Groups re-stitched from chunks (seams previously cut mid-word) | 3,502 |
| Chunks classified transcript-style (no caps, no punctuation) | 6,736 of 19,680 (34.2%) |
| Rows repunctuated | 6,736 |
| Short rows / duplicates dropped | 75 / 24 |
| **Rows written** | **19,581** |

The v2 cleaner rebuilds each document from its parts, scrubs URLs, strips verbal
fillers (um, uh, hmm, …), restores punctuation and casing on transcripts
(`repunctuate.py`, using `fullstop-punctuation-multilang-large` plus a casing
heuristic), and re-chunks on sentence boundaries so no chunk starts mid-word.
Phrases like "you know" and "I mean" are deliberately kept because Lemkin also
uses them in writing. If repunctuation changes the word count by more than 5%,
the original text is kept.

**RAG corpus rebuilt from the cleaned set (`rebuild_chroma.py`).** The
`lemkin_content` collection was dropped and re-embedded from the cleaned
`dataset.jsonl` with the same `nomic-embed-text` model:
19,581 rows upserted, 0 failures, collection count 19,581 (448 s). Retrieval in
`generate.py` needed no changes.

---

## 5. Model changes

**Base and method.** QLoRA on `unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit`
(Unsloth + TRL), on a single 16 GB RTX 5070 Ti (Blackwell, sm_120), Windows host.
The base model stays frozen; only a LoRA adapter is trained.

**Configuration across attempts** (from `host_finetune/config.py` at each commit):

| Setting | Attempt 1 (`a60785a`) | Attempt 2 (`8570fe5`, gibberish) | FTv2 / working (`7a26d4e`, `40567e3`) |
|---|---|---|---|
| `MAX_SEQ_LENGTH` | 1024 | 512 | 512 |
| LoRA rank / alpha | 16 / 32 | 8 / 16 | 16 / 32 |
| Epochs | 3 | 1 | 1 |
| Learning rate | 2e-4 | 1e-4 | 1e-4 |
| Optimizer | default | default | default in FTv2; `paged_adamw_8bit` from `40567e3` |
| SFT label chunk size | not chunked | 1,600 chars | 1,600 chars |

Other changes made along the way:
- LoRA targets all attention and MLP projections (`q,k,v,o,gate,up,down`); dropout 0 so Unsloth's fast patch path stays on.
- Explicit NF4 + double-quant `BitsAndBytesConfig`.
- Packing is turned off automatically when `MAX_SEQ_LENGTH < 1024`, because packed batches could exceed the cap and break Unsloth's fused cross-entropy on Windows.
- VRAM controls: `UNSLOTH_CE_LOSS_TARGET_GB`, expandable-segments allocator settings, a 0.93 CUDA memory fraction cap, and per-step VRAM logging. The cap exists to stop silent spill into Windows shared memory, which slows training 5–10×.
- Checkpoints every 200 steps, keeping 4; `probe_max_seq.py` was added to find the largest sequence length that fits.
- Loss is computed on **assistant tokens only** (`train_on_responses_only`), and the Ollama `Modelfile` template matches the training chat template (`llama_chat_format.py`) to avoid train/serve mismatch.

### The gibberish bug and its fix

**Symptom.** The merged model generated nonsense (repeated tokens such as
"vibe vibe vibe…") even though training loss looked normal and the merged weights
had no NaN/Inf.

**Diagnosis (`diagnose_pipeline.py`, `test_A`…`test_D` logs).** The script splits
the pipeline into stages and uses a decision matrix to find where output breaks:

| Test | What it checks | Output in the committed logs |
|---|---|---|
| A | Base bnb-4bit model, no LoRA | Clean, sensible text |
| D | Base + LoRA adapter, **no merge** | Clean, sensible text |
| Unsloth-exported GGUF | Q4_K_M GGUF from Unsloth's `save_pretrained_gguf` | Labelled "FT GGUF (broken)" in `gguf_compare_output.txt`; per the `merge_and_export.py` docstring it generated garbage |
| C (after fix) | GGUF built with the new merge path, run via `llama-cli` | Clean, sensible text |

So the base model and the LoRA adapter were fine; the fault was in the
**merge/export step**.

**Root cause (documented in `merge_and_export.py`).** On torch 2.10.0+cu128 with a
Blackwell GPU, `bitsandbytes` C++ extensions cannot load ("Skipping import of cpp
extensions due to incompatible torch version"). Unsloth's `save_pretrained_gguf`
calls `dequantize_4bit`, which then takes a Python fallback that produces
numerically wrong but statistically plausible weights. Training uses a different,
working CUDA path, so the loss curve gave no warning.

**Fix.** Do not dequantize anything. `merge_and_export.py` now:
1. Loads the **un-quantized** base (`unsloth/Meta-Llama-3.1-8B-Instruct`) in bf16 on **CPU** (`MERGE_DEVICE=cpu`; slow but reliable).
2. Applies the adapter with `peft.PeftModel` and calls `merge_and_unload()`.
3. Saves bf16 safetensors.
4. Runs llama.cpp's own `convert_hf_to_gguf.py` and `llama-quantize` to produce the Q4_K_M GGUF (about 4.9 GB).
5. `register_ollama.py` then registers it as **`lemkin-clone`**.

---

## 6. Evaluation harness (`host_finetune/eval_rag.py`, `7219f57`, extended in `0990c93`)

For each prompt in an eval set it builds the same RAG prompt as `generate.py` for
every model under test (for example `llama3.1` vs `lemkin-clone`), generates, and scores:
- **G-Eval** rubric via an Ollama judge, anchored on the corpus persona profile and real post excerpts.
- **BERTScore, BLEU, ROUGE-L** against real Lemkin outputs.
- **Style consistency**: TF-IDF + logistic-regression authorship classifier.
- **RAGAS** faithfulness and answer relevancy, best-effort.

Outputs go to `host_finetune/output/eval/` (`traces.jsonl`, `geval.jsonl`, `summary.json`).

---

## 7. Application-level changes

- `generate.py` builds prompts from the persona profile, a per-format spec (`formats/format_specs.yaml`: LinkedIn post, blog draft, X thread, YouTube script), and retrieved Chroma context (capped at 5,000 characters, footer-scrubbed).
- The Streamlit UI now lists models by querying Ollama's `/api/tags`, and puts `llama3.1` and `lemkin-clone` first, so the fine-tune can be selected under **Advanced** without code changes.
- Generation timeout raised to 900 s to cover cold model loads and long prompts.

---

## 8. Open items

1. **No measured before/after results are committed.** Run `eval_rag.py` for `llama3.1` vs `lemkin-clone` and record `summary.json` here. In the committed diagnostic logs the fine-tuned model's answers read like generic Llama output, so style improvement is unproven.
2. **RAG has two sources of truth.** The DAG's `load_to_chroma` loads Spark chunks from MinIO, while `rebuild_chroma.py` loads the cleaned dataset. Re-running the DAG can overwrite the cleaned collection. This is directly relevant to **PUK-12** (integrating the new RAG with the data pipeline).
3. **Cleaning v2 runs by hand on the host**, not inside the Airflow DAG, so cleaned data is not reproduced by a normal DAG run (relevant to **PUK-7**).
4. **Hyperparameter documentation** for the last run is tracked separately in **PUK-9**.
5. **Artifacts are gitignored** (`host_finetune/output/lemkin_lora`, `lemkin-clone`, `lemkin-clone_gguf`) and the committed logs show they were produced on a teammate's machine. The trained adapter and GGUF are not in the repo.
6. **Hard-coded dev credentials** (`minioadmin`, Airflow `admin/admin`) must be replaced before the cloud deployment (**PUK-14**).
7. The Streamlit frontend is being replaced (**PUK-5**); the model selector and format inputs above are the behaviour the new frontend must reproduce.
8. `persona_pipeline/` appears to be an earlier scaffold (DuckDB, Colab QLoRA, Mac venv) and is not used by the Docker stack. Confirm and archive or delete.
