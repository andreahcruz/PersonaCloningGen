# Persona cloning ETL pipeline (Jason Lemkin)

Dockerized stack: Airflow, Spark, MinIO, Chroma, Streamlit. **Local Lemkin JSONL** (blog, LinkedIn, X, YouTube transcripts) is ingested into MinIO, cleaned and chunked in Spark, embedded with **Ollama** (`nomic-embed-text`), then loaded into **Chroma**, with a **Streamlit** UI for RAG-style generation.

## Data files (place under `./data`)

The Airflow task `extract_to_minio` reads these files from the host `./data` folder (mounted at `/opt/airflow/data` in the scheduler):

| File | Role |
|------|------|
| `jasonlemkin_blog.jsonl` | SaaStr blog posts (`title`, `date`, `url`, `content`) |
| `jasonlemkinlinkedin.jsonl` | LinkedIn posts (`content`, `published_text`, …) |
| `jasonlk_originals.jsonl` | X posts (`text`, `url`, `created_at`; replies/reposts skipped; very short stubs skipped) |
| `jasonmlemkinyoutubetranscripts.jsonl` | Jason channel transcripts (`video_title`, `video_url`, `transcript_text`, …) |
| `saastryoutubetranscripts.jsonl` | SaaStr channel transcripts (same shape) |

Rows without usable body text are skipped (e.g. YouTube rows with `transcript_text: null`). After a DAG run, **`./data/persona_profile.json`** is written on the host for Streamlit / `generate.py`.

## Prerequisites

1. **Docker** — Docker Desktop (or Docker Engine + Compose v2) running.
2. **Ollama on the host** — Containers call Ollama at `host.docker.internal:11434`. Install [Ollama](https://ollama.com), then pull the models the pipeline expects:
   - `ollama pull nomic-embed-text` (embeddings in Spark + RAG)
   - `ollama pull llama3.1` (default generation model in `generate.py` / Streamlit)

## One-time setup

```bash
cd /path/to/298P
mkdir -p data
# Copy your Lemkin JSONL files into ./data (see table above)
```

Optional: `cp .env.example .env` if you add custom env overrides.

## Start the stack

```bash
docker compose up --build
```

Use `docker-compose` instead of `docker compose` if you only have the older CLI.

First startup can take several minutes (images build, Airflow DB init, admin user). When things settle:

| Service | URL |
|--------|-----|
| **Airflow** | http://localhost:8080 — user `admin`, password `admin` (from `docker-compose.yml`) |
| **Streamlit** | http://localhost:8501 |
| **MinIO console** | http://localhost:9001 — `minioadmin` / `minioadmin` |
| **Spark UI** | http://localhost:8081 |
| **Chroma** | API on port **8000** (used by the app, not a browser UI) |

Keep **`ollama serve`** running on your machine while Spark/Airflow embed (default URL `http://localhost:11434`).

MinIO buckets used: **`lemkin-raw`** (normalized JSON under `raw/`) and **`lemkin-processed`** (Spark output under `chunks/`, plus `persona_profile.json`).

## Run the pipeline (Airflow)

1. Open the Airflow UI at http://localhost:8080.
2. Find DAG **`lemkin_content_pipeline`**.
3. **Unpause** it, then **Trigger** (play button) for a manual run.

Tasks: **extract_to_minio** → **trigger_spark_clean** → **load_to_chroma**. **extract_persona** runs after raw load (in parallel with Spark) and writes persona stats to MinIO and `./data/persona_profile.json`.

Rebuild Airflow images after dependency changes: `docker compose build --no-cache airflow-webserver airflow-scheduler airflow-init`.

Spark needs Ollama for `nomic-embed-text`.

## Streamlit UI

With the stack up, open **http://localhost:8501**. The app uses **Chroma over HTTP** (`chroma:8000`) and `./data/persona_profile.json`. Run the DAG through **extract_persona** (or copy a persona file) before generating.

The Streamlit **Docker** image installs **`requirements-streamlit.txt`** only (not the full `requirements.txt`), so Chroma’s Pydantic/FastAPI stack can use **`email-validator>=2`** without conflicting with **Airflow**’s older `email-validator<2` pin in the monolithic requirements file.

**Chroma “default_tenant” / tenant errors** — The **Chroma server image** and **`chromadb` pip version** are pinned together in `docker-compose.yml` and the Dockerfiles. If you still see *Could not connect to tenant default_tenant*, the **`chroma-data` volume** may be from an older server layout: stop the stack, remove that volume (Docker Desktop → Volumes, or `docker volume rm <project>_chroma-data`), bring Chroma back up, and **re-run `load_to_chroma`** to repopulate the collection.

**Streamlit: *Collection lemkin_content does not exist*** — Chroma is empty for the current tenant/database. In Airflow, trigger **`lemkin_content_pipeline`** through **`load_to_chroma`** (upstream: **`extract_to_minio`** → **`trigger_spark_clean`** if chunks are missing in MinIO). Rebuild/restart Airflow after Dockerfile changes so **`chromadb==1.5.5`** matches the Chroma container.

## Optional: CLI generation (outside Docker)

Point at Chroma on localhost (port **8000** published from the container) and set:

```bash
set CHROMA_USE_HTTP=true
set CHROMA_HOST=localhost
set CHROMA_PORT=8000
set CHROMA_COLLECTION_NAME=lemkin_content
pip install -r requirements.txt
python generate.py --format linkedin_post --topic "..." --audience "..." --goal "..." --cta "..." --k 8 --out outputs/example.md
```

Create an `outputs/` folder if you want files there.

## Fine-tuning (StyleAdaptedLM-style LoRA on the host GPU)

Adds a fine-tuned generation model on top of the existing RAG pipeline. The base
model stays frozen (no catastrophic forgetting), a small LoRA adapter is trained
on your full corpus reframed as instruction/response pairs, the merged result is
exported to GGUF, and Ollama serves it as a custom model named **`lemkin-clone`**
that the existing Streamlit UI / `generate.py` can select.

### Prerequisites

- NVIDIA GPU with 12 GB+ VRAM (developed on RTX 5070 Ti, 16 GB Blackwell sm_120).
- NVIDIA driver 555+ on the **host** (not Docker). Driver-bundled CUDA 12.x or 13.x both work; the wheels target CUDA 12.8 runtime.
- **Python 3.11** on the host. Newer (3.12, 3.13, 3.14) won't work — Unsloth and bitsandbytes ship wheels for 3.10–3.11. Install via `winget install Python.Python.3.11` if needed.
- `ollama serve` running on the host (the same one the Docker stack already calls via `host.docker.internal`).
- The Docker stack already running (Airflow + MinIO + Spark + Chroma).

### One-time host setup

The trainer lives in `host_finetune/` and runs **outside Docker** so it can use
the GPU directly. **Install order matters** — see the comment at the top of
[host_finetune/requirements.txt](host_finetune/requirements.txt) for why.

```powershell
py -3.11 -m venv host_finetune\.venv
host_finetune\.venv\Scripts\activate
python -m pip install --upgrade pip
# Step 1: install a torch in the Unsloth-compatible range FROM the cu128 index.
# If you skip this step, pip will silently downgrade torch to a CPU-only
# PyPI wheel when resolving Unsloth's `torch<2.11` constraint.
pip install "torch>=2.10.0,<2.11.0" "torchvision>=0.25.0,<0.26.0" --index-url https://download.pytorch.org/whl/cu128
# Step 2: everything else (Unsloth, peft, trl, datasets, bitsandbytes, boto3, ...).
pip install -r host_finetune\requirements.txt
ollama pull llama3.1
```

Sanity-check the GPU stack before training:

```powershell
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

Expected: `2.10.0+cu128 True NVIDIA GeForce RTX 5070 Ti` (or whatever your card is).
If it prints `False`, your torch got downgraded to CPU-only — re-run step 1 with
`--force-reinstall` added.

(Linux/macOS: `python3.11 -m venv host_finetune/.venv && source host_finetune/.venv/bin/activate && ...`)

### Run the fine-tune flow

1. Trigger the Airflow DAG (`lemkin_content_pipeline`) end-to-end. The Spark job
   now writes a training dataset under
   `s3://lemkin-processed/training/dataset_jsonl/`, and the new
   **`mark_training_ready`** task promotes it to a stable
   `s3://lemkin-processed/training/dataset.jsonl` plus a `_READY` sentinel.
2. With the venv active, run the watcher on the host:
   ```powershell
   python -m host_finetune.watcher --once
   ```
   It downloads the dataset, runs `host_finetune.finetune` (QLoRA on
   `unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit`, ~30–60 min on a 5070 Ti),
   `host_finetune.merge_and_export` (merges + writes Q4_K_M GGUF), and
   `host_finetune.register_ollama` (`ollama create lemkin-clone -f Modelfile`).
3. Open Streamlit at <http://localhost:8501>. Under **Advanced**, set
   *Generate model* to **`lemkin-clone`** and generate as usual. RAG retrieval
   still runs against the same Chroma collection — the model itself just
   produces more native-sounding output.

For continuous polling (re-train every time the DAG runs):
```powershell
python -m host_finetune.watcher
```

To re-train without a new sentinel (e.g. after tweaking hyperparams):
```powershell
python -m host_finetune.watcher --once --force
```

### Tunable knobs

All read from environment variables in `host_finetune/config.py`:

| Variable | Default | Notes |
|---|---|---|
| `HF_MODEL_NAME` | `unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit` | Drop to `unsloth/Llama-3.2-3B-Instruct-bnb-4bit` if you OOM. |
| `PER_DEVICE_BATCH` / `GRAD_ACCUM` | `2` / `8` | Effective batch 16; lower batch first if OOM. |
| `NUM_EPOCHS` | `3` | 1–2 is usually enough for style transfer; 3+ risks overfitting on small corpora. |
| `LORA_R` / `LORA_ALPHA` | `16` / `32` | Standard StyleAdaptedLM defaults. |
| `GGUF_QUANT` | `q4_k_m` | `q5_k_m` for higher fidelity at ~25 % more disk. |
| `OLLAMA_MODEL_NAME` | `lemkin-clone` | Change if you want to keep multiple personas around. |
| `MINIO_ENDPOINT_HOST` | `http://localhost:9000` | Override only if you remapped MinIO's host port. |

### Fine-tune troubleshooting

- **`AssertionError: Torch not compiled with CUDA enabled` after install.** Pip
  silently swapped your cu128 torch for a CPU-only PyPI wheel during dependency
  resolution. Recover with:
  ```powershell
  pip install --force-reinstall "torch>=2.10.0,<2.11.0" "torchvision>=0.25.0,<0.26.0" --index-url https://download.pytorch.org/whl/cu128
  ```
  Verify `torch.cuda.is_available()` is True before re-running training.
- **`datasets X requires fsspec[http]<=2025.9.0, but you have fsspec 2026.x`.**
  Force-reinstalling torch pulls in too-new fsspec. Pin it back:
  ```powershell
  pip install "fsspec<=2025.9.0"
  ```
- **`RuntimeError: No or negligible GPU memory available for fused cross entropy.`**
  On Windows, `torch.cuda.mem_get_info()` often reports almost no *free* VRAM
  while the 8B weights are loaded, so Unsloth's fused CE auto-tuner aborts.
  `host_finetune/finetune.py` sets `UNSLOTH_CE_LOSS_TARGET_GB=2` before import.
  If training still fails, try `3`, or free VRAM by closing games/browsers, then:
  ```powershell
  $env:UNSLOTH_CE_LOSS_TARGET_GB="3"
  python -m host_finetune.finetune
  ```
- **OOM during training.** Drop `HF_MODEL_NAME` to Llama 3.2 3B, lower
  `PER_DEVICE_BATCH` to 1 (and raise `GRAD_ACCUM` to 16 to keep effective batch
  size constant), or shorten `MAX_SEQ_LENGTH` to 768.
- **`save_pretrained_gguf` is slow on first run.** Unsloth builds llama.cpp into
  its cache the first time; subsequent runs reuse it. Resulting GGUF is ~5 GB
  for Llama 3.1 8B at Q4_K_M.
- **`ollama create` says "model not found".** Ensure `ollama serve` is running
  before the watcher reaches the `register_ollama` step. The watcher chains
  scripts in order, so you can re-run just the register step if needed:
  `python -m host_finetune.register_ollama`.
- **Streamlit dropdown still shows the old options.** Streamlit caches the
  module — refresh the browser tab or restart the `streamlit` container:
  `docker compose restart streamlit`.

## Troubleshooting

- If **`airflow-init`** errors because the admin user already exists, that is normal on later runs; the webserver and scheduler should still work.
- If something fails on first boot, check container logs for `airflow-init`, `airflow-webserver`, and `spark-worker`.
- **`minio-setup`** and **`airflow-init`** exit after finishing; not staying “running” in Docker Desktop is expected.
- **`trigger_spark_clean` fails with `Could not parse Master URL`** — Airflow’s Spark hook builds `--master` from the connection. For a **standalone** cluster, use **`host`: `spark://spark-master:7077`** and **no separate port**. The default in `docker-compose.yml` is **`local[*]`** (all Spark work in the Airflow container) to avoid driver/worker Spark JAR mismatches. Recreate Airflow containers after changing `AIRFLOW_CONN_SPARK_DEFAULT`.
- **`InvalidClassException: org.apache.spark.scheduler.Task`** — The Spark **driver** (pip `pyspark` in the Airflow image) and **executors** (Bitnami `spark-worker`) were different builds. Use **`local[*]`** (default) or run `spark-submit` from the same image as the workers with matching `SPARK_HOME`.
- **`ClassNotFoundException: org.apache.hadoop.fs.s3a.S3AFileSystem`** — The Airflow image uses PySpark without S3A JARs. The DAG’s `SparkSubmitOperator` passes `--packages` for `hadoop-aws` and the AWS bundle (first run downloads from Maven; the scheduler container needs outbound internet, or pre-cache the JARs).
- **`load_to_chroma` / `TypeError: 'type' object is not subscriptable` in `posthog.types`** — The Airflow image must be **Python 3.10** (`Dockerfile.airflow` uses `apache/airflow:2.8.1-python3.10`) and pin **`posthog<3`**. If the log still shows **`python3.8`** in paths, you are on an **old container**: run `docker compose build --no-cache` then `docker compose up -d --force-recreate airflow-webserver airflow-scheduler` and confirm `docker compose exec airflow-scheduler python --version` prints **3.10.x**.

### Debugging a failed Airflow task yourself

1. In the Airflow UI, open the DAG → click the red task → **Log**. The traceback and `spark-submit` output are there, not in `docker compose` stdout.
2. From the project directory:  
   `docker compose exec airflow-scheduler ls /opt/airflow/logs/dag_id=lemkin_content_pipeline/`  
   then open the latest `task_id=trigger_spark_clean/attempt=*.log` with `cat`.
3. With **`local[*]`**, the job does not use `spark-worker`; confirm MinIO and Ollama are reachable. For standalone Spark, check **http://localhost:8081** and that `spark-master` / `spark-worker` are running.
4. Tail scheduler: `docker compose logs -f airflow-scheduler` (less detail than the task log for SparkSubmit).
