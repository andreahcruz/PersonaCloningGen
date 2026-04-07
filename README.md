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
