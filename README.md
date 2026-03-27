# Persona cloning ETL pipeline

Dockerized stack: Airflow, Spark, MinIO, Chroma, Streamlit. Paul Graham essays are extracted, cleaned, embedded (Ollama), loaded to Snowflake and Chroma, with a Streamlit UI for RAG-style generation.

## Prerequisites

1. **Docker** — Docker Desktop (or Docker Engine + Compose v2) running.
2. **Ollama on the host** — Containers call Ollama at `host.docker.internal:11434`. Install [Ollama](https://ollama.com), then pull the models the pipeline expects:
   - `ollama pull nomic-embed-text` (embeddings in Spark + RAG)
   - `ollama pull llama3.1` (default generation model in `generate.py` / Streamlit)
3. **Snowflake** — Put real values in `.env` (same keys as `.env.example`). The DAG loads processed data into Snowflake.

## One-time setup

```bash
cd /path/to/298etlpipeline
cp .env.example .env
# Edit .env with your Snowflake account, user, password, database, schema, warehouse
mkdir -p data
```

`.env` is gitignored; secrets are not committed.

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

## Run the pipeline (Airflow)

1. Open the Airflow UI at http://localhost:8080.
2. Find DAG **`pg_essay_pipeline`**.
3. **Unpause** it, then **Trigger** (play button) for a manual run.

Tasks run in order: Hugging Face → MinIO → Spark clean/embed → Snowflake → Chroma → persona extraction. Spark needs Ollama for `nomic-embed-text`.

## Streamlit UI

With the stack up, open **http://localhost:8501**. Adjust the Ollama URL field if needed (from inside Docker, the default points at the host Ollama).

## Optional: CLI generation (outside Docker)

After data exists in Chroma (pipeline has run at least through the Chroma step):

```bash
pip install -r requirements.txt
python generate.py --format linkedin_post --topic "..." --audience "..." --goal "..." --cta "..." --k 8 --out outputs/example.md
```

Create an `outputs/` folder if you want files there.

## Troubleshooting

- If **`airflow-init`** errors because the admin user already exists, that is normal on later runs; the webserver and scheduler should still work.
- If something fails on first boot, check container logs for `airflow-init`, `airflow-webserver`, and `spark-worker`.
