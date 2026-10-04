# Private EC2 PersonaRAG + QLoRA demo

This deploys two separately selectable methods: **M3 PersonaRAG with base `llama3.1`** and
**M4 QLoRA as `lemkin-qlora`**. The only browser-facing service is the Next.js app, bound to
EC2 loopback and viewed from a laptop through SSH.

## Before launch

1. Keep the project spend limit at **$20** and create budget alerts at **$5**, **$10**, and **$15**. The alerts must have a notification email.
2. Launch one x86 `t3.xlarge` with 60 GB encrypted EBS. Restrict the security group to SSH (port 22) from the operator's current IP. Do not open ports 3000, 8000, 8001, 9000, 9001, 11434, or 5432.
3. Clone branch `andreah/deploy-with-react` on the instance and create a private `.env` from `.env.example`. Fill in strong MinIO, Postgres, and Airflow credentials; do not commit this file.
4. Copy the local `artifacts/chroma-data.tar` archive and `lemkin-repaired-Q4_K_M.gguf` to the EC2 host. Never add either artifact to Git.

## Start and load the pipeline

```bash
docker compose --env-file .env -f docker-compose.yml -f docker-compose.aws.yml config -q
docker compose --env-file .env -f docker-compose.yml -f docker-compose.aws.yml up -d --build
docker compose --env-file .env -f docker-compose.yml -f docker-compose.aws.yml exec ollama ollama pull nomic-embed-text
docker compose --env-file .env -f docker-compose.yml -f docker-compose.aws.yml exec ollama ollama pull llama3.1
```

Start Chroma once, stop it, restore the existing index, then start the complete stack:

```bash
docker compose --env-file .env -f docker-compose.yml -f docker-compose.aws.yml up -d chroma
docker compose --env-file .env -f docker-compose.yml -f docker-compose.aws.yml stop chroma
CHROMA_VOLUME=298p_chroma-data scripts/restore_chroma_data.sh /path/to/chroma-data.tar
mkdir -p models
cp /path/to/lemkin-repaired-Q4_K_M.gguf models/
docker compose --env-file .env -f docker-compose.yml -f docker-compose.aws.yml up -d --build
docker compose --env-file .env -f docker-compose.yml -f docker-compose.aws.yml exec ollama ollama create lemkin-qlora -f /models/Modelfile
```

Do **not** run the full corpus DAG on EC2: it would rebuild the copied 37,525-chunk index. Run only the existing isolated pipeline sample to prove the AWS ingestion/embedding path without replacing `lemkin_content`.

Check the backend after the DAG finishes:

```bash
docker compose --env-file .env -f docker-compose.yml -f docker-compose.aws.yml exec backend \
  python scripts/capture_personarag_smoke.py \
  --method personarag --output /app/experiments/EXP-YYYYMMDD-001-ec2-personarag
docker compose --env-file .env -f docker-compose.yml -f docker-compose.aws.yml exec backend \
  python scripts/capture_personarag_smoke.py \
  --method qlora --output /app/experiments/EXP-YYYYMMDD-002-ec2-qlora
```

The health response must report the persona file, Chroma collection, `nomic-embed-text`, `llama3.1`, and `lemkin-qlora` ready. The smoke scripts store the fixed request, response, request ID, source IDs, model, and latency under the host's `experiments/` folder. Add the collection count, commit SHA, and screenshots to that same experiment folder.

## Show the private demo

On the laptop, open an SSH tunnel (replace the host and key path):

```bash
ssh -i /path/to/key.pem -L 3000:127.0.0.1:3000 -L 8080:127.0.0.1:8080 ubuntu@EC2_PUBLIC_DNS
```

Open `http://localhost:3000/generate`. It calls Next.js, which privately proxies `/api` to FastAPI; FastAPI privately calls Chroma and Ollama. Run the same brief through both method choices, capturing the PersonaRAG draft and retrieved-source cards alongside the QLoRA draft. Open `http://localhost:8080` through the same tunnel for the Airflow sample success screen.

## Shut down

Stop the instance when the demo ends to stop compute charges. After copying any experiment artifacts you need, terminate it and delete any unneeded EBS volume or snapshots.
