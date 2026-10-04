# Private EC2 PersonaRAG demo

This deploys **M3 PersonaRAG with base `llama3.1`**, not Kevin's QLoRA adapter. The only browser-facing service is the Next.js app, bound to EC2 loopback and viewed from a laptop through SSH.

## Before launch

1. In AWS Billing, create budget alerts at **$25**, **$50**, and **$80**. The alerts must have a notification email.
2. Launch one x86 `t3.xlarge` with 60 GB encrypted EBS. Restrict the security group to SSH (port 22) from the operator's current IP. Do not open ports 3000, 8000, 8001, 9000, 9001, 11434, or 5432.
3. Clone branch `andreah/deploy-with-react` on the instance and create a private `.env` from `.env.example`. Fill in strong MinIO, Postgres, and Airflow credentials; do not commit this file.

## Start and load the pipeline

```bash
docker compose --env-file .env -f docker-compose.yml -f docker-compose.aws.yml config -q
docker compose --env-file .env -f docker-compose.yml -f docker-compose.aws.yml up -d --build
docker compose --env-file .env -f docker-compose.yml -f docker-compose.aws.yml exec ollama ollama pull nomic-embed-text
docker compose --env-file .env -f docker-compose.yml -f docker-compose.aws.yml exec ollama ollama pull llama3.1
```

Set `SPARK_EMBED_PARTITIONS=2` in `.env`. In the private Airflow UI, run the Lemkin pipeline against `data/cleaned/` and wait for every task to turn green. It copies the persona profile to `data/deploy/persona_profile.json`, uploads the raw and processed data to MinIO, and rebuilds Chroma collection `lemkin_content` through HTTP.

Check the backend after the DAG finishes:

```bash
docker compose --env-file .env -f docker-compose.yml -f docker-compose.aws.yml exec backend \
  python scripts/capture_personarag_smoke.py \
  --output /app/experiments/EXP-YYYYMMDD-001-ec2-personarag
```

The health response must report the persona file, Chroma collection, `nomic-embed-text`, and `llama3.1` ready. The smoke script stores the fixed request, response, request ID, source IDs, model, and latency under the host's `experiments/` folder. Add the collection count, commit SHA, and screenshots to that same experiment folder.

## Show the private demo

On the laptop, open an SSH tunnel (replace the host and key path):

```bash
ssh -i /path/to/key.pem -L 3000:127.0.0.1:3000 -L 8080:127.0.0.1:8080 ubuntu@EC2_PUBLIC_DNS
```

Open `http://localhost:3000/generate`. It calls Next.js, which privately proxies `/api` to FastAPI; FastAPI privately calls Chroma and Ollama. Capture the generated draft and the retrieved-source cards. Open `http://localhost:8080` through the same tunnel for the Airflow success screen.

## Shut down

Stop the instance when the demo ends to stop compute charges. After copying any experiment artifacts you need, terminate it and delete any unneeded EBS volume or snapshots.
