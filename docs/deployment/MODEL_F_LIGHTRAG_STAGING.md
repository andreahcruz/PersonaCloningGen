# CPU Model F / LightRAG staging

This deployment path adds **LightRAG + Relabel** beside Base Llama, PersonaRAG,
and direct QLoRA. It does not replace the Chroma corpus or create a new
fine-tune. The final writer remains the existing Ollama GGUF `lemkin-qlora`.

## Scope and limits

- The supplied `storage_expand_v1_snapshot` is a partial frozen snapshot.
  Its indexing report records 2,713 graph-indexed documents; do not describe
  it as the completed full corpus.
- The graph audit is not loaded by the service.
- The first deployed profile is `source`: raw source passages are citable;
  graph-generated entity/relationship summaries are intentionally excluded
  from both source cards and the writer prompt.
- This is CPU-compatible. Retrieval plus generation can be slow, which is why
  the web relay remains at ten minutes and the internal retrieval request has
  a 480-second limit.

## Artifact staging on the candidate

Keep the original transfer directory on the operator machine. Do not put a
snapshot, adapter, or audit into Git.

```bash
# On the deployment host, outside ~/298P.
sudo install -d -m 0755 /opt/model-f/verified \
  /opt/model-f/runtime/storage_expand_v1_snapshot

# Upload only the verified archive, then compare its digest to the handoff.
shasum -a 256 /path/to/lightrag_storage_expand_v1_snapshot.zip

sudo unzip -q /path/to/lightrag_storage_expand_v1_snapshot.zip -d /opt/model-f/verified

# This image command recomputes every hash from snapshot_hashes.json.  The
# one-command environment override mounts the immutable verified copy; it does
# not modify the deployment's .env file or start a service.
LIGHTRAG_RUNTIME_DIR=/opt/model-f/verified/storage_expand_v1_snapshot \
docker compose --env-file .env -f docker-compose.yml -f docker-compose.aws.yml run --rm \
  --no-deps lightrag-retriever python -c \
  'from lightrag_service.retriever import verify_snapshot; print(verify_snapshot("/runtime"))'

# Make the directory LightRAG is permitted to touch. The verified copy remains
# untouched as the rollback baseline.
sudo cp -a /opt/model-f/verified/storage_expand_v1_snapshot/. \
  /opt/model-f/runtime/storage_expand_v1_snapshot/
```

Never query the verified copy directly. The normal deployment `.env` continues
to point at the runtime path shown below.

## Enable only after artifact verification

Add these deployment-only values to the host `.env` after the service has a
healthy runtime copy:

```text
MODEL_F_ENABLED=true
LIGHTRAG_RUNTIME_DIR=/opt/model-f/runtime/storage_expand_v1_snapshot
LIGHTRAG_TIMEOUT=480
```

Pull the two retrieval models once inside the private Ollama container:

```bash
docker compose --env-file .env -f docker-compose.yml -f docker-compose.aws.yml \
  exec ollama ollama pull nomic-embed-text
docker compose --env-file .env -f docker-compose.yml -f docker-compose.aws.yml \
  exec ollama ollama pull llama3.1
```

Then rebuild only the affected services:

```bash
docker compose --env-file .env -f docker-compose.yml -f docker-compose.aws.yml \
  up -d --build lightrag-retriever backend web public-web
```

## Candidate checks

1. `GET /model-f/status` reports `available: true` and 768 dimensions.
2. Call the private retriever from the backend network with a known query and
   confirm it yields real source passages.
3. Generate one Model F draft. It must report the existing `lemkin-qlora`
   writer model and show source cards.
4. Re-run one Base Llama, PersonaRAG, and direct QLoRA request.
5. Keep Model F experimental until the four-arm and 120-prompt studies have
   measured a selected graph-context profile.

## Rollback

Set `MODEL_F_ENABLED=false` and rebuild `backend`/`web`. The Model F card
disappears while every existing generation path remains unchanged. The
verified snapshot and the existing stopped instance remain rollback assets.
