# Frontend API contract (Lemkin Studio)

Linear: **PUK-5** (frontend). Audience: whoever owns the backend (PUK-8, PUK-16, PUK-12).

The new web app in `web/` talks to the backend only through one file, `web/lib/api.ts`. This
document lists every endpoint it calls, which ones already exist on `QLoRAFT`, and which ones are
still needed. Nothing here requires the frontend to change when an endpoint lands. Until then the
app falls back gracefully and says so on screen.

Base URL: `NEXT_PUBLIC_API_BASE` (default `/api`, expecting a proxy that strips the prefix, as in
the nginx config on `pukhraj/integration-docs`). Local dev without a proxy: `http://localhost:8001`.

## Status at a glance

| Endpoint | Status | Used by | If missing |
|---|---|---|---|
| `GET /health` | Exists | System health | Page shows an error |
| `GET /formats` | Exists | Generator | Uses the four built-in formats |
| `POST /generate` | Exists, response too thin | Generator | Works, but no sources and no server latency |
| `GET /health/dependencies` | Proposed | System health, dashboard | Shows only the basic API check |
| `GET /models` | Proposed | Generator model picker | Offers `lemkin-clone` and `llama3.1` |
| `GET /sources` | Proposed | Sources page | "Not available yet" message |
| `GET /statistics` | Proposed | Dashboard | Dashboard shows a notice |
| `GET /evaluation` | Proposed | Evaluation page | Shows the reported run bundled with the app |

## Endpoints

### `GET /health` (exists)

```json
{ "status": "ok" }
```

### `GET /formats` (exists)

```json
["linkedin_post", "blog_draft", "x_thread", "youtube_script"]
```

The UI ignores any id it does not have a label for.

### `POST /generate` (exists, needs a richer response)

Request. The frontend sends these fields today. `model` and `temperature` are new and optional:

```json
{
  "format": "linkedin_post",
  "topic": "Hiring your first salesperson",
  "audience": "B2B SaaS founders",
  "goal": "Share one sharp takeaway",
  "cta": "none",
  "k": 8,
  "model": "lemkin-clone",
  "temperature": 0.7
}
```

Response the backend returns today. The UI accepts it:

```json
{ "draft": "..." }
```

Response the UI would like. It accepts either shape:

```json
{
  "content": "Generated content...",
  "model": "lemkin-clone",
  "latency_ms": 4820,
  "request_id": "a1b2c3",
  "sources": [
    {
      "id": "chunk-123",
      "text": "Retrieved passage...",
      "score": 0.87,
      "source_type": "blog",
      "title": "Optional title",
      "url": "https://example.com/optional",
      "date": "2019-04-02"
    }
  ]
}
```

- `source_type` is one of `blog`, `linkedin`, `x`, `youtube`. Anything else shows as "Other".
- `score` is a similarity from 0 to 1. Use `null` if it is not available.
- Errors: HTTP 4xx or 5xx with `{ "detail": "message", "hint": "optional fix text" }`. The UI shows
  `detail` and `hint`. The typed errors and hints already exist in `generate.py` (`LemkinError.hint`),
  so the backend can pass them straight through.
- `model` and `temperature` are ignored by the current backend. `generate.py` has no temperature
  argument yet, so it needs a small change before the temperature slider does anything.

### `GET /health/dependencies` (proposed)

Wraps `check_dependencies()` from `generate.py`.

```json
{
  "services": [
    { "name": "FastAPI", "status": "ok", "detail": "" },
    { "name": "Ollama", "status": "ok", "detail": "" },
    { "name": "ChromaDB", "status": "ok", "detail": "38105 chunks" },
    { "name": "Embedding model", "status": "ok", "detail": "nomic-embed-text" },
    { "name": "Generation model", "status": "degraded", "detail": "lemkin-clone not pulled" },
    { "name": "Airflow", "status": "unknown", "detail": "" },
    { "name": "MinIO", "status": "down", "detail": "connection refused" }
  ]
}
```

`status` is one of `ok`, `degraded`, `down`, `unknown`. Names are shown as given. The dashboard
picks out `Ollama`, `ChromaDB` and `Generation model`. This overlaps PUK-16.

### `GET /models` (proposed)

```json
{
  "models": [
    { "name": "lemkin-clone", "kind": "fine-tuned", "available": true },
    { "name": "llama3.1", "kind": "base", "available": true },
    { "name": "nomic-embed-text", "kind": "embedding", "available": true }
  ]
}
```

`kind` is `fine-tuned`, `base` or `embedding`. Embedding models are hidden from the picker.

### `GET /sources?q=hiring&type=blog&limit=30` (proposed)

`type` is `all`, `blog`, `linkedin`, `x` or `youtube`. `q` may be empty. Same source shape as in
`/generate`. With a non-empty `q` the score is the similarity to the query.

```json
{ "items": [ { "id": "chunk-1", "text": "...", "score": 0.8, "source_type": "blog" } ] }
```

### `GET /statistics` (proposed)

```json
{
  "total_documents": 31443,
  "searchable_chunks": 38105,
  "training_examples": 19961
}
```

These are the figures from the 2026-09-18 full pipeline run. Chunks can come from the Chroma
collection count. Documents and training examples can come from MinIO or the persona profile.

### `GET /evaluation` (proposed)

```json
{
  "runs": [
    {
      "id": "run-1",
      "label": "Fine-tuned + RAG",
      "note": "Where the numbers came from",
      "prompts": 30,
      "metrics": [
        { "key": "overall", "label": "Overall rubric", "value": 0.61 },
        { "key": "faithfulness", "label": "Faithfulness", "value": null }
      ]
    }
  ],
  "matrix": [
    { "system": "Base Llama", "available": false },
    { "system": "Base + RAG", "available": false },
    { "system": "Fine-tuned", "available": false },
    { "system": "Fine-tuned + RAG", "available": false }
  ]
}
```

`value` is 0 to 1, or `null` for pending. Until this endpoint exists the page shows the one run
that is committed today: `host_finetune/output/eval/gold_rag_summary_with_gold.json` on branch
`metricsft` (10 prompts, model not recorded in the file). The four-way comparison depends on PUK-13.

## CORS

The current backend allows all origins, so the browser can call it directly in local dev. Behind
nginx the `/api` proxy avoids CORS entirely. Restrict `allow_origins` before any cloud deployment.
