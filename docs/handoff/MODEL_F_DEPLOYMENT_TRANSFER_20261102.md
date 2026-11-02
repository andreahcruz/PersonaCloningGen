# Model F deployment transfer — 2026-11-02

This is the handoff for integrating Model F into the React deployment. It does not deploy the service and it does not merge branches.

Check out the tip of `kevin/lightrag-model-f` and pull. The implementation commit is `c04e77cf438e7d845c640d2727250ce642243204`. This guide was added in `e97c602dc9b76fa52ebdd7823591795cee4110e9`. The tip after that guide excludes failed graph documents from experiment hits. `git rev-parse HEAD` on this branch is the commit to use.

Previous pushed commit named in the last handoff: `817e159b99b904a3171a0495e7a1c9433fe98137`. `54ceb5dc4d929c3c1234803ea75308a0153b4d7c` (Workbook 2 catch-up note) was already on the remote before this push. `c04e77cf438e7d845c640d2727250ce642243204` is the new code commit.

## A. GitHub

| Item | Value |
|---|---|
| Repository | https://github.com/andreahcruz/PersonaCloningGen |
| Clone remote in this workspace | `https://github.com/andreahcruz/298P.git` (GitHub redirects that URL to PersonaCloningGen). The remote URL was not changed. |
| Feature branch | `kevin/lightrag-model-f` |
| Implementation commit | `c04e77cf438e7d845c640d2727250ce642243204` |
| Deployment branch | `andreah/deploy-with-react` |
| Merge base | `ea1d58b30cafcb4cf18a03ded39963f64ce6aee7` |
| Compare URL | https://github.com/andreahcruz/PersonaCloningGen/compare/andreah/deploy-with-react...kevin/lightrag-model-f?expand=1 |

No merge was performed. A read-only `git merge-tree` against `origin/andreah/deploy-with-react` reported conflicts in:

- `.dockerignore` (add/add)
- `.gitignore` (content)
- `README.md` (content)
- `backend/Dockerfile` (add/add)
- `backend/main.py` (add/add)
- `docker-compose.yml` (content)

`host_finetune/compare_adapters.py` and `host_finetune/lightrag_backend.py` were not in that conflict list. `andreah/deploy-with-react` has no `model-f-context` usage. Its `backend/main.py` is the React API (PersonaRAG plus an Ollama GGUF tag `lemkin-qlora`). This branch's `backend/main.py` is an older short API and also has no LightRAG route. Keep the deploy branch's React service files when resolving those six conflicts.

Related docs already on the branch:

- `docs/architecture/LIGHTRAG_MODEL_F.md`
- `docs/implementation/MODEL_F_GRAPH_CONTEXT_IMPLEMENTATION.md`
- `docs/implementation/MODEL_F_FOUR_ARM_EXPERIMENT.md`
- `docs/implementation/MODEL_F_POST_INDEX_CHECKLIST.md`
- `docs/evaluation/MODEL_F_EXPERIMENT_SPEC.md`
- `docs/evaluation/MODEL_F_EVALUATION_INFRASTRUCTURE.md`
- `docs/evaluation/UNIFIED_120_COMPARISON.md`
- `docs/handoff/MODEL_F_HANDOFF_MANIFEST.md`
- `docs/handoff/BRANCH_CATCHUP_FOR_TEAMMATE.md` (written while indexing was still running; the snapshot section below supersedes its indexing status)
- `docs/deployment/DEPLOYMENT_HANDOFF.md` and `docs/deployment/ARTIFACTS.md` (older Relabel + Chroma package, not this graph)

## B. Required artifacts

Send these files from `C:\Users\okevi\DATA298_model_f_transfer`. Originals under the repository were not modified. The zips were built by reading those originals. Each zip member of the snapshot was hashed and matched `snapshot_hashes.json`.

| Artifact | Exact location to send | Size | Hash or manifest | Transfer method | Destination expected by deployment | Dependencies |
|---|---|---|---|---|---|---|
| LightRAG snapshot zip | `C:\Users\okevi\DATA298_model_f_transfer\lightrag_storage_expand_v1_snapshot.zip` | 481,019,443 bytes | SHA-256 `4e7a00a977efe0c4ab121b0356704ecd4472cb17d0a96533f8c80a5ad500b9ea`. 32 members. Inner directory `storage_expand_v1_snapshot/`. File manifest hash-of-hashes `1f50c096a001218c19db11f3d74d657111d490c8b0467cb25144b785037d9e6d` | Private copy (disk, drive, or a private channel). Not GitHub. | Extract outside the repo. Query a runtime copy, for example `/opt/model-f/runtime/storage_expand_v1_snapshot`. Keep one untouched extract as the verified baseline. | LightRAG 1.5.7, Ollama `nomic-embed-text` and `llama3.1:latest` |
| Relabel QLoRA adapter zip | `C:\Users\okevi\DATA298_model_f_transfer\relabel_qlora_adapter.zip` | 158,836,161 bytes | SHA-256 `644041cbe47e0a1ce69997e790be799b5ed42862be2854b792e94a24dd93777f`. Six files under `lemkin_lora_relabel/`. Weight file SHA-256 `850e9aa88d04fa4e7c9490d5d974715c8a1018c47c7ada340d642b4b2afea530` | Private copy. Not GitHub. | Extract to the configured adapter directory. Code default: `host_finetune/output/lemkin_lora_relabel` | Base model `unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit`, Unsloth/PEFT, a CUDA machine for generation |
| Graph audit (optional for boot, needed for the limitation note) | `C:\Users\okevi\DATA298_model_f_transfer\graph_audit\storage_expand_v1_snapshot_graph_audit.json` | 5,628,431 bytes | SHA-256 `2672aee28b5564876f6f8d547745acdc3eda7780e6b5df5d4812f5def2e0b4dd` | Private copy. Contains derived entity names, descriptions, and cache lines. | Not loaded by the server. Keep beside the run notes. | None |
| This guide | `docs/handoff/MODEL_F_DEPLOYMENT_TRANSFER_20261102.md` on the branch | in Git | the tip commit of `kevin/lightrag-model-f` | `git pull` | repository | none |

Unpacked snapshot directory (the zip source, left in place):

`C:\Users\okevi\SJSU\DATA298\298P\artifacts\lightrag\model_f\storage_expand_v1_snapshot`

989,760,345 bytes, 32 files. `snapshot_hashes.json` records 31 files. The hash file itself is the 32nd. Recomputed SHA-256 of those 31 files matched the recorded manifest on 2026-11-02. Embedding dimension of the first chunk vector is 768.

Adapter source (left in place, checkpoints not included in the zip):

`C:\Users\okevi\SJSU\DATA298\298P\host_finetune\output\lemkin_lora_relabel`

| File | Bytes | SHA-256 |
|---|---|---|
| `adapter_config.json` | 1,309 | `f768bb552e20d9af8dfdb413a3bcd011110d80a3c097b3ae16214543da1899e7` |
| `adapter_model.safetensors` | 167,832,240 | `850e9aa88d04fa4e7c9490d5d974715c8a1018c47c7ada340d642b4b2afea530` |
| `chat_template.jinja` | 4,723 | `118bacd5dcb2cc49fdeee41524226720679225194aea902ea0b9c58dae57a752` |
| `README.md` | 1,623 | `10032257dff7ec3a509edd937fb97ad11012a73866065e99ca49718711765f63` |
| `tokenizer.json` | 17,209,920 | `6b9e4e7fb171f92fd137b777cc2714bf87d11576700a1dcd7a399e7bbe39537b` |
| `tokenizer_config.json` | 52,733 | `3f22f363e0e195edb7b2ad06959af59a0e8a8cfc6215b77045a68b60548491cb` |

The six files are what `FastLanguageModel.from_pretrained` loads from the adapter directory. Checkpoint folders under the same directory (`checkpoint-2000`, `checkpoint-2400`, `checkpoint-2674`) are training history. They are not in the zip. The full directory on this machine is 997,815,445 bytes.

### What else the runtime needs

| Piece | Identity | On GitHub? | How the recipient gets it | Required? |
|---|---|---|---|---|
| Base model | `unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit`. `adapter_config.json` field `revision` is null. Training manifest `experiments/EXP-20261003-005-relabel-qlora` records the repo id, rank 16, alpha 32, dropout 0. The Hugging Face cache on this machine resolves `refs/main` to `f15c379fb32bb402fa06a7ae9aecb1febf4b79ec` (about 5.72 GB). That revision is the cached snapshot here. It is not a pin written into the adapter. | No | First Unsloth load downloads it. A GPU with the CUDA torch build in `host_finetune/requirements.txt` is required to generate. | Required for generation |
| Tokenizer and chat template | Shipped inside the adapter zip: `tokenizer.json`, `tokenizer_config.json`, `chat_template.jinja`. Llama 3.1 special tokens. `tokenizer_config.json` records `model_max_length` 512, which is the training length. Generation sets `max_seq_length` 2048 in `compare_adapters.py`. | No, except whatever older tokenizer files history already tracked | Inside `relabel_qlora_adapter.zip` | Required |
| Query embedding | Ollama `nomic-embed-text:latest`, id `0a109f422b47`, 274 MB, 768 dimensions | No | `ollama pull nomic-embed-text` | Required for retrieval |
| Keyword / extraction model | Ollama `llama3.1:latest`, id `46e0c10c039e`, 4.9 GB. Index config in the snapshot used this tag at `num_ctx` 16384, temperature 0, gleaning 1. | No | `ollama pull llama3.1` | Required for a LightRAG query |
| LightRAG package | `lightrag-hku==1.5.7`, then `opentelemetry-api==1.41.1` so Chroma 1.5.5 still imports | The requirement file is on the branch: `host_finetune/requirements-lightrag.txt` | pip, from the host fine-tune environment | Required |
| Writing judge | `qwen2.5:14b` | No | `ollama pull qwen2.5:14b` | Optional. Scoring only. |
| Train-only Chroma | `host_finetune/output/chroma_lemkin_train_only` | No | Already described in `docs/deployment/ARTIFACTS.md` | Optional. This is Model E. Model F does not read it. |
| Ollama GGUF tags `lemkin-relabel`, `lemkin-qlora`, `lemkin-clone` | Separate exports | No | Do not use them as the Model F adapter | Not the Model F generator |

The snapshot contains source passages and derived graph text (`catalog.jsonl`, `kv_store_full_docs.json`, `kv_store_text_chunks.json`, `kv_store_llm_response_cache.json`, the GraphML file, and the vector JSON files). Those internal files are required for LightRAG to open the store. They were left in the archive. The archive does not include the raw cleaned corpora, historical extraction caches outside this snapshot, `.env` files, or unrelated checkpoints.

The recipient cannot regenerate this snapshot without re-indexing. Re-indexing is not equivalent. The adapter cannot be downloaded. Retraining will not reproduce `850e9aa8…`.

### Verify after transfer

From the machine that received the folder:

```powershell
Get-FileHash -Algorithm SHA256 .\lightrag_storage_expand_v1_snapshot.zip
Get-FileHash -Algorithm SHA256 .\relabel_qlora_adapter.zip
Get-FileHash -Algorithm SHA256 .\graph_audit\storage_expand_v1_snapshot_graph_audit.json
```

Expected digests:

- snapshot zip `4e7a00a977efe0c4ab121b0356704ecd4472cb17d0a96533f8c80a5ad500b9ea`
- adapter zip `644041cbe47e0a1ce69997e790be799b5ed42862be2854b792e94a24dd93777f`
- audit json `2672aee28b5564876f6f8d547745acdc3eda7780e6b5df5d4812f5def2e0b4dd`

Extract the snapshot zip. It has 32 members and the top folder `storage_expand_v1_snapshot`. Then, from the repository root, point verify at the extracted copy:

```powershell
host_finetune\.venv\Scripts\python.exe -X utf8 -m host_finetune.model_f_experiment verify --snapshot PATH\TO\EXTRACTED\storage_expand_v1_snapshot
```

`verify` recomputes the 31 recorded hashes, checks the required graph, catalog, and vector files, and requires a 768-dimensional chunk vector. It refuses a path whose directory parts include the live store name `storage_expand_v1`. The extracted folder name `storage_expand_v1_snapshot` is accepted.

Copy that extract to a runtime directory before any query. LightRAG storage initialization can rewrite files in the working directory. The verified extract and the zip stay untouched.

Extract the adapter zip so the directory contains `adapter_model.safetensors` at its root. Confirm that file's SHA-256 is `850e9aa88d04fa4e7c9490d5d974715c8a1018c47c7ada340d642b4b2afea530`.

## C. Runtime architecture

Model F generation on this branch is `host_finetune/compare_adapters.py` `load_generator`. It calls Unsloth `FastLanguageModel.from_pretrained` on the adapter directory, `load_in_4bit=True`, `max_seq_length=2048`. The adapter config names the base model `unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit`. The tokenizer and `chat_template.jinja` in that directory supply the Llama 3.1 chat template. Decoding used by the Model F runners is temperature 0.7, `top_p` 0.9, repetition penalty 1.15, seed base 42.

Retrieval for the four-arm runner and the 120-prompt runner is `host_finetune/model_f_experiment/retrieval.py` `retrieve_raw`. That function opens the snapshot directory, checks that Ollama has `llama3.1:latest` and `nomic-embed-text`, and calls LightRAG `aquery_data` with mode `hybrid`, `top_k` 20, `only_need_context=True`, rerank off, and `num_ctx` 16384. `nomic-embed-text` embeds the query. `llama3.1:latest` is the LightRAG keyword model. The index was built with the same embedding model, the same extraction tag, and `num_ctx` 16384 (`reports/indexing_summary.json` inside the snapshot).

`query_documents` in `lightrag_backend.py` returns verbatim chunks as hits. Entity rows are returned on `entity_descriptions` and are excluded from hits. Relationship rows stay on `relationship_candidates` until a source span is accepted. Graph-context profiles are applied later by `host_finetune/model_f_context`.

The CLI flag `--retrieval-backend` still defaults to `dense`. `--model-f-context` still defaults to `off`. `--lightrag-mode` defaults to `hybrid` only after the backend is set to `lightrag`. `--lightrag-workdir` defaults to `artifacts/lightrag/model_f/storage`, which is the old small store, not this snapshot. `build_lightrag_retriever` (the `compare_adapters.py` CLI path) does not pass `num_ctx`, so that path uses `NUM_CTX` 32768 in `lightrag_backend.py`. The experiment runners pass 16384. A Model F demonstration uses the experiment retrieval path, or an explicit `num_ctx=16384`, on a runtime copy of this snapshot.

### Current deployment backend

`origin/andreah/deploy-with-react` `backend/main.py` still loads `generate.py`: a Chroma-style index at `INDEX_PATH`, Ollama embeddings, and Ollama generation. The `method` field is `personarag` or `qlora`. The QLoRA branch calls Ollama tag `lemkin-qlora` with temperature 0. That file has no LightRAG working directory, no hybrid query, no graph-context profile, and no Unsloth Relabel load. `backend/requirements.txt` on that branch is FastAPI, Chroma, and PyYAML. It does not install LightRAG or Unsloth.

This branch's `backend/main.py` is also not a Model F service. Model F is the `host_finetune` CLI and experiment runner. Wiring it into FastAPI is the integration work. Until that route exists, the React app cannot serve Model F by pointing `QLORA_MODEL` at a GGUF tag.

The existing CPU demo comment on the deploy branch says that box holds one 8B Ollama model at a time. The Relabel path here needs a CUDA machine for the 4-bit Unsloth load, plus Ollama for retrieval. Those can be the same GPU machine. They are not the current CPU GGUF path.

## D. Model F configuration

Use these settings when the route is demonstrating Model F:

| Setting | Value |
|---|---|
| Retrieval backend | `lightrag` |
| Query mode | `hybrid` |
| Snapshot | runtime copy of `storage_expand_v1_snapshot` |
| Embedding | `nomic-embed-text`, 768 dimensions |
| Keyword model | `llama3.1:latest` |
| LightRAG `num_ctx` | 16384 |
| `top_k` | 20 |
| Evidence budget | 4 |
| Max chunks per source on the `query_documents` path | 1 |
| Generator | `host_finetune/output/lemkin_lora_relabel` |
| Base | `unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit` |
| Quantization | 4-bit |
| Generation window | 2048 tokens |
| Prompt family | `model-f-graph-context-v1` |
| Decoding | temperature 0.7, top_p 0.9, repetition penalty 1.15, seed `42 + gold_index * 4 + medium_index` |
| Output budgets | blog 768, LinkedIn 320, X 160, talk 768 |

Graph-context profiles:

| CLI `--model-f-context` | Experiment arm | What is rendered |
|---|---|---|
| `source` | `F-SOURCE` | primary source claims |
| `relationships` | `F-REL` | primary sources plus source-supported relationships |
| `descriptions` | `F-DESC` | primary sources plus derived descriptions |
| `multihop` | `F-HOP` | primary sources plus multi-hop paths |
| `off` | none | current non-graph prompt. This is the CLI default. A Model F demo sets one of the four profiles above. |

The profile has not been selected. Selection is the 36-question run `EXP-20261101-009-model-f-four-arm`, which has not been executed. For a smoke test, call `source` so the route exercises hybrid retrieval and the graph-context contract, and call `descriptions` once so entity-description text is actually rendered. Record the profile name on the response. Leave the existing Model E route on its current dense / PersonaRAG behavior.

Local evaluation and the deployment demo differ in these ways:

- The experiment runners refuse a path that contains `storage_expand_v1` and they call `verify` before generation. A service should do the same check against the runtime copy at startup.
- The experiment runners set `num_ctx` 16384. The `compare_adapters.py` CLI retriever currently uses 32768.
- The deploy QLoRA path uses temperature 0 and an Ollama GGUF tag. Model F uses temperature 0.7 and the PEFT adapter.
- Talk and blog evidence caps in the four-arm budgets are fitted to the 2048 window (557 source tokens and 83 graph tokens). That fit is in `host_finetune/model_f_experiment/arms.py`.
- Gold references are evaluator text. They are not inserted into the Model F prompt.

## E. Deployment integration

1. Create an integration branch from `andreah/deploy-with-react`. Do not commit on `andreah/deploy-with-react` itself until the smoke tests below pass.
2. Bring in the Model F Python from `kevin/lightrag-model-f`: `host_finetune/model_f_context/`, `host_finetune/model_f_experiment/`, `host_finetune/eval_bench/`, `host_finetune/lightrag_backend.py`, `host_finetune/compare_adapters.py`, and the tests listed in section F. Keep the React app, Docker files, and the current generate route from `andreah/deploy-with-react`.
3. Resolve the six conflicts in section A by keeping the deploy branch's service behavior. `backend/main.py` should keep the React request and response shape. Add a Model F path beside it. A clean textual merge of `compare_adapters.py` still needs a read, because this branch's copy is the Model F generator.
4. Place the extracted snapshot outside the repository. Set the Model F workdir to a runtime copy of that extract.
5. Place the six adapter files in the directory the Model F route will pass to `load_generator`. The code default is `host_finetune/output/lemkin_lora_relabel`.
6. Install `host_finetune/requirements.txt` and `host_finetune/requirements-lightrag.txt` into the generation environment. Re-apply `opentelemetry-api==1.41.1` after LightRAG, as that requirements file says. Pull `nomic-embed-text` and `llama3.1`. Let Unsloth download the 4-bit base model on a CUDA machine.
7. Check the three SHA-256 values in section B, then run `verify` on the extracted snapshot.
8. Start the process with the runtime copy as the LightRAG working directory. Confirm startup prints or logs the snapshot path, mode `hybrid`, and a profile other than `off`.
9. Hybrid retrieval smoke, on the runtime copy only:

```powershell
host_finetune\.venv\Scripts\python.exe -X utf8 -c "import asyncio, json; from pathlib import Path; from host_finetune.model_f_experiment.retrieval import retrieve_raw; payload = asyncio.run(retrieve_raw('What price did Capital One announce for acquiring Brex?', Path(r'PATH\TO\RUNTIME\COPY'))); print(json.dumps({'status': payload.get('status'), 'keys': sorted(payload.keys())}))"
```

Ollama must be running. This call can write inside the runtime copy. It does not load the Relabel weights.

10. Relabel generation smoke, one topic, explicit profile, runtime copy:

```powershell
host_finetune\.venv\Scripts\python.exe -X utf8 -m host_finetune.compare_adapters --adapters relabel=PATH\TO\lemkin_lora_relabel --retrieval --retrieval-backend lightrag --lightrag-workdir PATH\TO\RUNTIME\COPY --lightrag-mode hybrid --model-f-context source --topic-ids gold_001 --skip-score --out-dir PATH\TO\SMOKE_OUT
```

This loads the 4-bit model on GPU. The CLI retriever uses `num_ctx` 32768, so treat this as a service smoke, not as the 120-prompt measurement. The measurement path is `python -m host_finetune.model_f_experiment`, which uses 16384.

11. Open the smoke trace and confirm `model_f_context.profile` is `source` and `graph_context_contract` is `model-f-graph-context-v1`. Run one more topic with `--model-f-context descriptions` and confirm the trace contains the descriptions channel. If that channel is missing, the route is not exercising F-DESC.

Keep the existing Model E / PersonaRAG route unchanged until both smokes pass. The Model F route is additive.

## F. Validation status and known limitations

| Work | Status |
|---|---|
| Graph-context pipeline, four profiles, snapshot/verify/audit commands, four-arm runner | IMPLEMENTED |
| Entity descriptions returned by `query_documents` and read by F-DESC without entering hits | IMPLEMENTED. Fixture-tested. Not live-validated on this snapshot. |
| 120-prompt panel (`unified-plan`, `unified-run`, `unified-score`) | IMPLEMENTED. CPU fixture-tested. No Model F drafts have been generated. |
| CPU tests on 2026-11-02 | TESTED. 68 passed, 2 DeprecationWarning from SWIG types, in `tests/test_model_f_unified.py`, `tests/test_model_f_graph_context.py`, `tests/test_model_f_experiment.py`, `tests/test_eval_bench.py`, `tests/test_eval_bench_historical.py`, `tests/test_lightrag_retrieval.py`. |
| Snapshot hash recheck on 2026-11-02 | The 31 recorded file hashes matched. This was a read of the existing snapshot. |
| Hybrid query and Relabel generation on this snapshot | Not run for this handoff. |
| Four-arm experiment `EXP-20261101-009-model-f-four-arm` | Not MEASURED. |
| Unified 120-prompt panel `EXP-20261102-001-model-f-unified-panel` | Not MEASURED. Historical Base, PersonaRAG, Relabel-off, and Relabel+RAG rows stay in their own files. |
| Deployment service | Not integrated and not validated. |

Index coverage, from the snapshot ledger opened read-only and from `reports/indexing_summary.json` inside the verified snapshot:

| Status | Documents |
|---|---|
| completed | 2,713 |
| pending | 10,269 |
| interrupted | 153 |
| failed_permanent | 5 |
| processing | 0 |

Completed documents by `source_medium`: blog 1,890, LinkedIn 370, X 453. LinkedIn plus X is 823. The normal indexing process has exited. Deferred and interrupted documents were not retried. The pending rows stay pending. This partial store is the baseline to deploy and evaluate.

`reports/indexing_summary.json` records 27,797 graph nodes, 47,897 edges, and 9,269 successful chunks, with `graph_count_source` `measured_after_shutdown_flush`. A direct count of the `data` array in `vdb_chunks.json` is 9,266 rows, and the first vector has length 768. Both numbers are from the hash-matched snapshot. The 3-row gap was not repaired.

Relationship recovery on this snapshot is `experiments/EXP-20261102-model-f-repair/raw/recovery_summary.json`: mode `dry-run`, policy `relation-normalization-v1`, 413 `insert_candidate`, 987 `already_in_graph`, 3,326 `needs_review`, 462 `rejected_endpoint_absent`. Nothing was inserted. The ledger was not applied and is not in the transfer zip.

The graph audit file beside the snapshot contains 25,184 findings. `recovery_applied` is false and `graph_modified` is false. The audit's own impact labels are:

| Impact label | Findings |
|---|---|
| `affects_retrieval_and_generation` | 8,143 |
| `affects_retrieval` | 2,508 |
| `cosmetic_or_cache_only` | 14,533 |

The largest kinds are `weak_entity_in_graph` 7,236, `schema_labels` 3,323, `comma_between_keywords_and_description` 3,104, `omitted_keywords` 3,064, `omitted_description` 2,873, and `alias_group` 2,498. Those impact labels are the audit's categories. They have not had a separate severity review. Do not treat the cache-only rows as edits to the graph.

The original expansion directory `artifacts/lightrag/model_f/storage_expand_v1` and the verified snapshot were not modified by this handoff. Indexing was not resumed. Embeddings were not regenerated. Relationship repair was not applied.

There is no Model F versus Model E result. Do not describe Model F as better than the other architectures.

## G. Message you can send

Branch: `kevin/lightrag-model-f` on https://github.com/andreahcruz/PersonaCloningGen. Pull the tip. The Model F code commit is `c04e77cf438e7d845c640d2727250ce642243204`. The guide is `docs/handoff/MODEL_F_DEPLOYMENT_TRANSFER_20261102.md` on that branch. Since `817e159`, the remote already had the Workbook 2 catch-up note (`54ceb5d`). This push adds entity-description pass-through (descriptions stay out of citable hits) and the 120-prompt panel runner. The panel does not choose the profile. CPU tests: 68 passed.

I will send three files from `C:\Users\okevi\DATA298_model_f_transfer`:

- `lightrag_storage_expand_v1_snapshot.zip` (481,019,443 bytes, SHA-256 `4e7a00a977efe0c4ab121b0356704ecd4472cb17d0a96533f8c80a5ad500b9ea`)
- `relabel_qlora_adapter.zip` (158,836,161 bytes, SHA-256 `644041cbe47e0a1ce69997e790be799b5ed42862be2854b792e94a24dd93777f`)
- `graph_audit\storage_expand_v1_snapshot_graph_audit.json` (5,628,431 bytes, SHA-256 `2672aee28b5564876f6f8d547745acdc3eda7780e6b5df5d4812f5def2e0b4dd`)

Check those hashes, extract outside the repo, and run `python -m host_finetune.model_f_experiment verify` on the extracted snapshot. Copy it again before any query. The adapter zip is the Relabel PEFT directory, not an Ollama GGUF. Generation also needs `unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit` on a CUDA machine. Retrieval needs Ollama `nomic-embed-text` and `llama3.1:latest`.

You can start now: an integration branch from `andreah/deploy-with-react`, bring over the `host_finetune` Model F modules, and keep the React route. Merge-tree conflicts are `.dockerignore`, `.gitignore`, `README.md`, `backend/Dockerfile`, `backend/main.py`, and `docker-compose.yml`. The deploy backend still has no LightRAG path. A Model F route has to set `--retrieval-backend lightrag`, mode `hybrid`, the runtime snapshot, and an explicit profile (`source` for the first smoke, `descriptions` to show that channel). Leave Model E on the existing route until those smokes pass.

Still pending on my side: the 36-question four-arm run and the 120-prompt panel. No Model F score exists yet. The snapshot is partial on purpose: 2,713 completed documents (1,890 blogs and 823 LinkedIn/X), with 10,269 pending and 153 interrupted left as they are. The 413 relationship repairs were a dry run and were not applied. The graph audit lists 25,184 findings and still needs a severity pass. The verified snapshot and the live expansion directory were left unchanged.

## H. Commands left unrun on purpose

Indexing was not resumed. `storage_expand_v1` was not opened for writes. The snapshot and the adapter were not edited. Relationship repair was not applied. No new embeddings were created. No GPU generation was run. Nothing was uploaded. No merge into `andreah/deploy-with-react` was made.
