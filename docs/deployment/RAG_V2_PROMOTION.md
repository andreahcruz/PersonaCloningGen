# RAG v2 promotion

Status: cut over for new `compare_adapters` inference. v1 remains in place.

## Provenance

The embedding Airflow run is the source of the index. Its manifests still say `fae3a95` and `dirty=true`. This promotion does not rewrite those manifests.

| Field | Value |
|---|---|
| Source Airflow run | `v2-embed-airflow-20261005` |
| Source RAG SHA-256 | `240e7bccfb2458e05f99f33a2643f2e7ebff0ecf68af65d615155965d4a49b49` |
| Source code SHA | `fae3a951237621345fbfbbfbb980d84d20e65cbb` |
| Source dirty | true |
| Promotion checkpoint SHA | `08f8436580bcdf82c5285c3e8bb3664385a19b27` |
| Embedding model | `nomic-embed-text` |
| Embedding digest | `0a109f422b47e3a30ba2b10eca18548e944e8a23073ee3f3e947efcf3c45e59f` |
| Ollama | 0.34.1 |
| Dimension | 768 |
| Documents | 21,193 |
| Vector contract | `float32_one_ulp`: 15,751 float32-identical rows, 5,442 within one float32 ULP, 0 outside that limit. Minimum cosine `0.9999999999999964`. Status: CURRENT EMBEDDING BEHAVIOR REPRODUCED. |
| Logical Chroma parity | ID, document, metadata, and vector mismatches all 0 |
| Retrieval at shadow validation | 29 exact rankings and 1 equal-distance cutoff tie (`gold_013`, distance `0.24499374628067017`) |
| Retrieval at pre-cutover comparison | 30/30 exact top-4 rankings, k=4, no reranker, no medium filter |
| QLoRA smoke | `gold_001` retrieved `train_20783`, `train_26362`, `train_21878`, `train_25912` and generated blog, LinkedIn, X, and talk. Exit 0. No `lemkin_content` fallback. |
| Cutover | 2026-10-05, `host_finetune/rag_active.json` set to `v2` |

## Indexes

| | Path | Collection |
|---|---|---|
| v1, untouched | `host_finetune/output/chroma_lemkin_train_only` | `lemkin_train_only` |
| v2, promoted | `host_finetune/output/chroma_lemkin_train_only_v2` | `lemkin_train_only_v2_shadow` |

v2 is a full directory copy of `artifacts/refactor/spark_v2/v2-embed-airflow-20261005/chroma`. The copy was byte-identical to that shadow directory before any client opened it. The collection name was kept because renaming it would rewrite the validated database. The stable path is not the Airflow run directory.

`chroma.sqlite3` for v1 remained `a87c12d31eda36cc81f831bb6b5114d3c545e5e98e1fb20836a7ca718b1e7bb9`.

## Active selection

`host_finetune/rag_active.json` is the only switch. `compare_adapters` reads it when `--collection` and `--chroma-path` are omitted. Startup logs the version, path, collection, and document count.

Rollback is that same file:

```json
{ "version": "v1" }
```

No rebuild, re-embedding, or database restore. After a rollback, a new process logs `RAG version=v1` and the v1 path. This was checked by switching the file to v1 and back to v2. The deployed file is v2.

## What was not cut over

`generate.py`, Streamlit, and `lemkin_content_pipeline` still refer to legacy `lemkin_content`. `host_finetune/watcher.py` still watches the old training sentinel. Those paths were not repointed and were not deleted.

## Provenance caveats

These limits are historical records. They are not current transformation mismatches.

- The historical balance invocation command is unrecorded.
- The historical command that first built `lemkin_train_only` is unrecorded.
- Some historical experiment worktrees were dirty.
- The successful embedding Airflow run `v2-embed-airflow-20261005` recorded source SHA `fae3a95` and `dirty=true`. That dirty flag stays in the run manifests.

## Airflow boundary

The shadow DAG writes a versioned run directory and stops at parity. It does not copy that directory onto v1 or v2. A later promotion task can call the same directory copy after the parity gates. It must refuse `chroma_lemkin_train_only` as a destination.
