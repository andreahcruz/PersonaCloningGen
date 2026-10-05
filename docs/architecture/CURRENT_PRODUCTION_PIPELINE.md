# Current production pipeline

This is the active path for `compare_adapters`. The frozen Python artifacts remain the parity oracle that proved the Airflow/Spark v2 implementation. They are no longer the only execution path.

## Data pipeline

Five frozen raw sources

→ Airflow/Spark v2 orchestration (`lemkin_canonical_v2_shadow`)

→ canonical cleaning, 32,835 → 30,370

→ SFT build and promo filter, 30,370 → 42,771

→ canonical FIT, 42,771 → 43,012

→ balance, 43,012 → 31,328

→ group split

→ Relabel, 26,545 total

→ Relabel train, 21,377

→ RAG governance, 21,193

→ `nomic-embed-text`, 768 dimensions

→ Chroma v2, collection `lemkin_train_only_v2_shadow`

The embedding call is `POST /api/embed` with batches of 16 and `num_gpu=0`. The embedded string is the stored RAG document, which is the stripped Relabel output. The vector contract is float32 within one unit in the last place. The Airflow run that produced this index is `v2-embed-airflow-20261005`. Its manifests record code SHA `fae3a951237621345fbfbbfbb980d84d20e65cbb` and `dirty=true`.

## Generation pipeline

User topic

→ `nomic-embed-text` query embedding

→ active RAG v2

→ cosine retrieval

→ k=4

→ factual excerpts

→ Relabel QLoRA adapter

→ generated medium-specific response

There is no reranker and no medium filter.

## Active model and RAG

| Piece | Location |
|---|---|
| QLoRA adapter | `host_finetune/output/lemkin_lora_relabel` |
| RAG selector | `host_finetune/rag_active.json` |
| Active version | `v2` |
| Active path | `host_finetune/output/chroma_lemkin_train_only_v2` |
| Active collection | `lemkin_train_only_v2_shadow` |
| Documents | 21,193 |
| Held-out id | `train_26148` is absent |

`compare_adapters` reads the selector when `--collection` and `--chroma-path` are omitted. An explicit flag overrides that selection. `lemkin_content` is not a valid version.

## Rollback

v2 to v1 is one edit of `host_finetune/rag_active.json` to `{"version": "v1"}`. v1 stays at `host_finetune/output/chroma_lemkin_train_only`, collection `lemkin_train_only`, 21,193 documents. Database SHA-256 `a87c12d31eda36cc81f831bb6b5114d3c545e5e98e1fb20836a7ca718b1e7bb9`. No rebuild, re-embedding, or restore. The switch was checked v2 → v1 → v2, and the file was left on v2.

Details and the promotion copy record are in `docs/deployment/RAG_V2_PROMOTION.md`.

## Legacy

`lemkin_content` and `dags/lemkin_pipeline_dag.py` are historical. They are not the active `compare_adapters` RAG path.

`generate.py`, Streamlit, and `host_finetune/watcher.py` have not been migrated. They can still follow the legacy collection or the old training sentinel.

## Provenance caveats

The historical balance invocation is unrecorded. The historical command that first built the v1 index is unrecorded. Some historical experiment worktrees were dirty. The successful v2 embedding Airflow run was dirty. Those are provenance limits, not mismatches in the current transformation.
