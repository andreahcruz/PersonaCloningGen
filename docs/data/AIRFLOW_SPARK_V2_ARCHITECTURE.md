# Airflow / Spark v2 architecture

Status: shadow pipeline through a staging Chroma index. It reproduces the Python oracle through the 21,193-document corpus, embeds those documents with the current CPU `nomic-embed-text` call, and writes `lemkin_train_only_v2_shadow`. It does not replace `lemkin_content_pipeline` or `lemkin_train_only`. Active `compare_adapters` inference uses the promoted copy in `docs/architecture/CURRENT_PRODUCTION_PIPELINE.md`. `lemkin_content` stays legacy.

## Current versus target

| | Current demo path | v2 shadow path |
|---|---|---|
| Orchestration | DAG `lemkin_content_pipeline` | DAG `lemkin_canonical_v2_shadow` |
| Clean | DAG normalizers plus `spark_jobs/clean_and_embed.py` | `canonical_cleaning.clean_record` inside Spark `mapPartitions` |
| Chunking | about 500 words for retrieval | frozen SFT chunker, promo filter, then `canonical_fit` |
| Balance, split, relabel, RAG | not this DAG | `select_balanced_rows`, `assign_groups`, `transform_rows`, `select_train_documents` |
| Embeddings | Ollama inside Spark | `POST /api/embed` for `nomic-embed-text` with `num_gpu=0`, batch 16, no prefix |
| Chroma | `lemkin_content` | new collection `lemkin_train_only_v2_shadow` under the run directory; production is never opened for write |
| Training handoff | `host_finetune/watcher.py` watches MinIO `training/_READY` | not used |
| Output | MinIO `lemkin-raw` and `lemkin-processed` | `artifacts/refactor/spark_v2/<run_id>/` |

Infrastructure that stays: Compose Airflow, the local Spark master, MinIO, and the mounted `dags/` and `spark_jobs/` directories. The Airflow services also mount `host_finetune/` read-only, `artifacts/` read-write, `experiments/` read-only, `.git` read-only, and the host Hugging Face cache read-only at `/home/airflow/.cache/huggingface`. `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1` are set. Those mounts do not change the old DAG's tasks.

The Bitnami Spark workers are not the v2 execution path. Pip PySpark 3.5.0 runs `local[*]` inside the Airflow container, which avoids the JAR mismatch called out in the Compose file.

Transformation behavior that does not stay: 500-word chunks, Spark-local HTML cleanup, Ollama embedding, and `lemkin_content`.

## Stage graph

```text
validate_raw
    → canonical_clean          32,835 → 30,370     Spark mapPartitions
    → clean_parity
    → build_nopromo            30,370 → 42,771     ordered Python
    → nopromo_parity
    → canonical_fit            42,771 → 43,012     one ordered sequence
    → fit_parity
    → balance                  43,012 → 31,328
    → balance_parity
    → split                    25,062 / 3,133 / 3,133
    → split_parity
    → relabel                  26,545
    → relabel_parity
    → rag_governance           21,193
    → rag_parity
    → embed_rag                21,193 vectors
    → embedding_parity
    → build_chroma_shadow      lemkin_train_only_v2_shadow
    → logical_chroma_parity
    → retrieval_parity         gold topics, k=4, one shared query vector
    → parity_summary
```

`embedding_parity` runs before the shadow collection is built. A vector-contract failure stops the run before Chroma publication. The DAG does not copy or replace production Chroma. Promotion is a separate manual copy of a validated shadow directory to `host_finetune/output/chroma_lemkin_train_only_v2`, recorded in `docs/deployment/RAG_V2_PROMOTION.md`. A future promotion task could wrap that copy, and it still must not overwrite `chroma_lemkin_train_only`.

Airflow orders the tasks, records the run id, and stops the run when a stage raises `ParityError`. A parity task reads the previous manifest and fails the run when that manifest is missing or not `PASS`. The algorithms stay in `host_finetune`. The DAG callables and `python -m host_finetune.spark_v2` call the same stage functions.

## Execution and order

Raw files are read in `SOURCE_SPECS` order. Each row keeps `source_ordinal` and the physical `raw_record_index`.

Cleaning runs in Spark partitions when the executor is `spark`. `clean_partition` calls `clean_record` once per row. `mapPartitions` runs that function. An accumulator must equal the requested partition count, or the stage fails. There is no fallback to the in-process cleaner. After the partitions return, `restore_order` sorts by `(source_ordinal, raw_record_index)` before any cleaned file is written. Partition arrival order is not the corpus order.

Measured in the Airflow container with PySpark 3.5.0, master `local[*]`, Python `/usr/local/bin/python`, and `canonical_cleaning.py` imported from `/opt/airflow/host_finetune`. Partition counts 1, 2, 4, and 8 each matched the ordered cleaner and the frozen oracle. Worker partitions ran equaled the requested count. Fallback was false.

SFT and the promo filter run in process, in cleaned-file order, through the existing helpers in `build_sft_from_cleaned_sources` and `filter_event_promos`.

Fit calls `canonical_fit.fit_rows` on the full ordered nopromo list. Lead-in merges look at adjacent rows, so this stage is one sequence. It is not split across partitions. The tokenizer is the mounted snapshot `f15c379fb32bb402fa06a7ae9aecb1febf4b79ec`, loaded with `local_files_only=True`. Model weights are not loaded.

Balance calls `select_balanced_rows` and writes explicit CRLF JSONL. Split calls `assign_groups`. Relabel calls `transform_rows` with the same local tokenizer. RAG calls `blocked_group_ids`, `headline_overlap_groups`, and `select_train_documents`.

Embedding calls `canonical_embeddings.embed_documents` on the stored document string, which is the stripped Relabel output. The request is the same body `rebuild_chroma._embed_batch` posts for the train-only index: model `nomic-embed-text`, `options.num_gpu=0`, batches of 16, no `search_document` prefix, and no extra normalization. The historical command that first built `lemkin_train_only` is still unrecorded. A vector matches when the fresh vector, cast to float32, is within one float32 unit in the last place of the stored production vector. That limit comes from the sample, where every component was identical or exactly one step away. On the full 21,193 rows the same limit held: 15,751 float32 vectors were bit-identical and 5,442 differed by one unit in the last place. Retrieval uses the 30 gold topics at k=4 with one shared query vector. Two topics, `gold_013` and `gold_028`, swap the fourth hit with another document at the identical distance. Brute-force cosine places those pairs in a tie, so that swap is not counted as a different result.

The shadow collection is written under `artifacts/refactor/spark_v2/<run_id>/chroma` with cosine distance and metadata `group_id`, `medium`, `row_id`, and `split`. Production vectors are read from a byte copy of `host_finetune/output/chroma_lemkin_train_only`. The live database is only hashed.

## Local command

```text
python -m host_finetune.spark_v2 --run-id local --through parity_summary --executor ordered
python -m host_finetune.spark_v2 --executor spark --partition-counts 1,2,4,8 --run-id spark-proof
```

`spark_jobs/v2_canonical.py` calls that same module.

## Manifest

Each stage writes `manifests/<stage>.json` with `run_id`, `stage`, `executor`, input and output artifact paths, row counts, hashes, `git_sha`, `dirty`, `config_version` (`spark-v2-shadow-embed-v1`), `status`, and `mismatches`. A nonzero mismatch raises `ParityError`, and Airflow does not start the next task.

## Measured Airflow run

Run id `v2-rag-airflow-20261005`. Every task state was success, including `parity_summary`.

Clean used executor `spark` and kept 30,370 rows. Membership, order, content, and identity mismatches were 0.

Fit wrote 43,012 rows with SHA-256 `bf00e53fd2401f2d6bea6ac39cb69829fc21761de3a4f7225fed7ecd75082fba`.

Balance wrote 31,328 rows with SHA-256 `6e99b414733eafdfa27a5e94dd26beff1498bbfd58af132251e24044fef4f057`.

Split counts were train 25,062, validation 3,133, test 3,133. Crossing groups were 0.

Relabel wrote 26,545 rows with SHA-256 `1231835ecc5dcb608fefa0357d325affb2afbdc3aeade2a63b1dc42b2920897b`.

RAG governance wrote 21,193 rows. Gold exclusions were 183. The headline exclusion was `train_26148`. Role counts were unchanged 11,611, opening 1,687, continuation 7,895. Missing ids, unexpected ids, and document, group, medium, and split mismatches against the persisted collection were 0. Nothing was embedded in that run.

A separate poller process exited after that run had already reached success. That exit is monitoring-process termination, not an Airflow task failure.

The frozen datasets and the production collections were not rewritten.

## Measured embedding run

Run id `v2-embed-airflow-20261005`. Every task state was success, including `embed_rag`, `embedding_parity`, `build_chroma_shadow`, `logical_chroma_parity`, `retrieval_parity`, and `parity_summary`.

The RAG artifact SHA-256 was again `240e7bccfb2458e05f99f33a2643f2e7ebff0ecf68af65d615155965d4a49b49` for 21,193 rows. Embeddings used installed `nomic-embed-text` digest `0a109f422b47e3a30ba2b10eca18548e944e8a23073ee3f3e947efcf3c45e59f` through Ollama 0.34.1. The model was not pulled. Dimension 768. Float32 comparison: 15,751 exact, 5,442 within one unit in the last place, 0 outside that limit. The shadow collection `lemkin_train_only_v2_shadow` has 21,193 rows, cosine distance, and `train_26148` absent. Served vectors matched production on all 21,193 ids. Thirty gold topics were queried at k=4. Twenty-nine top-k lists matched in order. `gold_013` swapped the fourth hit with another document at the same distance, `0.24499374628067017`. No score differed. Production `chroma.sqlite3` hash stayed `a87c12d31eda36cc81f831bb6b5114d3c545e5e98e1fb20836a7ca718b1e7bb9`.
