# Airflow / Spark v2 architecture

Status: shadow pipeline through RAG governance. It reproduces the Python oracle through the 21,193-document corpus. It does not replace `lemkin_content_pipeline`, and it does not embed.

## Current versus target

| | Current demo path | v2 shadow path |
|---|---|---|
| Orchestration | DAG `lemkin_content_pipeline` | DAG `lemkin_canonical_v2_shadow` |
| Clean | DAG normalizers plus `spark_jobs/clean_and_embed.py` | `canonical_cleaning.clean_record` inside Spark `mapPartitions` |
| Chunking | about 500 words for retrieval | frozen SFT chunker, promo filter, then `canonical_fit` |
| Balance, split, relabel, RAG | not this DAG | `select_balanced_rows`, `assign_groups`, `transform_rows`, `select_train_documents` |
| Embeddings | Ollama inside Spark | not in this stage |
| Chroma | `lemkin_content` | not written; the persisted `lemkin_train_only` database is opened SQLite `mode=ro` for comparison only |
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
    → parity_summary
```

Embeddings and Chroma publication are named and not implemented.

Airflow orders the tasks, records the run id, and stops the run when a stage raises `ParityError`. A parity task reads the previous manifest and fails the run when that manifest is missing or not `PASS`. The algorithms stay in `host_finetune`. The DAG callables and `python -m host_finetune.spark_v2` call the same stage functions.

## Execution and order

Raw files are read in `SOURCE_SPECS` order. Each row keeps `source_ordinal` and the physical `raw_record_index`.

Cleaning runs in Spark partitions when the executor is `spark`. `clean_partition` calls `clean_record` once per row. `mapPartitions` runs that function. An accumulator must equal the requested partition count, or the stage fails. There is no fallback to the in-process cleaner. After the partitions return, `restore_order` sorts by `(source_ordinal, raw_record_index)` before any cleaned file is written. Partition arrival order is not the corpus order.

Measured in the Airflow container with PySpark 3.5.0, master `local[*]`, Python `/usr/local/bin/python`, and `canonical_cleaning.py` imported from `/opt/airflow/host_finetune`. Partition counts 1, 2, 4, and 8 each matched the ordered cleaner and the frozen oracle. Worker partitions ran equaled the requested count. Fallback was false.

SFT and the promo filter run in process, in cleaned-file order, through the existing helpers in `build_sft_from_cleaned_sources` and `filter_event_promos`.

Fit calls `canonical_fit.fit_rows` on the full ordered nopromo list. Lead-in merges look at adjacent rows, so this stage is one sequence. It is not split across partitions. The tokenizer is the mounted snapshot `f15c379fb32bb402fa06a7ae9aecb1febf4b79ec`, loaded with `local_files_only=True`. Model weights are not loaded.

Balance calls `select_balanced_rows` and writes explicit CRLF JSONL. Split calls `assign_groups`. Relabel calls `transform_rows` with the same local tokenizer. RAG calls `blocked_group_ids`, `headline_overlap_groups`, and `select_train_documents`.

## Local command

```text
python -m host_finetune.spark_v2 --run-id local --through parity_summary --executor ordered
python -m host_finetune.spark_v2 --executor spark --partition-counts 1,2,4,8 --run-id spark-proof
```

`spark_jobs/v2_canonical.py` calls that same module.

## Manifest

Each stage writes `manifests/<stage>.json` with `run_id`, `stage`, `executor`, input and output artifact paths, row counts, hashes, `git_sha`, `dirty`, `config_version` (`spark-v2-shadow-rag-v1`), `status`, and `mismatches`. A nonzero mismatch raises `ParityError`, and Airflow does not start the next task.

## Measured Airflow run

Run id `v2-rag-airflow-20261005`. Every task state was success, including `parity_summary`.

Clean used executor `spark` and kept 30,370 rows. Membership, order, content, and identity mismatches were 0.

Fit wrote 43,012 rows with SHA-256 `bf00e53fd2401f2d6bea6ac39cb69829fc21761de3a4f7225fed7ecd75082fba`.

Balance wrote 31,328 rows with SHA-256 `6e99b414733eafdfa27a5e94dd26beff1498bbfd58af132251e24044fef4f057`.

Split counts were train 25,062, validation 3,133, test 3,133. Crossing groups were 0.

Relabel wrote 26,545 rows with SHA-256 `1231835ecc5dcb608fefa0357d325affb2afbdc3aeade2a63b1dc42b2920897b`.

RAG governance wrote 21,193 rows. Gold exclusions were 183. The headline exclusion was `train_26148`. Role counts were unchanged 11,611, opening 1,687, continuation 7,895. Missing ids, unexpected ids, and document, group, medium, and split mismatches against the persisted collection were 0. Nothing was embedded.

The frozen datasets and the production collections were not rewritten.
