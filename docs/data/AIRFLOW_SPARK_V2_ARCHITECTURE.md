# Airflow / Spark v2 architecture

Status: shadow pipeline through the fit stage. It reproduces the Python oracle. It does not replace `lemkin_content_pipeline`, and it does not embed.

## Current versus target

| | Current demo path | v2 shadow path |
|---|---|---|
| Orchestration | DAG `lemkin_content_pipeline` | DAG `lemkin_canonical_v2_shadow` |
| Clean | DAG normalizers plus `spark_jobs/clean_and_embed.py` | `canonical_cleaning.clean_record` |
| Chunking | about 500 words for retrieval | the frozen SFT chunker and promo filter, then `canonical_fit` |
| Embeddings | Ollama inside Spark | not in this stage |
| Chroma | `lemkin_content` | not written |
| Training handoff | `host_finetune/watcher.py` watches MinIO `training/_READY` | not used |
| Output | MinIO `lemkin-raw` and `lemkin-processed` | `artifacts/refactor/spark_v2/<run_id>/` |

Infrastructure that stays: Compose Airflow, the local Spark master, MinIO, and the mounted `dags/` and `spark_jobs/` directories. The Airflow service also mounts `host_finetune/` read-only so the shadow DAG can import the canonical modules. That mount does not change the old DAG's tasks.

Transformation behavior that does not stay: 500-word chunks, Spark-local HTML cleanup, Ollama embedding, and `lemkin_content`.

## Stage graph

```text
validate_raw
    → canonical_clean          32,835 → 30,370
    → build_nopromo            30,370 → 42,771
    → canonical_fit            42,771 → 43,012
    → parity_summary
```

Later stages are named and not implemented: balance, split, relabel, rag governance, embeddings, staging Chroma.

Airflow orders the tasks, records the run id, and stops the run when a stage raises `ParityError`. The algorithms stay in `host_finetune`.

## Execution and order

Raw files are read in `SOURCE_SPECS` order. Each row keeps `source_ordinal` and the physical `raw_record_index`.

Cleaning may run in Spark partitions. `clean_partition` calls `clean_record` once per row. `mapPartitions` would initialize nothing heavier than that. After the partitions return, `restore_order` sorts by `(source_ordinal, raw_record_index)` before any cleaned file is written. The shadow default is the ordered in-process call of that same function. `--executor spark` uses a local Spark context and then the same sort. Partition arrival order is not the corpus order.

SFT and the promo filter run in process, in cleaned-file order, through the existing helpers in `build_sft_from_cleaned_sources` and `filter_event_promos`. They are not reimplemented in Spark SQL.

Fit calls `canonical_fit.fit_rows` on the full ordered nopromo list. Lead-in merges look at adjacent rows, so this stage is one sequence. It is not split across partitions. The tokenizer is the local snapshot `f15c379fb32bb402fa06a7ae9aecb1febf4b79ec`, loaded with `local_files_only=True`. Model weights are not loaded.

## Local command

```text
python -m host_finetune.spark_v2 --run-id local --through parity_summary
```

`spark_jobs/v2_canonical.py` calls that same module. The DAG tasks call it too.

## Manifest

Each stage writes `manifests/<stage>.json` with `run_id`, `stage`, row counts, hashes, `git_sha`, `dirty`, `config_version` (`spark-v2-shadow-fit-v1`), `status`, and `mismatches`. `status` is `FAIL` only when the process records a failure before raising. A nonzero mismatch raises `ParityError`, and Airflow does not start the next task.

## Measured shadow parity

Clean: 30,370 kept, membership 0, order 0, content 0, identity 0.

Nopromo: 42,771, content and lineage mismatches 0.

Fit: 43,012, SHA-256 `bf00e53fd2401f2d6bea6ac39cb69829fc21761de3a4f7225fed7ecd75082fba`, lineage mismatches 0.

The frozen nopromo file, the frozen fit file, and the production collections were not rewritten.
