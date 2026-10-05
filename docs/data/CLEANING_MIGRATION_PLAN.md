# Cleaning ownership migration plan

Status: L2 record-policy inversion is in place. Airflow and Spark were not migrated. Persisted Chroma was not audited.

`clean_record` calls `scrub_corpus_text` and owns the record gates. `clean_row` delegates to `clean_record` for the five frozen source specs and returns the historical tuple. `apply_source_policy` remains the one gate implementation for a caller that passes arguments outside those specs. `clean_file` was not run. `data/cleaned/` was not rewritten.

`docs/architecture/PIPELINE_TARGET_LAYERS.md` Phase D remains true for the scrub call: there is one `scrub_corpus_text`. The record-policy inversion below is done. Airflow and Spark were left on their existing imports. Do not create a second scrub module. Do not run `python -m host_finetune.clean_scraped`. That entry point writes `data/cleaned/`.

## Call graph after the L2 inversion

```text
scrub_corpus_text
    ↑
canonical_cleaning.apply_source_policy
    ↑
canonical_cleaning.clean_record
    ↑
clean_scraped.clean_row          historical tuple
    ↑
clean_file / prepare_complete_sft

build_provenance_sidecars.py     → clean_record
build_identity_crosswalk.py      → clean_record

NOT MIGRATED
dags/lemkin_pipeline_dag.py      → scrub_corpus_text, with its own X gates
spark_jobs/clean_and_embed.py    → strip_footer_noise only
generate.py, clean_dataset.py, clean_dataset_v2.py → strip_footer_noise
```

| Symbol | Class | Role |
|---|---|---|
| `clean_record` / `apply_source_policy` | CURRENT FROZEN PATH | Record gates, drop codes, identity copy. |
| `clean_row` | CURRENT FROZEN PATH, shim | Historical tuple. Delegates to `clean_record` for the five source specs. |
| `clean_file`, `clean_scraped` | CURRENT FROZEN PATH | Writers of `data/cleaned/`. Not re-run. |
| `scrub_corpus_text` and its helpers | CURRENT FROZEN PATH | The one text-level definition. File not moved. |
| `build_provenance_sidecars.py`, `build_identity_crosswalk.py` | REFACTOR-ONLY | Call `clean_record`. They do not write `data/cleaned/`. |
| `dags/lemkin_pipeline_dag.py` `_normalize_*` | AIRFLOW CALLER | Still calls `scrub_corpus_text` directly. Not migrated. |
| `spark_jobs/clean_and_embed.py` | SPARK CALLER | Still uses `strip_footer_noise` plus its own HTML strip and 500-word chunks. Not migrated. |
| `generate.py` | HISTORICAL PATH | `strip_footer_noise` while formatting `lemkin_content` context. Not the QLoRA path. |
| `host_finetune/clean_dataset.py` | HISTORICAL PATH | `strip_footer_noise` on a training file. Not L2. |
| `host_finetune/clean_dataset_v2.py` | HISTORICAL PATH | Same helper, plus Ollama repunctuation. Not L2. |
| `host_finetune/prepare_complete_sft.py` | HISTORICAL PATH | Still calls the `clean_row` shim. Not nopromo. |
| `count_sources.py`, `count_dataset_by_source.py` | UNUSED / DEAD for this lineage | Own reply filters and an HTML strip used by audits. Not the cleaner. |
| `experiments/EXP-*/config/clean_scraped.py` and `corpus_footer_scrub.py` | EXPERIMENT COPY | Frozen snapshots. Not edited. |
| `tests/test_corpus_footer_scrub.py` | TEST | Text-level rules. |
| `tests/test_canonical_cleaning.py`, `tests/test_cleaning_contract.py` | TEST | Corpus parity and the ownership contract. |

## Two kinds of shared logic

Text-level scrub stays in `spark_jobs/corpus_footer_scrub.py`:

- footer removal
- URL repair
- HTML unescape
- LinkedIn feed-chrome detection and `hashtag #` token strip
- L2 event-promo decision, because it runs after those text steps inside `scrub_corpus_text`

Record-level policy moves into `canonical_cleaning.py`:

- which field is the body, and which field is the title
- X reply or repost/quote rejection
- X empty body, then the 25-character floor
- renaming a YouTube `empty` result to `empty_transcript`
- stable drop codes
- copying `source_id` and `raw_record_id` from the raw record

`scrub_corpus_text` remains the lower-level shared module. Moving that file would break the Airflow Spark submit, which imports `corpus_footer_scrub` from `/opt/airflow/spark_jobs`. The module docstring records that mount. A new `canonical_text_scrub` module is not the narrow owner.

## Target call graph

Option B. Option A, a new scrub module under the canonical cleaner, is rejected for this migration.

```text
spark_jobs/corpus_footer_scrub.py
    scrub_corpus_text
    strip_footer_noise, repair_broken_urls, unescape_html
    is_event_promo, is_linkedin_feed_chrome, strip_linkedin_chrome
        ↑
host_finetune/canonical_cleaning.py
    clean_record
    record gates, field selection, DROP_CODES, identity copy
        ↑
host_finetune/clean_scraped.py
    clean_row          temporary shim, old tuple
    clean_file         writer, still calls clean_row
    clean_scraped      CLI, still calls clean_file

Unchanged side imports of the scrub module:
    DAG _normalize_*
    Spark clean_and_embed.strip_footer_noise
    generate.py, clean_dataset.py, clean_dataset_v2.py
```

After the shim commit, `canonical_cleaning` must not import `clean_row`. `clean_row` may import `clean_record`. That is the only legal direction. The intermediate commit that copies the gates into `canonical_cleaning` while `clean_row` still has them is temporary duplication. The next commit deletes the copy inside `clean_row`.

Do not point the DAG or Spark at `clean_record` in the L2 commits. Their output schema is not the cleaned JSONL schema.

## Duplicate-responsibility audit

| Responsibility | Implementations | Active on the frozen path? | Target owner |
|---|---|---|---|
| Footer, URL repair, HTML unescape, LinkedIn chrome, L2 event promo | `scrub_corpus_text` | Yes | Stay in `corpus_footer_scrub.py` |
| Same functions, frozen copies | `experiments/EXP-20261003-00{1,2,3}*/config/corpus_footer_scrub.py` | No. Snapshots. | Leave in the experiment directory |
| X reply/repost and 25-character floor | `clean_row`; copied in the DAG `_normalize_x` | `clean_row` is the frozen path. The DAG copy is Airflow. | `canonical_cleaning` for L2. DAG may call that helper later without changing its schema. |
| Empty transcript rename | `clean_row` only | Yes | `canonical_cleaning` |
| BeautifulSoup HTML strip | `spark_jobs/clean_and_embed.py`; a separate strip in `count_dataset_by_source.py` | No | Not L2. Do not merge with `unescape_html`. |
| SFT promo filter | `filter_event_promos.py` | Downstream of cleaning | Stays L3 |
| Complete-source promo reuse | `prepare_complete_sft.py` calls `is_event_promo` | No | Stays on the shim. Not a second L2. |
| Record API and drop codes | `canonical_cleaning.py` via `clean_row` | Wrapper only | `clean_record`, calling the scrubber directly |

## Compatibility shims

| Function | Current callers | Temporary behavior | Removal condition |
|---|---|---|---|
| `clean_row` | `canonical_cleaning` today; `prepare_complete_sft`; provenance sidecars; identity crosswalk; `clean_file` | After the shim commit, return the old `(row_or_none, frozen_reason, flags)` tuple by calling `clean_record`. | The name stays until every caller has moved. The inlined gate body is deleted in the shim commit. `canonical_cleaning` stops calling it one commit earlier. |
| `clean_file` | CLI only, plus the historical EXP-001 command | Keep writing through `clean_row`. Do not invoke it during the migration. | Not deleted. It is the writer, not a second rule set. |
| `clean_scraped` | `python -m host_finetune.clean_scraped` | Keep as the CLI over `clean_file`. | Not deleted. |
| `scrub_corpus_text` | `clean_row` today; DAG; after migration, `canonical_cleaning` | Remains the text implementation. | Not deleted. |
| `strip_footer_noise` and the other scrub helpers | Spark, `generate.py`, `clean_dataset.py`, `clean_dataset_v2.py`, tests | Stay exported from `corpus_footer_scrub.py`. | Not deleted. Spark and the demo generator import them by that module name. |

Experiment copies are not shims. They are evidence of past runs. Leave them.

## Semantic contract and byte contract

Semantic cleaning contract, required:

- same kept raw records
- same order
- same fields and field values
- same cleaned text
- same first drop reason
- `source_id` and `raw_record_id` unchanged

Compatibility artifact contract, required for the frozen files, separate from the semantic contract:

- UTF-8
- historical JSON key order
- explicit CRLF, including a final CRLF
- recorded file hashes where this repo already checks them

The internal canonical result is a Python object. It must not use the host default newline. The shadow preview file is explicit LF so the artifact is portable. Byte parity against `data/cleaned/` is explicit CRLF. On the current checkout every frozen cleaned file matches CRLF and does not match LF. A newline difference is not a cleaning mismatch, and a cleaning mismatch is not fixed by changing the newline.

## Acceptance tests

Corpus:

| Check | Value |
|---|---:|
| Raw records | 32,835 |
| Cleaned records | 30,370 |
| Drops | 2,465 |
| Blog cleaned | 3,466 |
| LinkedIn cleaned | 2,006 |
| X cleaned | 24,277 |
| Jason YouTube cleaned | 428 |
| SaaStr YouTube cleaned | 193 |
| `event_promo` | 450 |
| `linkedin_feed_chrome` | 623 |
| `too_short` | 1,203 |
| `empty` | 150 |
| `empty_transcript` | 39 |
| Shadow cleaned → existing SFT and promo filter, in memory | 42,771 rows, 0 mismatches |

Also required:

- membership, order, and parsed content, including JSON key order
- explicit CRLF byte match against each frozen cleaned file
- `identity_breaks` = 0
- SaaStr rows still present after cleaning
- `verify_frozen_parity` blocking failures = 0
- in-memory RAG selection still 21,193, with `train_26148` absent

`tests/test_cleaning_contract.py` is the before-state. It compares `clean_record` with `clean_row` on synthetic rows, including the first gate when several could apply. Those expected reasons stay frozen. Do not edit them to match a drifted implementation.

Out of the cleaner, and out of this migration:

- SFT word floors and `split_long_document`
- `filter_event_promos`
- `fit_sft_to_context`
- balance
- group and split
- relabel
- gold overlap, headline overlap, and RAG selection

## Migration commits

The L2 inversion in this task followed steps 2 and 3 below, after the contract tests already existed. Step 2 was checked against `clean_row` on all 32,835 raw records before step 3. Airflow/Spark alignment and deletion of experiment snapshots were not done. `clean_file` was not run.

The original sequence, kept here so a later Airflow change does not skip the gates:

1. **Contract tests.** This document and `tests/test_cleaning_contract.py`. No production edit. Already the before-state.

2. **Give `clean_record` the record policy.** Call `scrub_corpus_text` directly. Stop importing `clean_row`. Leave `clean_row` unchanged. The gate lines exist in both modules for this one commit. Run the contract tests. They must still agree. If they do not, revert this commit. Do not update expected values.

3. **Turn `clean_row` into the shim.** Delete the inlined gates. Return the old tuple from `clean_record`, including the flags dict the current callers read. Do not change `clean_file` paths, mode, or newline behavior. Do not run it. Add a static test that `canonical_cleaning` does not import `clean_row`. Run the same acceptance tests.

4. **Acceptance run, no behavior edit.** Targeted pytest, `verify_frozen_parity`, and the in-memory RAG count of 21,193. Confirm `data/cleaned/`, nopromo, the balanced file, and the Relabel dataset hashes did not move.

5. **Optional caller switch.** Point the provenance sidecar and the identity crosswalk at `clean_record`. Leave `prepare_complete_sft` on the `clean_row` shim until that dataset is reviewed on its own. It is not the nopromo path.

6. **Airflow and Spark, separate and optional.** The DAG may call a shared X pre-scrub helper if its MinIO schema stays `title/date/url/text/source`. Spark keeps importing `strip_footer_noise` from `spark_jobs/corpus_footer_scrub.py`. Do not route Spark through `clean_record`. Its HTML strip and 500-word chunker are not L2.

7. **Deletion.** Delete the duplicated gate body inside `clean_row` in commit 3, not later. Do not delete `scrub_corpus_text`, `clean_file`, `clean_scraped`, the DAG, Spark, or any `experiments/` snapshot. Those last items are not duplicate L2 owners of the frozen cleaned files.

## Rollback

Stop and revert if any of these appear. Do not repair them by editing the expected fixtures.

- any raw→cleaned membership mismatch
- any cleaned-text mismatch
- any row-order mismatch
- any drop-reason mismatch
- any `source_id` or `raw_record_id` change
- cleaned count other than 30,370
- nopromo count other than 42,771, or any nopromo field mismatch
- a frozen hash moves (`data/cleaned/`, nopromo, balanced, Relabel dataset, Relabel split)
- `verify_frozen_parity` reports a blocking failure
- in-memory RAG selection moves away from 21,193 or `train_26148` reappears
- `clean_file` or `clean_scraped` writes during the migration
- a circular import between `canonical_cleaning` and `clean_scraped`

## Airflow and Spark impact

| Caller | In the frozen Relabel/RAG lineage? | Still in the repo? | Import break if the scrub file stays put? | Later action |
|---|---|---|---|---|
| `dags/lemkin_pipeline_dag.py` | No. It builds MinIO input for `lemkin_content`. | Yes. Compose mounts it. | No, if `scrub_corpus_text` stays in `spark_jobs/corpus_footer_scrub.py`. | School/demo orchestration. Optional later delegation of the copied X gates. Do not change it in the L2 commits. |
| `spark_jobs/clean_and_embed.py` | No. `lemkin_content` chunks. | Yes. The DAG submits it. | No, if `strip_footer_noise` stays importable as `corpus_footer_scrub`. | Keep. Do not treat its HTML strip or 500-word chunker as the canonical cleaner. |
| `generate.py` | No. Default collection is `lemkin_content`. | Yes. | No, under the same condition. | Keep the helper import. |
| `clean_dataset.py`, `clean_dataset_v2.py` | No. Training-file scrubs. | Yes. `finetune` can call `clean_dataset`. | No. | Keep. The Relabel adapter is already trained. |

Changing scrub ownership by moving the file would break those imports. Leaving the file in place does not.

## Known deferred issues

- 8,988 fit rows were unpaired before the tokenizer replay. Exact parentage is now in `docs/data/FIT_STAGE_REPRODUCTION.md`. Do not edit `fit_sft_to_context.py` as part of the cleaning migration.
- Persisted Chroma parity is a read-only PASS: 21,193 documents, `train_26148` absent. The historical embedding command is still unrecorded.
- The historical balance write command and the embedding-build command remain unrecorded.
- `reply_or_repost` and `empty_after_scrub` are real gates and did not fire on this corpus. The contract tests cover them with synthetic rows. The corpus counts stay 2,465.
