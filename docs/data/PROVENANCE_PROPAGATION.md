# Provenance propagation

Sidecar manifests under `artifacts/refactor/provenance/`. The frozen stage files were not rewritten. These previews are not production artifacts.

## Identity definitions

| ID | Meaning |
|---|---|
| `source_id` | External item: blog URL, tweet id, LinkedIn URN or URL, or channel family plus `video_id`. |
| `raw_record_id` | Canonical capture identity. Hash of the content-bearing fields only. Not a hash of every byte in the raw JSONL record. |
| `raw_payload_sha256` | SHA-256 of the JSON object bytes sliced from that JSONL line, excluding the line terminator. Includes scrape timestamps and original key order. Not rebuilt with `json.dumps`. |
| `document_id` | v1 normalized document. One-to-one with `raw_record_id`. |
| `model_row_id` | One fitted chunk: `document_id`, fit version, chunk ordinal, fitted output hash. |
| `group_id` | Frozen split-ownership id from EXP-008. Not redesigned. |
| `relabel_row_id` | Relabel rendering of that model row. |
| `rag_row_id` | Equal to `relabel_row_id`. Eligibility is a separate field. |
| `experiment_row_id` | `train_<EXP-004 index>`. `train_26148` is index 26148. |

File SHA-256 values in `coverage.json` are the container byte guarantee. A per-record payload hash does not replace them.

## Sidecar architecture

```text
raw_manifest.preview.jsonl
    one raw JSONL record
cleaned_manifest.preview.jsonl
    same raw record, kept or dropped
sft_manifest.preview.jsonl
    one nopromo chunk
fit_manifest.preview.jsonl
    one keep-breaks row
balanced_manifest.preview.jsonl
    every fit row, selected true or false
split_manifest.preview.jsonl
    every selected row, with frozen group_id and split
relabel_manifest.preview.jsonl
    every EXP-004 row
rag_manifest.preview.jsonl
    every train Relabel row, with eligibility
```

Lookup: `python -m host_finetune.trace_provenance --index 26148`.

## Lineage

```text
RAW
  source_id + raw_record_id + raw_payload_sha256
    ↓ clean_record, in memory
CLEANED
  same source_id and raw_record_id
    ↓ chunk + promo filter, in memory
SFT nopromo
  document_id, sft chunk ordinal
    ↓ frozen fit file, joined by source_file + source_line
FIT512
  model_row_id
    ↓ select_balanced_rows
BALANCED
  same model_row_id on surviving rows
    ↓ EXP-008 assignments
SPLIT
  group_id + split
    ↓ EXP-004 dispositions
RELABEL
  relabel_row_id
    ├── SFT view: all train rows
    └── RAG governance view
          rag_row_id, eligible, exclusion reasons
```

`experiment_row_id` stays beside this chain.

## Stage counts

| Transition | Input | Output | Dropped | Expanded | Ambiguous | Unmapped |
|---|---:|---:|---:|---:|---:|---:|
| RAW → CLEANED | 32,835 | 30,370 | 2,465 | 0 | 0 | 0 |
| CLEANED → SFT | 30,370 | 42,771 | 5,642 | 3,312 | 0 | 0 |
| SFT → FIT | 42,771 | 43,012 | 0 | 751 | 8,988 | 0 |
| FIT → BALANCE | 43,012 | 31,328 | 11,684 | 0 | 0 | 0 |
| BALANCE → SPLIT | 31,328 | 31,328 | 0 | 0 | 0 | 0 |
| SPLIT → RELABEL | 31,328 | 26,545 | 4,783 | 0 | 0 | 0 |
| RELABEL train → RAG | 21,377 | 21,193 | 184 | 0 | 0 | 0 |

RAW → CLEANED drops are explicit cleaner reasons: event promo 450, LinkedIn feed chrome 623, too short 1,203, empty 150, empty transcript 39. No raw capture produced two cleaned lines. No cleaned line lacked a raw capture. The in-memory replay matched the frozen cleaned files.

CLEANED → SFT drops 5,518 cleaned lines under the word floor and 124 whose chunks were all removed. Promo filter drops 170 chunks. One chunk is dropped after a trailing CTA strip. 3,312 cleaned lines become more than one nopromo row. The replay matches `dataset_from_cleaned_sources_nopromo.jsonl` in order, including output text.

SFT → FIT does not drop a whole document: every nopromo `source_file` and `source_line` is present in the keep-breaks file, and every fit row has that locator. 751 extra fit rows come from documents that grew. 510 SFT rows sit in documents that shrank. The net change is 43,012 − 42,771 = 241. 8,988 fit rows belong to documents whose chunk texts are not the same sequence as the nopromo chunks. Those rows keep the document id and receive no single parent SFT index. The tokenizer was not loaded, so the fit split was not replayed.

FIT → BALANCE drops 5,492 SaaStr rows and 6,192 X rows over the cap. 31,328 rows stay. Their `model_row_id` values are copied from the fit sidecar. Renamed ids: 0.

BALANCE → SPLIT copies EXP-008 `group_id` and `split` onto those 31,328 rows. Groups that cross train, validation, and test: 0.

SPLIT → RELABEL keeps 26,545 rows and drops 4,783. Each kept row has one `prior_row_index`. `model_row_id` is the parent fit id. `group_id` is unchanged.

RELABEL train → RAG starts from 21,377 train rows, excludes 183 gold-overlap-family rows and `train_26148`, and leaves 21,193 eligible. `relabel_row_id` is not rewritten. Chroma was not opened.

## Legacy compatibility

| Legacy | Sidecar field |
|---|---|
| raw file and record order | `source_file`, `raw_record_index` |
| cleaned line | `cleaned_line`, then `legacy_source_line` |
| nopromo position | `sft_row_index` |
| fit position | `fit_row_index`, `chunk_ordinal` |
| balanced position | `balanced_row_index`, `legacy_prior_row_index` |
| EXP-008 `group_id` and `split` | copied through |
| EXP-004 index | `legacy_exp004_row_index` |
| `train_<index>` | `legacy_experiment_row_id` |

## Unresolved anomalies

- 8,988 fit rows do not have a one-chunk parent inside their nopromo document. Document lineage is intact. Chunk-to-chunk lineage is unpaired. That is the fit stage, not the cleaner.
- 954 LinkedIn Relabel rows still use `content_fallback` because the raw post has no URN or URL.
- Eighteen Jason YouTube videos have more than one raw capture. None of those pairs both survive as separate captures in the Relabel file.
- The historical balance write command and the embedding-build command remain unrecorded.
- Persisted `lemkin_train_only` matches the in-memory 21,193-row selection. The historical command that embedded it is still unrecorded.

A shadow cleaner now lives in `host_finetune/canonical_cleaning.py`. `clean_record` owns the L2 record policy and calls `scrub_corpus_text`. `clean_row` is a compatibility shim. Provenance replay calls `clean_record` and still stores the historical reason string. The production writer was not run, and `data/cleaned/` was not replaced. Airflow and Spark were not switched. The contract is `docs/data/CANONICAL_CLEANING_SPEC.md`. The migration note is `docs/data/CLEANING_MIGRATION_PLAN.md`. The 8,988 unpaired fit rows remain a fit-stage issue.
