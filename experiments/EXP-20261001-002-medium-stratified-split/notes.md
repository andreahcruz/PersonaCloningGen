# EXP-20261001-002 — Medium-stratified split

Question: after long essays are split on section headings when they do not fit, can validation and test each keep about 10 percent of every medium instead of filling up with leftover X posts?

Held fixed: raw files in `data/` and the cleaned copies in `data/cleaned/`. `host_finetune/data/dataset.jsonl` was not overwritten. No fine-tune was started. EXP-20261001-001 was not edited. Its manifest hashed the previous sentence-only fit file, which this rebuild replaced.

## Method

`split_long_document` leaves a document whole when it fits in 1,600 characters. A longer document is cut on numbered or markdown headings when at least two of those headings exist. A section that is still too long is split on sentence boundaries. Talks have no such headings, so they stay on the sentence splitter. The Llama 3.1 chat-template fit then packs any remaining overflow into 512 tokens.

`split_groups.py` still keeps one source line, and near-duplicate text, inside one group. Each `source` value then receives its own 80/10/10 row quota.

`finetune.py` defaults to `host_finetune/data/dataset_from_cleaned_sources_fit512.jsonl` and this assignment. `DATASET_LOCAL` remains the MinIO download path.

## Result

SFT build: 42,942 rows, including 5,494 chunks whose instruction names a section heading. Promo filter: 42,942 → 42,771. Token fit: 42,771 → 42,877, with 0 rows dropped. Audit: 42,877 examples, max 512, over-limit 0.

Split: 34,302 train / 4,287 validation / 4,288 test. No group crosses a split.

Validation medium counts, and their share of that medium:

| Medium | Validation | Test | Share of that medium in validation |
|---|---:|---:|---:|
| blog | 1,466 | 1,451 | 10.1% |
| LinkedIn | 125 | 126 | 12.5% |
| X | 1,977 | 1,992 | 9.8% |
| Jason talks | 178 | 178 | 10.5% |
| SaaStr talks | 541 | 541 | 9.9% |

Validation word-count mean is 133, against 121 in train. The previous unstratified validation mean was 35.
