# Canonical cleaning spec

Shadow implementation. The production cleaner is unchanged. Frozen files under `data/cleaned/` were not rewritten.

## Boundary

| Layer | What it is | What it is not |
|---|---|---|
| L0/L1 normalization | `source_id`, `raw_record_id`, and field names. Identity is computed from the raw record. | Text scrub |
| L2 canonical cleaning | The frozen `clean_row` gates plus `scrub_corpus_text`. One raw capture becomes one kept cleaned record or one drop. | SFT word floors, chunking, context fit, balance, split, relabel, RAG eligibility |
| L3 model-dataset filtering | Word floors in `build_sft_from_cleaned_sources`, the later promo filter in `filter_event_promos`, fit, balance, and RAG exclusions. | A second copy of the L2 scrub |

Event promo exists in both layers, and they are different rules. L2 drops a short named-event pitch, or a sponsor title, inside `scrub_corpus_text`. L3 `filter_event_promos` drops SFT chunks whose instruction title is a direct event announcement, and it strips a trailing CTA. The shadow cleaner reproduces L2 only. The nopromo check then runs the existing L3 functions in memory.

SaaStr rows are not removed as a class during cleaning. Balance drops that medium later.

## Frozen cleaning contract

`host_finetune.clean_scraped.clean_row` is the frozen row transform. It calls `spark_jobs.corpus_footer_scrub.scrub_corpus_text`. The first gate that fires is the only drop reason.

Order:

1. X reply or repost/quote: `reply_or_repost`.
2. X empty body: `empty`. X body under 25 characters: `too_short`.
3. Empty body for other media: `empty`. YouTube renames that to `empty_transcript`.
4. LinkedIn text matching `Feed post`: `linkedin_feed_chrome`.
5. LinkedIn `hashtag #` token strip. YouTube HTML unescape. Footer strip. Broken-URL repair.
6. Event promo on the scrubbed text: `event_promo`.
7. Empty after scrub: `empty_after_scrub`.

A kept row is the original JSON object with the text field replaced. Key order stays the order of the raw object.

Stable codes: `REPLY_OR_REPOST`, `TOO_SHORT`, `EMPTY_TEXT`, `EMPTY_TRANSCRIPT`, `LINKEDIN_FEED_CHROME`, `EVENT_PROMO`, `EMPTY_AFTER_SCRUB`.

On the current raw corpus the drops are event promo 450, LinkedIn feed chrome 623, too short 1,203, empty 150, empty transcript 39. Total 2,465. Reply/repost and empty-after-scrub do not fire on this corpus. The rules remain.

## Source behavior

| Source | Text field | Extra scrub | Notes |
|---|---|---|---|
| Blog | `content`, title `title` | Footer and URL repair. Event promo uses the title. | `Dear SaaStr` titles are kept by the promo rule. |
| X | `text` | Reply/repost gate, then 25-character minimum, then the shared scrub. | Tweet id still identifies a dropped row. |
| LinkedIn | `content` | Feed-chrome drop and hashtag-token strip. | No URN or URL is invented. Fallback identity stays `content_fallback`. |
| Jason YouTube | `transcript_text`, title `video_title` | HTML unescape. Empty transcript is `empty_transcript`. | Two captures of one `video_id` keep one `source_id` and two `raw_record_id` values. |
| SaaStr YouTube | Same as Jason | Same contract. | Surviving rows stay in the cleaned file. Balance drops the medium later. |

`source_id` and `raw_record_id` are copied from the raw record. Cleaning may change `cleaned_text`, `cleaned_text_sha256`, status, and drop reason. It does not rename the capture.

## Duplicates that did not produce the frozen cleaned files

| Responsibility | Current code | Frozen cleaned path? | Duplicate? |
|---|---|---|---|
| Shared text scrub | `spark_jobs/corpus_footer_scrub.py` | Yes. Called by `clean_row`. | Also imported by the Airflow DAG and available to Spark. |
| Row gates and cleaned JSONL shape | `host_finetune/clean_scraped.py` `clean_row` / `clean_file` | Yes. `clean_file` wrote `data/cleaned/`. | Experiment config copies exist. They are not the live module. |
| DAG normalizers | `dags/lemkin_pipeline_dag.py` | No. Different output schema. | A second caller of the same scrubber. |
| Spark HTML strip and 500-word chunks | `spark_jobs/clean_and_embed.py` | No. That path feeds `lemkin_content`. | Separate chunker. |
| SFT word floor and chunking | `build_sft_from_cleaned_sources.py`, `sft_chunk_utils.py` | Downstream of cleaned files. | Not L2. |
| SFT promo filter | `filter_event_promos.py` | Downstream. It produces nopromo from the SFT file. | Not the L2 event-promo rule. |
| Complete-SFT preparation | `prepare_complete_sft.py` | No. A different dataset. | Reuses promo helpers. |

## Shadow implementation

`host_finetune/canonical_cleaning.py` owns the record policy. `clean_record` calls `spark_jobs.corpus_footer_scrub.scrub_corpus_text` and does not import `clean_row`. `clean_file` is unchanged and was not run.

`host_finetune/build_shadow_cleaned.py` reads the five raw files and writes:

- `artifacts/refactor/cleaning/canonical_cleaned.preview.jsonl` — kept records, five sources in frozen order, explicit LF
- `artifacts/refactor/cleaning/canonical_cleaned_manifest.preview.jsonl` — one row per raw record, text stored as SHA-256
- `artifacts/refactor/cleaning/parity_report.json`

The preview file uses LF so the artifact does not depend on the host text mode. Byte parity is measured separately with explicit LF and explicit CRLF against each frozen cleaned file. On this checkout every frozen cleaned file matches explicit CRLF and does not match explicit LF. That is the same serialization contract as the balanced SFT file. A newline difference is not a cleaning difference.

## Parity

Membership, order, and content match `data/cleaned/` for all five sources: same parsed objects and the same JSON key order. Kept count 30,370. Dropped count 2,465.

Those same shadow records, passed through the existing SFT chunker and promo filter in memory, reproduce the frozen nopromo file: 42,771 rows, including instruction, output, `source_file`, and `source_line`.

Fit is out of scope. The 8,988 unpaired fit chunks stay a later-stage issue. The tokenizer was not loaded.

## Consolidation status

Record-level L2 policy now lives in `canonical_cleaning.clean_record`. It calls `scrub_corpus_text` directly. `clean_scraped.clean_row` is a compatibility shim that returns the historical `(row, reason, flags)` tuple. The scrubber file was not moved. Airflow and Spark were not switched. Persisted Chroma was not audited.

## Future ownership

Not implemented. The migration plan is the sequence. This table is the target, not the current code.

| Responsibility | Current owner | Future owner | Move now? | Why |
|---|---|---|---|---|
| Raw identity | `canonical_ids.py` | `canonical_ids.py` | No | Already the identity contract. |
| Shared text scrub | `corpus_footer_scrub.py` | One scrub module called by the canonical cleaner | Not yet | The function is already the single regex definition. Moving the file would touch Spark and Airflow. |
| Source gates and drop codes | `canonical_cleaning.clean_record` | `canonical_cleaning.clean_record` | Done for L2 | `clean_row` now delegates. The scrubber file stayed put. |
| Drop classification | `canonical_cleaning.DROP_CODES` plus the historical strings | `canonical_cleaning.DROP_CODES` | Done for the map | The shim still returns the historical reason string. |
| SFT word floor | `build_sft_from_cleaned_sources.py` | That module, or a later model-policy module | No | It is L3. It is not cleaning. |
| SFT promo policy | `filter_event_promos.py` | That module | No | Different rule from L2 event promo. |
| Chunking | `sft_chunk_utils.py` | `sft_chunk_utils.py` | No | Model-dataset construction. |
| Context fit | `fit_sft_to_context.py` | `fit_sft_to_context.py` | No | The unpaired chunk map is a fit issue. |

The L2 record owner is `canonical_cleaning.clean_record`. `clean_row` is the compatibility shim. `clean_file` remains the writer and was not run against `data/cleaned/`. Airflow and Spark still import the scrub module themselves.
