# Canonical identity spec

Phase C design, corrected so external identity and raw capture identity are different. Identity version: `identity-v2`.

This document does not migrate the pipeline. Frozen JSONL, `group_id`, split assignment, and RAG metadata stay as they are. `host_finetune/canonical_ids.py` is pure and has no file IO. The preview crosswalk under `artifacts/refactor/` is a sidecar, not a production artifact.

## Three layers, plus experiment position

| ID | Question it answers |
|---|---|
| `source_id` | Which external item is this? Article, tweet, LinkedIn post, or video. |
| `raw_record_id` | Which exact ingested capture is this? |
| `document_id` | Which normalized document did that one capture become? |
| `model_row_id` | Which fitted chunk of that document is this? |
| `group_id` | Which frozen split-ownership component is this? Algorithm unchanged. |
| `relabel_row_id` | Which Relabel rendering of that fitted chunk is this? |
| `rag_row_id` | The RAG content id for that Relabel row. Eligibility is not part of it. |
| `experiment_row_id` | Which frozen file position was used? `train_26148` is `experiment_row_id(26148)`. |

`train_26148` is an experiment index. It is not a `source_id`.

Two captures of video `abc123` share `source_id` and have different `raw_record_id` values when the raw transcript, or any other canonical raw field, differs.

## document_id

v1 maps one-to-one onto `raw_record_id`. Each ingested capture becomes one normalized document. `document_id` is SHA-256 of `document-v1` plus that `raw_record_id`. It does not collapse to `source_id`, so two transcripts of one video stay two documents. Normalization version is metadata. A later rule that merges captures would be a new document prefix, not a silent change.

`model_row_id` is hashed from `document_id`, not from `source_id`, so the fitted row traces to the capture that produced it.

## Source rules

`source_id` is SHA-256 of UTF-8 `source-v2`, medium, kind, and key, joined by newlines. Scrape timestamps, `published_text`, `fetched_at_utc`, `listing_source`, and `matched_keywords` are not inputs. Cleaned text is not an input.

| Source | source_id rule | raw_record_id rule | Fallback behavior |
|---|---|---|---|
| Blog | Canonical article URL. Kind `platform_url`. 3,502 raw URLs are unique. Content and `scraped_at` do not change it. | Canonical raw payload: `content`, `date`, `tags`, `title`, `url`. `scraped_at` is excluded. | No platform-id fallback. A blog row without a URL has no `source_id`. |
| X | Tweet `id`. Kind `platform_tweet_id`. 25,997 raw ids are unique. Missing or changed `text` does not change it. | Canonical raw payload: `created_at`, `id`, `is_reply`, `is_repost_or_quote`, `text`, `url`, `username`. `scraped_at` and `scraped_from` are excluded. | A row with no tweet id is unmapped. Text is not promoted to a fake tweet id. |
| LinkedIn | URN when present. Kind `platform_urn`. Else a post URL. Kind `platform_url`. | Canonical raw payload: `author`, `author_source`, `content`, `format`, `url`, `urn`. `published_text` and `char_count` are excluded. A body change changes `raw_record_id` even when the URN keeps `source_id`. | Rows with no URN and no URL use kind `content_fallback`. `external_id` is null. `source_id` is a hash of the raw body under that kind. That hash is not a platform id. |
| Jason YouTube | Medium `youtube_jason` plus `video_id`. Kind `platform_video_id`. | Canonical raw payload includes the transcript and the other non-volatile video fields listed below. `fetched_at_utc`, `listing_source`, and `matched_keywords` are excluded. | 18 `video_id` values have more than one raw capture, and those captures have different transcripts. They share `source_id` and not `raw_record_id`. Cleaning keeps more than one capture for only 2 of those 18. |
| SaaStr YouTube | Medium `youtube_saastr` plus `video_id`. Kind `platform_video_id`. Current video ids are unique inside this file. | Same payload rule as Jason, with medium `youtube_saastr` in the hash. | Same video id on the Jason channel is a different `source_id` because the medium differs. |

URL canonicalization lowercases the scheme and host, drops the fragment, strips one trailing slash, and applies NFC. Path case is preserved.

## Raw-record hash contract

`raw_record_id` is the canonical capture identity. It is not a forensic hash of every byte in the original JSONL record. Declared volatile fields, including scrape timestamps, stay in the file and stay out of this hash.

`raw_payload_sha256` is SHA-256 of the JSON object bytes as stored on that JSONL line, with the line terminator removed. The bytes are sliced from the file. They are not rebuilt with `json.dumps`. A scrape-time change alters `raw_payload_sha256` and leaves `raw_record_id` unchanged. The file-level SHA-256 is still the guarantee for the whole container.

`raw_record_id` = SHA-256 of UTF-8 text:

```text
raw-record-v1
<medium>
<canonical JSON>
```

joined by `\n` between those three parts. The JSON object itself contains no record separator.

Canonical JSON:

- Only the schema fields for that medium. Extra keys, including `cleaned_text` and scrape timestamps, are dropped.
- A missing key and JSON null both become null. Empty string stays empty string.
- Strings are Unicode NFC. Lists keep their order. Nested objects are kept.
- Booleans and integers keep JSON types. NaN becomes null.
- Serialization is `json.dumps(..., ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)`.
- Encoding is UTF-8. Non-ASCII is not escaped.
- No platform newline is inserted. A newline inside a source string is a JSON escape, not a file newline.

YouTube payload fields: `channel`, `description`, `transcript_error`, `transcript_language`, `transcript_segments_count`, `transcript_source`, `transcript_text`, `upload_date`, `upload_date_iso`, `video_id`, `video_title`, `video_url`.

Cleaning does not recompute `source_id` or `raw_record_id`. Both are fixed from the raw record. A cleaned copy of the body is a different string and would be a different capture if it were hashed as raw content. The pipeline must copy the ids forward instead of hashing cleaned text.

## Derived lineage

```text
external source
    ↓
source_id
    ↓
one or more raw captures
    ↓
raw_record_id
    ↓
document_id          (v1: one document per capture)
    ↓
model_row_id         (document_id, fit version, chunk ordinal, output hash)
    ↓
relabel_row_id       (model_row_id, relabel version, role, instruction hash)
    ├── SFT view     (all train rows, 21,377)
    └── RAG governance
            ↓
         rag_row_id  (equals relabel_row_id)

experiment_row_id    (train_<EXP-004 row index>; compatibility only)
group_id             (frozen split_groups; not redesigned)
```

Chunk ordinal is the appearance order of fitted rows that share one cleaned `source_file` and `source_line`, assigned on the keep-breaks file before balance. Balance may drop a row. It is not an input to `model_row_id`.

`rag_row_id` does not include eligibility. Gold-overlap and `train_26148` stay governance attributes.

Frozen `group_id` remains the SHA-256 of the minimum normalized title in the `split_groups.assign_groups` component.

## Schemas

Identity fields are the ids. Audit fields record how the row was built and must not be folded into those ids unless the version table says so.

### L0 raw manifest

| Field | Role |
|---|---|
| `source_id` | identity |
| `source_identity_kind` | identity class: `platform_url`, `platform_tweet_id`, `platform_urn`, `platform_video_id`, or `content_fallback` |
| `external_id` | platform key, or null for `content_fallback` |
| `raw_record_id` | identity |
| `source` | audit |
| `source_file` | audit |
| `raw_record_index` | audit position at ingest. Not part of either id |
| `raw_sha256` | audit hash of the canonical payload |
| `schema_version` | `raw-v1` |
| `ingested_at` | audit. Not part of either id |

### L1 normalized

`source_id`, `raw_record_id`, `document_id`, `source`, `medium`, `title`, `text`, `published_at` when a real date exists, `url` or `external_id`, `schema_version` = `normalize-v1`. LinkedIn `published_text` may be copied for display and is not an id.

### L2 cleaned

`source_id`, `raw_record_id`, `document_id`, `cleaned_text`, `cleaning_version`, `cleaned_sha256`, `status`, `reason`.

### L3 model-ready

`source_id`, `raw_record_id`, `document_id`, `model_row_id`, `chunk_ordinal`, `instruction`, `output`, `sft_build_version`, `context_fit_version`, legacy `source_file`, legacy `source_line`.

### L4 split

`model_row_id`, frozen `group_id`, `split`, `split_version`, `group_version`.

### L5 Relabel

`model_row_id`, `relabel_row_id`, `sft_role`, `prior_group`, `instruction`, `output`, `group_id`, `split`, `relabel_version`.

### L5B RAG governance

`relabel_row_id`, `rag_row_id`, `eligible`, `exclusion_reasons`, `split`, `group_id`, `medium`, `rag_governance_version`. Eligibility changes do not rename `relabel_row_id` or `rag_row_id`. The frozen Chroma id remains `experiment_row_id` until a later migration.

## Legacy crosswalk

Preview path: `artifacts/refactor/canonical_identity_crosswalk.preview.jsonl`. Nothing in Airflow, `watcher.py`, training, embedding, or RAG reads `artifacts/refactor/`. `watcher.py` polls MinIO.

| Legacy | Canonical |
|---|---|
| `source_file` | `legacy_source_file` |
| `source_line` (cleaned-file line) | `legacy_source_line`, then the raw row paired with that line |
| EXP-004 row position | `legacy_exp004_row_index` |
| `train_<index>` | `legacy_experiment_row_id` |
| `group_id` | `legacy_group_id` |
| raw row from the in-memory cleaner replay | `source_id`, `raw_record_id`, `document_id` |
| keep-breaks chunk order | `chunk_ordinal`, then `model_row_id` |
| Relabel role and instruction | `relabel_row_id` |
| same Relabel id | `rag_row_id` |

`source_line` is not a raw line number. The replay calls `clean_row` only. It does not call `clean_file` or `clean_scraped`. A count or text mismatch rejects that whole source file instead of zip-aligning a shifted tail.

Mapping status:

| Status | Meaning |
|---|---|
| `EXACT_NATURAL_KEY` | Cleaner replay matched, and `source_identity_kind` is a platform id |
| `CONTENT_FALLBACK` | Cleaner replay matched, and the source id is the typed LinkedIn body fallback |
| `AMBIGUOUS` | Replay count or text did not match the frozen cleaned file, so no pairing was guessed |
| `UNMAPPED` | Replay matched the file, but this cleaned line was not in it, or the raw row had no usable identity |

`EXACT_RAW_REPLAY` is the method behind the first two statuses (`mapping_method=raw_cleaner_replay`), and the model-chunk status when the fitted output matches one keep-breaks ordinal. A fallback hash is never reported as a natural platform id.

## Versioning

| Version | Changes `source_id` | Changes `raw_record_id` | Changes `document_id` | Changes `model_row_id` | Changes `relabel_row_id` | Changes `rag_row_id` |
|---|---|---|---|---|---|---|
| `raw_schema_version` | only if the external-key rule changes | only if the payload field set changes | follows `raw_record_id` | follows `document_id` | follows `model_row_id` | follows `relabel_row_id` |
| `identity_version` | when `source-v2` changes | when `raw-record-v1` changes | when `document-v1` changes | when `model-row-v1` changes | when `relabel-row-v1` changes | follows `relabel_row_id` |
| `normalization_version` | no | no | no in v1 | no | no | no |
| `cleaning_version` | no | no | no | only if fitted output bytes change | follows that | follows that |
| `sft_build_version` | no | no | no | only through the output hash | follows that | follows that |
| `context_fit_version` | no | no | no | yes | follows that | follows that |
| `balance_version` | no | no | no | no | no | no |
| `group_version` | no | no | no | no | no | no |
| `split_version` | no | no | no | no | no | no |
| `relabel_version` | no | no | no | no | yes | follows that |
| `rag_governance_version` | no | no | no | no | no | no |
| `embedding_version` | no | no | no | no | no | no |

## Invariants

- Same canonical blog URL means the same `source_id` when only the body changes.
- Same tweet id means the same `source_id` when text is missing or changed.
- A LinkedIn URN wins over the body. The body-only id is `content_fallback` and `external_id` is null.
- Same channel family and `video_id` means the same `source_id` when the transcript changes.
- A changed raw transcript or LinkedIn body changes `raw_record_id`.
- `scraped_at`, `fetched_at_utc`, and `published_text` do not change `raw_record_id`.
- Dict insertion order and host newlines do not change `raw_record_id`.
- Cleaning is not an input to `source_id` or `raw_record_id`.
- `model_row_id` changes with fitted output, chunk ordinal, or fit version.
- Balance, split, and RAG eligibility do not rename surviving content ids.
- `experiment_row_id(26148)` is `train_26148` and is not a `source_id`.
- Every Relabel row appears once in the preview. Status counts sum to 26,545.

## Not in this phase

No JSONL rewrite. No id injection into frozen rows. No change to `select_balanced_rows`, `split_groups`, `relabel_continuations`, cleaners, or RAG selectors. No Chroma open, no embedding, no training. Cleaning consolidation is not started.
