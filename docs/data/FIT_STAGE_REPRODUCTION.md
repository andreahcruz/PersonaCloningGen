# Fit-stage reproduction

Status: **MEASURED** for this worktree. The keep-breaks file was replayed in memory from the nopromo SFT file with the local Llama 3.1 tokenizer snapshot. The replay matches the frozen file byte for byte. The same bytes are now produced by `host_finetune/canonical_fit.py`. `fit_sft_to_context.py` is the compatibility command line. The contract is `docs/data/CANONICAL_FIT_SPEC.md`.

This stage is an L3 model-dataset transformation. It changes chunk boundaries and instruction text under a tokenizer budget. It is not L2 cleaning.

## Frozen input

| Item | Value |
|---|---|
| Path | `host_finetune/data/dataset_from_cleaned_sources_nopromo.jsonl` |
| Rows | 42,771 |
| SHA-256 | `55b9680657ed72a424d47ea7102214d9b2c6693a535572e02520e406d57f8241` |
| Serialization | UTF-8 JSONL, one object per line |

The historical invocation is recorded in `experiments/EXP-20261001-008-repaired-balanced-split/config/commands.txt`:

```text
host_finetune\.venv\Scripts\python.exe -u -m host_finetune.fit_sft_to_context --dataset host_finetune/data/dataset_from_cleaned_sources_nopromo.jsonl --tokenizer unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit --out host_finetune/data/dataset_from_cleaned_sources_fit512_keepbreaks.jsonl
```

An older command in EXP-20261001-002 wrote `dataset_from_cleaned_sources_fit512.jsonl`. That file is not the keep-breaks parent.

The command names the tokenizer repository and does not pin a revision. The local Hugging Face cache contains one snapshot of that repository, and `refs/main` points at the same snapshot. Replay with that snapshot is byte-identical to the frozen output, which identifies the snapshot that produced the file.

## Tokenizer contract

| Item | Value |
|---|---|
| Identifier | `unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit` |
| Revision | `f15c379fb32bb402fa06a7ae9aecb1febf4b79ec` |
| Local source | the snapshot directory under the Hugging Face hub cache |
| Load | `AutoTokenizer.from_pretrained`, `local_files_only=True`, `trust_remote_code=True` |
| Device | CPU. Model weights, Unsloth, and CUDA were not loaded |
| Chat template | user instruction plus assistant output, `tokenize=False`, `add_generation_prompt=False` |
| Special tokens | the template string is encoded with `add_special_tokens=False` |
| Max sequence length | 512 |
| Prompt token cap | 120 |
| Packing limit | 504, leaving room for a `(part k of N)` label |

## Transformation

The replay calls the existing functions. It does not reimplement packing.

1. Read every non-blank nopromo JSON object, in order.
2. `merge_lead_in_rows` attaches a row whose rstripped output ends in `:` onto the next row of the same `source_file` and `source_line`. The lead-in row is not emitted. Its instruction is discarded. The surviving row keeps its own instruction and source fields. The output is `lead.rstrip() + "\n\n" + next.lstrip()`. Chains continue when the combined output still ends in `:`.
3. For each surviving row, `style_prefix` reads the medium from the instruction, `extract_base_topic` strips a trailing `(part k of N)`, and `_short_topic` caps the topic at 120 tokens.
4. `_split_output` packs the answer on paragraph, line, sentence, and word boundaries at a 504-token limit, then drops an empty piece or a piece that still exceeds 512 tokens before the part label.
5. A multi-part answer gets `(part k of N)` on the topic. `N` is the part count before the final 512 check. A part whose labeled form exceeds 512 is not written.
6. Each written object keeps the surviving row's keys, moves `instruction` and `output` to the end, and is `json.dumps(..., ensure_ascii=False)`.

On this corpus the final 512 check dropped nothing, and no merged row produced an empty part list.

## Count arithmetic

```text
42,771 input parents
   - 706 lead-in rows consumed
= 42,065 merged rows
   - 0 merged rows with zero outputs
   + 947 additional parts
= 43,012 fit rows
```

Net change: `947 - 706 = 241`.

Merged-row cardinality:

| Outputs from the merged row | Merged rows | Fit rows |
|---:|---:|---:|
| 1 | 41,155 | 41,155 |
| 2 | 878 | 1,756 |
| 3 | 27 | 81 |
| 4 | 5 | 20 |

Parent cardinality counts a lead-in parent on every fit row that contains its text. A combined row therefore increments more than one parent. The output column is parent mentions, not unique fit rows.

| Outputs per parent | Parents | Parent mentions |
|---:|---:|---:|
| 0 | 0 | 0 |
| 1 | 41,733 | 41,733 |
| 2 | 956 | 1,912 |
| 3 | 71 | 213 |
| 4 | 11 | 44 |

Parents with zero outputs: 0. Expanded parents (two or more fit rows): 1,038. Additional fit rows from multi-part merged rows: 947.

The earlier sidecar figures of 8,988 unpaired rows, 751 extra rows, and 510 contracted rows came from comparing nopromo chunk text sequences inside a `source_file` and `source_line`. They are not the tokenizer result.

## Relationship types

Observed from the replay, one label per fit row:

| Type | Rows | Meaning |
|---|---:|---|
| `UNCHANGED` | 37,517 | One parent. One part. Output equals the rstripped parent output. |
| `REFLOWED` | 3,072 | One parent. One part. Output words match the parent, and the whitespace does not. |
| `SPLIT` | 1,619 | One parent. Two or more parts. |
| `COMBINED` | 566 | Two or more parents from a lead-in merge. One part. |
| `COMBINED_SPLIT` | 238 | Two or more parents, then two or more parts. |

No fit row has an empty parent list. No fit row combines parents from different captures. `parent_sft_row_index` is set only when there is one parent. Combined rows store `parent_sft_row_indices` and leave the singular index null.

`model_row_id` is unchanged. It still hashes `document_id`, `fit512-keepbreaks-v1`, the appearance ordinal of that fit row inside its `source_file` and `source_line`, and the output text. Recomputing those ids from the replay matched the existing fit manifest for all 43,012 rows.

## Frozen output and byte contract

| Item | Value |
|---|---|
| Path | `host_finetune/data/dataset_from_cleaned_sources_fit512_keepbreaks.jsonl` |
| Rows | 43,012 |
| Encoding | UTF-8, no BOM |
| Newline | CRLF, including a final CRLF |
| Serializer | `json.dumps(record, ensure_ascii=False)` |
| Key order | original keys, with `instruction` and `output` last |
| SHA-256 | `bf00e53fd2401f2d6bea6ac39cb69829fc21761de3a4f7225fed7ecd75082fba` |
| Size | 42,331,180 bytes |

The hash matches `docs/data/artifact_hashes.csv`. LF serialization does not match the file. Semantic equality and byte equality were checked separately, and both passed.

## Lineage artifact

`artifacts/refactor/provenance/fit_lineage_exact.preview.jsonl` has one row per fit row. It stores parent SFT indexes, capture ids, relationship, chunk ordinal, output hashes, and token counts. It does not store answer text.

`fit_manifest.preview.jsonl` now marks every row `chunk_alignment=exact_replay`. `coverage.json` records the SFT → FIT transition as 42,771 → 43,012 with ambiguous 0. A later provenance rebuild overlays this sidecar and will not restore the old text-sequence pairing.

## Remaining uncertainty

The historical command did not record `pip freeze`, the tokenizer revision, or the machine. The byte-identical replay against the only local snapshot closes that gap for this cache. A different machine would need this snapshot, not a newer hub revision.

Balance invocation, the embedding-build command, and dirty historical worktrees are separate provenance gaps. They are not fit-stage ambiguity.
