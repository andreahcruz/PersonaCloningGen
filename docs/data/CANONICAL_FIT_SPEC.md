# Canonical fit specification

Status: **TESTED** against the frozen keep-breaks artifact. The canonical module reproduces that file byte for byte. It does not replace the artifact, and it does not change the packing rules.

This is an L3 model-dataset transformation. It is not L2 cleaning.

## Boundary

`host_finetune/canonical_fit.py` owns the transformation. `fit_rows` is pure: it takes rows and a tokenizer, and it returns records plus parent indexes. It does not write a file.

`host_finetune/fit_sft_to_context.py` is the compatibility command line. It accepts the historical arguments, calls `fit_rows`, and writes `json.dumps(..., ensure_ascii=False)` plus a newline through text mode. The historical helper names (`merge_lead_in_rows`, `_split_output`, `atomic_blocks`, `move_dangling_lead_ins`) are re-exported from the canonical module. There is one packer.

Instruction parsing stays in `sft_chunk_utils.style_prefix`, `extract_base_topic`, and `_split_sentences`. Those helpers are older than the fit stage and are not a second fit implementation.

## Configuration

`FitConfig` records the frozen values:

| Setting | Value |
|---|---|
| `fit_version` | `fit512-keepbreaks-v1` |
| `max_tokens` | 512 |
| `prompt_cap` | 120 |
| `part_label_reserve` | 8 |
| packing limit | 504 |
| chat template | `tokenize=False`, `add_generation_prompt=False` |
| encoding | `add_special_tokens=False` |
| `trust_remote_code` | true |

The historical command named `unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit` and did not pin a revision. `historical_revision` is therefore unrecorded. The compatibility replay revision is `f15c379fb32bb402fa06a7ae9aecb1febf4b79ec`. Loading that local snapshot with `local_files_only=True` produces the frozen bytes. The loader does not download a replacement.

## Algorithm

The canonical code is the frozen algorithm, not a new packer.

1. A row whose rstripped output ends in `:` is attached to the next row with the same `source_file` and `source_line`. The lead-in is not emitted. Its instruction is discarded. Chains continue while the combined output ends in `:`. 706 lead-ins are consumed. 42,065 merged rows remain. No parent is dropped.
2. The topic is shortened only when the prompt exceeds 120 tokens.
3. Answers are packed on paragraph, line, sentence, and word boundaries at 504 tokens. A colon lead-in stays with the following list. Newlines are kept. Paragraph stripping can change whitespace while the word sequence stays the same. Those rows are `REFLOWED`, not `UNCHANGED`.
4. A multi-part answer receives `(part k of N)` on the topic. `N` is the part count before the final 512 check. A labeled part over 512 is not emitted. On this corpus that check drops nothing.
5. `instruction` and `output` are written last. Other keys stay in their input order, taken from the surviving row.

## Lineage

Each output carries `parent_indices`, `fit_chunk_ordinal`, `part_ordinal`, `relationship_type`, `token_count_before`, and `token_count_after`.

`fit_chunk_ordinal` is the appearance order inside one `source_file` and `source_line`. `model_row_id` is not redefined. It still hashes `document_id`, `fit512-keepbreaks-v1`, that ordinal, and the output text.

Relationship counts on the frozen file:

| Type | Rows |
|---|---:|
| `UNCHANGED` | 37,517 |
| `REFLOWED` | 3,072 |
| `SPLIT` | 1,619 |
| `COMBINED` | 566 |
| `COMBINED_SPLIT` | 238 |

A combined row lists every consumed parent. Parent mentions are not unique fit rows.

## Frozen bytes

| Item | Value |
|---|---|
| Input | `host_finetune/data/dataset_from_cleaned_sources_nopromo.jsonl` |
| Input rows | 42,771 |
| Input SHA-256 | `55b9680657ed72a424d47ea7102214d9b2c6693a535572e02520e406d57f8241` |
| Output | `host_finetune/data/dataset_from_cleaned_sources_fit512_keepbreaks.jsonl` |
| Output rows | 43,012 |
| Encoding | UTF-8, no BOM |
| Newline | CRLF, including a final CRLF |
| Output SHA-256 | `bf00e53fd2401f2d6bea6ac39cb69829fc21761de3a4f7225fed7ecd75082fba` |

The independent canonical replay matched this hash before `fit_sft_to_context.py` delegated to it. The same replay matched the historical function output and `fit_lineage_exact.preview.jsonl` with zero parent, ordinal, and relationship mismatches.

## What this is not

A later experiment may change the context window, the lead-in rule, the reflow, or the packing target. That experiment needs its own fit version. It is not a silent edit to this module. `model_row_id` changes when the fit version, the ordinal, or the output text changes.
