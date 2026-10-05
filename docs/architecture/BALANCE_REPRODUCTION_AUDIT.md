# Balance file reproduction audit

The frozen file `host_finetune/data/dataset_fit512_keepbreaks_balanced.jsonl` is a filtered copy of `host_finetune/data/dataset_from_cleaned_sources_fit512_keepbreaks.jsonl`. The filter is the current `select_balanced_rows`. An earlier parity hash used LF and did not match. The historical bytes are CRLF, written explicitly.

## What matches

`select_balanced_rows` on the keep-breaks parent returns the same 31,328 parsed rows in the same order as the frozen file. Intersection is 31,328. Frozen-only and replay-only are 0. No provenance identity is duplicated. Paired JSON objects are equal, including key order:

`input`, `source`, `source_file`, `source_line`, `instruction`, `output`

That key order is what `fit_sft_to_context.py` writes. It copies the source row, drops `instruction` and `output`, then puts those two keys back at the end.

Counts match the EXP-008 note: blog 14,315, X 14,316, SaaStr dropped 5,492, parent 43,012 rows.

## What the hash difference is

Both files are UTF-8 without a BOM. Every record ends in CRLF, and the file ends in CRLF. There are no bare CR bytes.

| Serializer | SHA-256 | Matches frozen bytes |
|---|---|---|
| `json.dumps(row, ensure_ascii=False) + "\n"` | `60481f38fee42dbfa6c980e65cb7d6ce620e8f9a920f422701487f1922f7486a` | no |
| `json.dumps(row, ensure_ascii=False) + "\r\n"` | `6e99b414733eafdfa27a5e94dd26beff1498bbfd58af132251e24044fef4f057` | yes |
| selected parent lines joined with `\r\n` and a final CRLF | `6e99b414…` | yes |
| `ensure_ascii=True`, compact separators, and `sort_keys=True` | other hashes | no |

The LF form hashes to `60481f38…`. That is not a different row set. The parity harness now builds the artifact with `b"\r\n".join(...) + b"\r\n"` inside `explicit_crlf_jsonl_bytes`. It does not use `os.linesep` or text mode. Policy, membership, order, and that byte hash are separate passes. The missing historical command stays a separate unverified row.

Python `Path.open("w", encoding="utf-8")` on Windows translates `\n` to `\r\n` unless `newline="\n"` is set. `fit_sft_to_context.py` and `queue_followup_trains.build_balanced` both write `json.dumps(..., ensure_ascii=False) + "\n"` through that default. The keep-breaks parent has the same CRLF pattern, one CRLF per row.

## Writer

No committed command creates `dataset_fit512_keepbreaks_balanced.jsonl`. EXP-008's command list fits the nopromo file, then splits the balanced file. `build_balanced` reads `dataset_from_cleaned_sources_fit512.jsonl` and writes `dataset_fit512_no_saastr_xcap.jsonl`. `queue_followup_trains.py` matches commit `fd16806` and has not changed. Both data files are ignored by `host_finetune/.gitignore`, so git has no blob for them.

The bytes are verified: current `select_balanced_rows`, then `json.dumps(row, ensure_ascii=False)` and CRLF. The named script that applied that function to the keep-breaks parent was not found.

## Level

LEVEL 1. The exact SHA-256 is reproduced from the keep-breaks parent and `select_balanced_rows` when the output newline is CRLF.

Do not replace the frozen file. A later canonical writer has to emit the CRLF bytes explicitly. Matching the hash by opening the file in Windows text mode is not the contract, because Linux and macOS would then write LF and miss `6e99b414…`.
