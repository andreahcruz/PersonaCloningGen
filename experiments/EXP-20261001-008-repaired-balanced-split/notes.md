# EXP-20261001-008 — Repaired balanced split

Question: can the 512-token fit keep paragraph breaks and keep a colon lead-in with the list that follows it, then still produce a per-medium 80/10/10 split?

Held fixed: raw files in `data/`, `data/openai_ft/lemkin_gold_eval.jsonl`, and the previous fit file `host_finetune/data/dataset_from_cleaned_sources_fit512.jsonl`. This run writes a new fit file. It does not replace the file the earlier adapters trained on.

## What changed

`fit_sft_to_context.py` no longer cuts an answer every 120 characters and joins those slices with spaces. It packs paragraphs, keeps newlines, keeps a heading or a colon lead-in with the following list, and prepends a colon lead-in onto the next chunk of the same `source_file` and `source_line`.

Input was `host_finetune/data/dataset_from_cleaned_sources_nopromo.jsonl` (42,771 rows). After attaching same-document lead-ins, 42,065 rows were fit into 43,012. Of those, 30,819 still contain a newline. Outputs that end in `:` fell from 1,445 in the old fit file to 782, and 669 of those are short closings such as a trailing "here:" rather than a list that was cut away.

The balanced subset drops SaaStr talks and caps X at the blog row count: 31,328 rows (blog 14,315, X kept 14,316, SaaStr dropped 5,492). Split is 25,062 / 3,133 / 3,133. No near-duplicate or exact group crosses a split. Dataset sha256 `6e99b414733eafdfa27a5e94dd26beff1498bbfd58af132251e24044fef4f057`.
