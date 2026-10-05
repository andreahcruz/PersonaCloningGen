# EXP-20261005-003 — Relabel + frozen factual RAG, full gold set

Question: can the final Relabel plus frozen factual RAG path produce all 120 gold drafts?

Held fixed: `lemkin_train_only` (21,193 docs), nomic-embed-text, cosine, k=4, topic-only query, Relabel at trained scale 1.0, and the same decoding budgets (blog 768, LinkedIn 320, X 160, talk 768). Unsupported numbers are diagnostic only. A source-copy failure may retry once. Scoring was not run.

Initial pass, 2026-10-05T07:30:11Z to 2026-10-05T08:04:38Z, exit 2: 104 generated drafts and four placeholder rows. gold_009, gold_015, gold_025, and gold_029 failed atomic-evidence parsing. The byte copy of that file is `backup/relabel.jsonl.before-recovery` (sha256 `aec13324527b6e99c403b2592fa62a48518a466d3a06de96f69abe987896d90f`, 108 file rows).

Recovery did not rerun the full command. gold_009 and gold_015 were generated into `experiments/EXP-20261005-003-relabel-rag-final-recovery`. gold_025 and gold_029 were generated into `experiments/EXP-20261005-003-relabel-rag-final-recovery-rest` after the parser accepted their closed claims. Seeds use each topic's original gold-file index. None of the 104 completed drafts were regenerated.

Merged result: `raw/relabel.jsonl` has 120 drafts, 30 topics, and 30 of each medium (sha256 `266507aca9effe008596ca64dc199d2e0de32bafbd7a9765d738adb1d34ec550`). The four placeholder rows are not in that file. Sixty drafts have an unsupported-quantity diagnostic. Two drafts are source-copy flagged. Stop reasons: 103 eos, 17 length. The merge check is `metrics/merge_audit.json`.

## Evaluation

Scored on 2026-10-05T09:16:57Z with `unified_eval.v1` and blinded `qwen2.5:14b` (temperature 0, GPU). The generation file was not edited. The retrieval-off Relabel comparison reuses the 120 rows already scored in `EXP-20261004-008-writing-evidence`. Cells and seeds match.

Eligible drafts: 72 of 120 here, 77 of 120 for retrieval-off. Completion failures are 36 versus 41. RAG grounding applies only here: 495 of 498 verifiable claims are supported, and 14 drafts fail that gate. Eligible blog voice distance is 0.7765 versus 0.7119 (lower is closer to Lemkin). Eligible writing quality is 3.0602 versus 3.1537. Persona Utility, on the blog drafts that pass every gate, is 0.6878 versus 0.705. The selector ranks retrieval-off first because blog voice is the first ranking key. Unsupported numbers and source-copy flags stay diagnostics. Per-draft scores are in `metrics/rows.jsonl`. The matched comparison is `metrics/comparison.json`.

## Handoff

Before this experiment was committed, config files in this directory and in the two recovery directories stored a machine-local Chroma path and a machine-local scorer command. Those strings were rewritten to repo-relative paths. `raw/relabel.jsonl` was not edited. Its sha256 remains `266507aca9effe008596ca64dc199d2e0de32bafbd7a9765d738adb1d34ec550`.
