# Whole-source SFT preparation

Status: IMPLEMENTED, TESTED and MEASURED. Structural invariants independently checked; semantic quality and authorship NOT VALIDATED. No model training or retrieval change occurred.

## What changed

`host_finetune/prepare_complete_sft.py` rebuilds candidates from complete cleaned-source documents, with raw lineage established by replaying the existing cleaner. It never re-fits truncated fragments. Targets preserve the complete trimmed cleaned body, source hashes, character spans and original paragraphs. It records all matching raw records. Input hashes are checked before and after execution.

Frozen EXP-20261001-008 ownership is joined by source file/line and checked against the prior dataset hash. All 18,677 owned full documents participate in normalized-exact, token-Jaccard >=0.8 and original-family/source-identity checks, including otherwise rejected material. Entire cross-split families are quarantined; no assignments are changed. Gold topic/reference/fact lexical matches propagate to related families. Unassigned documents cannot enter the output.

Topic instructions come from source titles, not the opening answer text. Untitled social posts and unverified-speaker transcripts are deferred. Promotional, guest-recap, missing-context and ambiguous-ending heuristics quarantine entire documents with reason codes. Long documents are deferred without truncation. A punctuation ending is a triage heuristic, not proof of semantic completion. All output is explicitly a review candidate; training defaults remain unchanged.

## Registered evidence

- EXP-20261003-001: 1,498 candidates (867/318/313 train/validation/test). Sample exposed event advertisements and guest/embedded content.
- EXP-20261003-002: 1,142 candidates (653/245/244). Structural verification passed; sample exposed link digests, missing visuals and broken words.
- EXP-20261003-003: 1,011 candidates (601/206/204). Structural verification passed for all rows and 288,434 cross-split pairs, with zero exact/Jaccard >=0.8 crossings and zero normalized exact duplicates within the candidate. All 34 audit-inventoried artifacts remained byte-identical. Tests: 129 passed, 4 skipped.

The v3 budgets retain 295 rows at 512 tokens, 735 at 1024, 1,011 at 2048 and 1,176 at 4096, counting the full serialized chat, after other eligibility checks. This is eligibility, not measured GPU feasibility. The tokenizer-only run loads no model weights.

Each run records 13 cross-split-family documents and 39 gold-overlap-family documents quarantined across the owned source pool. Reason counts overlap and cannot be summed as distinct rejected rows. The 3,199 graph links include frozen-family/source-identity edges, not just detected duplicates.

V3 is blog-only and is not a controlled replacement for earlier mixed-medium SFT. Its split proportions reflect filtering of frozen chunk-balanced ownership; they were not rebalanced. Lexical checks do not rule out semantic or partial-text leakage. Gold was checked mechanically; no test bodies were used to tune filters.

## Content review gate

The fixed sample contains eight train and eight validation examples selected by source-ID sort. This is a deterministic diagnostic sample, not a random quality estimate. Review annotations are in the v3 run's `raw/content_review.jsonl`. Multiple surviving examples still contain dangling embed introductions, third-party summaries, or split words. Individual annotations assess textual suitability only, not independent authorship or factual correctness. No clean-corpus or model-quality claim follows from the structural pass.

Next: recover author and paragraph/link/embed structure from original source pages into a new derived layer, or explicitly annotate suitable source documents. Preserve raw snapshots, inherited splits and full provenance. Do not silently join ambiguous broken words or fabricate missing charts. Establish a reviewed cohort before a longer-context pilot. Separate train-only style exemplars from factual retrieval; existing full-corpus indexes must not be used as leakage-free experimental controls.

## Reproduction

Run from repository root with the existing transformer environment. Substitute a NEW experiment directory; preparation refuses to overwrite any directory. The exact executed commands, tokenizer hashes and package snapshot are stored in each manifest/config directory.

```powershell
& host_finetune/.venv/Scripts/python.exe -X utf8 -m host_finetune.prepare_complete_sft `
  --prior-dataset host_finetune/data/dataset_fit512_keepbreaks_balanced.jsonl `
  --assignments experiments/EXP-20261001-008-repaired-balanced-split/raw/split_assignments.jsonl `
  --tokenizer C:/Users/okevi/.cache/huggingface/hub/models--unsloth--Meta-Llama-3.1-8B-Instruct-bnb-4bit/snapshots/f15c379fb32bb402fa06a7ae9aecb1febf4b79ec `
  --out experiments/EXP-YYYYMMDD-NNN-complete-source-sft
& host_finetune/.venv/Scripts/python.exe -X utf8 -m host_finetune.verify_complete_sft experiments/EXP-YYYYMMDD-NNN-complete-source-sft
& .venv-test/Scripts/python.exe -m pytest -q
```

The independent verifier compares direct chat-template tokenization with recorded lengths, brute-forces cross-split pairs, checks full source spans/hashes and inherited assignments, and verifies partition exports and test exclusion from the review sample. It supports list and mapping tokenizer return types. It does not independently verify authorship, raw cleaner correctness, factual claims or semantic leakage. Rubric traceability: D1-1 preprocessing, D1-2 splits, D1-3 analytics, WB-4.4 evaluation and WB-5.3 solution inputs.
