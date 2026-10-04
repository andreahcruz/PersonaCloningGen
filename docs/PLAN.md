# Plan and next task

Current priority (2026-10-04): EXP-20261003-005 has finished and saved `lemkin_lora_relabel`. The separate CPU build of `lemkin_train_only` finished with 21,194 documents and did not replace `lemkin_content`. EXP-20261004-003 is the retrieval-off repaired-versus-relabel comparison and must finish before any retrieval-on generation. The CPU retrieval harness can exist beside that run: `--retrieval` and `--fixed-style` default off, so the EXP-003 prompts stay the one-line instructions. Source recovery remains useful and does not block that comparison. EXP-20261004-002 records remaining data risks. No second train has been launched. The finished index is not a retrieval-quality result.

Updated 2026-10-04. EXP-20261003-004 relabeled the existing 512-token rows (MEASURED). EXP-20261003-005 finished that file (train loss 1.797, best eval loss 1.3316 at step 2,674). The CPU index of its train split into `lemkin_train_only` finished with 21,194 documents at 03:55. Whole-source preparation remains IMPLEMENTED, TESTED and MEASURED, and its content-quality gate has not passed.

## Exact next task

**Select on early-stop and mid-sentence rates.** EXP-20261004-003 finished retrieval off, and EXP-20261004-004 through 006 added retrieval-on and seeds 43 and 44. [RELABEL_FT_NEXT_STEPS.md](evaluation/RELABEL_FT_NEXT_STEPS.md) records the selection rule: those two rates decide, and human voice preference is not collected. Log BLEU and ROUGE and do not decide on them. Do not overwrite `lemkin_lora_repaired`. Retrieval on raised the early-stop rate, so it is not the next step for fewer early stops.

**In parallel, do not promote the whole-source candidate until attribution and scrape structure are resolved.** Preparation code and three recorded candidates now exist; see [results and reproduction](data/COMPLETE_SOURCE_SFT.md). The v3 fixed sample still contains missing embeds, guest summaries and scrape artifacts. Recover author/body/block metadata from original pages in a separate derived layer, or explicitly annotate complete author-written documents. Do not endlessly expand regexes and call the remaining corpus clean. Use train/validation for policy development and preserve held-out ownership.

Preparation acceptance criteria, now implemented: reconstruct complete candidate posts from source documents, not already-chunked fit512 data; inspect cleaned-source lineage against raw files and hashes; preserve existing source-family train/validation/test ownership through provenance joins. Report missing, excluded and conflicting ownership. Quarantine every member of newly detected cross-split families; do not reshuffle or choose an arbitrary winner. Any later reassignment still requires explicit approval.

Create only a new versioned derived dataset. Prefer complete, attributable material; quarantine uncertain speech and promotional/template content with reason codes. DEC-004 records the implemented conservative candidate policy; source snapshots record exact thresholds. Preserve authentic paragraphing, lists and punctuation. A `.?!` ending is only a heuristic; lists and short posts require contextual completeness checks.

Keep full posts within the verified chat budget, explicitly label self-contained sections, or defer long material. Do not label middle fragments as full posts. Detect title/instruction-only targets and topics copied from the target. Do not silently synthesize completions or neutral summaries. Record source ID/hash/span, medium, cleaning version, and disposition.

## Acceptance gates

1. Raw files, old SFT, adapters and frozen assignments remain byte-identical. Every emitted row maps to an existing family/split; conflicts are reviewable.
2. Measure full serialized chat length with a pinned tokenizer/revision, without hidden truncation. Compare eligibility at 512/1024/2048 and optionally 4096. A 2048 pilot is a candidate, not a guaranteed GPU fit or universal minimum.
3. Tests cover oversized sentences, lists, Unicode, repeated titles, ambiguous attribution, unassigned sources, duplicate families and overflow. Review a fixed train/validation sample for semantic completeness. Do not tune on test examples.
4. Run exact and near-duplicate checks, including cross-platform families and gold references. Apply split ownership to profile/exemplar/index construction. Existing Jaccard checks are one diagnostic, not a semantic guarantee.
5. Register the preparation run with the complete `experiments/README.md` manifest, hashes, environment, commands, dispositions, counts, examples and warnings. Data-preparation success is not model-quality success.

Rubric: D1-1 preprocessing, D1-2 splits, D1-3 analytics, WB-4.4 evaluation, WB-5.3 solution inputs.

## Subsequent work

1. Calibrate independent voice/completion/grounding/copying metrics; fix split-aware reference loading and classifier evaluation. Capture token counts and stop reasons before attributing all cutoffs to learned EOS.
2. Register a short longer-context feasibility pilot, initially retaining rank 16/alpha 32 and assistant-only loss. Keep LR/dropout/epochs/optimizer/packing changes explicit. No rank sweep before a sound data/evaluation baseline.
3. Compare adapter/base under identical tasks, prompts and backend settings; then fixed exemplars, train-only style retrieval, and factual retrieval as separate additions.
4. Add an ontology-aware branch with typed schema, source support, and structured-question tests against ordinary vector retrieval. LightRAG is an option, not the definition of ontology-RAG.
5. Build the demo from recorded evidence and limitations. Saved rubric lists Oct 6 noon; avoid depending on an untested graph or long retraining run.

The user approved the preparation task and, separately, the relabel-and-retrain plan in DEC-006. EXP-008 family ownership was kept for rows that survived the filter. Rows were dropped, not moved between splits. The whole-source candidate is still not training-ready. Live index replacement and the architecture comparisons have not been run.
