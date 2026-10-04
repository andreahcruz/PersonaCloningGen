# Relabeled fine-tune: next steps

Updated 2026-10-04. EXP-20261003-004/005 are the user's active data/training branch. The earlier whole-source v3 corpus is a separate conservative candidate, not a replacement for this run. Training inputs, trainer code, checkpoints and process were not modified by this review.

## Interpretation

Relabeling first sections and continuations is a sensible test of instruction/target scope alignment. Authentic complete social posts need not end in punctuation. The earlier recommendation that semantic completeness matters still applies, but a section task can validly end before the source post ends. A whole-source-only policy is not required for every SFT example.

The supplied summary's causal claim that incorrect EOS supervision caused the generation failures is a hypothesis supported by the data audit, not a controlled model result. Short EOS termination can be correct; reaching the generation budget can mean truncation. Training loss measures prediction of the supplied targets, not voice fidelity. Relabeling plus filtering changes tasks, medium mix, exposure and optimizer-step count, so a gain would support this combined intervention rather than isolate relabeling alone.

## Read-only audit evidence

EXP-20261004-002-relabel-audit independently checks the frozen output against its prior rows: all targets are unchanged and all inherited splits/groups are preserved. Counts match 21,377 train, 2,588 validation and 2,580 test. Training comprises 9,434 blog, 713 LinkedIn, 11,190 X and 40 YouTube rows.

Training diagnostics:

- 40 transcript rows remain: the current code drops categories of malformed transcripts, not every transcript. Three remain in validation and one in test. Speaker attribution remains unresolved.
- 12 continuation prompts have no previous-section quote after prompt-budget fitting (three in validation). These have a weaker task specification.
- 72 section targets end in a colon or ellipsis. These need semantic review; punctuation matching does not establish that the requested unit is complete.
- 11,890 prompts contain the target's first 12 whitespace-delimited words, including 10,749 unchanged X rows. This is a diagnostic for target-derived prompts, not cross-split leakage or proof that every match is invalid. It is especially serious where the task purports to generate new content from a topic.
- Reusing the v3 source audit flags 183 training rows from gold-overlap families and 13 from full-source cross-split families. These are row counts, not unique document counts or pair counts. Preserving the old groups does not eliminate newly detected full-source family leakage. The existing 30 gold topics are development prompts, not an untouched final test.
- Source-level warnings also touch 353 guest-recap rows, 384 broken-word rows, 159 missing-visual rows and 36 roundup rows. These overlapping flags require source review; they are not automatic proof of invalid targets.
- Whole-target/source-body checking found no training mismatch among unchanged rows; one held-out test row was mechanically flagged. No test body was manually inspected or used to tune policy.

The first audit attempt (EXP-20261004-001) failed because Python splitlines treated a Unicode separator inside JSON content as a record boundary. The reader now splits physical newlines; a regression test passes. Both audit attempts are retained. All audited inputs remained unchanged in the successful run.

## Fixes implemented around this run

`audit_relabel_run.py` provides a CPU-only, new-directory-only audit with hashes and an experiment manifest. `generation_diagnostics.py` and `compare_adapters.py` now retain effective EOS IDs, last generated ID, prompt/generated lengths, stop reason, early-EOS diagnostic, and deterministic per-prompt seeds. Native Ollama done_reason is retained without pretending that its generic stop means an observed EOS. Legacy traces without stopping metadata return unavailable rather than zero. Seeds and backend differences are recorded; seeds alone do not make different backends equivalent.

The comparison summary warns against selecting a model using its legacy aggregate lexical/format score. Existing X prompts request a post but legacy scoring expects a thread. Other formats also demand structures absent from the actual prompt. Legacy scores remain for compatibility and must be treated as secondary diagnostics. New telemetry unit tests and the full CPU suite passed: 141 passed, 4 skipped. GPU generation with the revised comparison code has not been run while training occupies the GPU.

## After training finishes

1. Check successful completion and saved adapter files. Record checkpoint identity, final hashes, training environment package snapshot and actual training arguments. The current manifest names an environment directory rather than a package snapshot; preserve the actual environment before later changes. Do not infer completion from loss or elapsed time.
2. Register a new comparison experiment. Compare the unadapted instruction model, repaired adapter and relabeled adapter with the same base revision, tokenizer/chat template, backend/quantization, prompt set and generation settings. Record separate adapter identities; do not compare old stochastic outputs with a new seeded run as if fully controlled.
3. Freeze fresh development briefs by medium: X, LinkedIn and blog first. Include full-post requests, opening requests and continuations with supplied context; score these task families separately. Treat talk as exploratory while transcript supervision is sparse/unverified. Use self-contained factual briefs to reduce topic-knowledge confounding. No target text should be hidden inside a supposedly neutral topic prompt.
4. Start with a modest paired development panel (proposed: 20 topics in each written medium). Use identical requests for all three models. When multiple generations are used, retain seeds and aggregate/uncertainty by topic, not each generation as an independent sample. Do not tune on held-out test sources. The familiar 30-topic set can serve as a regression panel with its known limitations.
5. Select on the measured stop rates, by medium. Human voice preference is not collected. Lower validation loss can shortlist checkpoints; it does not decide stopping.

## Selection rule

DEC-007 removed the blinded voice review. The comparison does not write a preference sheet. An adapter is selected by two rates on the same prompts: early `<|eot_id|>` before half of `max_new_tokens`, and answers that are not sentence-final. On EXP-20261004-003, retrieval off, relabel won both (early 0.550 vs 0.650, mid-sentence 0.342 vs 0.633). BLEU, ROUGE, and the gold lexical overall stay in the log and do not select.

Supporting diagnostics:

- True stop reason and token length, stratified by medium. Early `<|eot_id|>` and a missing sentence ending are the selection rates. A sentence-final answer shorter than half the budget still counts as an early stop.
- Stylometry against authentic medium/length-matched reference distributions: sentence/paragraph length, question frequency, pronouns, contractions, numbers and list structure. Use distributions, not a requirement that every answer matches a fixed template.
- Copying: exact/long n-gram overlap with training targets, excluding phrases already supplied in the prompt. Manually examine highest-overlap cases.
- BLEU/ROUGE/BERTScore against an arbitrary reference are secondary content-similarity diagnostics. They do not select an adapter.
- `eval_rag.py` is not a selection measure. Its authorship score is not split-aware, and the judge has not been calibrated. Do not run it to choose an adapter.

Method references: [human evaluation review](https://aclanthology.org/2021.gem-1.6/) distinguishes style, meaning preservation and fluency; [scikit-learn leakage guidance](https://scikit-learn.org/stable/common_pitfalls.html) explains fitting preprocessing inside CV.

## Next dataset revision, only if comparison warrants it

Keep this run immutable. Write a separate version with independently derived or human-reviewed social topic briefs, explicit opening/next-section/conclusion roles, intact necessary continuation context, source-backed formatting repair and reviewable attribution. Drop an overflowed continuation if its required context cannot fit rather than silently discarding that context. Use complete paragraph/list boundaries; a colon introducing an absent list should not count as a completed opening. Record any generated topic labels and inspect target leakage. Filter full-source duplicate/gold-overlap families without reassigning ownership. Recover original HTML before joining ambiguous broken words; remove guest/roundup material from style supervision unless its task and attribution are explicit.

Prioritize clean complete written posts and coherent sections. A small longer-context blog pilot may be appropriate if full-post coherence remains weak; increasing rank is not the first remedy. Do not launch another full training run merely because data has imperfections: first measure whether the current model meets the intended writing tasks.

## Vertical slice after the FT decision

As of 2026-10-04 the train-only index code exists and a CPU embed into `lemkin_train_only` finished with 21,194 documents. That collection is not `lemkin_content`, the trainer did not retrieve it, and Streamlit does not use it by default. Finishing the embed does not finish the vertical.

Build an adapter-only path first: versioned source/data manifest -> selected adapter -> one generation service -> existing UI -> logged evaluation traces. Validate identical fixed prompts before and after export/deployment, including chat template, stop behavior, context budget, latency and model version. Keep train/test and source provenance accessible in the pipeline.

Then add factual retrieval as an explicit toggle and compare FT alone against FT plus RAG under the same briefs. Evaluate evidence relevance, citation support, unsupported claims, and the same stop rates. A production knowledge index and an evaluation-safe index need documented inclusion boundaries. RAG addresses evidence access; whether it improves the application is a measured question. Train-only style exemplars and any ontology-aware retrieval extension should be separate later ablations so their effects can be identified.
