# Relabeled continuation SFT file

Question: can the repaired 512-token training rows be filtered and relabeled so `<|eot_id|>` marks the end of the requested text, without refitting or rewriting raw posts?

The prior file `host_finetune/data/dataset_fit512_keepbreaks_balanced.jsonl` (sha256 `6e99b414733eafdfa27a5e94dd26beff1498bbfd58af132251e24044fef4f057`) was not modified. Targets were copied through. No row was re-chunked.

## Train split

Of 25,062 train rows, 21,377 remain.

| Action | Rows | What they are |
|---|---:|---|
| Unchanged | 11,616 | 10,974 whole X posts and 642 whole blog or LinkedIn posts, including posts with no period |
| Opening prompt | 1,711 | First slice of a longer post. The prompt asks for the opening section only |
| Continuation prompt | 8,050 | Later slices that end on a sentence, a colon, or a heading. The prompt quotes the previous ending |
| Dropped | 3,685 | 1,148 word-level cuts, 1,332 other mid-sentence or paragraph cuts, 1,015 last slices that are not a sentence, and 190 transcript fragments |

The 2,247 last slices split into 1,232 continuations and 1,015 drops. Group 3 is 8,430, which is 115 above the summary table's 8,315. Those 115 end on the sentence regex, including an ellipsis, so they were relabeled with the other sentence-complete fragments. The paragraph-cut bucket is 1,332, which is 115 below that table. Word-level cuts stay 1,148. Groups 1, 2, 5, 6, and the last-slice total match the audit.

Validation (2,588) and test (2,580) used the same rules. Each surviving row kept the split of its source document. No document was moved between splits.

Every emitted chat string fits in 512 tokens. No relabeled prompt still says "Write a blog post / LinkedIn post / X post / talk in the style of Jason Lemkin about:".

Dataset sha256 `1231835ecc5dcb608fefa0357d325affb2afbdc3aeade2a63b1dc42b2920897b`. This file is the training input. It is not a trained adapter.
