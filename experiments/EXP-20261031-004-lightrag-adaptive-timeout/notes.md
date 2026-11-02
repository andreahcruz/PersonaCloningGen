# EXP-20261031-004 — adaptive document timeouts

Question: can the normal indexing queue stop a stalled extraction without leaving a LightRAG worker free to write the graph, and without changing the expansion graph's configuration?

Held fixed: graph-ready v2, fingerprint `771baa30dd8f79c30c4c51f296385b2ec2c652b33ff1031a099a7e16f139260b`, the pilot, Chroma, the Relabel adapter, and `rag_active.json`. No indexing process was running when this work started. The expansion ledger still has 73 completed, 1 permanent failure, 3 interrupted, and 13,063 pending documents.

## What the ledger shows

Completed documents: 73. Their runtimes sum to 1,173.2 seconds. Median 7.2, p75 14.4, p90 26.7, p95 33.4, p99 128.1, maximum 356.4. Blogs are slower than LinkedIn and X posts in the normal range. Actual chunk count tracks that normal range: one indexed chunk has median 6.3 seconds, and the slowest ordinary multi-chunk success is under 40 seconds.

Six other attempts sum to 2,616.7 seconds. Every one is `process_interruption` at about 436 seconds, which is the 420-second document cap plus cancel wait. That is 69 percent of recorded attempt time. Those attempts did not become successful-document runtimes. Two documents later completed after an earlier non-complete attempt. The stalls are not explained by word count: the failed documents are 318 to 610 words, and the 356-second success is a 262-word blog estimated as one chunk.

## Cancellation

LightRAG 1.5.7 runs the Ollama call inside a priority-queue worker task. Cancelling the pipeline task does not cancel that inner call. The worker waits until the call returns or the 1,200-second worker timeout fires. The project LLM wrapper now cancels its own inner call when the document budget fires, unless that document's graph merge has already started. Embeddings are not cancelled, because a cancelled document still has to flush chunk vectors. The priority-queue test shows the inner sleep is cancelled and the next call starts only after it has stopped.

## Policy

`adaptive-v1`: 90 seconds for an estimated one-chunk document, 180 for two or three, 420 for four or more. Floor 60. Ceiling 420. `--retry-timeouts` is the explicit second queue and uses 420 seconds. It is not on by default. `--timeout-policy fixed` restores one budget, `--max-document-seconds`, default 420. The policy is stored on the session summary, not in the fingerprint.

## Isolated benchmark

A temporary store, not `storage_expand_v1`, indexed the validation batch with the adaptive policy. Three documents completed in 28.3, 6.0, and 4.0 seconds. `gd_756878d4d219336489c49168` was still extracting at the 180-second medium budget. The process recorded the cancellation, finalized storage, shut the workers down, and exited 0. It did not wait for a 1,200-second worker timeout. The same document has completed in about 15 seconds on an earlier attempt, so this is a measured case of the policy deferring a document that can succeed. The first benchmark process also crashed while flushing embeddings because the cancel flag was still set; that path was corrected before the clean rerun. This is not a measured corpus speedup.

Counterfactual from the six historical 436-second attempts, if each had stopped at 180 seconds: about 1,537 seconds. That figure is not a measured speedup. It does not include the 356-second document that did finish and would have been deferred at 90 seconds.

Pytest: 26 passed.
