# Resumable LightRAG indexing

Question: can the graph-ready v2 corpus be indexed across sessions without repeating completed documents, and can an interrupted document be recovered without copying the pilot graph?

Held fixed: `llama3.1:latest`, `nomic-embed-text` at 768 dimensions, project entity guidance, `num_ctx` 16384, chunk token size 400, gleaning 1, LightRAG 1.5.7. `storage_pilot160`, both Chroma indexes, `rag_active.json`, and the Relabel adapter were not modified. The expansion directory is `artifacts/lightrag/model_f/storage_expand_v1`.

## Status

The indexer is IMPLEMENTED. The checkpoint tests are TESTED: 9 passed in `tests/test_indexing_checkpoint.py`. The live Ollama sessions below are MEASURED. The expansion is not VALIDATED as a retrieval improvement over the pilot, and it is not the full corpus.

## What was indexed

Prepare saw 13,140 graph-ready v2 documents, selected 13,132, and deferred the 8 pilot failures from EXP-20261029-004. The file hash matches the earlier graph-ready v2 hash. Documents excluded when that file was built are not rows in this ledger. EXP-20261027-001 measured 573 quarantined documents on the v1 derivative; that count was not recomputed here.

Five documents reached a verified graph commit: 61 nodes, 41 edges, embedding dimension 768. Text-chunk rows for those five documents: 8. The vector store has 10 chunk vectors. The extra two belong to the failed document below and are not a graph commit.

| Document | Medium | Words | Chunks | Seconds | Result |
|---|---|---:|---:|---:|---|
| `gd_09e160ccf7663729c581d1e2` | blog | 296 | 2 | 25.215 | completed, 24 entities, 19 relations |
| `gd_0270bf4a3abebc35861d117f` | linkedin | 129 | 1 | 5.223 | completed, 4 entities, 6 relations |
| `gd_014b45cb99cfee97ee337556` | x | 12 | 1 | 3.552 | completed, 3 entities, 2 relations |
| `gd_756878d4d219336489c49168` | blog, omission | 594 | 3 | 15.291 | completed on the second attempt |
| `gd_0e1afddaca819c17829b209e` | blog | 161 | 1 | 8.340 | completed, 11 entities, 0 relations |

Measured LLM time on the completed attempts was 21.59, 3.78, 2.13, 11.79, and 6.84 seconds. Measured embedding time was 3.50, 1.33, 1.31, 3.36, and 1.36 seconds. The LLM-versus-embedding split is the bottleneck on every completed document. The kind split inside the LLM time is estimated from prompt text.

`gd_0e1afddaca819c17829b209e` logged relation rows with 4 of 5 fields. LightRAG dropped those relations and still committed the entities. That is a format warning on a completed document, not a failed graph write.

## Interruption and recovery

`gd_756878d4d219336489c49168` was cancelled at 436.127 seconds while chunk 3 of 3 was still extracting. LightRAG status stayed `processing`, `kg_write_state` stayed `pre_graph`, and there were no entity anchors. The first close of that session shut the embedding workers down before the chunk flush and failed to finalize `chunks_vdb`. The runner now flushes storage before it shuts the workers down. The retry purged the 3 staged chunk ids, rebuilt the document, and completed in 15.291 seconds. Graph growth from 31 nodes and 27 edges to 51 nodes and 41 edges matches the new document, including one merge into an existing `AI` node. The three earlier documents were not reprocessed.

`gd_002be28d89a6ca9924e73be2` is a 318-word blog, not a long document. Two attempts were cancelled at 436.12 and 436.07 seconds. Measured LLM time on both attempts was 0. The extraction call did not return. The second attempt purged 2 staged chunks, found no anchors, and was cancelled again before a chunk finished. The same session retried it immediately and used the rest of the runtime budget. That retry has been removed: a document-timer cancel now sets `defer_reason=expensive_timeout`, and at the attempt cap the status becomes `failed_permanent`. This document was sealed to `failed_permanent`. Its LightRAG status was then set to `failed`, with `kg_write_state` still `pre_graph`, so the next ingest does not auto-resume it. Its chunk vectors were not deleted.

A mix-mode context query for the SaaStr Annual change returned status `success`, with hits `gd_09e160ccf7663729c581d1e2`, `gd_014b45cb99cfee97ee337556`, and `gd_756878d4d219336489c49168`. The existing relationship checker accepted 6 candidates and rejected 11. That checker is not a score of the graph write.

## Coverage and the next session

Completed graph documents: 5 of 13,132 selected. Pending in the default queue start at `gd_087a4675a11cf57ae08ea7cf`, a 203-word blog. The mean completed runtime is 11.5 seconds. The approximate remaining time from that mean is about 42 hours and does not include deferred documents or the 436-second cancels. One hung extraction costs the whole per-document limit, so that mean is not a schedule.

The next snapshot point is 25 completed documents. Do not pass `--retry-timeouts` or `--include-deferred` on an unattended run. The explicit longer command is in `docs/architecture/LIGHTRAG_INDEXING_AND_CHECKPOINTING.md`.

## Failures and warnings

- `gd_002be28d89a6ca9924e73be2`: two document-timer cancels, then sealed failed. Chunk-vector residue remains. No graph mutation was recorded.
- Relation format warnings (4 of 5 fields) on `gd_756878d4d219336489c49168` chunk 2 and on `gd_0e1afddaca819c17829b209e`. Entities were kept. Those relations were not stored.
- Cancelling an in-flight extraction logs `Queue is shutting down` from a leftover task after storage finalize. That message is not a second graph write.
- The worktree was dirty. This run did not commit. EXP-20261030-012 and EXP-20261031-001 were left in place.
