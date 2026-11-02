# Indexing quality audit

Operational status is separate from a judgment that the extracted graph is correct.
A completed graph document has a processed LightRAG record, chunk vectors of the configured dimension, source-text provenance, and a persisted entity anchor.
Vector-only documents are not graph-indexed.

## Coverage

- Eligible documents in the ledger: 13140
- Selected for the default queue: 13132
- Completed graph-indexed: 5
- Vector-only completed: 0
- Deferred: 9
- Quarantined: 0

## Pilot graph

The 160-document pilot at artifacts/lightrag/model_f/storage_pilot160 was not copied. EXP-20261029-004 records that 26 of its documents were extracted under num_ctx 32768 and the finishing pass used 16384, and the store contains a duplicate status remnant. This expansion starts empty with one recorded configuration: project entity guidance, num_ctx 16384, chunk token size 400, gleaning 1, llm timeout 600 (worker timeout 1200), nomic-embed-text at 768 dimensions.

## Failures by category

- process_interruption: 1. Examples: `gd_002be28d89a6ca9924e73be2`

Earlier attempts, including ones a later attempt completed:

- `gd_756878d4d219336489c49168` attempt 1 interrupted (process_interruption)
- `gd_002be28d89a6ca9924e73be2` attempt 1 interrupted (process_interruption)
- `gd_002be28d89a6ca9924e73be2` attempt 2 interrupted (process_interruption)

## Exclusions

Graph-ready policy exclusions are quarantined when they appear in the loaded file.
Exact text-hash duplicates are skipped so LightRAG does not create a `dup-` status row.
The eight EXP-20261029-004 failures stay deferred under `pilot160_not_retried` and are not retried by a default session.
Short documents are not dropped. Very long documents stay eligible at a lower priority because extraction cost grows with chunks.

## Recoverable cases

Retryable failures have no recorded graph mutation. The next session may delete that document's LightRAG row and enqueue it again.
A document with `kg_write_state=graph_mutation_started` or a node whose `file_path` contains its id is quarantined instead of deleted.
Interrupted documents are retried through LightRAG's own resume when the status is still processing, which purges stored chunk ids before rebuilding.
