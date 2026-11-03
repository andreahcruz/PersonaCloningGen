# Indexing quality audit

Operational status is separate from a judgment that the extracted graph is correct.
A completed graph document has a processed LightRAG record, chunk vectors of the configured dimension, source-text provenance, and a persisted entity anchor.
Vector-only documents are not graph-indexed.

## Coverage

- Eligible documents in the ledger: 13140
- Selected for the default queue: 13132
- Completed graph-indexed: 2713
- Vector-only completed: 0
- Deferred: 163
- Quarantined: 0

## Pilot graph

The 160-document pilot at artifacts/lightrag/model_f/storage_pilot160 was not copied. EXP-20261029-004 records that 26 of its documents were extracted under num_ctx 32768 and the finishing pass used 16384, and the store contains a duplicate status remnant. This expansion starts empty with one recorded configuration: project entity guidance, num_ctx 16384, chunk token size 400, gleaning 1, llm timeout 600 (worker timeout 1200), nomic-embed-text at 768 dimensions.

## Failures by category

- document_parsing_preparation: 2. Examples: `gd_d9367d2354469b61c124d3c2`, `gd_9df8d7ed77b488b1b9daa14f`
- process_interruption: 155. Examples: `gd_002be28d89a6ca9924e73be2`, `gd_4c65a0a559794e0816949bf4`, `gd_26afab739fd7d88d62176a45`, `gd_10c67f9686ead125655f8f69`, `gd_07cc9ccfb5f7311fd5566e41`, `gd_3092e091743fc01c2546edd2`, `gd_1f23d54c4b7c97be63dfc27a`, `gd_11d253afe7ce6c30d5552f3e` and 147 more
- tokenization_chunking: 1. Examples: `gd_860e3a31c459f5a4ebfa33dd`

Earlier attempts, including ones a later attempt completed:

- `gd_756878d4d219336489c49168` attempt 1 interrupted (process_interruption)
- `gd_002be28d89a6ca9924e73be2` attempt 1 interrupted (process_interruption)
- `gd_002be28d89a6ca9924e73be2` attempt 2 interrupted (process_interruption)
- `gd_4c65a0a559794e0816949bf4` attempt 1 interrupted (process_interruption)
- `gd_26afab739fd7d88d62176a45` attempt 1 interrupted (process_interruption)
- `gd_10c67f9686ead125655f8f69` attempt 1 interrupted (process_interruption)
- `gd_07cc9ccfb5f7311fd5566e41` attempt 1 interrupted (process_interruption)
- `gd_3092e091743fc01c2546edd2` attempt 1 interrupted (process_interruption)
- `gd_1f23d54c4b7c97be63dfc27a` attempt 1 interrupted (process_interruption)
- `gd_11d253afe7ce6c30d5552f3e` attempt 1 interrupted (process_interruption)
- `gd_42f286ba2859c0216cb984f5` attempt 1 interrupted (process_interruption)
- `gd_2779ddc38f43b82ddc8b9f1f` attempt 1 interrupted (process_interruption)
- 151 more attempt rows are in `stage_timings.csv`.

## Exclusions

Graph-ready policy exclusions are quarantined when they appear in the loaded file.
Exact text-hash duplicates are skipped so LightRAG does not create a `dup-` status row.
The eight EXP-20261029-004 failures stay deferred under `pilot160_not_retried` and are not retried by a default session.
Short documents are not dropped. Very long documents stay eligible at a lower priority because extraction cost grows with chunks.

## Recoverable cases

Retryable failures have no recorded graph mutation. The next session may delete that document's LightRAG row and enqueue it again.
A document with `kg_write_state=graph_mutation_started` or a node whose `file_path` contains its id is quarantined instead of deleted.
Interrupted documents are retried through LightRAG's own resume when the status is still processing, which purges stored chunk ids before rebuilding.
