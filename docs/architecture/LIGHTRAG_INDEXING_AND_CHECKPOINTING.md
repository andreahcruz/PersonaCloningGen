# LightRAG indexing and checkpointing

This note describes the resumable indexer on branch `kevin/lightrag-model-f`. The behavior below was read from installed `lightrag-hku` 1.5.7 in `host_finetune/.venv`, not from a newer LightRAG release. Status labels follow `AGENTS.md`: the runner is **IMPLEMENTED**; the isolated stub tests are **TESTED**; a live Ollama session is **MEASURED** only after its experiment directory records the outputs.

The expansion store is `artifacts/lightrag/model_f/storage_expand_v1`. It does not replace Model E, the frozen Relabel adapter, Chroma v1 or v2, `rag_active.json`, or the pilot graph at `artifacts/lightrag/model_f/storage_pilot160`.

## What is stored

LightRAG 1.5.7, with this project's empty workspace, writes these files under the workdir:

| File | Contents |
|---|---|
| `kv_store_full_docs.json` | Original document text, keyed by document id |
| `kv_store_text_chunks.json` | Chunk text and chunk ids |
| `kv_store_doc_status.json` | Per-document status. `processed` is the success commit |
| `kv_store_full_entities.json` | Entity names anchored to a document |
| `kv_store_full_relations.json` | Relation pairs anchored to a document |
| `kv_store_entity_chunks.json` / `kv_store_relation_chunks.json` | Chunk ids attached to entities and relations |
| `kv_store_llm_response_cache.json` | Cached extraction and summary responses |
| `vdb_chunks.json` | Chunk embeddings |
| `vdb_entities.json` | Entity embeddings |
| `vdb_relationships.json` | Relationship embeddings |
| `graph_chunk_entity_relation.graphml` | Knowledge-graph nodes and edges |

Defaults are `JsonKVStorage`, `JsonDocStatusStorage`, `NanoVectorDBStorage`, and `NetworkXStorage`. A non-empty `workspace` would nest those files and use a different pipeline lock. The expansion uses workspace `""` so the layout matches the pilot.

Four different objects are easy to conflate:

- **Chunk embeddings** are vectors of the source chunks. They support the naive and mix retrieval modes. They are not graph edges.
- **Entity embeddings** are vectors of extracted entity names and descriptions.
- **Relationship embeddings** are vectors of extracted relationship descriptions.
- **Knowledge-graph edges** are the graph links in the GraphML file. An edge can exist only after entity and relation extraction and merge. A chunk vector does not create an edge.

Embedding model is `nomic-embed-text`, 768 dimensions. Inside LightRAG the query prefix is `search_query: ` and the document prefix is `search_document: `. Frozen Chroma does not use those prefixes and is not rebuilt by this indexer.

## How a document is indexed

`LightRAG.ainsert` always chunks and always extracts. It does not accept an option to skip the graph. The runner therefore does not call `ainsert`. It calls `apipeline_enqueue_documents` and then `apipeline_process_enqueue_documents`, one document at a time.

Verified order inside `process_single_document` for version 1.5.7:

1. Chunk with the legacy fixed-token chunker (`chunk_token_size` 400). The library overlap default is 100. This project does not override it, and the configuration record says `library_default_not_overridden`.
2. Persist document status `processing`, the chunk vectors, and the chunk text together.
3. Extract entities and relations with the LLM, unless the document was enqueued with process option `!`.
4. Merge nodes and edges into the shared graph. Phase 0 writes entity and relation anchors before the first graph mutation. The document metadata field `kg_write_state` moves from `pre_graph` to `graph_mutation_started` at that boundary.
5. Flush every derived store in `_insert_done`.
6. Write document status `processed`. That status write is the commit. Chunks written in step 2 are not a successful graph index.

Document status values in this version are `pending`, `parsing`, `analyzing`, `processing`, `processed`, and `failed`. `preprocessed` is deprecated. Auto-resume inside LightRAG covers `pending`, `processing`, `parsing`, and `analyzing`. A `failed` row is not auto-retried. Re-enqueueing an id that already exists in `doc_status` does not reprocess it, and a second content-hash collision can create a separate `dup-` failed row. The runner never enqueues an id that is already present.

### Vector-only mode

Process option `!` (`PROCESS_OPTION_SKIP_KG`) is present in 1.5.7. `parse_process_options` sets `skip_kg`. After chunks are persisted, entity and relation extraction is skipped. The document can still reach `processed`. `chunking_explicit` stays false for `!` alone, so the same legacy chunker is used.

That mode is available as `--mode vector_only`. It must use its own workdir, suggested name `storage_expand_v1_vector`. A vector-only `processed` document is not a graph document. The ledger records `vector_only` separately from `graph_extracted`. The configuration fingerprint includes `store_role`, so the two stores cannot share a ledger.

### What consumes GPU time

Chunking is local. The slow calls are Ollama:

- LLM extraction and the optional gleaning pass (`llama3.1:latest`, temperature 0, `num_ctx` 16384, gleaning 1).
- LLM merge summaries when extracted entities or relations are folded into existing nodes.
- Embedding calls for chunks, and, after extraction, for entities and relationships (`nomic-embed-text`).

`llm_model_max_async` and `embedding_func_max_async` are both 1. The LightRAG worker timeout is `llm_timeout * 2`. With `default_llm_timeout` 600, a stuck extraction can run for 1200 seconds. That matches the seven 1,200-second worker timeouts in the pilot. The runner's own `--max-document-seconds` (default 420) stops a document before that worker cap when the session is still alive to cancel it.

VRAM samples come from `nvidia-smi` at document boundaries. They are samples, not a peak trace. Per-stage times that the project callables observe are measured. The split of one LLM call into "extraction" versus "gleaning" versus "merge summary" is estimated from prompt text and labeled that way. Token and chunk counts in the ledger are `characters / 4` and `ceil(tokens / 400)`. They are estimates, not tokenizer counts.

## Why the pilot was not copied

`storage_pilot160` stays read-only. EXP-20261029-004 records 152 processed documents, 8 failures, 1,791 nodes, 2,000 edges, 348 chunk vectors, and 10,768 seconds. Twenty-six of those documents were extracted with `num_ctx` 32768 before the finishing pass used 16384. The store also contains a `dup-` status remnant, and failed documents can leave chunk rows because chunks are written before extraction. Copying that directory would not be a verified snapshot, and mixing the two context settings in one graph would make a later size comparison uninterpretable.

The eight pilot failure ids stay in the new ledger with `defer_reason=pilot160_not_retried`. They are not in the default queue. `--include-deferred` is an explicit later choice, not the default. A timeout in the pilot also hit short documents, so length alone is not treated as the cause.

The expansion starts empty with one recorded configuration: project entity guidance, `num_ctx` 16384, chunk size 400, gleaning 1, LLM timeout 600, worker timeout 1200, `nomic-embed-text` at 768 dimensions, async 1.

## Ledger

`indexing_ledger.sqlite` sits beside the LightRAG files. It is the operational record. LightRAG's own `doc_status` remains the graph commit record. A document is `completed` in the ledger only after the adapter's persistence check passes.

SQLite uses WAL and `BEGIN IMMEDIATE`. Attempts are append-only. A later reconcile inserts a note and does not rewrite the earlier attempt. The ledger refuses a corpus-hash change, a configuration-fingerprint change, and a source-hash change on a completed document.

States:

| Status | Meaning |
|---|---|
| `pending` | Eligible and not yet attempted |
| `processing` | A session has claimed it |
| `completed` | Persistence check passed for the store role |
| `failed_retryable` | Failed before a graph mutation, and attempts remain |
| `failed_permanent` | Not safe to retry, or the attempt budget is exhausted |
| `quarantined` | Source or residue problem. Not a quality grade of the prose |
| `skipped` | Exact `text_sha256` duplicate of a higher-priority document |
| `interrupted` | The process stopped before a verified commit |

`completed` plus `graph_extracted` means the graph check passed. `completed` plus `vector_only` means only the chunk-vector check passed. Provenance is a separate flag: the source text hash and the graph `file_path` token must match the document id. Operational status is not a judgment that the extracted relations are factually correct.

Graph verification requires LightRAG status `processed`, at least one chunk, embedding dimension 768, the source provenance string, a `full_entities` row, and, when the entity count is greater than zero, at least one graph node whose `file_path` contains the document id as a separated token. Zero extracted entities can still be `verified_empty` and `completed`, with a quality note. That is an empty extraction, not a failed write. Vector-only verification requires `skip_kg`, zero entity anchors, and zero nodes for that document.

## Scheduler

`--prepare` reads `artifacts/lightrag/model_f/graph_ready_v2/documents.jsonl` and writes every row into the ledger. The 2026-10-31 prepare of that file saw 13,140 documents. The file hash is `d7bd01a01f288dda3582eddcfd2fc1539574f97b6e168b1124a0b43b76d9a118`, the same hash recorded for graph-ready v2 in earlier experiments. Documents excluded when v2 was built are not in this file. EXP-20261027-001 measured 573 quarantined documents (540 exact-duplicate sources, 25 other authors, 8 unattributed transcripts) on the v1 derivative. Those exclusions are not re-applied here and are not new measurements.

Selection priority is a scheduling heuristic:

- Attribution: `author_account` 40, `named_author` 38, `publication_page` 28, `third_party_quotation` 6.
- Medium: blog 18, LinkedIn 16, X 6.
- Length bands that keep short posts and do not treat the longest blogs as the most useful.
- A complexity penalty after 3 estimated chunks.
- Small bonuses for numeric surfaces and `cleaned_document` reconstruction.
- Penalties for an omission marker, a continuation fragment, and a third-party quotation flag.

Exact text-hash duplicates are `skipped`. The first 40 words of a near-duplicate opening are downranked by 25, not dropped. The queue is then round-robin across `medium:topic` buckets (hiring, pricing, fundraising, retention, ai, otherwise general) so one topic does not occupy the front of the queue. Historical pilot ids are stored and not selected.

The validation batch is the first selected pending row of each kind: an X post of at least 12 words, a LinkedIn post of at least 80 words, a blog of 1–3 estimated chunks without an omission marker, and one omission document of at most 6 estimated chunks. It is written to `validation_batch_ids.json`.

## Stopping

Create `STOP` in the workdir, or press Ctrl+C once.

1. The session stops claiming new documents.
2. The current document is allowed to finish inside the grace period (default 30 seconds) and, if `kg_write_state` is already `graph_mutation_started`, one extra merge wait (default 180 seconds).
3. The ledger records the outcome and LightRAG storage is finalized.
4. The process prints the session summary.

A per-document time limit is not a worker timeout. The runner cancels the pipeline, records `process_interruption`, and sets `defer_reason` to `expensive_timeout`. The same session does not pick that document up again. A default resume skips it. `--retry-timeouts` allows another attempt while the status is still `interrupted`. After `--max-attempts` the status becomes `failed_permanent`. A Ctrl+C stop is different: the interrupted document stays in the queue for the next resume.

A second Ctrl+C requests cancel. LightRAG checks cancellation between stages, not inside an in-flight HTTP call, so the runner cancels the pipeline task after the grace period. That cancel can leave the document `processing` inside LightRAG. It is not a verified clean merge boundary. If a graph mutation had already started, the document is quarantined instead of being deleted and retried. `adelete_by_doc_id` can rebuild shared entities with the LLM, so the runner deletes a failed document only when `kg_write_state` is still `pre_graph` and the entity-anchor count is zero.

On close, the runner flushes LightRAG storage first, then shuts down the role-LLM and embedding priority-queue workers. A cancelled document can still have chunk upserts that need those workers during the flush. `finalize_storages` does not join the workers, so the shutdown still has to run afterwards. Without it, a cancelled session can fail to exit.

There is one writer. `indexing.lock` is held for the session. A second `apipeline_process_enqueue_documents` sees the pipeline busy and returns without processing.

## Restart

`--resume` checks, in order:

1. The ledger file exists.
2. The configuration fingerprint matches the workdir. A different model, context, chunk size, guidance, or store role is refused.
3. The graph-ready file hash matches the hash stored at prepare.
4. Rows left in `processing` are reconciled with LightRAG. Completed and verified rows are marked completed. A pre-graph failure can be retried. Graph residue is quarantined. The original attempt row stays.
5. Completed documents are not selected again.
6. `failed_retryable` and `interrupted` rows are eligible when they are selected, under the attempt cap (default 2), and not deferred. Extraction timeouts are stored with `defer_reason=expensive_timeout` and stay out of the default queue unless `--retry-timeouts` is passed.
7. Backoff of `min(30, 2**(attempt-1))` seconds applies only when the same failed id would be selected again immediately.

An unbounded resume is refused. Pass `--max-documents`, `--max-runtime-minutes`, or `--stop-after-current`.

`--snapshot DEST` copies the workdir only when no ledger row is `processing` and LightRAG has no `processing`, `parsing`, or `analyzing` document. The destination must not already exist. The copy ignores `STOP` and `indexing.lock`. It is a file copy after a clean shutdown, not a storage-engine snapshot. Do not copy the directory while a session is running. Do not snapshot the pilot.

## Failure categories

The classifier maps an error message and LightRAG's stage hint to one of: source validation, document parsing or preparation, tokenization or chunking, context-window overflow, LLM extraction timeout, invalid extraction format, entity or relation merge, embedding-model failure, embedding dimension mismatch, vector-store write, graph-store write, persistence, and process interruption.

Permanent categories are source validation, context-window overflow, and embedding dimension mismatch. The others can be retried only before a graph mutation and only inside the attempt cap. A timeout is still classified as an extraction timeout when the scheduler declines to run it again.

## Retrieval and coverage

Adding documents changes the corpus, not only the retrieval algorithm. A later comparison of this graph with dense retrieval has to use the same source ids. Improvements after more documents are indexed are not, by themselves, evidence that the graph is better than dense search. Query-time comparison stays on the existing evaluation path. This indexer does not edit `rag_active.json` and does not point the default retriever at the expansion.

Corpus coverage is `completed_graph_indexed / selected`. The selected count (13,132 after the 2026-10-31 prepare, with 8 deferred) is the indexing queue. The eligible count is every non-quarantined row in the graph-ready file. Documents excluded before that file was built are outside both numbers.

A retrieval smoke test calls the existing query helper against this workdir only. It does not change the active RAG configuration.

## Commands (PowerShell)

Run from the repository root. The interpreter is `host_finetune\.venv\Scripts\python.exe`.

Prepare the ledger once:

```powershell
host_finetune\.venv\Scripts\python.exe -X utf8 -m host_finetune.indexing_session --prepare
```

Show the next documents without calling a model:

```powershell
host_finetune\.venv\Scripts\python.exe -X utf8 -m host_finetune.indexing_session --dry-run --max-documents 20
```

Conservative live session. This is the default shape. It is not an overnight run:

```powershell
host_finetune\.venv\Scripts\python.exe -X utf8 -m host_finetune.indexing_session --resume --max-documents 10 --max-runtime-minutes 60 --max-document-seconds 420
```

Validation sample only:

```powershell
host_finetune\.venv\Scripts\python.exe -X utf8 -m host_finetune.indexing_session --resume --validation-batch --max-documents 4 --max-document-seconds 420
```

Stop: press Ctrl+C once, or:

```powershell
New-Item -ItemType File -Path artifacts\lightrag\model_f\storage_expand_v1\STOP
```

Remove `STOP` before the next resume. A second Ctrl+C cancels the current document.

Resume, status, and reports:

```powershell
host_finetune\.venv\Scripts\python.exe -X utf8 -m host_finetune.indexing_session --status
host_finetune\.venv\Scripts\python.exe -X utf8 -m host_finetune.indexing_session --export-report artifacts\lightrag\model_f\storage_expand_v1\reports
host_finetune\.venv\Scripts\python.exe -X utf8 -m host_finetune.indexing_session --resume --max-documents 25 --max-runtime-minutes 180
```

An explicit longer unattended session, still bounded. Do not omit the limits:

```powershell
host_finetune\.venv\Scripts\python.exe -X utf8 -m host_finetune.indexing_session --resume --max-documents 50 --max-runtime-minutes 480 --max-document-seconds 420
```

Snapshot after a clean stop, into a new directory:

```powershell
host_finetune\.venv\Scripts\python.exe -X utf8 -m host_finetune.indexing_session --snapshot artifacts\lightrag\model_f\storage_expand_v1_snap_50
```

Vector-only, separate directory, still bounded:

```powershell
host_finetune\.venv\Scripts\python.exe -X utf8 -m host_finetune.indexing_session --prepare --mode vector_only --workdir artifacts\lightrag\model_f\storage_expand_v1_vector
host_finetune\.venv\Scripts\python.exe -X utf8 -m host_finetune.indexing_session --resume --mode vector_only --workdir artifacts\lightrag\model_f\storage_expand_v1_vector --max-documents 10
```

Reports written by `--export-report` or at the end of a session:

- `indexing_summary.json`
- `document_status.csv`
- `failures.csv`
- `stage_timings.csv`
- `index_manifest.json`
- `quality_audit.md`

The remaining-time field is labeled approximate. It is the mean completed runtime times the selected documents still queued. It does not include deferred documents, and it does not predict a timeout.

## Quality, orphans, and omission markers

`--audit` reads the ledger and LightRAG files and writes `reconciliation.json`, `graph_quality.json`, `orphan_vectors.json`, `runtime_statistics.json`, and `next_resume.txt`. It does not delete vectors or graph rows.

A completed document with entities and no committed relations stays completed. The quality file records that separately. A zero-relation document is an indexing failure only when extraction did not finish. Malformed relation warnings, counted from LightRAG's own log lines on later sessions, explain a zero-relation result when they are present. The first sessions attached that counter to the root logger. LightRAG's logger does not propagate, so those stored zeros are not a measured absence of malformed lines. Later sessions attach the counter to the `lightrag` logger. Session logs remain the count for the documents indexed before that correction.

Failed-document chunk vectors can remain after a cancelled extraction. The two vectors for `gd_002be28d89a6ca9924e73be2` have `file_path` equal to that id. `normalize_lightrag_data` excludes those chunks, and the relationship checker rejects a failed document. Entity and relationship descriptions are not returned as source hits. The vectors were not deleted.

Omission markers stay inside graph-ready v2 and inside `storage_expand_v1`. Query time drops a chunk whose verbatim span contains the marker. Splitting on the marker changes document identity, so that representation is a separate corpus, `graph_ready_v2_segments`, with segment ids prefixed `gs_`. Its configuration record adds `omission_segments`, which changes the fingerprint. A default resume of `storage_expand_v1` does not include that field and will refuse a segmented corpus. Do not insert segmented text into the current graph.

A document under about 800 words can still stall. `gd_002be28d89a6ca9924e73be2` is 318 words. It is deferred and does not block the next selected document. The adaptive budgets below are the normal-queue limits; `--timeout-policy fixed` keeps 420 seconds for every document.

LightRAG processes every pending or processing document in one call. Before that call, any other unfinished document whose graph merge has not started is marked `failed` and left in storage. That keeps a deferred timeout from being extracted again in the same session. A document whose merge already started is not rewritten by this hold.

## Adaptive timeouts

The default command uses timeout policy `adaptive-v1`. It is not part of the configuration fingerprint. `--timeout-policy fixed` restores one budget for every document, which is `--max-document-seconds` and still defaults to 420.

| Estimated chunks | Normal-queue budget |
|---|---|
| 1 | 90 seconds |
| 2–3 | 180 seconds |
| 4 or more | 420 seconds |

An explicit `--retry-timeouts` run is the only path that uses the 420-second retry budget. It is not enabled on an ordinary resume. A document that exceeds its normal budget is deferred with `expensive_timeout` and the next selected document continues after the in-flight Ollama call is cancelled. The call is cancelled in the project wrapper. Cancelling the LightRAG pipeline task alone leaves the priority-queue worker's HTTP request running until the 1,200-second worker cap. The wrapper does not cancel a call once that document's graph merge has started.

The chunk estimate is characters/4. It is not an exact token count. The budgets sit above the p95 of the 73 completed documents measured in the expansion ledger (33 seconds) and below the 436-second stalls. One completed document, `gd_0cf4677ed9c64844dce49e18`, took 356 seconds, so a 90-second budget would have deferred a document that did finish. That is the premature-deferral case. It can be retried only with `--retry-timeouts`.

`--status` reports the normal queue, the deferred retry queue, tier counts, and two approximate ETAs. The rate includes unsuccessful attempt time. It is not a completion date.

## Blog-only expansion queue

The expansion graph already contains LinkedIn and X documents. New scheduling can be limited to blogs without deleting those documents or rebuilding the graph. The resulting index is a mixed-media seed graph with blog-only expansion. It is not a blog-exclusive graph. A clean blog-only rebuild would be a separate experiment.

`--selection-policy blog-only` passes `source_medium = blog` into the existing ledger query. It does not change `selected` flags, so an all-media process and a blog-only process do not rewrite each other's queue. The default `--selection-policy all-media` is the previous `--resume` behavior. A blog-only process keeps that filter for the whole session. Deferred blogs stay out of the normal queue unless `--retry-timeouts` is passed.

The versioned record is `artifacts/lightrag/model_f/storage_expand_v1/selection_blog_only_v1.jsonl`. It does not replace `selection_manifest.jsonl`. Status fields in the JSONL are a snapshot. Resume reads the live ledger. The identity hash covers rank, document id, source hash, and original selection rank, so it stays the same when statuses change.

Policy `blog-only-v1` is not part of configuration fingerprint `771baa30dd8f79c30c4c51f296385b2ec2c652b33ff1031a099a7e16f139260b`. Extraction, embeddings, chunk size, and graph-ready v2 text are unchanged. Graph-ready v2 segments are still not indexed into this store.

Blogs are the long-form source. This queue assumes that much of the author's explanatory material is in those posts. It does not show that blogs contain every fact, example, or relationship present in LinkedIn or X. Pending posts in those media stay in the ledger and are simply not scheduled.

Do not start `--selection-policy blog-only` while another indexing process holds `indexing.lock`.

```powershell
host_finetune\.venv\Scripts\python.exe -X utf8 -m host_finetune.indexing_session --status --selection-policy blog-only
host_finetune\.venv\Scripts\python.exe -X utf8 -m host_finetune.indexing_session --resume --selection-policy blog-only --max-documents 13000 --max-runtime-minutes 480 --timeout-policy adaptive
```

Stop with one Ctrl+C. Resume with the same blog-only command. Omit `--retry-timeouts` on an unattended run. All-media resume is the same command without `--selection-policy`.

## Measured on 2026-10-31

This is the first bounded expansion, recorded as EXP-20261031-002. Five documents are verified graph commits: 61 nodes, 41 edges, 768-dimensional vectors. `gd_002be28d89a6ca9924e73be2` was cancelled twice by the 420-second document limit and is `failed_permanent` in the ledger and `failed` in LightRAG, with `kg_write_state` still `pre_graph`. Do not pass `--retry-timeouts` or `--include-deferred` on an unattended run.

EXP-20261031-003 continued the same store and the same configuration fingerprint. After the queue hold was added, one 18-document session brought the verified graph commits to 25: 266 nodes and 217 edges. `gd_4c65a0a559794e0816949bf4` hit the 420-second limit once, is deferred, and was marked `failed` in LightRAG without deleting its rows. The next safe command is in `storage_expand_v1/reports/next_resume.txt`. The eight-hour session was not started.

## What is still uncertain

- Mid-document cancel during an HTTP call is not a clean commit. Recovery is tested for a stub interruption and must be re-checked when a live cancel leaves `processing`.
- A document that extracts zero entities can be marked completed. That is a successful empty graph write, not evidence the source had no facts.
- Stage labels come from error text plus LightRAG's stage hint. An unusual message can be mislabeled. The attempt row keeps the original message.
- Copying a live workdir is not a consistent snapshot. Use `--snapshot` after shutdown.
- Whether a larger graph retrieves better than the 152-document pilot is not answered by indexing more documents. That needs a matched-corpus comparison that holds the retrieval method fixed and does not attribute the document-count change to the graph alone.
- Chunk overlap is the library default. Changing it, the extraction model, the prompt, `num_ctx`, or the embedding model requires a new workdir. The resume guard refuses a fingerprint mismatch.
