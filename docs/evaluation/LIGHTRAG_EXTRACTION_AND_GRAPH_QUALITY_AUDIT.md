# LightRAG extraction and graph quality audit

Status: the diagnostic functions are **TESTED**. The cache and graph counts below are a **MEASURED** read-only snapshot taken while blog-only indexing was still writing `artifacts/lightrag/model_f/storage_expand_v1`. They will move as that process commits more documents. No edge was inserted, no prompt was changed, and the configuration fingerprint is still `771baa30dd8f79c30c4c51f296385b2ec2c652b33ff1031a099a7e16f139260b`.

The workspace does not contain one 4,946-line console capture, so the warning totals quoted from that capture were not added to this census. Console captures also repeat the same record across extraction and gleaning. The unit below is a cached model response.

## 1. Executive summary

The installed prompt and the installed parser agree. A relation row is five fields: `relation`, source, target, keywords, description, separated by `<|#|>`. Llama 3.1 often emits a different shape. The parser drops every row whose field count is not five, and it also drops a five-field row whose endpoints are the same string.

On the snapshot, 3,949 unique malformed records were dropped. A deterministic repair exists for 2,564 of them. The other 1,385 include keyword-only rows and rows whose extra fields do not match a known pattern. Name co-occurrence in the source chunk is a separate filter: 3,195 of 3,594 repaired rows contain both endpoint strings. That filter is not a labeled judgment that the relationship is true.

The graph also stores weak endpoints the project prompt told the model to omit: `You`, `I`, `Monday`, `Company`, `Customer`, years, and currency amounts. Some of those edges are real advice to the reader. Some are generic co-occurrence. A copied keyword string appears on more than one edge. Retrieval can surface a bare number without the company, metric, and period attached to the node name.

The blog-only session should continue. The first repair is an offline dry-run over the existing cache, after this process stops, and it should not be wired into the running indexer.

## 2. Confirmed root causes

1. **Schema echo.** On six-field and seven-field rows the model writes the prompt's own labels as columns: `relationship_keywords` and, on seven-field rows, `relationship_description`. The real keywords and description are the following fields. `operate.py` `_handle_single_relationship_extraction` requires `len(record_attributes) == 5` and discards the row. Examples: `gd_a753d199b9c468b7c9b06a64` (SaaStr, Okta, SaaS, Calendly's document), `gd_c19cd54fed6d5f7cfae5bbc2`, `gd_c761713807815a161628791f`.

2. **Comma used as the fifth delimiter.** Four-field SDR rows in `gd_cbce5e6622c8c301403e5d87` look like `relation <|#|> SDR <|#|> AE <|#|> Collaboration, SDRs work closely with AEs...`. Keywords and description were packed into one field. The prompt says keywords are comma-separated and must not use `<|#|>`. The model followed the comma rule and then omitted the delimiter before the description.

3. **Keyword list with no claim.** `gd_a48dad30cf3032e3f22fcde5` and `gd_b5598f901756b268dfaf7770` emit four fields whose last field is a short keyword list (`Product offering, conversational AI chatbot`, `involved, leadership`). There is no description to recover. Inventing one would create a claim.

4. **Unstructured extra fields.** The eight-field Interest-rates / ZIRP row in `gd_bd97c14fa0476835df73618f` splits a question across columns (`Post-ZIRP`, `High interest rates`, `Maybe`, the question, `Maybe`). Joining those columns, or keeping the first five, would invent a boundary.

5. **Prompt compliance, not a prompt/parser mismatch.** The system prompt in `lightrag/prompt.py` (`entity_extraction_system_prompt`, output format lines for a five-part relation) matches the parser. The project entity guidance, visible in cached `original_prompt` values, already says to omit weekdays, years, percentages, currency amounts, and generic nouns such as Company and Customer, and to avoid `I` and `you`. The model still emits them. Gleaning is asked to correct malformed rows. In this snapshot it repeated 1,419 initial malformed identities.

6. **Accepted edges can still be weak.** The parser accepts any well-formed five-field row with different endpoints and a non-empty description. It does not check the chunk.

## 3. Remaining hypotheses

- Whether a schema-label repair is true to the source is not labeled. Co-occurrence of the two names is necessary and was measured. A human or a source-verified predicate check still has to accept each edge before insertion.
- The identical keyword string on `You -> Toast` and `You -> HubSpot` looks like a copied template. The generated description text is not a sentence in the chunk store. That does not, by itself, prove the underlying claim is absent from the chunk.
- `Mongo` and `MongoDB` may be the same product in some posts and different mentions in others. The graph does not show which.
- A chunk task that raises `PipelineCancelledException` after it has already merged can leave a partial write. The ledger's mutation check is the control. This audit did not find a new case of that write in the live store, and it did not interrupt the live process to look for one.

## 4. Code and prompt references

Installed package: `lightrag-hku` 1.5.7 under `host_finetune/.venv`.

| Piece | Location | Behavior |
|---|---|---|
| Relation schema | `lightrag/prompt.py`, `entity_extraction_system_prompt` | `relation{tuple_delimiter}source{tuple_delimiter}target{tuple_delimiter}keywords{tuple_delimiter}description` |
| Delimiter | `PROMPTS["DEFAULT_TUPLE_DELIMITER"]` | `<|#|>` |
| Gleaning prompt | `entity_continue_extraction_user_prompt` | Asks for missed or corrected rows. One continuation is configured (`entity_extract_max_gleaning` 1). |
| Parser | `lightrag/operate.py`, `_handle_single_relationship_extraction` | Accepts only five fields. Drops `source == target`. |
| Warning | same function | `found {n}/5 fields on RELATION` |
| Record split | `_process_extraction_result` | Newline, then `<|#|>`. `fix_tuple_delimiter_corruption` repairs broken delimiters, not missing fields. |
| Chunk-task cancel | `operate.py` around the `asyncio.wait(..., FIRST_EXCEPTION)` extraction wait | Cancels pending tasks and waits. It does not read their exceptions. |
| Merge-task drain | `lightrag/utils.py`, `wait_tasks_with_drain` | `gather(..., return_exceptions=True)` retrieves sibling exceptions. |
| Source-id cap | `constants.py`, `DEFAULT_MAX_SOURCE_IDS_PER_ENTITY = 200`, method `KEEP` | `operate.py` logs `Skipped {name}: KEEP old chunks` and leaves the old summary in place. |
| File-path cap | `DEFAULT_MAX_FILE_PATHS = 75` | The constant comment says the file-path field is for display. The log line is `Limited {name}: file_path N -> 75 (KEEP)`. |
| Project guidance | cached `original_prompt`, hashed into the fingerprint as `entity_guidance_sha256` | Tells the model to omit generic nouns, pronouns, weekdays, years, and bare amounts. |
| Indexer cancel | `host_finetune/indexing_timeout.py`, `abortable` | Cancels and awaits the inner generation task. Embeddings are not cancelled. |

Changing `entity_guidance` or `entity_extract_max_gleaning` changes the fingerprint. Resume would then refuse this store. A parser used only on cached text does not change the fingerprint. Wiring that parser into the running indexer would mix two storage policies in one graph without a fingerprint change. That is why it stays offline.

## 5. Raw cache examples

These are the model's fields after splitting on `<|#|>`. They come from `kv_store_llm_response_cache.json`.

`gd_cbce5e6622c8c301403e5d87`, four fields, comma packing, initial and again in gleaning:

`relation | SDR | AE | Collaboration, SDRs work closely with AEs to learn what "great" looks like.`

`gd_a753d199b9c468b7c9b06a64`, six fields, schema echo:

`relation | SaaStr | Okta | Relationship_keywords | provides insights | The SaaStr community relies on Okta's data to understand SaaS adoption and usage patterns.`

`gd_c761713807815a161628791f`, seven fields, schema echo:

`relation | Salesforce | Procurement | relationship_keywords | automate, discounting, procurement | relationship_description | Salesforce automates any discounting at the low end by marking prices up 20%.`

`gd_bd97c14fa0476835df73618f`, eight fields, no safe join:

`relation | Interest rates | ZIRP | Post-ZIRP | High interest rates | Maybe | Does the downfall in SaaS multiples correlate with the rise of interest rates? | Maybe`

`gd_a48dad30cf3032e3f22fcde5`, four fields, keywords only:

`relation | Freemium | ChatGPT | Product offering, conversational AI chatbot`

`gd_b5598f901756b268dfaf7770`, four fields, keywords only, repeated in gleaning:

`relation | CEO | Board | involved, leadership`

## 6. Field-count analysis

Snapshot of 7,839 extract-cache responses (3,921 initial, 3,918 gleaning) and 4,067 chunk texts.

| Field count | Rows |
|---:|---:|
| 5 | 37,170 |
| 4 | 3,599 |
| 6 | 959 |
| 7 | 843 |
| 8 | 21 |
| 3, 9, 16, 17 | 8 |

Of the five-field rows, 36,612 are accepted and 558 are dropped because the endpoints match (`SaaS -> SaaS` is this case).

| Kind | Rows | Repair |
|---|---:|---|
| Accepted five-field | 36,612 | already stored when merge succeeds |
| Schema labels (6 or 7) | 1,406 | drop the label columns |
| Comma between keywords and description | 1,337 | split the last field on the first comma |
| Description present, keywords omitted | 923 | keep the sentence, leave keywords empty |
| Delimiter inside the tail, pattern unknown | 397 | leave dropped |
| Keywords and no description | 1,229 | leave dropped |
| Four fields, ambiguous | 110 | leave dropped |
| Extra fields, ambiguous | 24 | leave dropped |
| Self-relation | 558 | leave dropped |
| Fewer than four fields | 4 | leave dropped |

Unique malformed identities, keyed by chunk, unordered endpoints, and field count: **3,949**. Unique identities with a deterministic repair: **2,564**. Identities left dropped: **1,385**. Only **32** malformed identities already have an accepted five-field row for the same pair in the same chunk, so a repair is usually a missing edge rather than a second copy of an edge the parser already kept.

Gleaning repeated 1,419 malformed identities from the initial pass. Responses also contain 5,333 duplicate relation rows inside a single completion. Duplicate suppression happens later, when edges are keyed by the endpoint pair. The warning line is earlier, so one bad row can be logged twice and still be one edge.

Rows are not unique edges. The console count of 377 warnings is a slice of this stream. It is not added to 3,949.

## 7. Unique relationship-loss estimate

| Quantity | Value | What it is |
|---|---:|---|
| Unique malformed records | 3,949 | dropped shapes |
| Deterministic syntactic repairs | 2,564 | 65% of those identities |
| Left unrecovered | 1,385 | no safe field mapping |
| Repaired rows checked against chunk text | 3,594 | a row can repeat across gleaning |
| Both endpoint strings occur in that chunk | 3,195 | 89% of checked repaired rows |
| A description word also occurs in the chunk | 3,055 | 85% of checked repaired rows |
| Chunks with at least one malformed relation | 700 | |
| Chunks whose relation rows were all rejected | 403 | entities may still have been stored |

The 89% and 85% figures are overlap filters. They are not precision or recall. There is no labeled set of relations for this corpus. A repaired row whose names are absent from the chunk (399 checked rows) must not be inserted.

The Freemium / board-meeting documents are mostly the unrecovered keyword-only shape. The SaaStr, startup, Salesforce, and SDR documents are mostly the recoverable shapes. The ZIRP row stays unrecovered.

## 8. Semantic graph-quality findings

Point-in-time GraphML read during the same session: **12,888 nodes, 20,983 edges**. The ledger's last shutdown flush is older than this file. The in-memory graph can be slightly ahead of the file.

Among accepted five-field rows, frequent weak endpoints include `Company` 596, `Customer` 552, `Founder` 738, `You` 209, `Monday` 208, `I` 119, and the years `2021` through `2025`. The project prompt told the model not to create those as entities. The parser stored them when the row had five fields.

Persisted examples:

| Edge | Reading |
|---|---|
| `SaaStr Annual 2024 -> You`, keyword `Warning` | The description restates advice to the reader. The chunk uses "you". The edge is supported as advice and weak as a fact about a person named You. Class **C**. |
| `I -> You`, keywords `Address,Reference,same_person` | Pronouns the prompt forbids. Class **C**. |
| `You -> Toast` and `You -> HubSpot` | Same keyword string on both edges (`easy,higher value,simple product,start,stickier`). The description is generated text, not a source sentence. Class **F** for the copied payload. The underlying claim was not labeled. |
| `You -> Them`, `You -> The best`, `You -> The next place` | Endpoints are pronouns or discourse phrases. Class **C**. |
| Generic `Company -> Customer` rows in the accepted stream | The prompt lists both as nouns to omit. An edge between them is co-occurrence of generic words. Class **B** when the description only says that a company has customers. |

Class key: **A** supported and specific, **B** supported and generic, **C** weak or ambiguous, **D** unsupported, **E** wrong speaker, **F** generation or format artifact.

This sample is not a rate. It is large enough to show that endpoint names alone are a poor filter, and that a well-formed description can still be generic or copied. Deleting every generic edge would also delete useful context such as a named company discussed as "the customer" in a story. The safer cut is retrieval-time: skip edges whose endpoints are pronouns, weekdays, or bare discourse phrases, and keep the stored graph.

Co-occurrence without a predicate is common because the parser never checks the chunk. The existing `source-verified-relationship-v1` checker in `graph_relationships.py` already requires both names and a predicate word in one passage. That checker is the right gate for any repaired edge. It was not run as a corpus score here, because that would be a new labeling pass.

## 9. Numeric and temporal grounding

The project prompt says not to create an entity for a year, a percentage, or a currency amount, and to put a number in a description only when the input states it. The graph still has node names such as `2025`, `2022`, `23%`, `$150m`, `$100/month`, and `$1m+ ARR`. A name scan found **840** nodes whose names contain a year, a percent, or a currency-like token. That count includes real titles such as `SaaStr Annual 2024`. It is an upper bound on numeric nodes, not a count of bare numbers.

A bare node such as `23%` or `2022` can be retrieved without a company, a metric, a denominator, or a period unless those facts happen to be in the node description. Several nodes put the context in the name itself (`110% net rev retention from SMBs`, `50% customers from word-of-mouth`). Those are in better shape. Standalone year and percent nodes are not.

`numeric_fact_gaps` flags a numeric name whose description lacks a year, a unit, or a capitalized name. It is a heuristic for review, not a correction. Historical numbers were not rewritten.

Retrieval safety that does not move the graph: when a hit's entity name is only a year, a percent, or a currency amount, require the returned description or the same source chunk to name an organization and a metric. Otherwise withhold that node and return the chunk. Claim-level checks belong in the existing source-verified relationship path, not in another graph hop.

## 10. Entity normalization

Observed as separate nodes:

| Pair | Safe automatic merge now? |
|---|---|
| `HubSpot` / `Hubspot` | Case fold only, at retrieval |
| `ServiceNow` / `SeviceNow` | Typo. Retrieval alias after one confirmed pair. Do not rewrite the node in this session. |
| `Salesforce` / `@salesforce`, `Okta` / `OKTA`, `MongoDB` / `@MongoDB` | Punctuation fold at retrieval |
| `Mongo` / `MongoDB` | Not safe. Distinct mentions exist. |
| `SDR` / `Sales Development Representative` | Plausible alias. Context can also mean a different expansion. Keep a reviewed list. |
| `AE` / `Account Executive` | Same constraint. `AE` is short and collides with other uses. |

Do this at retrieval. Extraction-time canonicalization would change future documents and the fingerprint if it is added to entity guidance. Ingestion-time merges would rewrite vectors and the graph together; that is a migration, not a filter. No nodes were merged.

## 11. Provenance limits

Two different caps show up in logs.

**Chunk ids, cap 200, method KEEP.** After an entity or relation already has 200 source chunks, a new chunk does not update the stored summary. The log is `Skipped SaaStr: KEEP old chunks ... (limit 200)`. The full id list is written to the chunk-tracking store before that skip. The graph node's summary and its capped `source_id` stay on the older chunks. Later blog posts about a frequent entity such as SaaStr or Salesforce can be embedded as chunks and still be missing from that entity's summary. Query paths that trust the summary under-represent newer documents. Query paths that search chunk vectors can still find them.

**File paths, cap 75.** `Limited Salesforce: file_path 76+ -> 75 (KEEP)` truncates the display list of file paths on the node. The library comment marks this field as display metadata. It is not evidence that the chunk text was deleted.

Raising either cap in the running process would change merge behavior for the rest of the corpus and force more summary rewrites. Leave the caps. A separate provenance ledger is justified for the 200-chunk summary skip: the chunk-tracking rows already hold a longer id list than the graph node. A repair can read that list. It should not expand the summary cap in place.

## 12. Extraction-to-storage accounting

| Stage | What changes the count |
|---|---|
| Raw model rows | Includes entities, relations, gleaning repeats, and in-response duplicates |
| Parsed relations | Non-relation lines are ignored. Field count must be five. Empty descriptions are dropped. |
| Validated by the parser | `source == target` is dropped. Names are normalized. |
| Deduplicated | The same unordered pair in one chunk becomes one merge candidate with a list of descriptions |
| Merge candidate | A new graph edge, or an update to an existing edge. Weight stays 1.0 for these text rows. Descriptions are concatenated with `<SEP>`. |
| Persisted edge | One GraphML edge per pair. An upsert does not create a second edge. |
| Relationship vector | Updated when the merged description changes. A skipped KEEP summary does not refresh the vector. |

Extracted rows outnumber merge candidates because of duplicates, gleaning repeats, malformed drops, and self-relations. Merge candidates outnumber new edges because many pairs already exist. Nothing in this accounting shows an accepted five-field row being discarded after validation, apart from the KEEP skip once a pair is already at the chunk cap, and apart from a cancelled merge that the ledger records as interrupted or quarantined.

## 13. Async cancellation

The message `Task exception was never retrieved` with `PipelineCancelledException` matches the chunk-extraction wait in `operate.py`. That wait cancels sibling tasks and then `asyncio.wait`s them. `wait` does not retrieve a finished task's exception. `PipelineCancelledException` is a normal `Exception`, so the event loop logs it when the task is destroyed. `CancelledError` is not logged the same way. The merge path does retrieve sibling exceptions through `wait_tasks_with_drain`.

The project wrapper `abortable` owns the inner generation task, cancels it, and awaits it. `close` drains in-flight generation, then finalizes storages. Embeddings are left to finish so a flush is not cancelled. The document supervisor awaits the pipeline task. Another unfinished document is parked so the next ingest does not resume it. A document whose graph merge has started is not generation-cancelled; the ledger records mutation and does not treat that document as a clean retry.

Classification: the seven unretrieved-task lines in the captured log are a **cleanup defect** in the chunk-wait path. They are not, by themselves, a demonstrated second writer. The consistency risk that remains is a merge that started before cancellation. That case is already represented in the ledger as an interrupted or quarantined document, and this audit did not try to reproduce it against the live store.

`retrieve_task_exceptions` in `indexing_extraction_audit.py` gathers tasks with `return_exceptions=True`. The regression test uses that helper on isolated tasks. It does not call Ollama and does not open the expansion store. The installed package was not patched, so the live process still has the chunk-wait behavior.

## 14. Candidate repairs

| Strategy | Cause | Benefit | False-edge risk | GPU | New embeddings | Replay completed docs | Fingerprint | Effort |
|---|---|---|---|---|---|---|---|---|
| A. Prompt rewrite | Model ignores the five-field contract and the omit-list | Fewer bad rows on future documents | Lower if the model complies. Unmeasured. | Yes, for every new document | Yes, for those documents | No, old cache stays | Changes if guidance text changes. Resume of this store would refuse. | Medium |
| B. Conservative parse of cached rows | Schema labels, comma packing, omitted keywords | Recovers a defined subset with no extraction | High if tails are joined arbitrarily. Low if limited to the two patterns in section 6 and gated by the chunk. | No extraction | Only for edges that are actually inserted | No | Unchanged if it stays outside the indexer | Small |
| C. JSON schema extraction | Free text is brittle | Stronger future contract | New failure mode if JSON is partial | Yes | Yes | Would be a new store | New configuration | Large |
| D. Gleaning change | Gleaning repeats bad rows | Saves one call per chunk on future work | Might drop the rows gleaning does add. 3,918 gleaning responses exist. No A/B was run. | Changes future cost | No for old docs | No | `entity_extract_max_gleaning` is in the fingerprint | Small, blocked on this store |
| E. Alias table at retrieval | Case, `@`, and one typo | Better recall of the same name | `Mongo`/`MongoDB` and short acronyms can collapse distinct mentions | No | No | No | Unchanged | Small if the table stays short |
| F. Quality filter at retrieval | Weak and generic edges | Less misleading context | Can hide a useful generic mention | No | No | No | Unchanged | Small |
| G. Cache repair into the graph | Dropped rows in section 6 | Adds missing edges | Any repair that skips the chunk check | Embeddings only | Yes, for new relationship vectors | No re-extraction | Repair policy must be versioned beside the fingerprint | Medium, after a snapshot |
| H. Retrieval safeguards only | Weak nodes and bare numbers | Improves answers while the graph stays | None for stored data | No | No | No | Unchanged | Small |

A parser fix does not restore historical edges by itself. The rows are already absent from the graph. Restoration is a separate insert.

## 15. Offline tests

`tests/test_indexing_extraction_audit.py`, **7 passed**. Isolated fixtures cover omitted keywords, keyword-only refusal, schema-label removal, refusal to join the ZIRP-style tail, self-relations, gleaning repeats, alias grouping, numeric gap flags, and retrieval of a cancellation exception after `asyncio.wait`. No live extraction, no graph write, no site-package edit.

The fixture set is the evaluation set. It has known field shapes. It does not assign truth labels to corpus edges, so corpus precision and recall are not reported.

## 16. GPU and runtime

This audit used no GPU. A later cache repair that only rewrites fields uses no extraction GPU. Inserting an accepted repaired edge updates the relationship vector and needs `nomic-embed-text` for that description. That cost is one embedding per inserted description, not a document re-extraction. A prompt change applies only to documents indexed after the change, and it cannot be applied to this store without a new fingerprint. A clean reindex of the blogs would repeat the entire extraction and embedding cost. It is not required to recover the schema-label and comma-packed rows.

## 17. Rollout

1. Let the current blog-only process finish or stop at its configured limit. Do not restart it with a new parser.
2. After the lock is free, snapshot `storage_expand_v1`.
3. Dry-run the repair ledger from the cache. Keep schema-label, comma-split, and omitted-keyword repairs. Drop keyword-only rows and unknown tails. Require both endpoint strings in the chunk. Run `validate_relationships` before any insert.
4. Write a versioned repair policy and an idempotent ledger. Insert only the edges that pass. Update relationship vectors for those edges. Roll back by restoring the snapshot.
5. Add retrieval filters for pronouns, weekdays, and bare numeric nodes. Add the small alias table. Do not merge stored nodes in the first rollout.
6. Keep gleaning at 1 and keep the entity guidance text until a separate store can compare them.

## 18. Limitations

The snapshot moved during the read because indexing continued. Counts are reproducible from the cache that was on disk at that moment, and the next checkpoint will be larger. Overlap with the chunk is not semantic truth. The 840 numeric-name count includes titles that contain a year. Graph node and edge counts are the file on disk, not a quiesced export. The seven unretrieved-task lines from the console capture were not re-counted from a file in this workspace. No repaired edge was inserted.

## Decision table

| Issue | Severity | Confirmed cause | Preferred fix | GPU | Historical repair | Priority |
|---|---|---|---|---|---|---|
| 6- and 7-field schema labels | High | Model echoes prompt field names; parser requires five fields | Offline repair, then chunk gate | Embeddings only if inserted | Yes, from cache | 1 |
| 4-field comma packing | High | Keywords and description share one field | Split on the first comma when the right side is a sentence | Same | Yes, from cache | 1 |
| Keyword-only 4-field rows | Medium | No description was generated | Leave dropped | No | No | 4 |
| ZIRP-style extra fields | Low | Unstructured split | Leave dropped | No | No | 5 |
| `You`, `Monday`, generic Company/Customer | High for retrieval | Model ignores the omit list; parser accepts five fields | Retrieval filter | No | Leave the nodes | 2 |
| Bare year and percent nodes | High for retrieval | Same omit-list miss | Withhold the node unless description or chunk has entity and metric | No | Do not rewrite numbers | 2 |
| `SeviceNow`, case and `@` | Medium | No alias layer | Retrieval aliases | No | Do not merge nodes yet | 3 |
| `Mongo`/`MongoDB`, `SDR`, `AE` | Medium | Ambiguous | Reviewed list only | No | No | 4 |
| Gleaning repeats malformed rows | Medium | Continuation copies the bad shape | Do not change gleaning on this store | Would change future calls | No | 5 |
| KEEP 200 summary skip | Medium | Library cap | Read chunk-tracking ids; do not raise the cap | No | Optional summary rebuild later | 4 |
| File-path cap 75 | Low | Display field | Leave it | No | No | 6 |
| Unretrieved `PipelineCancelledException` | Low | Chunk wait does not read sibling exceptions | Keep the drain helper; patch upstream only in a future dependency bump | No | No | 5 |

## Answers

1. The principal defect is a combination. The prompt and the parser agree. The model emits schema labels, packed fields, and omitted descriptions. The parser then drops the row. Weak accepted edges are a second defect: the parser checks shape only.

2. **3,949** unique malformed records in this snapshot. **32** of them already have an accepted row for the same pair in the same chunk.

3. **2,564 / 3,949 (65%)** have a deterministic field repair that needs no LLM call. **3,195 / 3,594** checked repaired rows also contain both names in the chunk. The rest of the 65% still needs that chunk gate, and none of it is a labeled semantic recall.

4. Yes. Pronoun edges, weekday edges, generic company/customer edges, and copied keyword strings can mislead retrieval. Some of those descriptions still restate something the chunk said to the reader.

5. Yes. Years, percents, and currency amounts exist as nodes. A bare node does not carry company, metric, and period unless the description or the chunk is returned with it.

6. Yes, after this session stops: the offline schema-label and comma repair, as a dry-run. Do not load it into the running process.

7. Existing chunk and entity vectors stay. A repaired edge that is actually inserted needs a relationship-vector update for that edge. Documents that already parsed cleanly do not need re-embedding.

8. Yes. Cache repair plus a versioned ledger can add edges idempotently after a snapshot. It cannot silently rewrite the meaning of edges that are already stored.

9. No. A clean reindex is a different experiment. It is not required to recover the rows already sitting in the cache.

10. Yes. Continue the blog-only session. The failures drop individual rows. They do not invalidate completed documents or the fingerprint.

11. Fix the schema-label and comma-packed rows first, offline, from cache, with a chunk gate.

12. After the current process exits, snapshot the store and run the dry-run repair ledger. Do not resume with a changed prompt, a changed gleaning count, or a live parser patch.

## Three fixes, by quality per hour

1. **Offline recovery of schema-label and comma-packed rows from the existing cache**, inserted only when both names occur in the chunk and the source-verified predicate check passes. No extraction GPU. The largest pile of missing claims.

2. **Retrieval filters for pronouns, weekdays, and bare numeric nodes**, plus the existing source-verified relationship gate. No GPU and no graph rewrite. This is the fastest protection against misleading context.

3. **A short retrieval alias table** for case, `@`, and `SeviceNow`/`ServiceNow`. Leave `Mongo`/`MongoDB` and the short role acronyms unmerged until each pair is reviewed.

## Implementation after the indexer stopped

The blog-only process exited with `stop_reason=sigint` after a completed flush: 13,401 nodes, 22,027 edges, 12 storages finalized, lock file absent. The last completed document in that session was `gd_6c080774239412140ab932ec`. Completed documents before the next resume keep the extraction that was stored for them.

`relation-normalization-v1` now rewrites schema-label rows and comma-packed rows in memory before `lightrag.operate._process_extraction_result`. The installed package and the cache file are not edited. The configuration fingerprint is unchanged. Sentence-only four-field rows and the eight-field ZIRP row stay rejected.

A dry-run over the cache after that checkpoint, using the same normalizer and the source-verified window rule, produced 2,344 unique repaired identities: 1,225 schema-label and 1,119 comma-packed. Of those, 416 are insert candidates, 1,537 need review, 215 lack an endpoint in the chunk, and 176 supported pairs are already edges. Repeating the dry-run reproduced ledger `3feeb06480fa557534b2227665b87a56abfa9134484ed6e3ec8b6a1957c03697`. No edge was inserted. The procedure is `docs/implementation/LIGHTRAG_RELATIONSHIP_REPAIR.md`.
