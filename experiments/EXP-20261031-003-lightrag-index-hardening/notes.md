# EXP-20261031-003 — index quality and recovery

Question: can the existing `storage_expand_v1` graph keep being resumed without treating a technical commit as a clean graph, and without letting a failed or timed-out document back into the next ingest?

Held fixed: graph-ready v2 text, configuration fingerprint `771baa30dd8f79c30c4c51f296385b2ec2c652b33ff1031a099a7e16f139260b`, the pilot store, Chroma, the Relabel adapter, and `rag_active.json`. Segmented text was written beside the corpus and was not inserted.

The first `--max-documents 20` command completed `gd_087a4675a11cf57ae08ea7cf` and `gd_00808ccaeb7073dfcf899559`, then the 420-second limit cancelled `gd_4c65a0a559794e0816949bf4`. LightRAG put that document back to pending, and the next call processed it together with `gd_0671f71cf02d19206bacd8c5`. That process was stopped after the graph had been flushed at 84 nodes and 55 edges. The follow-up command was capped at 18 documents so the two runs together stayed within 20 new commits. It held `gd_4c65a0a559794e0816949bf4` out, processed one document at a time, and finished with `stop_reason=max_documents`.

Measured after shutdown: 25 completed graph commits, 266 nodes, 217 edges, 21 completed documents with relations, 4 completed documents with entities and no relations, 0 anchor conflicts, 0 duplicate ledger, graph-node, or chunk-vector ids. Two chunk vectors for `gd_002be28d89a6ca9924e73be2` have no text-chunk row. Both carry that document id, and the query filter's uncovered-failed-vector list is empty. The vectors were not deleted.

`gd_4c65a0a559794e0816949bf4` is `interrupted` in the ledger with `defer_reason=expensive_timeout`, and `failed` in LightRAG with `kg_write_state` still `pre_graph`. It is not in the default queue. `gd_002be28d89a6ca9924e73be2` remains `failed_permanent`.

The ledger's malformed-relation counter was attached to the root logger during these sessions. LightRAG does not propagate to that logger, so a stored zero is not a measured zero. The session logs show relation-format warnings for five documents: `gd_087a4675a11cf57ae08ea7cf` (10 warnings, 0 relations), `gd_0019c5e0dad1cf6315645b67` (9, 0), `gd_003275b6850776457e113392` (8, 0), `gd_14530da4ea39350d8431fea6` (11 warnings, 8 relations kept), and `gd_00f0fd1277363a92e5cb2ada` (30 warnings, 19 relations kept). `gd_0e1afddaca819c17829b209e` was indexed before the counter existed; its zero relations stay `no_relations_committed_cause_not_recorded`. Those entity-only documents stay completed.

Omission segmentation of graph-ready v2 produced 13,583 segments from 13,140 documents: 12,785 unchanged ids and 798 segments from 355 marked documents. That file is `graph_ready_v2_segments`. It was not indexed.

The next default document is `gd_0f321f128e2fb24360869510`, a 60-word LinkedIn post. Mean completed runtime is 11.9 seconds. The approximate remaining compute for 13,105 queued documents is 156,507 seconds. That estimate excludes deferred documents and attempts that did not complete. The eight-hour session was not started.

Pytest: 20 passed, then the warning-counter correction passed with the checkpoint tests (10 passed in that later run).
