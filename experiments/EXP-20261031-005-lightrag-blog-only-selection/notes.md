# Blog-only expansion queue

The expansion already contains LinkedIn and X documents. Further indexing is scheduled from Jason Lemkin blogs only. That choice reduces the remaining queue. It does not show that blogs contain every fact from the other media.

The graph stays a mixed-media seed with blog-only expansion. Completed non-blog documents are not deleted or reprocessed. Pending LinkedIn and X documents stay in the ledger and drop out of the blog-only schedule. `selection_manifest.jsonl` is unchanged.

`--selection-policy blog-only` filters `next_document` by `source_medium`. It does not flip `selected`. The configuration fingerprint is unchanged. Adaptive timeouts are unchanged. Deferred blogs are not retried unless `--retry-timeouts` is passed. The one permanent blog failure stays failed.

A live three-document check was not run. PIDs 19396 and 23320 were still writing the all-media queue. The manifest was exported through a read-only SQLite connection. Its status fields are a snapshot and can lag the ledger by the documents the running job finishes after the export. The identity hash does not include status.

Do not start the blog-only resume until that process has stopped.
