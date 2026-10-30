# Pilot graph integrity, read only

Question: which stored chunks and edges should stay out of evidence without rewriting the finished graph?

The workdir `artifacts/lightrag/model_f/storage_pilot160` was not modified. No Ollama call and no indexing run.

Eight document ids are failed or skipped. Twenty-two stored chunks still belong to seven of those ids. The skipped 2,168-word blog has no stored chunks. No graph node or edge cites a failed document id. Naive and mix could previously return those leftover chunks. Query normalization now excludes them with reason `failed_extraction_chunk`.

Nineteen chunks contain an omission marker. Two hundred thirty-eight edges cite those chunks. A relationship drawn across the marker can join text the source did not keep together. Those chunks are now excluded at query time with reason `omission_marker_span`. The edges remain in the file.

Every stored chunk's `file_path` is in the catalog. One catalog document is a third-party quotation, `gd_043b769dd0d2fd156767e0ba`, the Anneka Gupta post. The weak-name rule flags 244 entity names: 17 generic, 219 numeric-name hits, 5 months, and 3 weekdays. The numeric rule matches any digit, including `B2B`. Entity descriptions are still not passed to the generator.

Query-time exclusions are `failed_extraction_chunk`, `omission_marker_span`, `generated_summary_not_a_source_span`, and `unknown_graph_document`.
