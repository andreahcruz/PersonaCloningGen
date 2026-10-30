# Three-document LightRAG smoke

Question: can local LightRAG 1.5.7 build a persistent graph from train-only graph-ready documents and answer in `naive`, `local`, `global`, `hybrid`, and `mix` without writing the final article?

Held fixed: the graph-ready file from EXP-20261027-001, seed 298, a 40 percent medium cap, evidence budget 4, one chunk per source, `llama3.1:latest` for extraction and keywords, `nomic-embed-text` at 768 dimensions. The Relabel adapter was not loaded. Chroma was not opened.

The command indexed one blog, one LinkedIn post, and one X post, then queried once. Exit code 0. Wall clock 118.4 seconds. The saved graph has 29 nodes and 29 edges. Querying reloaded that graph, so the storage survived a process restart.

All five modes returned success and the same three train-only sources. That is not evidence the modes retrieve different documents. The index only had those three. Before the source cap, the library logs were different: naive used 4 chunks, global kept 3 chunks with 16 entities and 20 relations, and local, hybrid, and mix reported about 20 entities and 27 relations. Mix also ran the vector chunk search. Entity and relationship summaries were excluded from the evidence hits.

The short X post was extracted into edges whose names include `Tuesday` and `93%-95% Correct`. Those strings are in the source sentence. They are poor entity names. They did not become citable spans.

The LinkedIn document is filed under a Lemkin author field and quotes Anneka Gupta. The current quote rule does not remove that sentence.

Per-query latency was not written. The code now records it for a later run. No precision, recall, or generation score is claimed. The 160-document index and the full corpus index were not started.
