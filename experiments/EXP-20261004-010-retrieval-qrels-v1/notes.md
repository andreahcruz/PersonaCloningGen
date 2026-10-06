# EXP-20261004-010 — initial Dense RAG v1 qrel pool

Question: which train-owned documents does frozen Dense RAG v1 place in the top 20, and which of those should a reviewer grade before any retrieval architecture changes?

The generator was not loaded. No relevance grade was assigned. Ranked IR metrics are omitted.

## Pool

- Queries: 30
- Raw top-20 rows: 600
- Judgment units after collapsing same-group hits within a query: 499
- Unique document ids: 574
- Unique group ids: 442
- Collection count checked live: 21193
- `train_26148` present: False
- Chroma space: cosine
- Frozen baseline left unchanged: EXP-20261004-009-dense-rag-v1

Existing evidence-review sheets have 0 non-empty `on_topic` values and 0 non-empty `claim_support` values across 992 rows. Those sheets are not qrels.

Of the 600 raw hits, 297 are continuation rows, 42 LinkedIn hits have no URL because `jasonlemkinlinkedin.jsonl` does not store one, and none are flagged as held-out topic overlap. YouTube does not appear in the top 20.

## Diagnostics without grades

Top 20 over 30 queries:

- empty retrieval rate: 0.0
- duplicate document rate: 0.0
- duplicate group rate: 0.9666666666666667
- mean cosine distance: 0.27315787543853126
- median cosine distance: 0.2674688398838043
- fragment rate: 0.20666666666666667
- headline-only rate: 0.08166666666666667
- short-chunk rate: 0.23
- platform counts: {'x': 149, 'blog': 409, 'linkedin': 42}
- mean distinct groups at 4 / 10 / 20: 3.533333333333333 / 8.466666666666667 / 16.633333333333333
- mean extra same-group fraction at 4 / 10 / 20: 0.11666666666666667 / 0.15333333333333338 / 0.16833333333333333
- mean source diversity at 4 / 10 / 20: 2.033333333333333 / 2.4 / 2.566666666666667

## Dense RAG v2 chunking proposal, not built

Keep the train-only boundary, gold-overlap exclusions, `nomic-embed-text`, cosine search, topic-only query, `retrieval_k` 4, and no reranker. Change only the retrieval unit.

Do not use a fixed 500-word window. The frozen audit already shows a median row of 41 words and a p90 of 263, with 37.3% continuation rows and groups as large as 52 chunks.

1. Reconstruct each `group_id` in source order. Never merge two groups.
2. If the reconstructed post is 80 words or fewer, keep it as one chunk. That covers most X posts and short LinkedIn posts without splitting them.
3. If it is longer, split on headings and numbered items. A piece under 40 words is attached to the neighboring piece from the same post instead of being emitted as a `#7` fragment.
4. If a section still exceeds 180 words, split it on paragraph boundaries with a 30-word overlap. Use that overlap only inside a forced split, not between numbered items.
5. Chunk id is the SHA-256 of `group_id`, start character, and end character. Metadata keeps `group_id`, medium, the covered `row_id`s, split, and the character span.

180 words sits between the current median and p90, so a typical blog section stays intact and a long post becomes a handful of chunks rather than dozens of SFT rows.

## Shared pool

The v1 top 20 is the start of the pool, not the relevance universe. A document neither system retrieved is unlabeled, not irrelevant. See README.md for the union rule. The next human step is to grade `qrels_candidates.jsonl` and append accepted rows to `qrels_judgments.jsonl`.
