# EXP-20261004-009 — Factual Dense RAG v1, retrieval only

Question: what does the frozen dense retriever return, and what is wrong with the chunks, before any relevance labels exist?

The generator was not loaded. No relevance grades were assigned. Precision, MRR, nDCG, and Recall are omitted on purpose.

## Retriever held fixed

Factual Dense RAG v1. Collection `lemkin_train_only`, 21,193 documents, all `split=train`, `train_26148` absent. Embedder `nomic-embed-text` on CPU. Chroma space cosine. Query is the gold topic string. `retrieval_k=4`. No medium filter. No reranker. Each document is one relabel SFT train-row output, with metadata `group_id`, `medium`, `row_id`, and `split`.

## What this run can say

Unjudged diagnostics over all 30 gold topics, top 4:

- empty retrieval rate 0
- duplicate document-id rate 0
- 11 of 30 queries returned two chunks from the same `group_id`
- mean cosine distance 0.243, median 0.238
- top-4 slots: blog 69, X 39, LinkedIn 12, YouTube 0
- only `gold_001` was all X

X is 11,185 of 21,193 documents, but it does not dominate these top-4 lists. Blog does.

## Corpus counts

| Issue | Count | Share of 21,193 |
| --- | ---: | ---: |
| very short, under 20 words | 5,141 | 24.3% |
| under 40 words | 10,427 | 49.2% |
| starts as `#N` or `N.` | 1,525 | 7.2% |
| headline or link, under 30 words with a URL | 4,333 | 20.4% |
| no sentence-ending punctuation | 7,036 | 33.2% |
| continuation rows | 7,895 | 37.3% |
| documents sharing a group with another chunk | 11,137 | 52.6% |

Word counts: p10 15, median 41, p90 263. Largest group has 52 chunks. YouTube transcripts: 40 documents.

`gold_002` still returns Monday.com evidence, including the mid-list chunk `train_9156` (`#7. Monday.com...`). `gold_001` still returns loosely related X posts at distances about 0.31, which is expected while the social-contract article stays held out.

## Qrels status

No query-to-document relevance file exists. Gold `expected_facts` and `gold_reference` are generation rubrics. Evidence-review sheets leave `on_topic` and `claim_support` blank. Those cannot support Recall@k.

## Proposed pooled qrels, not yet labeled

1. Use the 30 held-out gold topics. Do not put the held-out source back into the index.
2. Retrieve top 20 from this frozen dense configuration.
3. Collapse candidates that share a `group_id` before judging, and keep one representative chunk plus the sibling ids.
4. Grade the remaining candidates 0 irrelevant, 1 tangential, 2 useful topical evidence, 3 direct support for an important claim.
5. Store grades separately from any generation output, with judge model, prompt, candidate text, score, and human-review status if a model assists.
6. Treat the result as judged-pool qrels. Report pool Precision, MRR, nDCG, and pool recall. Do not call that corpus Recall until a later pass argues the pool is complete enough.

## Failures

None in the scan or the 30 CPU queries. Ranked IR metrics were not computed.
