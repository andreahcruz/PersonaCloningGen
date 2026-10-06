# EXP-20261004-010 retrieval qrels v1

Initial candidate pool for frozen Factual Dense RAG v1. No relevance grades are stored yet.

`raw/raw_top20_traces.jsonl` is every retrieved hit, including same-group duplicates.
`qrels_candidates.jsonl` is the labeling view: one row per `group_id` within a query, with sibling document ids kept on the row.
`qrels_judgments.jsonl` is append-only and is empty until a reviewer accepts a grade.

## How to add a judgment

Append one JSON object. Do not edit older lines.

```json
{"query_id":"gold_002","document_id":"train_10049","relevance_grade":3,"reviewer":"name","review_status":"accepted","candidate_text_hash":"...","source_config":"factual_dense_rag_v1","pool_origin":"dense_rag_v1","notes":""}
```

`review_status` must be `accepted` before `host_finetune/retrieval_eval.py` may treat the grade as a qrel. `proposed` rows stay out of ranked metrics.

## Expanding the pool later

When Dense RAG v2, a hybrid retriever, or a reranker exists:

1. Retrieve that system's top 20 for the same 30 topics.
2. Union its candidates with this pool.
3. Match on `query_id` plus `document_id` plus `candidate_text_hash` only when the retrieval unit is the same text.
4. Append candidates whose text hash is new. Do not delete v1 rows.
5. Judge only the new units.
6. Leave any still-unjudged candidate unlabeled. Do not mark it irrelevant because another system retrieved it, and do not mark a v2 chunk irrelevant because v1 did not retrieve it.

Same-group grade inheritance applies only inside one `pool_origin`. A v2 chunk can share a `group_id` with a judged v1 row and still needs its own judgment.

Pool Precision, MRR, nDCG, and pool recall are valid only for the judged portion of this expanding pool. They are not corpus Recall@k.
