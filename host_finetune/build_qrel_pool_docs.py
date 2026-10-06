"""Text artifacts for the qrel-pool experiment. No retrieval and no grades."""

SCHEMA = """{
  "relevance_grade": {
    "0": "irrelevant",
    "1": "tangentially related",
    "2": "useful topical evidence",
    "3": "directly supports an important claim, position, example, or fact for the query"
  },
  "quality_flags_are_not_grades": [
    "fragment",
    "continuation",
    "duplicate_of",
    "near_duplicate",
    "source_family_redundant",
    "headline_only",
    "URL_heavy",
    "insufficient_context",
    "held_out_overlap_suspected",
    "reviewer_notes"
  ],
  "review_status": ["unlabeled", "proposed", "accepted", "rejected"],
  "candidate_identity": ["query_id", "document_id", "pool_origin", "candidate_text_hash"],
  "judgment_row": [
    "query_id",
    "document_id",
    "relevance_grade",
    "reviewer",
    "review_status",
    "candidate_text_hash",
    "source_config",
    "pool_origin",
    "notes"
  ],
  "rules": [
    "A quality flag does not set or imply relevance_grade.",
    "A missing judgment is unlabeled. It is not grade 0.",
    "Append judgments. Do not overwrite an accepted row.",
    "Same-group siblings inherit a grade only inside one pool_origin, and only after the representative judgment is accepted. An explicit sibling judgment wins.",
    "A later pool_origin such as dense_rag_v2 does not inherit dense_rag_v1 grades, even when group_id matches.",
    "Judged-pool recall is not corpus recall."
  ]
}
"""

JUDGE_PROMPT = """# Proposed relevance judge

This prompt is stored only. It was not run.

Do not execute it on a GPU in the current phase. A CPU run still needs an explicit decision before it starts, because its grades would be proposals rather than accepted qrels.

## Model

Not selected. `qwen2.5:14b` is the project's writing-quality judge and was not used here.

## Status of any future output

`review_status` must be `proposed` until a person sets it to `accepted`.
Proposed grades are excluded from Precision, MRR, nDCG, and pool recall.

## Instruction

You are labeling retrieval evidence for a Jason Lemkin / SaaStr question.
You are not writing the answer.

Grade the candidate on this scale:

- 0 irrelevant
- 1 tangentially related
- 2 useful topical evidence
- 3 directly supports an important claim, position, example, or fact for the query

Ignore chunk-quality flags when choosing the grade. A fragment can be grade 3. A complete post can be grade 0.

Return JSON with keys `relevance_grade`, `notes`.
Do not invent citations that are not in the candidate.

## Inputs to preserve with every proposal

- judge model name
- this prompt
- query_id and topic
- document_id
- candidate text
- candidate_text_hash
- proposed grade
- review_status: proposed
"""

README = """# EXP-20261004-010 retrieval qrels v1

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
"""


def notes_for(body: dict, diagnostics: dict) -> str:
    return f"""# EXP-20261004-010 — initial Dense RAG v1 qrel pool

Question: which train-owned documents does frozen Dense RAG v1 place in the top 20, and which of those should a reviewer grade before any retrieval architecture changes?

The generator was not loaded. No relevance grade was assigned. Ranked IR metrics are omitted.

## Pool

- Queries: {body['n_queries']}
- Raw top-20 rows: {body['raw_candidate_rows']}
- Judgment units after collapsing same-group hits within a query: {body['judgment_units_after_group_dedup']}
- Unique document ids: {body['unique_document_ids']}
- Unique group ids: {body['unique_group_ids']}
- Collection count checked live: {body['document_count']}
- `train_26148` present: {body['train_26148_present']}
- Chroma space: {body['distance_metric']}
- Frozen baseline left unchanged: {body['frozen_baseline_experiment']}

Existing evidence-review sheets have {body['existing_labels']['nonempty_on_topic']} non-empty `on_topic` values and {body['existing_labels']['nonempty_claim_support']} non-empty `claim_support` values across {body['existing_labels']['evidence_review_rows']} rows. Those sheets are not qrels.

## Diagnostics without grades

Top 20 over {diagnostics['n_queries']} queries:

- empty retrieval rate: {diagnostics['empty_retrieval_rate']}
- duplicate document rate: {diagnostics['duplicate_document_rate']}
- duplicate group rate: {diagnostics['duplicate_group_rate']}
- mean cosine distance: {diagnostics['mean_cosine_distance']}
- median cosine distance: {diagnostics['median_cosine_distance']}
- fragment rate: {diagnostics['fragment_rate']}
- headline-only rate: {diagnostics['headline_only_rate']}
- short-chunk rate: {diagnostics['short_chunk_rate']}
- platform counts: {diagnostics['platform_counts']}
- mean distinct groups at 4 / 10 / 20: {diagnostics.get('mean_distinct_groups_at_4')} / {diagnostics.get('mean_distinct_groups_at_10')} / {diagnostics.get('mean_distinct_groups_at_20')}
- mean extra same-group fraction at 4 / 10 / 20: {diagnostics.get('mean_extra_same_group_fraction_at_4')} / {diagnostics.get('mean_extra_same_group_fraction_at_10')} / {diagnostics.get('mean_extra_same_group_fraction_at_20')}
- mean source diversity at 4 / 10 / 20: {diagnostics.get('mean_source_diversity_at_4')} / {diagnostics.get('mean_source_diversity_at_10')} / {diagnostics.get('mean_source_diversity_at_20')}

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
"""
