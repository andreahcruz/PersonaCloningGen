# Proposed relevance judge

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
