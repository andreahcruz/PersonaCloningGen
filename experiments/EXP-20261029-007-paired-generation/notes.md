# Model E Chroma generation and unified scores

Question: on the same three topics and four media as EXP-20261029-006, what does full-corpus Chroma v2 produce with the frozen Relabel adapter, and how do the unified gates compare with the pilot arms?

Held fixed: `rag_active.json` version v2, collection `lemkin_train_only_v2_shadow` (21,193 documents), the Relabel adapter, seed 42, evidence budget 4, and the three topics already generated for pilot dense, LightRAG naive, and LightRAG mix. This run did not open the LightRAG workdir, did not start an index, and did not retrain the adapter. Gold expected facts were not in the prompts. The writing judge was not run.

## Chroma retrieval

The frozen payloads are in `raw/frozen_chroma_v2.jsonl`. They were written before the generator loaded.

The Brex query's second hit, `train_985`, states that Capital One announced it is acquiring Brex for $5.15 billion. The evidence stage then returned `unparseable` and skipped all four media for that topic. The same skip happened on the immediately preceding attempt, whose traces were not saved because the writer crashed on a Windows path. The saved file therefore has one rejected Brex row and eight drafts. `train_985` shows the price was retrieved. The missing draft is an evidence-parse failure, not an empty retrieval.

The speaker query's first hit, `train_12126`, quotes Anneka Gupta, Rubrik CPO, including the sentence about a personalized SaaS experience. The blog and talk still say the speaker was the narrator ("I did"). Neither draft names Anneka Gupta.

The VP-of-Sales query retrieved `train_22551`, the short line that criticizing a VP of Sales who cannot do better is counterproductive. The blog answers one side of the tension: a great VP generally brings in one or two top performers. The LinkedIn draft is the question repeated.

## Unified gates

`config/score_traces.py` scored the saved traces with `unified_eval`. The copying gate is an 8-word index over training text and fails a shared span of 12 or more words. No row failed it. The empty Brex skip also stays under that threshold because it has no 12-word span. The gate is not a claim that a draft is original in any broader sense. Writing quality is `judge_not_run`, so persona utility is null. Style distance is reported only for rows that pass every gate. Those eligible sets are not the same media mix, so the distances are not a ranking and are not an authorship score.

| Arm | Rows | Eligible | Completion | Format | Grounding | Topic echoes | Names Anneka | Contains 5.15 |
|---|---|---|---|---|---|---|---|---|
| Pilot dense | 12 | 5 | 0.9167 | 0.8333 | 0.5833 | 4 | 0 | 3 |
| Naive, parent ids | 12 | 3 | 0.9167 | 0.8333 | 0.4167 | 4 | 0 | 2 |
| Mix, parent ids | 12 | 2 | 0.9167 | 0.8333 | 0.3333 | 4 | 0 | 2 |
| Chroma v2 | 9 | 2 | 0.7778 | 0.7778 | 0.4444 | 3 | 0 | 0 |

Naive and mix produced identical text for the Brex topic and the VP-of-Sales topic in every medium. They differed on the speaker blog and talk. LinkedIn echoed the question in every arm, including Chroma. That is a generator behavior. The format gate misses an X echo when the word count is still inside the X band.

The chunk-id LightRAG run in EXP-20261029-006 accepted zero claims and is not this comparison. Its Brex blog stated $5.3 billion. After parent ids, accepted claims restored the $5.15 billion figure and numeric grounding still flagged other dollar amounts.

## What this does not show

LightRAG did not improve author voice on this set. It also did not keep the quoted speaker once the claim extractor had dropped the name. Full-corpus Chroma retrieved both the Brex price and the Anneka quotation, and the drafts that were written still failed those two checks. No precision or recall is claimed. The pilot graph was not expanded.
