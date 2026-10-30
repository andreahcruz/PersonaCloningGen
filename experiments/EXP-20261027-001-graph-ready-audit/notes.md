# Graph-ready audit

Question: can the frozen 21,193 train-only rows be turned into source documents for LightRAG without crossing splits or rewriting the Relabel file?

Held fixed: EXP-20261003-004 rows and assignments, EXP-20261003-003 gold-overlap dispositions, the gold eval file, and `data/cleaned/` as a read-only lookup. No Chroma client was opened. No model ran.

The selector matched the later RAG contract: 21,193 eligible rows, `train_26148` absent, 183 gold-overlap exclusions plus 1 headline exclusion. `docs/PROJECT_CONTEXT.md` still mentions the earlier 21,194 build. Both records stay.

Row accounting closed: 20,572 eligible rows are in included documents, 621 are in quarantine, and none are in both. 13,140 documents were kept. 11,917 of those are cleaned source bodies whose extra residual was at most 12 words. 948 are ordered joins of eligible chunks because the cleaned body contained more than that. 275 stayed single chunks.

No eligible train row shared a cleaned source line with a validation or test row. That is a measured count of zero, not a claim that semantic duplicates cannot exist across platforms. Deduplication here is exact normalized text only. 540 exact copies were dropped.

Blog documents in the derived set have median 576 words and 7 documents under 40 words. X documents have median 20 words and 9,266 under 40 words. Those X posts were kept on purpose. The October demo fragment rates were about the chunk index and were not recomputed.

All 2,042 included blogs are marked `not_verified` as personal statements. 538 LinkedIn rows match a Lemkin author field. 10,560 X rows match `jasonlk`. 25 other-author rows and 8 unlabeled transcripts were quarantined. 538 LinkedIn dates are relative strings and are flagged.

`raw/review_sample.jsonl` is nine rows read by the implementing agent. It is not a human relevance grade. The long AI-agent blog and the VP Sales LinkedIn post are coherent. One blog join is still a stack of sections because the cleaned residual rule refused the rest of the page. A third-person LinkedIn recap and an unlabeled transcript were correctly left out. Short X posts with a real number were kept.

The 160-id prototype sample is seed 298 with a 40 percent medium cap: 64 blog, 64 X, 32 LinkedIn. It is not a random sample of the 13,140 documents.

Warnings: the git worktree was dirty. The 30 MB `documents.jsonl` and the quarantine file stay in `artifacts/lightrag/model_f/graph_ready/` and are not part of this directory. Their SHA-256 values are in `manifest.yaml`.
