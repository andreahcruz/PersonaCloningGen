# EXP-20261004-015 — atomic evidence sanity

Question: can adapter-off extraction plus a source-span check and a second adapter-off support check give Relabel only verified claims, without the final draft copying 12 words of the original passages?

Held fixed: Factual Dense RAG v1 (`lemkin_train_only`, k=4, nomic-embed-text, topic-only query). Relabel. Final copy gate still compares drafts with the original retrieved texts, threshold 12, one retry at seed + 10000.

Evidence stage: greedy extraction, max 1536 new tokens, seed 0, adapter off. Each validated claim is checked with the same base model, max 24 new tokens. Support spans stay in provenance. The internal 12-word rule is not applied to claims.

Result: all 8 drafts passed the final copy gate on the first attempt. gold_001 kept one title fact and dropped three question-to-assertion claims. gold_002 kept six numeric claims and dropped six others. The verifier also dropped "Customer count is up 34%" even though that span states it. Final Monday.com drafts did not paste `train_10049`, but LinkedIn and X still introduced figures that were not in the verified set. gold_001 LinkedIn repeats the instruction closer. The full 120 was not started.

EXP-20261004-014 is the earlier attempt. Its gold_002 JSON was cut off at 768 tokens, and gold_001 had zero verified claims.
