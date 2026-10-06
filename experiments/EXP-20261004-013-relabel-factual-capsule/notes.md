# EXP-20261004-013 — evidence capsule sanity

Question: can one adapter-off distillation turn the frozen top-4 passages into a neutral evidence capsule that Relabel then uses, without the capsule or the final draft copying 12 contiguous source words?

Held fixed: Factual Dense RAG v1 (`lemkin_train_only`, k=4, nomic-embed-text, topic-only query, no medium filter). Relabel adapter. Generation temperature 0.7, top_p 0.9, repetition penalty 1.15, seed base 42. Final copy gate still compares the draft with the original retrieved texts. One retry, seed + 10000. No second model.

Change: one greedy distillation per topic with the Relabel adapter off (`do_sample=false`, max 384 new tokens, seed 0). All four media reuse that capsule. The final Relabel prompt does not contain the raw passages.

Result: stopped. gold_001 capsule overlap 9 (pass) and its four drafts passed on the first attempt. gold_002 capsule overlap 12 against `train_10774` (`nrr is likely around 85% for self service and 10 seat deals`). That capsule was not resampled, and gold_002 blog, LinkedIn, X, and talk were not generated. The full 120 was not started.

Additional warning: the gold_001 retrieved hits are short title stubs. The capsule elaborates claims that are not in those stubs (`value proposition`, `unique selling points`, `pain points`) and truncates before `train_25912`.
