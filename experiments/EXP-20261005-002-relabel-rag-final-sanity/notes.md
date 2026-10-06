# EXP-20261005-002 — final architecture sanity

Question: does full-strength Relabel plus frozen factual RAG and atomic evidence generate eight drafts without an engineering failure?

Held fixed: lemkin_train_only, k=4, topic-only query, atomic evidence, and the trained adapter scale. Unsupported numbers are recorded and do not reject, retry, or repair a draft. A source-copy failure may retry once.

Result: the pipeline ran. Eight drafts were saved, retrieval stayed train-only, raw passages stayed out of the prompts, and no source-copy flag fired. Six drafts carry an unsupported-quantity diagnostic. That is recorded, not treated as a failure. The full 120 was started after this check.
