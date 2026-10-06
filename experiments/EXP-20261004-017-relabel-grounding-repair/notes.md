# EXP-20261004-017 — grounding repair sanity

Question: if a Relabel draft fails numeric grounding, can one adapter-off edit of that same draft remove the unsupported quantities without a new Relabel sample?

Held fixed: Factual Dense RAG v1, atomic evidence, deterministic evidence checks, Relabel sampling, the 12-word source-copy gate, and the numeric gate. Numeric failure no longer starts a seed retry.

Change: one greedy repair on the same 4-bit Llama 3.1 with the Relabel adapter off. The repair sees the topic, the draft, the accepted claims, and the unsupported quantity surfaces. Both gates run again. A failed repair is rejected. Raw Relabel text is kept separate from the deployment text.

Result: not a go. All eight source-copy checks passed, and no cell was repaired more than once. Six deployment outputs passed numeric grounding. gold_001 blog still contained `$100m+`. gold_002 blog still contained `$500M`, `$10k`, `125%`, `120%`, and `100`. gold_002 talk passed both gates but appended an editor's note. The full 120 was not started.

The four-cell base diagnostic is not a deployment result. With the adapter off and the same evidence prompt, those four base draws had no unsupported quantities.
