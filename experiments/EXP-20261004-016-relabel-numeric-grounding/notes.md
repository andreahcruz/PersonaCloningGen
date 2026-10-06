# EXP-20261004-016 — numeric grounding sanity

Question: if deterministic checks replace the adapter-off verifier, and a separate numeric gate rejects drafts that add quantities outside the topic and the accepted claims, do all 8 Relabel drafts pass both deployment gates?

Held fixed: Factual Dense RAG v1, Relabel, the 12-word source-copy gate, one retry at seed + 10000.

Change: the LLM SUPPORTED/UNSUPPORTED verifier no longer decides what enters the prompt. A claim is kept only when its source id, span, numbers, and names check out, and question titles are not turned into assertions. The final closer is only the writing request. Unsupported quantities fail `unsupported_numeric_grounding`.

Result: not a go. Every attempt passed the source-copy gate. gold_001 LinkedIn no longer echoes the instruction closer. The 34% customer-count claim is now accepted. gold_002 blog, LinkedIn, and talk, and gold_001 blog, still introduced quantities that are not in the allowed set after the one retry. Those four cells are rejected. The full 120 was not started.
