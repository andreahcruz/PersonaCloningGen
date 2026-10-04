# Repaired versus relabel, retrieval on

Question: do excerpts from `lemkin_train_only` change early stops or mid-sentence stops relative to EXP-20261004-003?

Same 30 topics, four mediums, budgets, temperature, top_p, repetition penalty, and seed base 42. `--retrieval` is on. Collection `lemkin_train_only`. `lemkin_content` is not used. Both adapters are scored. The cutoff winner from EXP-003 is relabel; this run still scores both.

## Result

Exit 0. 120 answers each. Retrieval enabled.

| Adapter | Early stop | Mid-sentence |
|---|---:|---:|
| repaired | 0.725 | 0.375 |
| relabel | 0.667 | 0.258 |

By medium, early stop then mid-sentence: repaired blog 0.967 / 0.133, LinkedIn 0.367 / 0.633, X 0.633 / 0.500, talk 0.933 / 0.233. Relabel blog 1.000 / 0.000, LinkedIn 0.267 / 0.600, X 0.467 / 0.433, talk 0.933 / 0.000.

This run does not replace the EXP-003 cutoff pick.
