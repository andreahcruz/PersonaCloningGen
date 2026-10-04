# Repaired versus relabel, retrieval off

Question: after the label correction, does the new adapter stop early less often than the repaired adapter on the same 30 topics and four mediums?

Held fixed: one-line EXP-010 prompts, budgets blog 768 / LinkedIn 320 / X 160 / talk 768, temperature 0.7, top_p 0.9, repetition penalty 1.15, context 2048, seed base 42. Retrieval is off. `lemkin_train_only` is not queried. `lemkin_lora_repaired` is not overwritten.

Decision diagnostics are early `<|eot_id|>` before half of `max_new_tokens`, and the share of answers that are not sentence-final. BLEU, ROUGE, and the gold lexical overall are logged and do not choose the adapter. Blinded preference rows are written for a later human review. This run does not declare a voice winner.

## Result

Exit 0. 120 answers each. Retrieval off. Seed base 42.

| Adapter | Early stop | Mid-sentence |
|---|---:|---:|
| repaired | 0.650 | 0.633 |
| relabel | 0.550 | 0.342 |

By medium, early stop then mid-sentence: repaired blog 0.933 / 0.133, LinkedIn 0.100 / 0.733, X 0.567 / 0.800, talk 1.000 / 0.867. Relabel blog 0.567 / 0.033, LinkedIn 0.233 / 0.767, X 0.767 / 0.533, talk 0.633 / 0.033.

Cutoff rule: relabel wins because 0.550 is below 0.650 and 0.342 is not more than 5 percentage points worse than 0.633. This is a cutoff result only. No new fine-tune is started from this rule. Gold overall was 0.2713 repaired and is logged for relabel in `metrics/summary.json`; it does not decide.

Later retrieval-off seeds do not change this pick. Seed 43: repaired 0.608 / 0.600, relabel 0.550 / 0.300. Seed 44: repaired 0.617 / 0.642, relabel 0.600 / 0.325. Mean of the three 120-answer rates: repaired early 0.625 and mid-sentence 0.625; relabel early 0.567 and mid-sentence 0.322.
