# Comparison: Relabel + RAG seed 43 against PersonaRAG

Every difference is B minus A. Intervals are 95 percent, resampled by topic. p is a sign-flip test by topic. `p adj` corrects for how many tests were run (Holm). The four primary rows, which are the four parts of Persona Utility, are corrected together. Every other row is exploratory and is corrected as its own, larger family.

Paired drafts: 120 (only in PersonaRAG: 0, only in Relabel + RAG seed 43: 0).

## Gates

| Run | Drafts | Eligible (all gates) | Pass common gates | Completion fails | Copying fails | Format fails | Grounding fails |
|---|---|---|---|---|---|---|---|
| PersonaRAG | 120 | 57 (48%) | 71 (59%) | 49 | 0 | 0 | 20 of 120 |
| Relabel + RAG seed 43 | 120 | 72 (60%) | 82 (68%) | 36 | 1 | 3 | 16 of 120 |

Grounding is counted only where it applies. A run with retrieval has one extra gate, so its eligible count is not comparable to a run without retrieval. Use the common gates column.

Style distance role by medium (selector means it may rank models): blog selector, linkedin diagnostic_only, x diagnostic_only, talk diagnostic_only

## All drafts (no draft dropped)

Pairs in this view: 120

| Metric | Scope | n | PersonaRAG mean | Relabel + RAG seed 43 mean | Diff | 95% interval | p | p adj | Family | Verdict |
|---|---|---|---|---|---|---|---|---|---|---|
| grammar_lt_rate (lower is better) | all | 120 | 0.021 | 0.031 | 0.010 | -0.018 to 0.043 | 0.563 | 1.000 | exploratory | no clear difference |
| grammar_lt_rate (lower is better) | blog | 30 | 0.009 | 0.023 | 0.014 | 0.000 to 0.040 | 0.503 | 1.000 | exploratory | no clear difference |
| grammar_lt_rate (lower is better) | linkedin | 30 | 0.019 | 0.000 | -0.019 | -0.058 to 0.000 | 1.000 | 1.000 | exploratory | no clear difference |
| grammar_lt_rate (lower is better) | x | 30 | 0.028 | 0.000 | -0.028 | -0.084 to 0.000 | 1.000 | 1.000 | exploratory | no clear difference |
| grammar_lt_rate (lower is better) | talk | 30 | 0.027 | 0.101 | 0.074 | -0.012 to 0.192 | 0.179 | 1.000 | exploratory | no clear difference |
| grammar_lt_deviation (lower is better) | all | 90 | -0.205 | -0.216 | -0.011 | -0.033 to 0.002 | 0.500 | 1.000 | exploratory | no clear difference |
| grammar_lt_deviation (lower is better) | blog | 30 | -0.100 | -0.086 | 0.014 | 0.000 to 0.040 | 0.503 | 1.000 | exploratory | no clear difference |
| grammar_lt_deviation (lower is better) | linkedin | 30 | -0.111 | -0.130 | -0.019 | -0.058 to 0.000 | 1.000 | 1.000 | exploratory | no clear difference |
| grammar_lt_deviation (lower is better) | x | 30 | -0.404 | -0.432 | -0.028 | -0.084 to 0.000 | 1.000 | 1.000 | exploratory | no clear difference |

## Drafts that pass the common gates in both runs

Pairs in this view: 57

| Metric | Scope | n | PersonaRAG mean | Relabel + RAG seed 43 mean | Diff | 95% interval | p | p adj | Family | Verdict |
|---|---|---|---|---|---|---|---|---|---|---|
| grammar_lt_rate (lower is better) | all | 57 | 0.014 | 0.042 | 0.028 | 0.005 to 0.058 | 0.032 | 0.160 | exploratory | B worse (suggestive) |
| grammar_lt_rate (lower is better) | blog | 26 | 0.010 | 0.027 | 0.016 | 0.000 to 0.045 | 0.498 | 1.000 | exploratory | no clear difference |
| grammar_lt_rate (lower is better) | linkedin | 1 | 0.000 | 0.000 | 0.000 | n/a | n/a | n/a | exploratory | too few pairs |
| grammar_lt_rate (lower is better) | x | 3 | 0.000 | 0.000 | 0.000 | n/a | n/a | n/a | exploratory | too few pairs |
| grammar_lt_rate (lower is better) | talk | 27 | 0.020 | 0.062 | 0.042 | -0.010 to 0.111 | 0.191 | 0.765 | exploratory | no clear difference |
| grammar_lt_deviation (lower is better) | all | 30 | -0.133 | -0.119 | 0.014 | 0.000 to 0.038 | 0.498 | 1.000 | exploratory | no clear difference |
| grammar_lt_deviation (lower is better) | blog | 26 | -0.098 | -0.082 | 0.016 | 0.000 to 0.045 | 0.498 | 1.000 | exploratory | no clear difference |
| grammar_lt_deviation (lower is better) | linkedin | 1 | -0.130 | -0.130 | 0.000 | n/a | n/a | n/a | exploratory | too few pairs |
| grammar_lt_deviation (lower is better) | x | 3 | -0.432 | -0.432 | 0.000 | n/a | n/a | n/a | exploratory | too few pairs |

How to read the verdict: a difference is called clear only when the interval excludes zero and `p adj` is below 0.05. If it looks real on its own (interval excludes zero, p below 0.05) but `p adj` is not below 0.05, it is marked (suggestive): a lead worth a second run, not a finding. Rows by medium are exploratory unless listed as primary.

Limits: one run per arm unless more seeds are added, and the 30 gold topics are a development panel. A result that is not clear here is not proof of no difference.
