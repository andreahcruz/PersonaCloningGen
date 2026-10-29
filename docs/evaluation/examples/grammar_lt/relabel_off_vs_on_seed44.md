# Comparison: Relabel RAG on seed 44 against Relabel RAG off seed 44

Every difference is B minus A. Intervals are 95 percent, resampled by topic. p is a sign-flip test by topic. `p adj` corrects for how many tests were run (Holm). The four primary rows, which are the four parts of Persona Utility, are corrected together. Every other row is exploratory and is corrected as its own, larger family.

Paired drafts: 120 (only in Relabel RAG off seed 44: 0, only in Relabel RAG on seed 44: 0).

## Gates

| Run | Drafts | Eligible (all gates) | Pass common gates | Completion fails | Copying fails | Format fails | Grounding fails |
|---|---|---|---|---|---|---|---|
| Relabel RAG off seed 44 | 120 | 80 (67%) | 80 (67%) | 39 | 1 | 3 | 0 of 0 |
| Relabel RAG on seed 44 | 120 | 70 (58%) | 83 (69%) | 34 | 1 | 2 | 14 of 120 |

Grounding is counted only where it applies. A run with retrieval has one extra gate, so its eligible count is not comparable to a run without retrieval. Use the common gates column.

Style distance role by medium (selector means it may rank models): blog selector, linkedin diagnostic_only, x diagnostic_only, talk diagnostic_only

## All drafts (no draft dropped)

Pairs in this view: 120

| Metric | Scope | n | Relabel RAG off seed 44 mean | Relabel RAG on seed 44 mean | Diff | 95% interval | p | p adj | Family | Verdict |
|---|---|---|---|---|---|---|---|---|---|---|
| grammar_lt_rate (lower is better) | all | 120 | 0.055 | 0.043 | -0.012 | -0.053 to 0.025 | 0.575 | 1.000 | exploratory | no clear difference |
| grammar_lt_rate (lower is better) | blog | 30 | 0.069 | 0.038 | -0.030 | -0.109 to 0.043 | 0.526 | 1.000 | exploratory | no clear difference |
| grammar_lt_rate (lower is better) | linkedin | 30 | 0.053 | 0.041 | -0.012 | -0.068 to 0.044 | 0.941 | 1.000 | exploratory | no clear difference |
| grammar_lt_rate (lower is better) | x | 30 | 0.026 | 0.000 | -0.026 | -0.077 to 0.000 | 1.000 | 1.000 | exploratory | no clear difference |
| grammar_lt_rate (lower is better) | talk | 30 | 0.074 | 0.095 | 0.020 | -0.060 to 0.111 | 0.679 | 1.000 | exploratory | no clear difference |
| grammar_lt_deviation (lower is better) | all | 90 | -0.175 | -0.197 | -0.022 | -0.069 to 0.017 | 0.387 | 1.000 | exploratory | no clear difference |
| grammar_lt_deviation (lower is better) | blog | 30 | -0.040 | -0.070 | -0.030 | -0.109 to 0.043 | 0.526 | 1.000 | exploratory | no clear difference |
| grammar_lt_deviation (lower is better) | linkedin | 30 | -0.078 | -0.089 | -0.012 | -0.068 to 0.044 | 0.941 | 1.000 | exploratory | no clear difference |
| grammar_lt_deviation (lower is better) | x | 30 | -0.407 | -0.432 | -0.026 | -0.077 to 0.000 | 1.000 | 1.000 | exploratory | no clear difference |

## Drafts that pass the common gates in both runs

Pairs in this view: 67

| Metric | Scope | n | Relabel RAG off seed 44 mean | Relabel RAG on seed 44 mean | Diff | 95% interval | p | p adj | Family | Verdict |
|---|---|---|---|---|---|---|---|---|---|---|
| grammar_lt_rate (lower is better) | all | 67 | 0.054 | 0.053 | -0.001 | -0.046 to 0.047 | 0.950 | 1.000 | exploratory | no clear difference |
| grammar_lt_rate (lower is better) | blog | 26 | 0.068 | 0.027 | -0.040 | -0.124 to 0.032 | 0.402 | 1.000 | exploratory | no clear difference |
| grammar_lt_rate (lower is better) | linkedin | 4 | 0.000 | 0.000 | 0.000 | n/a | n/a | n/a | exploratory | too few pairs |
| grammar_lt_rate (lower is better) | x | 9 | 0.000 | 0.000 | 0.000 | 0.000 to 0.000 | 1.000 | 1.000 | exploratory | no clear difference |
| grammar_lt_rate (lower is better) | talk | 28 | 0.067 | 0.101 | 0.034 | -0.047 to 0.126 | 0.469 | 1.000 | exploratory | no clear difference |
| grammar_lt_deviation (lower is better) | all | 39 | -0.141 | -0.168 | -0.027 | -0.086 to 0.022 | 0.402 | 1.000 | exploratory | no clear difference |
| grammar_lt_deviation (lower is better) | blog | 26 | -0.041 | -0.082 | -0.040 | -0.124 to 0.032 | 0.402 | 1.000 | exploratory | no clear difference |
| grammar_lt_deviation (lower is better) | linkedin | 4 | -0.130 | -0.130 | 0.000 | n/a | n/a | n/a | exploratory | too few pairs |
| grammar_lt_deviation (lower is better) | x | 9 | -0.432 | -0.432 | 0.000 | 0.000 to 0.000 | 1.000 | 1.000 | exploratory | no clear difference |

How to read the verdict: a difference is called clear only when the interval excludes zero and `p adj` is below 0.05. If it looks real on its own (interval excludes zero, p below 0.05) but `p adj` is not below 0.05, it is marked (suggestive): a lead worth a second run, not a finding. Rows by medium are exploratory unless listed as primary.

Limits: one run per arm unless more seeds are added, and the 30 gold topics are a development panel. A result that is not clear here is not proof of no difference.
