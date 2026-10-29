# Comparison: Relabel RAG on seed 43 against Relabel RAG off seed 43

Every difference is B minus A. Intervals are 95 percent, resampled by topic. p is a sign-flip test by topic. `p adj` corrects for how many tests were run (Holm). The four primary rows, which are the four parts of Persona Utility, are corrected together. Every other row is exploratory and is corrected as its own, larger family.

Paired drafts: 120 (only in Relabel RAG off seed 43: 0, only in Relabel RAG on seed 43: 0).

## Gates

| Run | Drafts | Eligible (all gates) | Pass common gates | Completion fails | Copying fails | Format fails | Grounding fails |
|---|---|---|---|---|---|---|---|
| Relabel RAG off seed 43 | 120 | 81 (68%) | 81 (68%) | 36 | 1 | 7 | 0 of 0 |
| Relabel RAG on seed 43 | 120 | 72 (60%) | 82 (68%) | 36 | 1 | 3 | 16 of 120 |

Grounding is counted only where it applies. A run with retrieval has one extra gate, so its eligible count is not comparable to a run without retrieval. Use the common gates column.

Style distance role by medium (selector means it may rank models): blog selector, linkedin diagnostic_only, x diagnostic_only, talk diagnostic_only

## All drafts (no draft dropped)

Pairs in this view: 120

| Metric | Scope | n | Relabel RAG off seed 43 mean | Relabel RAG on seed 43 mean | Diff | 95% interval | p | p adj | Family | Verdict |
|---|---|---|---|---|---|---|---|---|---|---|
| grammar_lt_rate (lower is better) | all | 120 | 0.053 | 0.031 | -0.022 | -0.054 to 0.008 | 0.179 | 1.000 | exploratory | no clear difference |
| grammar_lt_rate (lower is better) | blog | 30 | 0.087 | 0.023 | -0.063 | -0.133 to 0.007 | 0.109 | 0.869 | exploratory | no clear difference |
| grammar_lt_rate (lower is better) | linkedin | 30 | 0.067 | 0.000 | -0.067 | -0.160 to 0.000 | 0.250 | 1.000 | exploratory | no clear difference |
| grammar_lt_rate (lower is better) | x | 30 | 0.000 | 0.000 | 0.000 | 0.000 to 0.000 | 1.000 | 1.000 | exploratory | no clear difference |
| grammar_lt_rate (lower is better) | talk | 30 | 0.059 | 0.101 | 0.042 | -0.052 to 0.151 | 0.426 | 1.000 | exploratory | no clear difference |
| grammar_lt_deviation (lower is better) | all | 90 | -0.173 | -0.216 | -0.043 | -0.081 to -0.013 | 0.011 | 0.100 | exploratory | B better (suggestive) |
| grammar_lt_deviation (lower is better) | blog | 30 | -0.022 | -0.086 | -0.063 | -0.133 to 0.007 | 0.109 | 0.869 | exploratory | no clear difference |
| grammar_lt_deviation (lower is better) | linkedin | 30 | -0.063 | -0.130 | -0.067 | -0.160 to 0.000 | 0.250 | 1.000 | exploratory | no clear difference |
| grammar_lt_deviation (lower is better) | x | 30 | -0.432 | -0.432 | 0.000 | 0.000 to 0.000 | 1.000 | 1.000 | exploratory | no clear difference |

## Drafts that pass the common gates in both runs

Pairs in this view: 68

| Metric | Scope | n | Relabel RAG off seed 43 mean | Relabel RAG on seed 43 mean | Diff | 95% interval | p | p adj | Family | Verdict |
|---|---|---|---|---|---|---|---|---|---|---|
| grammar_lt_rate (lower is better) | all | 68 | 0.077 | 0.055 | -0.022 | -0.075 to 0.032 | 0.444 | 1.000 | exploratory | no clear difference |
| grammar_lt_rate (lower is better) | blog | 27 | 0.084 | 0.026 | -0.059 | -0.132 to 0.014 | 0.158 | 0.947 | exploratory | no clear difference |
| grammar_lt_rate (lower is better) | linkedin | 3 | 0.390 | 0.000 | -0.390 | n/a | n/a | n/a | exploratory | too few pairs |
| grammar_lt_rate (lower is better) | x | 12 | 0.000 | 0.000 | 0.000 | 0.000 to 0.000 | 1.000 | 1.000 | exploratory | no clear difference |
| grammar_lt_rate (lower is better) | talk | 26 | 0.068 | 0.117 | 0.048 | -0.060 to 0.169 | 0.420 | 1.000 | exploratory | no clear difference |
| grammar_lt_deviation (lower is better) | all | 42 | -0.121 | -0.186 | -0.066 | -0.134 to -0.002 | 0.077 | 0.538 | exploratory | no clear difference |
| grammar_lt_deviation (lower is better) | blog | 27 | -0.024 | -0.083 | -0.059 | -0.132 to 0.014 | 0.158 | 0.947 | exploratory | no clear difference |
| grammar_lt_deviation (lower is better) | linkedin | 3 | 0.260 | -0.130 | -0.390 | n/a | n/a | n/a | exploratory | too few pairs |
| grammar_lt_deviation (lower is better) | x | 12 | -0.432 | -0.432 | 0.000 | 0.000 to 0.000 | 1.000 | 1.000 | exploratory | no clear difference |

How to read the verdict: a difference is called clear only when the interval excludes zero and `p adj` is below 0.05. If it looks real on its own (interval excludes zero, p below 0.05) but `p adj` is not below 0.05, it is marked (suggestive): a lead worth a second run, not a finding. Rows by medium are exploratory unless listed as primary.

Limits: one run per arm unless more seeds are added, and the 30 gold topics are a development panel. A result that is not clear here is not proof of no difference.
