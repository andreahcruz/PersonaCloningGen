# Comparison: PersonaRAG against Base Llama 3.1

Every difference is B minus A. Intervals are 95 percent, resampled by topic. p is a sign-flip test by topic. `p adj` corrects for how many tests were run (Holm). The four primary rows, which are the four parts of Persona Utility, are corrected together. Every other row is exploratory and is corrected as its own, larger family.

Paired drafts: 120 (only in Base Llama 3.1: 0, only in PersonaRAG: 0).

## Gates

| Run | Drafts | Eligible (all gates) | Pass common gates | Completion fails | Copying fails | Format fails | Grounding fails |
|---|---|---|---|---|---|---|---|
| Base Llama 3.1 | 120 | 45 (38%) | 45 (38%) | 75 | 0 | 0 | 0 of 0 |
| PersonaRAG | 120 | 57 (48%) | 71 (59%) | 49 | 0 | 0 | 20 of 120 |

Grounding is counted only where it applies. A run with retrieval has one extra gate, so its eligible count is not comparable to a run without retrieval. Use the common gates column.

Style distance role by medium (selector means it may rank models): blog selector, linkedin diagnostic_only, x diagnostic_only, talk diagnostic_only

## All drafts (no draft dropped)

Pairs in this view: 120

| Metric | Scope | n | Base Llama 3.1 mean | PersonaRAG mean | Diff | 95% interval | p | p adj | Family | Verdict |
|---|---|---|---|---|---|---|---|---|---|---|
| grammar_lt_rate (lower is better) | all | 120 | 0.035 | 0.021 | -0.014 | -0.043 to 0.014 | 0.365 | 1.000 | exploratory | no clear difference |
| grammar_lt_rate (lower is better) | blog | 30 | 0.037 | 0.009 | -0.028 | -0.071 to 0.011 | 0.250 | 1.000 | exploratory | no clear difference |
| grammar_lt_rate (lower is better) | linkedin | 30 | 0.054 | 0.019 | -0.035 | -0.098 to 0.031 | 0.377 | 1.000 | exploratory | no clear difference |
| grammar_lt_rate (lower is better) | x | 30 | 0.028 | 0.028 | 0.000 | -0.084 to 0.084 | 1.000 | 1.000 | exploratory | no clear difference |
| grammar_lt_rate (lower is better) | talk | 30 | 0.021 | 0.027 | 0.006 | -0.042 to 0.048 | 0.816 | 1.000 | exploratory | no clear difference |
| grammar_lt_deviation (lower is better) | all | 90 | -0.184 | -0.205 | -0.021 | -0.052 to 0.010 | 0.219 | 1.000 | exploratory | no clear difference |
| grammar_lt_deviation (lower is better) | blog | 30 | -0.072 | -0.100 | -0.028 | -0.071 to 0.011 | 0.250 | 1.000 | exploratory | no clear difference |
| grammar_lt_deviation (lower is better) | linkedin | 30 | -0.076 | -0.111 | -0.035 | -0.098 to 0.031 | 0.377 | 1.000 | exploratory | no clear difference |
| grammar_lt_deviation (lower is better) | x | 30 | -0.404 | -0.404 | 0.000 | -0.084 to 0.084 | 1.000 | 1.000 | exploratory | no clear difference |

## Drafts that pass the common gates in both runs

Pairs in this view: 37

| Metric | Scope | n | Base Llama 3.1 mean | PersonaRAG mean | Diff | 95% interval | p | p adj | Family | Verdict |
|---|---|---|---|---|---|---|---|---|---|---|
| grammar_lt_rate (lower is better) | all | 37 | 0.033 | 0.015 | -0.018 | -0.062 to 0.021 | 0.403 | 1.000 | exploratory | no clear difference |
| grammar_lt_rate (lower is better) | blog | 19 | 0.040 | 0.014 | -0.026 | -0.084 to 0.027 | 0.490 | 1.000 | exploratory | no clear difference |
| grammar_lt_rate (lower is better) | linkedin | 2 | 0.000 | 0.000 | 0.000 | n/a | n/a | n/a | exploratory | too few pairs |
| grammar_lt_rate (lower is better) | talk | 16 | 0.028 | 0.017 | -0.011 | -0.084 to 0.052 | 1.000 | 1.000 | exploratory | no clear difference |
| grammar_lt_deviation (lower is better) | all | 21 | -0.075 | -0.098 | -0.023 | -0.077 to 0.024 | 0.495 | 1.000 | exploratory | no clear difference |
| grammar_lt_deviation (lower is better) | blog | 19 | -0.069 | -0.095 | -0.026 | -0.084 to 0.027 | 0.490 | 1.000 | exploratory | no clear difference |
| grammar_lt_deviation (lower is better) | linkedin | 2 | -0.130 | -0.130 | 0.000 | n/a | n/a | n/a | exploratory | too few pairs |

How to read the verdict: a difference is called clear only when the interval excludes zero and `p adj` is below 0.05. If it looks real on its own (interval excludes zero, p below 0.05) but `p adj` is not below 0.05, it is marked (suggestive): a lead worth a second run, not a finding. Rows by medium are exploratory unless listed as primary.

Limits: one run per arm unless more seeds are added, and the 30 gold topics are a development panel. A result that is not clear here is not proof of no difference.
