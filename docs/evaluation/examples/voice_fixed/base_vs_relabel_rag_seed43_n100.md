# Comparison: Relabel RAG on seed 43 against Base Llama 3.1

Every difference is B minus A. Intervals are 95 percent, resampled by topic. p is a sign-flip test by topic. `p adj` corrects for how many tests were run (Holm). The four primary rows, which are the four parts of Persona Utility, are corrected together. Every other row is exploratory and is corrected as its own, larger family.

Paired drafts: 120 (only in Base Llama 3.1: 0, only in Relabel RAG on seed 43: 0).

## Gates

| Run | Drafts | Eligible (all gates) | Pass common gates | Completion fails | Copying fails | Format fails | Grounding fails |
|---|---|---|---|---|---|---|---|
| Base Llama 3.1 | 120 | 45 (38%) | 45 (38%) | 75 | 0 | 0 | 0 of 0 |
| Relabel RAG on seed 43 | 120 | 72 (60%) | 82 (68%) | 36 | 1 | 3 | 16 of 120 |

Grounding is counted only where it applies. A run with retrieval has one extra gate, so its eligible count is not comparable to a run without retrieval. Use the common gates column.

Style distance role by medium (selector means it may rank models): blog selector, linkedin diagnostic_only, x diagnostic_only, talk diagnostic_only

## All drafts (no draft dropped)

Pairs in this view: 120

| Metric | Scope | n | Base Llama 3.1 mean | Relabel RAG on seed 43 mean | Diff | 95% interval | p | p adj | Family | Verdict |
|---|---|---|---|---|---|---|---|---|---|---|
| style_distance (lower is better) | all | 90 | 1.043 | 0.760 | -0.283 | -0.355 to -0.207 | 0.000 | 0.001 | exploratory | B better |
| style_distance (lower is better) | blog | 30 | 1.867 | 0.845 | -1.022 | -1.160 to -0.878 | 0.000 | 0.000 | primary | B better |
| style_distance (lower is better) | linkedin | 30 | 0.740 | 0.664 | -0.076 | -0.190 to 0.061 | 0.255 | 0.436 | exploratory | no clear difference |
| style_distance (lower is better) | x | 30 | 0.523 | 0.770 | 0.248 | 0.176 to 0.322 | 0.000 | 0.001 | exploratory | B worse |
| style_distance_fixed (lower is better) | all | 53 | 1.381 | 0.893 | -0.488 | -0.559 to -0.415 | 0.000 | 0.001 | exploratory | B better |
| style_distance_fixed (lower is better) | blog | 27 | 2.028 | 1.106 | -0.922 | -1.040 to -0.800 | 0.000 | 0.001 | exploratory | B better |
| style_distance_fixed (lower is better) | linkedin | 26 | 0.710 | 0.672 | -0.038 | -0.097 to 0.021 | 0.218 | 0.436 | exploratory | no clear difference |

## Drafts that pass the common gates in both runs

Pairs in this view: 36

| Metric | Scope | n | Base Llama 3.1 mean | Relabel RAG on seed 43 mean | Diff | 95% interval | p | p adj | Family | Verdict |
|---|---|---|---|---|---|---|---|---|---|---|
| style_distance (lower is better) | all | 20 | 1.710 | 0.804 | -0.906 | -1.075 to -0.727 | 0.000 | 0.000 | exploratory | B better |
| style_distance (lower is better) | blog | 18 | 1.808 | 0.808 | -1.001 | -1.138 to -0.878 | 0.000 | 0.000 | primary | B better |
| style_distance (lower is better) | linkedin | 1 | 0.917 | 0.717 | -0.200 | n/a | n/a | n/a | exploratory | too few pairs |
| style_distance (lower is better) | x | 1 | 0.727 | 0.826 | 0.099 | n/a | n/a | n/a | exploratory | too few pairs |
| style_distance_fixed (lower is better) | all | 17 | 1.892 | 1.068 | -0.824 | -0.978 to -0.669 | 0.000 | 0.000 | exploratory | B better |
| style_distance_fixed (lower is better) | blog | 16 | 1.954 | 1.089 | -0.865 | -1.011 to -0.728 | 0.000 | 0.000 | exploratory | B better |
| style_distance_fixed (lower is better) | linkedin | 1 | 0.896 | 0.730 | -0.165 | n/a | n/a | n/a | exploratory | too few pairs |

How to read the verdict: a difference is called clear only when the interval excludes zero and `p adj` is below 0.05. If it looks real on its own (interval excludes zero, p below 0.05) but `p adj` is not below 0.05, it is marked (suggestive): a lead worth a second run, not a finding. Rows by medium are exploratory unless listed as primary.

Limits: one run per arm unless more seeds are added, and the 30 gold topics are a development panel. A result that is not clear here is not proof of no difference.
