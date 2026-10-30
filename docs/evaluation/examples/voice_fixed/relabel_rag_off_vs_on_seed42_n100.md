# Comparison: Relabel RAG on seed 42 against Relabel RAG off seed 42

Every difference is B minus A. Intervals are 95 percent, resampled by topic. p is a sign-flip test by topic. `p adj` corrects for how many tests were run (Holm). The four primary rows, which are the four parts of Persona Utility, are corrected together. Every other row is exploratory and is corrected as its own, larger family.

Paired drafts: 120 (only in Relabel RAG off seed 42: 0, only in Relabel RAG on seed 42: 0).

## Gates

| Run | Drafts | Eligible (all gates) | Pass common gates | Completion fails | Copying fails | Format fails | Grounding fails |
|---|---|---|---|---|---|---|---|
| Relabel RAG off seed 42 | 120 | 77 (64%) | 77 (64%) | 41 | 1 | 5 | 0 of 0 |
| Relabel RAG on seed 42 | 120 | 72 (60%) | 83 (69%) | 36 | 2 | 3 | 14 of 120 |

Grounding is counted only where it applies. A run with retrieval has one extra gate, so its eligible count is not comparable to a run without retrieval. Use the common gates column.

Style distance role by medium (selector means it may rank models): blog selector, linkedin diagnostic_only, x diagnostic_only, talk diagnostic_only

## All drafts (no draft dropped)

Pairs in this view: 120

| Metric | Scope | n | Relabel RAG off seed 42 mean | Relabel RAG on seed 42 mean | Diff | 95% interval | p | p adj | Family | Verdict |
|---|---|---|---|---|---|---|---|---|---|---|
| style_distance (lower is better) | all | 90 | 0.750 | 0.805 | 0.055 | 0.006 to 0.113 | 0.054 | 0.327 | exploratory | no clear difference |
| style_distance (lower is better) | blog | 30 | 0.709 | 0.814 | 0.104 | 0.023 to 0.213 | 0.015 | 0.015 | primary | B worse |
| style_distance (lower is better) | linkedin | 30 | 0.755 | 0.828 | 0.073 | -0.043 to 0.201 | 0.271 | 1.000 | exploratory | no clear difference |
| style_distance (lower is better) | x | 30 | 0.786 | 0.775 | -0.011 | -0.063 to 0.041 | 0.698 | 1.000 | exploratory | no clear difference |
| style_distance_fixed (lower is better) | all | 48 | 0.917 | 0.938 | 0.021 | -0.027 to 0.072 | 0.420 | 1.000 | exploratory | no clear difference |
| style_distance_fixed (lower is better) | blog | 29 | 1.083 | 1.131 | 0.049 | -0.024 to 0.125 | 0.227 | 1.000 | exploratory | no clear difference |
| style_distance_fixed (lower is better) | linkedin | 19 | 0.665 | 0.643 | -0.021 | -0.069 to 0.027 | 0.403 | 1.000 | exploratory | no clear difference |

## Drafts that pass the common gates in both runs

Pairs in this view: 72

| Metric | Scope | n | Relabel RAG off seed 42 mean | Relabel RAG on seed 42 mean | Diff | 95% interval | p | p adj | Family | Verdict |
|---|---|---|---|---|---|---|---|---|---|---|
| style_distance (lower is better) | all | 44 | 0.785 | 0.843 | 0.058 | 0.001 to 0.134 | 0.068 | 0.272 | exploratory | no clear difference |
| style_distance (lower is better) | blog | 27 | 0.714 | 0.774 | 0.060 | 0.009 to 0.111 | 0.032 | 0.032 | primary | B worse |
| style_distance (lower is better) | linkedin | 4 | 0.912 | 1.227 | 0.315 | n/a | n/a | n/a | exploratory | too few pairs |
| style_distance (lower is better) | x | 13 | 0.895 | 0.868 | -0.026 | -0.087 to 0.012 | 0.594 | 0.912 | exploratory | no clear difference |
| style_distance_fixed (lower is better) | all | 28 | 1.073 | 1.114 | 0.041 | -0.034 to 0.118 | 0.304 | 0.912 | exploratory | no clear difference |
| style_distance_fixed (lower is better) | blog | 27 | 1.087 | 1.128 | 0.041 | -0.037 to 0.120 | 0.329 | 0.912 | exploratory | no clear difference |
| style_distance_fixed (lower is better) | linkedin | 1 | 0.700 | 0.756 | 0.056 | n/a | n/a | n/a | exploratory | too few pairs |

How to read the verdict: a difference is called clear only when the interval excludes zero and `p adj` is below 0.05. If it looks real on its own (interval excludes zero, p below 0.05) but `p adj` is not below 0.05, it is marked (suggestive): a lead worth a second run, not a finding. Rows by medium are exploratory unless listed as primary.

Limits: one run per arm unless more seeds are added, and the 30 gold topics are a development panel. A result that is not clear here is not proof of no difference.
