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
| style_distance (lower is better) | all | 90 | 0.755 | 0.760 | 0.004 | -0.060 to 0.068 | 0.898 | 1.000 | exploratory | no clear difference |
| style_distance (lower is better) | blog | 30 | 0.746 | 0.845 | 0.099 | 0.010 to 0.191 | 0.040 | 0.040 | primary | B worse |
| style_distance (lower is better) | linkedin | 30 | 0.718 | 0.664 | -0.055 | -0.207 to 0.089 | 0.538 | 1.000 | exploratory | no clear difference |
| style_distance (lower is better) | x | 30 | 0.802 | 0.770 | -0.031 | -0.093 to 0.040 | 0.390 | 1.000 | exploratory | no clear difference |
| style_distance_fixed (lower is better) | all | 42 | 0.750 | 0.785 | 0.035 | -0.008 to 0.080 | 0.128 | 0.647 | exploratory | no clear difference |
| style_distance_fixed (lower is better) | blog | 25 | 0.849 | 0.898 | 0.049 | -0.012 to 0.103 | 0.108 | 0.647 | exploratory | no clear difference |
| style_distance_fixed (lower is better) | linkedin | 17 | 0.604 | 0.618 | 0.014 | -0.041 to 0.069 | 0.630 | 1.000 | exploratory | no clear difference |

## Drafts that pass the common gates in both runs

Pairs in this view: 68

| Metric | Scope | n | Relabel RAG off seed 43 mean | Relabel RAG on seed 43 mean | Diff | 95% interval | p | p adj | Family | Verdict |
|---|---|---|---|---|---|---|---|---|---|---|
| style_distance (lower is better) | all | 42 | 0.792 | 0.854 | 0.062 | -0.004 to 0.129 | 0.093 | 0.370 | exploratory | no clear difference |
| style_distance (lower is better) | blog | 27 | 0.742 | 0.819 | 0.077 | -0.012 to 0.165 | 0.099 | 0.099 | primary | no clear difference |
| style_distance (lower is better) | linkedin | 3 | 0.881 | 0.942 | 0.061 | n/a | n/a | n/a | exploratory | too few pairs |
| style_distance (lower is better) | x | 12 | 0.881 | 0.912 | 0.031 | -0.054 to 0.167 | 0.938 | 0.938 | exploratory | no clear difference |
| style_distance_fixed (lower is better) | all | 23 | 0.831 | 0.871 | 0.040 | -0.024 to 0.098 | 0.221 | 0.664 | exploratory | no clear difference |
| style_distance_fixed (lower is better) | blog | 22 | 0.844 | 0.883 | 0.040 | -0.027 to 0.100 | 0.246 | 0.664 | exploratory | no clear difference |
| style_distance_fixed (lower is better) | linkedin | 1 | 0.559 | 0.608 | 0.049 | n/a | n/a | n/a | exploratory | too few pairs |

How to read the verdict: a difference is called clear only when the interval excludes zero and `p adj` is below 0.05. If it looks real on its own (interval excludes zero, p below 0.05) but `p adj` is not below 0.05, it is marked (suggestive): a lead worth a second run, not a finding. Rows by medium are exploratory unless listed as primary.

Limits: one run per arm unless more seeds are added, and the 30 gold topics are a development panel. A result that is not clear here is not proof of no difference.
