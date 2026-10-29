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
| style_distance (lower is better) | all | 90 | 0.755 | 0.794 | 0.039 | -0.036 to 0.117 | 0.323 | 1.000 | exploratory | no clear difference |
| style_distance (lower is better) | blog | 30 | 0.742 | 0.862 | 0.121 | 0.008 to 0.263 | 0.078 | 0.078 | primary | no clear difference |
| style_distance (lower is better) | linkedin | 30 | 0.738 | 0.723 | -0.015 | -0.192 to 0.159 | 0.865 | 1.000 | exploratory | no clear difference |
| style_distance (lower is better) | x | 30 | 0.784 | 0.795 | 0.012 | -0.042 to 0.070 | 0.706 | 1.000 | exploratory | no clear difference |
| style_distance_fixed (lower is better) | all | 42 | 0.935 | 0.907 | -0.027 | -0.083 to 0.029 | 0.363 | 1.000 | exploratory | no clear difference |
| style_distance_fixed (lower is better) | blog | 26 | 1.118 | 1.047 | -0.071 | -0.146 to 0.003 | 0.079 | 0.472 | exploratory | no clear difference |
| style_distance_fixed (lower is better) | linkedin | 16 | 0.636 | 0.681 | 0.045 | -0.005 to 0.098 | 0.123 | 0.614 | exploratory | no clear difference |

## Drafts that pass the common gates in both runs

Pairs in this view: 67

| Metric | Scope | n | Relabel RAG off seed 44 mean | Relabel RAG on seed 44 mean | Diff | 95% interval | p | p adj | Family | Verdict |
|---|---|---|---|---|---|---|---|---|---|---|
| style_distance (lower is better) | all | 39 | 0.825 | 0.837 | 0.013 | -0.075 to 0.104 | 0.796 | 1.000 | exploratory | no clear difference |
| style_distance (lower is better) | blog | 26 | 0.707 | 0.791 | 0.084 | -0.021 to 0.198 | 0.168 | 0.168 | primary | no clear difference |
| style_distance (lower is better) | linkedin | 4 | 1.327 | 0.909 | -0.418 | n/a | n/a | n/a | exploratory | too few pairs |
| style_distance (lower is better) | x | 9 | 0.940 | 0.939 | -0.002 | -0.084 to 0.060 | 1.000 | 1.000 | exploratory | no clear difference |
| style_distance_fixed (lower is better) | all | 24 | 1.114 | 1.054 | -0.059 | -0.133 to 0.017 | 0.132 | 0.529 | exploratory | no clear difference |
| style_distance_fixed (lower is better) | blog | 24 | 1.114 | 1.054 | -0.059 | -0.133 to 0.017 | 0.132 | 0.529 | exploratory | no clear difference |

How to read the verdict: a difference is called clear only when the interval excludes zero and `p adj` is below 0.05. If it looks real on its own (interval excludes zero, p below 0.05) but `p adj` is not below 0.05, it is marked (suggestive): a lead worth a second run, not a finding. Rows by medium are exploratory unless listed as primary.

Limits: one run per arm unless more seeds are added, and the 30 gold topics are a development panel. A result that is not clear here is not proof of no difference.
