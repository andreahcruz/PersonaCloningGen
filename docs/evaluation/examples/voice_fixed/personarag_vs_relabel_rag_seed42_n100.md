# Comparison: Relabel RAG on seed 42 against PersonaRAG

Every difference is B minus A. Intervals are 95 percent, resampled by topic. p is a sign-flip test by topic. `p adj` corrects for how many tests were run (Holm). The four primary rows, which are the four parts of Persona Utility, are corrected together. Every other row is exploratory and is corrected as its own, larger family.

Paired drafts: 120 (only in PersonaRAG: 0, only in Relabel RAG on seed 42: 0).

## Gates

| Run | Drafts | Eligible (all gates) | Pass common gates | Completion fails | Copying fails | Format fails | Grounding fails |
|---|---|---|---|---|---|---|---|
| PersonaRAG | 120 | 57 (48%) | 71 (59%) | 49 | 0 | 0 | 20 of 120 |
| Relabel RAG on seed 42 | 120 | 72 (60%) | 83 (69%) | 36 | 2 | 3 | 14 of 120 |

Grounding is counted only where it applies. A run with retrieval has one extra gate, so its eligible count is not comparable to a run without retrieval. Use the common gates column.

Style distance role by medium (selector means it may rank models): blog selector, linkedin diagnostic_only, x diagnostic_only, talk diagnostic_only

## All drafts (no draft dropped)

Pairs in this view: 120

| Metric | Scope | n | PersonaRAG mean | Relabel RAG on seed 42 mean | Diff | 95% interval | p | p adj | Family | Verdict |
|---|---|---|---|---|---|---|---|---|---|---|
| style_distance (lower is better) | all | 90 | 0.906 | 0.805 | -0.101 | -0.167 to -0.025 | 0.011 | 0.032 | exploratory | B better |
| style_distance (lower is better) | blog | 30 | 1.552 | 0.814 | -0.739 | -0.879 to -0.589 | 0.000 | 0.000 | primary | B better |
| style_distance (lower is better) | linkedin | 30 | 0.682 | 0.828 | 0.145 | -0.009 to 0.323 | 0.098 | 0.197 | exploratory | no clear difference |
| style_distance (lower is better) | x | 30 | 0.484 | 0.775 | 0.291 | 0.223 to 0.363 | 0.000 | 0.001 | exploratory | B worse |
| style_distance_fixed (lower is better) | all | 49 | 1.299 | 0.930 | -0.369 | -0.507 to -0.243 | 0.000 | 0.001 | exploratory | B better |
| style_distance_fixed (lower is better) | blog | 29 | 1.725 | 1.131 | -0.594 | -0.780 to -0.409 | 0.000 | 0.001 | exploratory | B better |
| style_distance_fixed (lower is better) | linkedin | 20 | 0.681 | 0.638 | -0.043 | -0.099 to 0.013 | 0.166 | 0.197 | exploratory | no clear difference |

## Drafts that pass the common gates in both runs

Pairs in this view: 59

| Metric | Scope | n | PersonaRAG mean | Relabel RAG on seed 42 mean | Diff | 95% interval | p | p adj | Family | Verdict |
|---|---|---|---|---|---|---|---|---|---|---|
| style_distance (lower is better) | all | 30 | 1.416 | 0.769 | -0.647 | -0.828 to -0.483 | 0.000 | 0.000 | exploratory | B better |
| style_distance (lower is better) | blog | 26 | 1.543 | 0.750 | -0.792 | -0.912 to -0.682 | 0.000 | 0.000 | primary | B better |
| style_distance (lower is better) | linkedin | 2 | 0.722 | 0.663 | -0.058 | n/a | n/a | n/a | exploratory | too few pairs |
| style_distance (lower is better) | x | 2 | 0.466 | 1.125 | 0.659 | n/a | n/a | n/a | exploratory | too few pairs |
| style_distance_fixed (lower is better) | all | 28 | 1.639 | 1.095 | -0.544 | -0.760 to -0.349 | 0.000 | 0.000 | exploratory | B better |
| style_distance_fixed (lower is better) | blog | 26 | 1.716 | 1.122 | -0.594 | -0.800 to -0.398 | 0.000 | 0.000 | exploratory | B better |
| style_distance_fixed (lower is better) | linkedin | 2 | 0.640 | 0.740 | 0.100 | n/a | n/a | n/a | exploratory | too few pairs |

How to read the verdict: a difference is called clear only when the interval excludes zero and `p adj` is below 0.05. If it looks real on its own (interval excludes zero, p below 0.05) but `p adj` is not below 0.05, it is marked (suggestive): a lead worth a second run, not a finding. Rows by medium are exploratory unless listed as primary.

Limits: one run per arm unless more seeds are added, and the 30 gold topics are a development panel. A result that is not clear here is not proof of no difference.
