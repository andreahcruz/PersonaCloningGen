# Comparison: Relabel RAG on seed 44 against PersonaRAG

Every difference is B minus A. Intervals are 95 percent, resampled by topic. p is a sign-flip test by topic. `p adj` corrects for how many tests were run (Holm). The four primary rows, which are the four parts of Persona Utility, are corrected together. Every other row is exploratory and is corrected as its own, larger family.

Paired drafts: 120 (only in PersonaRAG: 0, only in Relabel RAG on seed 44: 0).

## Gates

| Run | Drafts | Eligible (all gates) | Pass common gates | Completion fails | Copying fails | Format fails | Grounding fails |
|---|---|---|---|---|---|---|---|
| PersonaRAG | 120 | 57 (48%) | 71 (59%) | 49 | 0 | 0 | 20 of 120 |
| Relabel RAG on seed 44 | 120 | 70 (58%) | 83 (69%) | 34 | 1 | 2 | 14 of 120 |

Grounding is counted only where it applies. A run with retrieval has one extra gate, so its eligible count is not comparable to a run without retrieval. Use the common gates column.

Style distance role by medium (selector means it may rank models): blog selector, linkedin diagnostic_only, x diagnostic_only, talk diagnostic_only

## All drafts (no draft dropped)

Pairs in this view: 120

| Metric | Scope | n | PersonaRAG mean | Relabel RAG on seed 44 mean | Diff | 95% interval | p | p adj | Family | Verdict |
|---|---|---|---|---|---|---|---|---|---|---|
| style_distance (lower is better) | all | 90 | 0.906 | 0.794 | -0.112 | -0.195 to -0.028 | 0.018 | 0.053 | exploratory | B better (suggestive) |
| style_distance (lower is better) | blog | 30 | 1.552 | 0.862 | -0.690 | -0.849 to -0.519 | 0.000 | 0.000 | primary | B better |
| style_distance (lower is better) | linkedin | 30 | 0.682 | 0.723 | 0.041 | -0.097 to 0.201 | 0.608 | 0.856 | exploratory | no clear difference |
| style_distance (lower is better) | x | 30 | 0.484 | 0.795 | 0.311 | 0.238 to 0.388 | 0.000 | 0.001 | exploratory | B worse |
| style_distance_fixed (lower is better) | all | 46 | 1.092 | 0.728 | -0.364 | -0.449 to -0.279 | 0.000 | 0.001 | exploratory | B better |
| style_distance_fixed (lower is better) | blog | 26 | 1.469 | 0.836 | -0.632 | -0.772 to -0.496 | 0.000 | 0.001 | exploratory | B better |
| style_distance_fixed (lower is better) | linkedin | 20 | 0.603 | 0.588 | -0.015 | -0.050 to 0.021 | 0.428 | 0.856 | exploratory | no clear difference |

## Drafts that pass the common gates in both runs

Pairs in this view: 60

| Metric | Scope | n | PersonaRAG mean | Relabel RAG on seed 44 mean | Diff | 95% interval | p | p adj | Family | Verdict |
|---|---|---|---|---|---|---|---|---|---|---|
| style_distance (lower is better) | all | 32 | 1.388 | 0.798 | -0.590 | -0.780 to -0.404 | 0.000 | 0.000 | exploratory | B better |
| style_distance (lower is better) | blog | 26 | 1.571 | 0.790 | -0.781 | -0.915 to -0.646 | 0.000 | 0.000 | primary | B better |
| style_distance (lower is better) | linkedin | 3 | 0.736 | 0.740 | 0.005 | n/a | n/a | n/a | exploratory | too few pairs |
| style_distance (lower is better) | x | 3 | 0.457 | 0.926 | 0.469 | n/a | n/a | n/a | exploratory | too few pairs |
| style_distance_fixed (lower is better) | all | 25 | 1.442 | 0.824 | -0.618 | -0.772 to -0.479 | 0.000 | 0.000 | exploratory | B better |
| style_distance_fixed (lower is better) | blog | 24 | 1.477 | 0.833 | -0.644 | -0.786 to -0.507 | 0.000 | 0.000 | exploratory | B better |
| style_distance_fixed (lower is better) | linkedin | 1 | 0.614 | 0.601 | -0.013 | n/a | n/a | n/a | exploratory | too few pairs |

How to read the verdict: a difference is called clear only when the interval excludes zero and `p adj` is below 0.05. If it looks real on its own (interval excludes zero, p below 0.05) but `p adj` is not below 0.05, it is marked (suggestive): a lead worth a second run, not a finding. Rows by medium are exploratory unless listed as primary.

Limits: one run per arm unless more seeds are added, and the 30 gold topics are a development panel. A result that is not clear here is not proof of no difference.
