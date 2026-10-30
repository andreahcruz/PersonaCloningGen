# Comparison: Relabel RAG on seed 43 against PersonaRAG

Every difference is B minus A. Intervals are 95 percent, resampled by topic. p is a sign-flip test by topic. `p adj` corrects for how many tests were run (Holm). The four primary rows, which are the four parts of Persona Utility, are corrected together. Every other row is exploratory and is corrected as its own, larger family.

Paired drafts: 120 (only in PersonaRAG: 0, only in Relabel RAG on seed 43: 0).

## Gates

| Run | Drafts | Eligible (all gates) | Pass common gates | Completion fails | Copying fails | Format fails | Grounding fails |
|---|---|---|---|---|---|---|---|
| PersonaRAG | 120 | 57 (48%) | 71 (59%) | 49 | 0 | 0 | 20 of 120 |
| Relabel RAG on seed 43 | 120 | 72 (60%) | 82 (68%) | 36 | 1 | 3 | 16 of 120 |

Grounding is counted only where it applies. A run with retrieval has one extra gate, so its eligible count is not comparable to a run without retrieval. Use the common gates column.

Style distance role by medium (selector means it may rank models): blog selector, linkedin diagnostic_only, x diagnostic_only, talk diagnostic_only

## All drafts (no draft dropped)

Pairs in this view: 120

| Metric | Scope | n | PersonaRAG mean | Relabel RAG on seed 43 mean | Diff | 95% interval | p | p adj | Family | Verdict |
|---|---|---|---|---|---|---|---|---|---|---|
| style_distance (lower is better) | all | 90 | 0.906 | 0.760 | -0.146 | -0.216 to -0.075 | 0.001 | 0.002 | exploratory | B better |
| style_distance (lower is better) | blog | 30 | 1.552 | 0.845 | -0.707 | -0.849 to -0.570 | 0.000 | 0.000 | primary | B better |
| style_distance (lower is better) | linkedin | 30 | 0.682 | 0.664 | -0.018 | -0.129 to 0.116 | 0.786 | 1.000 | exploratory | no clear difference |
| style_distance (lower is better) | x | 30 | 0.484 | 0.770 | 0.286 | 0.210 to 0.367 | 0.000 | 0.001 | exploratory | B worse |
| style_distance_fixed (lower is better) | all | 49 | 1.050 | 0.766 | -0.284 | -0.369 to -0.203 | 0.000 | 0.001 | exploratory | B better |
| style_distance_fixed (lower is better) | blog | 26 | 1.442 | 0.901 | -0.541 | -0.688 to -0.401 | 0.000 | 0.001 | exploratory | B better |
| style_distance_fixed (lower is better) | linkedin | 23 | 0.606 | 0.613 | 0.007 | -0.033 to 0.046 | 0.746 | 1.000 | exploratory | no clear difference |

## Drafts that pass the common gates in both runs

Pairs in this view: 57

| Metric | Scope | n | PersonaRAG mean | Relabel RAG on seed 43 mean | Diff | 95% interval | p | p adj | Family | Verdict |
|---|---|---|---|---|---|---|---|---|---|---|
| style_distance (lower is better) | all | 30 | 1.428 | 0.808 | -0.620 | -0.821 to -0.439 | 0.000 | 0.000 | exploratory | B better |
| style_distance (lower is better) | blog | 26 | 1.565 | 0.803 | -0.762 | -0.914 to -0.611 | 0.000 | 0.000 | primary | B better |
| style_distance (lower is better) | linkedin | 1 | 0.758 | 0.642 | -0.116 | n/a | n/a | n/a | exploratory | too few pairs |
| style_distance (lower is better) | x | 3 | 0.457 | 0.902 | 0.444 | n/a | n/a | n/a | exploratory | too few pairs |
| style_distance_fixed (lower is better) | all | 23 | 1.464 | 0.884 | -0.579 | -0.734 to -0.442 | 0.000 | 0.000 | exploratory | B better |
| style_distance_fixed (lower is better) | blog | 23 | 1.464 | 0.884 | -0.579 | -0.734 to -0.442 | 0.000 | 0.000 | exploratory | B better |

How to read the verdict: a difference is called clear only when the interval excludes zero and `p adj` is below 0.05. If it looks real on its own (interval excludes zero, p below 0.05) but `p adj` is not below 0.05, it is marked (suggestive): a lead worth a second run, not a finding. Rows by medium are exploratory unless listed as primary.

Limits: one run per arm unless more seeds are added, and the 30 gold topics are a development panel. A result that is not clear here is not proof of no difference.
