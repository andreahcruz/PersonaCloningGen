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
| style_distance (lower is better) | all | 90 | 1.043 | 0.906 | -0.137 | -0.182 to -0.092 | 0.000 | 0.001 | exploratory | B better |
| style_distance (lower is better) | blog | 30 | 1.867 | 1.552 | -0.315 | -0.437 to -0.196 | 0.000 | 0.000 | primary | B better |
| style_distance (lower is better) | linkedin | 30 | 0.740 | 0.682 | -0.057 | -0.117 to -0.002 | 0.065 | 0.195 | exploratory | no clear difference |
| style_distance (lower is better) | x | 30 | 0.523 | 0.484 | -0.039 | -0.079 to -0.001 | 0.065 | 0.195 | exploratory | no clear difference |
| style_distance_fixed (lower is better) | all | 60 | 1.369 | 1.194 | -0.175 | -0.255 to -0.095 | 0.000 | 0.001 | exploratory | B better |
| style_distance_fixed (lower is better) | blog | 30 | 2.028 | 1.709 | -0.319 | -0.489 to -0.144 | 0.001 | 0.004 | exploratory | B better |
| style_distance_fixed (lower is better) | linkedin | 30 | 0.710 | 0.680 | -0.031 | -0.077 to 0.016 | 0.210 | 0.210 | exploratory | no clear difference |

## Drafts that pass the common gates in both runs

Pairs in this view: 37

| Metric | Scope | n | Base Llama 3.1 mean | PersonaRAG mean | Diff | 95% interval | p | p adj | Family | Verdict |
|---|---|---|---|---|---|---|---|---|---|---|
| style_distance (lower is better) | all | 21 | 1.725 | 1.458 | -0.267 | -0.398 to -0.137 | 0.002 | 0.005 | exploratory | B better |
| style_distance (lower is better) | blog | 19 | 1.819 | 1.539 | -0.280 | -0.419 to -0.134 | 0.003 | 0.003 | primary | B better |
| style_distance (lower is better) | linkedin | 2 | 0.833 | 0.689 | -0.144 | n/a | n/a | n/a | exploratory | too few pairs |
| style_distance_fixed (lower is better) | all | 21 | 1.904 | 1.648 | -0.256 | -0.455 to -0.064 | 0.017 | 0.034 | exploratory | B better |
| style_distance_fixed (lower is better) | blog | 19 | 2.009 | 1.747 | -0.263 | -0.471 to -0.047 | 0.032 | 0.034 | exploratory | B better |
| style_distance_fixed (lower is better) | linkedin | 2 | 0.903 | 0.715 | -0.188 | n/a | n/a | n/a | exploratory | too few pairs |

How to read the verdict: a difference is called clear only when the interval excludes zero and `p adj` is below 0.05. If it looks real on its own (interval excludes zero, p below 0.05) but `p adj` is not below 0.05, it is marked (suggestive): a lead worth a second run, not a finding. Rows by medium are exploratory unless listed as primary.

Limits: one run per arm unless more seeds are added, and the 30 gold topics are a development panel. A result that is not clear here is not proof of no difference.
