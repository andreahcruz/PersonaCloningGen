# Comparison: Relabel + RAG seed 44 against PersonaRAG (base + retrieval)

Every difference is B minus A. Intervals are 95 percent, resampled by topic. p is a sign-flip test by topic. `p adj` corrects for how many tests were run (Holm). The four primary rows, which are the four parts of Persona Utility, are corrected together. Every other row is exploratory and is corrected as its own, larger family.

Paired drafts: 120 (only in PersonaRAG (base + retrieval): 0, only in Relabel + RAG seed 44: 0).

## Gates

| Run | Drafts | Eligible (all gates) | Pass common gates | Completion fails | Copying fails | Format fails | Grounding fails |
|---|---|---|---|---|---|---|---|
| PersonaRAG (base + retrieval) | 120 | 57 (48%) | 71 (59%) | 49 | 0 | 0 | 20 of 120 |
| Relabel + RAG seed 44 | 120 | 70 (58%) | 83 (69%) | 34 | 1 | 2 | 14 of 120 |

Grounding is counted only where it applies. A run with retrieval has one extra gate, so its eligible count is not comparable to a run without retrieval. Use the common gates column.

Style distance role by medium (selector means it may rank models): blog selector, linkedin diagnostic_only, x diagnostic_only, talk diagnostic_only

## All drafts (no draft dropped)

Pairs in this view: 120

| Metric | Scope | n | PersonaRAG (base + retrieval) mean | Relabel + RAG seed 44 mean | Diff | 95% interval | p | p adj | Family | Verdict |
|---|---|---|---|---|---|---|---|---|---|---|
| writing_quality | all | 120 | 3.599 | 2.881 | -0.718 | -0.864 to -0.571 | 0.000 | 0.000 | primary | B worse |
| writing_quality | blog | 30 | 3.722 | 3.367 | -0.356 | -0.600 to -0.133 | 0.006 | 0.112 | exploratory | B worse (suggestive) |
| writing_quality | linkedin | 30 | 3.761 | 3.056 | -0.706 | -1.061 to -0.350 | 0.001 | 0.020 | exploratory | B worse |
| writing_quality | x | 30 | 3.122 | 1.844 | -1.278 | -1.489 to -1.050 | 0.000 | 0.003 | exploratory | B worse |
| writing_quality | talk | 30 | 3.789 | 3.256 | -0.533 | -0.739 to -0.333 | 0.000 | 0.003 | exploratory | B worse |
| task_adherence | all | 120 | 0.725 | 0.705 | -0.019 | -0.043 to 0.001 | 0.099 | 0.198 | primary | no clear difference |
| task_adherence | blog | 30 | 0.746 | 0.692 | -0.053 | -0.094 to -0.020 | 0.003 | 0.057 | exploratory | B worse (suggestive) |
| task_adherence | linkedin | 30 | 0.710 | 0.751 | 0.040 | 0.019 to 0.062 | 0.001 | 0.032 | exploratory | B better |
| task_adherence | x | 30 | 0.712 | 0.719 | 0.008 | -0.012 to 0.029 | 0.476 | 1.000 | exploratory | no clear difference |
| task_adherence | talk | 30 | 0.732 | 0.659 | -0.073 | -0.130 to -0.027 | 0.002 | 0.048 | exploratory | B worse |
| stance_consistency | all | 87 | 0.989 | 0.969 | -0.019 | -0.057 to 0.019 | 0.444 | 0.444 | primary | no clear difference |
| stance_consistency | blog | 20 | 1.000 | 0.967 | -0.033 | -0.100 to 0.000 | 1.000 | 1.000 | exploratory | no clear difference |
| stance_consistency | linkedin | 19 | 1.000 | 0.982 | -0.018 | -0.053 to 0.000 | 1.000 | 1.000 | exploratory | no clear difference |
| stance_consistency | x | 22 | 1.000 | 0.924 | -0.076 | -0.182 to 0.000 | 0.252 | 1.000 | exploratory | no clear difference |
| stance_consistency | talk | 26 | 0.962 | 1.000 | 0.038 | 0.000 to 0.115 | 1.000 | 1.000 | exploratory | no clear difference |
| style_distance (lower is better) | all | 90 | 0.906 | 0.794 | -0.112 | -0.195 to -0.028 | 0.018 | 0.336 | exploratory | B better (suggestive) |
| style_distance (lower is better) | blog | 30 | 1.552 | 0.862 | -0.690 | -0.849 to -0.519 | 0.000 | 0.000 | primary | B better |
| style_distance (lower is better) | linkedin | 30 | 0.682 | 0.723 | 0.041 | -0.097 to 0.201 | 0.608 | 1.000 | exploratory | no clear difference |
| style_distance (lower is better) | x | 30 | 0.484 | 0.795 | 0.311 | 0.238 to 0.388 | 0.000 | 0.003 | exploratory | B worse |
| persona_utility | all | 8 | 0.625 | 0.668 | 0.043 | -0.014 to 0.092 | 0.180 | 1.000 | exploratory | no clear difference |
| persona_utility | blog | 8 | 0.625 | 0.668 | 0.043 | -0.014 to 0.092 | 0.180 | 1.000 | exploratory | no clear difference |
| claim_support | all | 93 | 1.000 | 0.998 | -0.002 | -0.007 to 0.000 | 1.000 | 1.000 | exploratory | no clear difference |
| claim_support | blog | 19 | 1.000 | 1.000 | 0.000 | 0.000 to 0.000 | 1.000 | 1.000 | exploratory | no clear difference |
| claim_support | linkedin | 23 | 1.000 | 1.000 | 0.000 | 0.000 to 0.000 | 1.000 | 1.000 | exploratory | no clear difference |
| claim_support | x | 26 | 1.000 | 0.992 | -0.008 | -0.023 to 0.000 | 1.000 | 1.000 | exploratory | no clear difference |
| claim_support | talk | 25 | 1.000 | 1.000 | 0.000 | 0.000 to 0.000 | 1.000 | 1.000 | exploratory | no clear difference |
| evidence_support | all | 91 | 0.967 | 0.937 | -0.030 | -0.075 to 0.015 | 0.209 | 1.000 | exploratory | no clear difference |
| evidence_support | blog | 22 | 0.977 | 0.955 | -0.023 | -0.091 to 0.045 | 1.000 | 1.000 | exploratory | no clear difference |
| evidence_support | linkedin | 20 | 1.000 | 0.963 | -0.037 | -0.083 to 0.000 | 0.252 | 1.000 | exploratory | no clear difference |
| evidence_support | x | 23 | 0.935 | 0.848 | -0.087 | -0.217 to 0.022 | 0.306 | 1.000 | exploratory | no clear difference |
| evidence_support | talk | 26 | 0.962 | 0.981 | 0.019 | -0.058 to 0.115 | 1.000 | 1.000 | exploratory | no clear difference |
| word_count | all | 120 | 296.475 | 184.400 | -112.075 | -131.851 to -92.800 | 0.000 | 0.003 | exploratory | B lower |
| word_count | blog | 30 | 390.900 | 241.267 | -149.633 | -189.367 to -110.998 | 0.000 | 0.003 | exploratory | B lower |
| word_count | linkedin | 30 | 241.967 | 175.767 | -66.200 | -101.734 to -32.133 | 0.001 | 0.024 | exploratory | B lower |
| word_count | x | 30 | 124.567 | 35.000 | -89.567 | -102.300 to -75.033 | 0.000 | 0.003 | exploratory | B lower |
| word_count | talk | 30 | 428.467 | 285.567 | -142.900 | -187.233 to -101.500 | 0.000 | 0.003 | exploratory | B lower |

## Drafts that pass the common gates in both runs

Pairs in this view: 60

| Metric | Scope | n | PersonaRAG (base + retrieval) mean | Relabel + RAG seed 44 mean | Diff | 95% interval | p | p adj | Family | Verdict |
|---|---|---|---|---|---|---|---|---|---|---|
| writing_quality | all | 60 | 3.717 | 3.253 | -0.464 | -0.655 to -0.269 | 0.000 | 0.001 | primary | B worse |
| writing_quality | blog | 26 | 3.699 | 3.442 | -0.256 | -0.481 to -0.045 | 0.030 | 0.327 | exploratory | B worse (suggestive) |
| writing_quality | linkedin | 3 | 3.444 | 2.722 | -0.722 | n/a | n/a | n/a | exploratory | too few pairs |
| writing_quality | x | 3 | 3.389 | 1.833 | -1.556 | n/a | n/a | n/a | exploratory | too few pairs |
| writing_quality | talk | 28 | 3.798 | 3.286 | -0.512 | -0.690 to -0.333 | 0.000 | 0.002 | exploratory | B worse |
| task_adherence | all | 60 | 0.731 | 0.691 | -0.040 | -0.063 to -0.017 | 0.003 | 0.005 | primary | B worse |
| task_adherence | blog | 26 | 0.740 | 0.699 | -0.040 | -0.065 to -0.015 | 0.004 | 0.059 | exploratory | B worse (suggestive) |
| task_adherence | linkedin | 3 | 0.717 | 0.719 | 0.003 | n/a | n/a | n/a | exploratory | too few pairs |
| task_adherence | x | 3 | 0.697 | 0.705 | 0.007 | n/a | n/a | n/a | exploratory | too few pairs |
| task_adherence | talk | 28 | 0.727 | 0.678 | -0.049 | -0.082 to -0.016 | 0.006 | 0.076 | exploratory | B worse (suggestive) |
| stance_consistency | all | 47 | 0.979 | 0.986 | 0.007 | -0.042 to 0.064 | 1.000 | 1.000 | primary | no clear difference |
| stance_consistency | blog | 18 | 1.000 | 0.963 | -0.037 | -0.111 to 0.000 | 1.000 | 1.000 | exploratory | no clear difference |
| stance_consistency | linkedin | 3 | 1.000 | 1.000 | 0.000 | n/a | n/a | n/a | exploratory | too few pairs |
| stance_consistency | x | 2 | 1.000 | 1.000 | 0.000 | n/a | n/a | n/a | exploratory | too few pairs |
| stance_consistency | talk | 24 | 0.958 | 1.000 | 0.042 | 0.000 to 0.125 | 1.000 | 1.000 | exploratory | no clear difference |
| style_distance (lower is better) | all | 32 | 1.388 | 0.798 | -0.590 | -0.780 to -0.404 | 0.000 | 0.002 | exploratory | B better |
| style_distance (lower is better) | blog | 26 | 1.571 | 0.790 | -0.781 | -0.915 to -0.646 | 0.000 | 0.000 | primary | B better |
| style_distance (lower is better) | linkedin | 3 | 0.736 | 0.740 | 0.005 | n/a | n/a | n/a | exploratory | too few pairs |
| style_distance (lower is better) | x | 3 | 0.457 | 0.926 | 0.469 | n/a | n/a | n/a | exploratory | too few pairs |
| persona_utility | all | 8 | 0.625 | 0.668 | 0.043 | -0.014 to 0.092 | 0.180 | 1.000 | exploratory | no clear difference |
| persona_utility | blog | 8 | 0.625 | 0.668 | 0.043 | -0.014 to 0.092 | 0.180 | 1.000 | exploratory | no clear difference |
| claim_support | all | 43 | 1.000 | 0.995 | -0.005 | -0.014 to 0.000 | 1.000 | 1.000 | exploratory | no clear difference |
| claim_support | blog | 16 | 1.000 | 1.000 | 0.000 | 0.000 to 0.000 | 1.000 | 1.000 | exploratory | no clear difference |
| claim_support | linkedin | 1 | 1.000 | 1.000 | 0.000 | n/a | n/a | n/a | exploratory | too few pairs |
| claim_support | x | 3 | 1.000 | 0.933 | -0.067 | n/a | n/a | n/a | exploratory | too few pairs |
| claim_support | talk | 23 | 1.000 | 1.000 | 0.000 | 0.000 to 0.000 | 1.000 | 1.000 | exploratory | no clear difference |
| evidence_support | all | 51 | 0.941 | 0.951 | 0.010 | -0.046 to 0.070 | 1.000 | 1.000 | exploratory | no clear difference |
| evidence_support | blog | 20 | 0.975 | 0.950 | -0.025 | -0.100 to 0.050 | 1.000 | 1.000 | exploratory | no clear difference |
| evidence_support | linkedin | 3 | 1.000 | 1.000 | 0.000 | n/a | n/a | n/a | exploratory | too few pairs |
| evidence_support | x | 3 | 0.500 | 0.667 | 0.167 | n/a | n/a | n/a | exploratory | too few pairs |
| evidence_support | talk | 25 | 0.960 | 0.980 | 0.020 | -0.060 to 0.120 | 1.000 | 1.000 | exploratory | no clear difference |
| word_count | all | 60 | 377.500 | 254.917 | -122.583 | -149.001 to -94.362 | 0.000 | 0.002 | exploratory | B lower |
| word_count | blog | 26 | 380.962 | 254.769 | -126.192 | -158.424 to -93.923 | 0.000 | 0.002 | exploratory | B lower |
| word_count | linkedin | 3 | 238.000 | 116.000 | -122.000 | n/a | n/a | n/a | exploratory | too few pairs |
| word_count | x | 3 | 125.000 | 36.667 | -88.333 | n/a | n/a | n/a | exploratory | too few pairs |
| word_count | talk | 28 | 416.286 | 293.321 | -122.964 | -158.143 to -88.429 | 0.000 | 0.002 | exploratory | B lower |

Skipped (no paired values in the files): grammar_rate, grammar_deviation

How to read the verdict: a difference is called clear only when the interval excludes zero and `p adj` is below 0.05. If it looks real on its own (interval excludes zero, p below 0.05) but `p adj` is not below 0.05, it is marked (suggestive): a lead worth a second run, not a finding. Rows by medium are exploratory unless listed as primary.

Limits: one run per arm unless more seeds are added, and the 30 gold topics are a development panel. A result that is not clear here is not proof of no difference.
