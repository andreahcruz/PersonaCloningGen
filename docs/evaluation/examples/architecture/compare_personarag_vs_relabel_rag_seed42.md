# Comparison: Relabel + RAG seed 42 against PersonaRAG (base + retrieval)

Every difference is B minus A. Intervals are 95 percent, resampled by topic. p is a sign-flip test by topic. `p adj` corrects for how many tests were run (Holm). The four primary rows, which are the four parts of Persona Utility, are corrected together. Every other row is exploratory and is corrected as its own, larger family.

Paired drafts: 120 (only in PersonaRAG (base + retrieval): 0, only in Relabel + RAG seed 42: 0).

## Gates

| Run | Drafts | Eligible (all gates) | Pass common gates | Completion fails | Copying fails | Format fails | Grounding fails |
|---|---|---|---|---|---|---|---|
| PersonaRAG (base + retrieval) | 120 | 57 (48%) | 71 (59%) | 49 | 0 | 0 | 20 of 120 |
| Relabel + RAG seed 42 | 120 | 72 (60%) | 83 (69%) | 36 | 2 | 3 | 14 of 120 |

Grounding is counted only where it applies. A run with retrieval has one extra gate, so its eligible count is not comparable to a run without retrieval. Use the common gates column.

Style distance role by medium (selector means it may rank models): blog selector, linkedin diagnostic_only, x diagnostic_only, talk diagnostic_only

## All drafts (no draft dropped)

Pairs in this view: 120

| Metric | Scope | n | PersonaRAG (base + retrieval) mean | Relabel + RAG seed 42 mean | Diff | 95% interval | p | p adj | Family | Verdict |
|---|---|---|---|---|---|---|---|---|---|---|
| writing_quality | all | 120 | 3.599 | 2.896 | -0.703 | -0.850 to -0.553 | 0.000 | 0.000 | primary | B worse |
| writing_quality | blog | 30 | 3.722 | 3.356 | -0.367 | -0.594 to -0.144 | 0.004 | 0.087 | exploratory | B worse (suggestive) |
| writing_quality | linkedin | 30 | 3.761 | 2.922 | -0.839 | -1.200 to -0.494 | 0.000 | 0.003 | exploratory | B worse |
| writing_quality | x | 30 | 3.122 | 2.006 | -1.117 | -1.350 to -0.867 | 0.000 | 0.003 | exploratory | B worse |
| writing_quality | talk | 30 | 3.789 | 3.300 | -0.489 | -0.694 to -0.278 | 0.000 | 0.007 | exploratory | B worse |
| task_adherence | all | 120 | 0.725 | 0.709 | -0.016 | -0.042 to 0.007 | 0.241 | 0.481 | primary | no clear difference |
| task_adherence | blog | 30 | 0.746 | 0.709 | -0.037 | -0.063 to -0.010 | 0.010 | 0.200 | exploratory | B worse (suggestive) |
| task_adherence | linkedin | 30 | 0.710 | 0.703 | -0.007 | -0.086 to 0.056 | 0.864 | 1.000 | exploratory | no clear difference |
| task_adherence | x | 30 | 0.712 | 0.719 | 0.008 | -0.013 to 0.029 | 0.481 | 1.000 | exploratory | no clear difference |
| task_adherence | talk | 30 | 0.732 | 0.706 | -0.026 | -0.063 to 0.009 | 0.161 | 1.000 | exploratory | no clear difference |
| stance_consistency | all | 87 | 0.989 | 0.970 | -0.018 | -0.056 to 0.019 | 0.410 | 0.481 | primary | no clear difference |
| stance_consistency | blog | 20 | 1.000 | 0.975 | -0.025 | -0.075 to 0.000 | 1.000 | 1.000 | exploratory | no clear difference |
| stance_consistency | linkedin | 24 | 1.000 | 0.938 | -0.062 | -0.167 to 0.000 | 0.495 | 1.000 | exploratory | no clear difference |
| stance_consistency | x | 19 | 1.000 | 0.982 | -0.018 | -0.053 to 0.000 | 1.000 | 1.000 | exploratory | no clear difference |
| stance_consistency | talk | 24 | 0.958 | 0.990 | 0.031 | -0.031 to 0.125 | 1.000 | 1.000 | exploratory | no clear difference |
| style_distance (lower is better) | all | 90 | 0.906 | 0.805 | -0.101 | -0.167 to -0.025 | 0.011 | 0.205 | exploratory | B better (suggestive) |
| style_distance (lower is better) | blog | 30 | 1.552 | 0.814 | -0.739 | -0.879 to -0.589 | 0.000 | 0.000 | primary | B better |
| style_distance (lower is better) | linkedin | 30 | 0.682 | 0.828 | 0.145 | -0.009 to 0.323 | 0.098 | 1.000 | exploratory | no clear difference |
| style_distance (lower is better) | x | 30 | 0.484 | 0.775 | 0.291 | 0.223 to 0.363 | 0.000 | 0.003 | exploratory | B worse |
| persona_utility | all | 10 | 0.628 | 0.675 | 0.047 | 0.027 to 0.068 | 0.008 | 0.172 | exploratory | B better (suggestive) |
| persona_utility | blog | 10 | 0.628 | 0.675 | 0.047 | 0.027 to 0.068 | 0.008 | 0.172 | exploratory | B better (suggestive) |
| claim_support | all | 93 | 1.000 | 0.995 | -0.005 | -0.012 to 0.000 | 0.504 | 1.000 | exploratory | no clear difference |
| claim_support | blog | 23 | 1.000 | 1.000 | 0.000 | 0.000 to 0.000 | 1.000 | 1.000 | exploratory | no clear difference |
| claim_support | linkedin | 23 | 1.000 | 1.000 | 0.000 | 0.000 to 0.000 | 1.000 | 1.000 | exploratory | no clear difference |
| claim_support | x | 25 | 1.000 | 0.992 | -0.008 | -0.024 to 0.000 | 1.000 | 1.000 | exploratory | no clear difference |
| claim_support | talk | 22 | 1.000 | 0.989 | -0.011 | -0.034 to 0.000 | 1.000 | 1.000 | exploratory | no clear difference |
| evidence_support | all | 86 | 0.965 | 0.961 | -0.004 | -0.045 to 0.036 | 0.901 | 1.000 | exploratory | no clear difference |
| evidence_support | blog | 21 | 0.976 | 0.944 | -0.032 | -0.127 to 0.048 | 0.754 | 1.000 | exploratory | no clear difference |
| evidence_support | linkedin | 23 | 1.000 | 0.980 | -0.020 | -0.050 to 0.000 | 0.499 | 1.000 | exploratory | no clear difference |
| evidence_support | x | 20 | 0.925 | 0.925 | 0.000 | -0.075 to 0.075 | 1.000 | 1.000 | exploratory | no clear difference |
| evidence_support | talk | 22 | 0.955 | 0.989 | 0.034 | -0.034 to 0.136 | 1.000 | 1.000 | exploratory | no clear difference |
| word_count | all | 120 | 296.475 | 183.067 | -113.408 | -130.500 to -96.266 | 0.000 | 0.003 | exploratory | B lower |
| word_count | blog | 30 | 390.900 | 259.700 | -131.200 | -170.667 to -92.265 | 0.000 | 0.003 | exploratory | B lower |
| word_count | linkedin | 30 | 241.967 | 151.933 | -90.033 | -122.433 to -58.899 | 0.000 | 0.003 | exploratory | B lower |
| word_count | x | 30 | 124.567 | 39.467 | -85.100 | -98.800 to -69.466 | 0.000 | 0.003 | exploratory | B lower |
| word_count | talk | 30 | 428.467 | 281.167 | -147.300 | -188.333 to -108.164 | 0.000 | 0.003 | exploratory | B lower |

## Drafts that pass the common gates in both runs

Pairs in this view: 59

| Metric | Scope | n | PersonaRAG (base + retrieval) mean | Relabel + RAG seed 42 mean | Diff | 95% interval | p | p adj | Family | Verdict |
|---|---|---|---|---|---|---|---|---|---|---|
| writing_quality | all | 59 | 3.715 | 3.282 | -0.432 | -0.616 to -0.244 | 0.000 | 0.001 | primary | B worse |
| writing_quality | blog | 26 | 3.692 | 3.378 | -0.314 | -0.545 to -0.083 | 0.018 | 0.199 | exploratory | B worse (suggestive) |
| writing_quality | linkedin | 2 | 3.333 | 3.250 | -0.083 | n/a | n/a | n/a | exploratory | too few pairs |
| writing_quality | x | 2 | 3.167 | 1.667 | -1.500 | n/a | n/a | n/a | exploratory | too few pairs |
| writing_quality | talk | 29 | 3.799 | 3.310 | -0.488 | -0.701 to -0.270 | 0.000 | 0.003 | exploratory | B worse |
| task_adherence | all | 59 | 0.732 | 0.707 | -0.026 | -0.053 to 0.002 | 0.082 | 0.164 | primary | no clear difference |
| task_adherence | blog | 26 | 0.741 | 0.704 | -0.037 | -0.066 to -0.006 | 0.029 | 0.286 | exploratory | B worse (suggestive) |
| task_adherence | linkedin | 2 | 0.711 | 0.750 | 0.039 | n/a | n/a | n/a | exploratory | too few pairs |
| task_adherence | x | 2 | 0.699 | 0.724 | 0.025 | n/a | n/a | n/a | exploratory | too few pairs |
| task_adherence | talk | 29 | 0.729 | 0.705 | -0.024 | -0.060 to 0.012 | 0.219 | 1.000 | exploratory | no clear difference |
| stance_consistency | all | 43 | 0.977 | 0.983 | 0.006 | -0.040 to 0.065 | 1.000 | 1.000 | primary | no clear difference |
| stance_consistency | blog | 17 | 1.000 | 0.971 | -0.029 | -0.088 to 0.000 | 1.000 | 1.000 | exploratory | no clear difference |
| stance_consistency | linkedin | 2 | 1.000 | 1.000 | 0.000 | n/a | n/a | n/a | exploratory | too few pairs |
| stance_consistency | x | 1 | 1.000 | 1.000 | 0.000 | n/a | n/a | n/a | exploratory | too few pairs |
| stance_consistency | talk | 23 | 0.957 | 0.989 | 0.033 | -0.033 to 0.130 | 1.000 | 1.000 | exploratory | no clear difference |
| style_distance (lower is better) | all | 30 | 1.416 | 0.769 | -0.647 | -0.828 to -0.483 | 0.000 | 0.002 | exploratory | B better |
| style_distance (lower is better) | blog | 26 | 1.543 | 0.750 | -0.792 | -0.912 to -0.682 | 0.000 | 0.000 | primary | B better |
| style_distance (lower is better) | linkedin | 2 | 0.722 | 0.663 | -0.058 | n/a | n/a | n/a | exploratory | too few pairs |
| style_distance (lower is better) | x | 2 | 0.466 | 1.125 | 0.659 | n/a | n/a | n/a | exploratory | too few pairs |
| persona_utility | all | 10 | 0.628 | 0.675 | 0.047 | 0.027 to 0.068 | 0.008 | 0.102 | exploratory | B better (suggestive) |
| persona_utility | blog | 10 | 0.628 | 0.675 | 0.047 | 0.027 to 0.068 | 0.008 | 0.102 | exploratory | B better (suggestive) |
| claim_support | all | 43 | 1.000 | 0.990 | -0.010 | -0.026 to 0.000 | 0.509 | 1.000 | exploratory | no clear difference |
| claim_support | blog | 19 | 1.000 | 1.000 | 0.000 | 0.000 to 0.000 | 1.000 | 1.000 | exploratory | no clear difference |
| claim_support | linkedin | 1 | 1.000 | 1.000 | 0.000 | n/a | n/a | n/a | exploratory | too few pairs |
| claim_support | x | 2 | 1.000 | 0.900 | -0.100 | n/a | n/a | n/a | exploratory | too few pairs |
| claim_support | talk | 21 | 1.000 | 0.988 | -0.012 | -0.036 to 0.000 | 1.000 | 1.000 | exploratory | no clear difference |
| evidence_support | all | 43 | 0.930 | 0.944 | 0.014 | -0.057 to 0.083 | 0.747 | 1.000 | exploratory | no clear difference |
| evidence_support | blog | 18 | 0.972 | 0.935 | -0.037 | -0.148 to 0.056 | 0.744 | 1.000 | exploratory | no clear difference |
| evidence_support | linkedin | 2 | 1.000 | 1.000 | 0.000 | n/a | n/a | n/a | exploratory | too few pairs |
| evidence_support | x | 2 | 0.250 | 0.500 | 0.250 | n/a | n/a | n/a | exploratory | too few pairs |
| evidence_support | talk | 21 | 0.952 | 0.988 | 0.036 | -0.036 to 0.143 | 1.000 | 1.000 | exploratory | no clear difference |
| word_count | all | 59 | 389.322 | 264.593 | -124.729 | -155.196 to -94.499 | 0.000 | 0.002 | exploratory | B lower |
| word_count | blog | 26 | 387.000 | 271.962 | -115.038 | -152.154 to -78.767 | 0.000 | 0.002 | exploratory | B lower |
| word_count | linkedin | 2 | 212.000 | 184.500 | -27.500 | n/a | n/a | n/a | exploratory | too few pairs |
| word_count | x | 2 | 127.000 | 15.000 | -112.000 | n/a | n/a | n/a | exploratory | too few pairs |
| word_count | talk | 29 | 421.724 | 280.724 | -141.000 | -183.172 to -101.275 | 0.000 | 0.002 | exploratory | B lower |

Skipped (no paired values in the files): grammar_rate, grammar_deviation

How to read the verdict: a difference is called clear only when the interval excludes zero and `p adj` is below 0.05. If it looks real on its own (interval excludes zero, p below 0.05) but `p adj` is not below 0.05, it is marked (suggestive): a lead worth a second run, not a finding. Rows by medium are exploratory unless listed as primary.

Limits: one run per arm unless more seeds are added, and the 30 gold topics are a development panel. A result that is not clear here is not proof of no difference.
