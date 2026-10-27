# Comparison: Relabel + RAG seed 43 against PersonaRAG (base + retrieval)

Every difference is B minus A. Intervals are 95 percent, resampled by topic. p is a sign-flip test by topic. `p adj` corrects for how many tests were run (Holm). The four primary rows, which are the four parts of Persona Utility, are corrected together. Every other row is exploratory and is corrected as its own, larger family.

Paired drafts: 120 (only in PersonaRAG (base + retrieval): 0, only in Relabel + RAG seed 43: 0).

## Gates

| Run | Drafts | Eligible (all gates) | Pass common gates | Completion fails | Copying fails | Format fails | Grounding fails |
|---|---|---|---|---|---|---|---|
| PersonaRAG (base + retrieval) | 120 | 57 (48%) | 71 (59%) | 49 | 0 | 0 | 20 of 120 |
| Relabel + RAG seed 43 | 120 | 72 (60%) | 82 (68%) | 36 | 1 | 3 | 16 of 120 |

Grounding is counted only where it applies. A run with retrieval has one extra gate, so its eligible count is not comparable to a run without retrieval. Use the common gates column.

Style distance role by medium (selector means it may rank models): blog selector, linkedin diagnostic_only, x diagnostic_only, talk diagnostic_only

## All drafts (no draft dropped)

Pairs in this view: 120

| Metric | Scope | n | PersonaRAG (base + retrieval) mean | Relabel + RAG seed 43 mean | Diff | 95% interval | p | p adj | Family | Verdict |
|---|---|---|---|---|---|---|---|---|---|---|
| writing_quality | all | 120 | 3.599 | 2.972 | -0.626 | -0.747 to -0.512 | 0.000 | 0.000 | primary | B worse |
| writing_quality | blog | 30 | 3.722 | 3.372 | -0.350 | -0.556 to -0.139 | 0.002 | 0.053 | exploratory | B worse (suggestive) |
| writing_quality | linkedin | 30 | 3.761 | 3.344 | -0.417 | -0.722 to -0.133 | 0.010 | 0.204 | exploratory | B worse (suggestive) |
| writing_quality | x | 30 | 3.122 | 2.000 | -1.122 | -1.378 to -0.867 | 0.000 | 0.003 | exploratory | B worse |
| writing_quality | talk | 30 | 3.789 | 3.172 | -0.617 | -0.822 to -0.417 | 0.000 | 0.003 | exploratory | B worse |
| task_adherence | all | 120 | 0.725 | 0.710 | -0.015 | -0.037 to 0.005 | 0.165 | 0.329 | primary | no clear difference |
| task_adherence | blog | 30 | 0.746 | 0.712 | -0.034 | -0.059 to -0.008 | 0.016 | 0.302 | exploratory | B worse (suggestive) |
| task_adherence | linkedin | 30 | 0.710 | 0.719 | 0.008 | -0.053 to 0.054 | 0.761 | 1.000 | exploratory | no clear difference |
| task_adherence | x | 30 | 0.712 | 0.722 | 0.011 | -0.010 to 0.032 | 0.339 | 1.000 | exploratory | no clear difference |
| task_adherence | talk | 30 | 0.732 | 0.686 | -0.046 | -0.096 to -0.004 | 0.053 | 0.956 | exploratory | no clear difference |
| stance_consistency | all | 88 | 0.989 | 0.966 | -0.023 | -0.068 to 0.017 | 0.441 | 0.441 | primary | no clear difference |
| stance_consistency | blog | 24 | 1.000 | 0.958 | -0.042 | -0.104 to 0.000 | 0.501 | 1.000 | exploratory | no clear difference |
| stance_consistency | linkedin | 21 | 1.000 | 1.000 | 0.000 | 0.000 to 0.000 | 1.000 | 1.000 | exploratory | no clear difference |
| stance_consistency | x | 18 | 1.000 | 1.000 | 0.000 | 0.000 to 0.000 | 1.000 | 1.000 | exploratory | no clear difference |
| stance_consistency | talk | 25 | 0.960 | 0.920 | -0.040 | -0.160 to 0.080 | 1.000 | 1.000 | exploratory | no clear difference |
| style_distance (lower is better) | all | 90 | 0.906 | 0.760 | -0.146 | -0.216 to -0.075 | 0.001 | 0.015 | exploratory | B better |
| style_distance (lower is better) | blog | 30 | 1.552 | 0.845 | -0.707 | -0.849 to -0.570 | 0.000 | 0.000 | primary | B better |
| style_distance (lower is better) | linkedin | 30 | 0.682 | 0.664 | -0.018 | -0.129 to 0.116 | 0.786 | 1.000 | exploratory | no clear difference |
| style_distance (lower is better) | x | 30 | 0.484 | 0.770 | 0.286 | 0.210 to 0.367 | 0.000 | 0.003 | exploratory | B worse |
| persona_utility | all | 15 | 0.629 | 0.675 | 0.046 | 0.020 to 0.068 | 0.004 | 0.095 | exploratory | B better (suggestive) |
| persona_utility | blog | 15 | 0.629 | 0.675 | 0.046 | 0.020 to 0.068 | 0.004 | 0.095 | exploratory | B better (suggestive) |
| claim_support | all | 93 | 1.000 | 0.998 | -0.002 | -0.007 to 0.000 | 1.000 | 1.000 | exploratory | no clear difference |
| claim_support | blog | 22 | 1.000 | 1.000 | 0.000 | 0.000 to 0.000 | 1.000 | 1.000 | exploratory | no clear difference |
| claim_support | linkedin | 21 | 1.000 | 1.000 | 0.000 | 0.000 to 0.000 | 1.000 | 1.000 | exploratory | no clear difference |
| claim_support | x | 26 | 1.000 | 0.992 | -0.008 | -0.023 to 0.000 | 1.000 | 1.000 | exploratory | no clear difference |
| claim_support | talk | 24 | 1.000 | 1.000 | 0.000 | 0.000 to 0.000 | 1.000 | 1.000 | exploratory | no clear difference |
| evidence_support | all | 84 | 0.964 | 0.931 | -0.034 | -0.080 to 0.013 | 0.243 | 1.000 | exploratory | no clear difference |
| evidence_support | blog | 24 | 0.979 | 0.938 | -0.042 | -0.104 to 0.000 | 0.500 | 1.000 | exploratory | no clear difference |
| evidence_support | linkedin | 18 | 1.000 | 0.972 | -0.028 | -0.083 to 0.000 | 1.000 | 1.000 | exploratory | no clear difference |
| evidence_support | x | 18 | 0.917 | 0.917 | 0.000 | -0.083 to 0.083 | 1.000 | 1.000 | exploratory | no clear difference |
| evidence_support | talk | 24 | 0.958 | 0.903 | -0.056 | -0.208 to 0.083 | 0.622 | 1.000 | exploratory | no clear difference |
| word_count | all | 120 | 296.475 | 190.058 | -106.417 | -127.025 to -86.483 | 0.000 | 0.003 | exploratory | B lower |
| word_count | blog | 30 | 390.900 | 243.600 | -147.300 | -197.834 to -99.300 | 0.000 | 0.003 | exploratory | B lower |
| word_count | linkedin | 30 | 241.967 | 193.900 | -48.067 | -77.833 to -21.833 | 0.002 | 0.038 | exploratory | B lower |
| word_count | x | 30 | 124.567 | 42.933 | -81.633 | -95.100 to -66.933 | 0.000 | 0.003 | exploratory | B lower |
| word_count | talk | 30 | 428.467 | 279.800 | -148.667 | -199.103 to -100.067 | 0.000 | 0.003 | exploratory | B lower |

## Drafts that pass the common gates in both runs

Pairs in this view: 57

| Metric | Scope | n | PersonaRAG (base + retrieval) mean | Relabel + RAG seed 43 mean | Diff | 95% interval | p | p adj | Family | Verdict |
|---|---|---|---|---|---|---|---|---|---|---|
| writing_quality | all | 57 | 3.731 | 3.284 | -0.447 | -0.576 to -0.319 | 0.000 | 0.000 | primary | B worse |
| writing_quality | blog | 26 | 3.699 | 3.397 | -0.301 | -0.519 to -0.083 | 0.014 | 0.153 | exploratory | B worse (suggestive) |
| writing_quality | linkedin | 1 | 3.833 | 3.833 | 0.000 | n/a | n/a | n/a | exploratory | too few pairs |
| writing_quality | x | 3 | 3.389 | 2.278 | -1.111 | n/a | n/a | n/a | exploratory | too few pairs |
| writing_quality | talk | 27 | 3.796 | 3.265 | -0.531 | -0.722 to -0.346 | 0.000 | 0.002 | exploratory | B worse |
| task_adherence | all | 57 | 0.732 | 0.703 | -0.029 | -0.055 to -0.002 | 0.048 | 0.095 | primary | B worse (suggestive) |
| task_adherence | blog | 26 | 0.742 | 0.707 | -0.034 | -0.062 to -0.006 | 0.029 | 0.290 | exploratory | B worse (suggestive) |
| task_adherence | linkedin | 1 | 0.675 | 0.775 | 0.100 | n/a | n/a | n/a | exploratory | too few pairs |
| task_adherence | x | 3 | 0.697 | 0.714 | 0.017 | n/a | n/a | n/a | exploratory | too few pairs |
| task_adherence | talk | 27 | 0.728 | 0.695 | -0.033 | -0.065 to -0.001 | 0.062 | 0.561 | exploratory | no clear difference |
| stance_consistency | all | 46 | 0.978 | 0.935 | -0.043 | -0.128 to 0.033 | 0.433 | 0.433 | primary | no clear difference |
| stance_consistency | blog | 20 | 1.000 | 0.950 | -0.050 | -0.125 to 0.000 | 0.495 | 1.000 | exploratory | no clear difference |
| stance_consistency | linkedin | 1 | 1.000 | 1.000 | 0.000 | n/a | n/a | n/a | exploratory | too few pairs |
| stance_consistency | x | 2 | 1.000 | 1.000 | 0.000 | n/a | n/a | n/a | exploratory | too few pairs |
| stance_consistency | talk | 23 | 0.957 | 0.913 | -0.043 | -0.174 to 0.087 | 1.000 | 1.000 | exploratory | no clear difference |
| style_distance (lower is better) | all | 30 | 1.428 | 0.808 | -0.620 | -0.821 to -0.439 | 0.000 | 0.002 | exploratory | B better |
| style_distance (lower is better) | blog | 26 | 1.565 | 0.803 | -0.762 | -0.914 to -0.611 | 0.000 | 0.000 | primary | B better |
| style_distance (lower is better) | linkedin | 1 | 0.758 | 0.642 | -0.116 | n/a | n/a | n/a | exploratory | too few pairs |
| style_distance (lower is better) | x | 3 | 0.457 | 0.902 | 0.444 | n/a | n/a | n/a | exploratory | too few pairs |
| persona_utility | all | 15 | 0.629 | 0.675 | 0.046 | 0.020 to 0.068 | 0.004 | 0.056 | exploratory | B better (suggestive) |
| persona_utility | blog | 15 | 0.629 | 0.675 | 0.046 | 0.020 to 0.068 | 0.004 | 0.056 | exploratory | B better (suggestive) |
| claim_support | all | 45 | 1.000 | 0.996 | -0.004 | -0.013 to 0.000 | 1.000 | 1.000 | exploratory | no clear difference |
| claim_support | blog | 19 | 1.000 | 1.000 | 0.000 | 0.000 to 0.000 | 1.000 | 1.000 | exploratory | no clear difference |
| claim_support | linkedin | 1 | 1.000 | 1.000 | 0.000 | n/a | n/a | n/a | exploratory | too few pairs |
| claim_support | x | 3 | 1.000 | 0.933 | -0.067 | n/a | n/a | n/a | exploratory | too few pairs |
| claim_support | talk | 22 | 1.000 | 1.000 | 0.000 | 0.000 to 0.000 | 1.000 | 1.000 | exploratory | no clear difference |
| evidence_support | all | 45 | 0.933 | 0.915 | -0.019 | -0.097 to 0.052 | 0.786 | 1.000 | exploratory | no clear difference |
| evidence_support | blog | 20 | 0.975 | 0.925 | -0.050 | -0.125 to 0.000 | 0.500 | 1.000 | exploratory | no clear difference |
| evidence_support | linkedin | 1 | 1.000 | 1.000 | 0.000 | n/a | n/a | n/a | exploratory | too few pairs |
| evidence_support | x | 3 | 0.500 | 0.667 | 0.167 | n/a | n/a | n/a | exploratory | too few pairs |
| evidence_support | talk | 21 | 0.952 | 0.937 | -0.016 | -0.159 to 0.127 | 1.000 | 1.000 | exploratory | no clear difference |
| word_count | all | 57 | 385.439 | 248.386 | -137.053 | -173.000 to -104.552 | 0.000 | 0.002 | exploratory | B lower |
| word_count | blog | 26 | 382.692 | 238.423 | -144.269 | -195.231 to -98.806 | 0.000 | 0.002 | exploratory | B lower |
| word_count | linkedin | 1 | 198.000 | 145.000 | -53.000 | n/a | n/a | n/a | exploratory | too few pairs |
| word_count | x | 3 | 125.000 | 35.000 | -90.000 | n/a | n/a | n/a | exploratory | too few pairs |
| word_count | talk | 27 | 423.963 | 285.519 | -138.444 | -175.704 to -102.481 | 0.000 | 0.002 | exploratory | B lower |

Skipped (no paired values in the files): grammar_rate, grammar_deviation

How to read the verdict: a difference is called clear only when the interval excludes zero and `p adj` is below 0.05. If it looks real on its own (interval excludes zero, p below 0.05) but `p adj` is not below 0.05, it is marked (suggestive): a lead worth a second run, not a finding. Rows by medium are exploratory unless listed as primary.

Limits: one run per arm unless more seeds are added, and the 30 gold topics are a development panel. A result that is not clear here is not proof of no difference.
