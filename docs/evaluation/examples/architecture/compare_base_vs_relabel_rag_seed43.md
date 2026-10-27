# Comparison: Relabel + RAG seed 43 against Base Llama 3.1 (no RAG)

Every difference is B minus A. Intervals are 95 percent, resampled by topic. p is a sign-flip test by topic. `p adj` corrects for how many tests were run (Holm). The four primary rows, which are the four parts of Persona Utility, are corrected together. Every other row is exploratory and is corrected as its own, larger family.

Paired drafts: 120 (only in Base Llama 3.1 (no RAG): 0, only in Relabel + RAG seed 43: 0).

## Gates

| Run | Drafts | Eligible (all gates) | Pass common gates | Completion fails | Copying fails | Format fails | Grounding fails |
|---|---|---|---|---|---|---|---|
| Base Llama 3.1 (no RAG) | 120 | 45 (38%) | 45 (38%) | 75 | 0 | 0 | 0 of 0 |
| Relabel + RAG seed 43 | 120 | 72 (60%) | 82 (68%) | 36 | 1 | 3 | 16 of 120 |

Grounding is counted only where it applies. A run with retrieval has one extra gate, so its eligible count is not comparable to a run without retrieval. Use the common gates column.

Style distance role by medium (selector means it may rank models): blog selector, linkedin diagnostic_only, x diagnostic_only, talk diagnostic_only

## All drafts (no draft dropped)

Pairs in this view: 120

| Metric | Scope | n | Base Llama 3.1 (no RAG) mean | Relabel + RAG seed 43 mean | Diff | 95% interval | p | p adj | Family | Verdict |
|---|---|---|---|---|---|---|---|---|---|---|
| writing_quality | all | 120 | 3.578 | 2.972 | -0.606 | -0.764 to -0.456 | 0.000 | 0.000 | primary | B worse |
| writing_quality | blog | 30 | 3.639 | 3.372 | -0.267 | -0.522 to 0.000 | 0.058 | 0.659 | exploratory | no clear difference |
| writing_quality | linkedin | 30 | 3.844 | 3.344 | -0.500 | -0.783 to -0.233 | 0.001 | 0.010 | exploratory | B worse |
| writing_quality | x | 30 | 3.061 | 2.000 | -1.061 | -1.361 to -0.756 | 0.000 | 0.003 | exploratory | B worse |
| writing_quality | talk | 30 | 3.767 | 3.172 | -0.594 | -0.794 to -0.394 | 0.000 | 0.003 | exploratory | B worse |
| task_adherence | all | 120 | 0.743 | 0.710 | -0.033 | -0.052 to -0.015 | 0.001 | 0.003 | primary | B worse |
| task_adherence | blog | 30 | 0.760 | 0.712 | -0.047 | -0.073 to -0.021 | 0.002 | 0.025 | exploratory | B worse |
| task_adherence | linkedin | 30 | 0.746 | 0.719 | -0.028 | -0.086 to 0.014 | 0.435 | 1.000 | exploratory | no clear difference |
| task_adherence | x | 30 | 0.704 | 0.722 | 0.018 | 0.001 to 0.036 | 0.055 | 0.659 | exploratory | no clear difference |
| task_adherence | talk | 30 | 0.763 | 0.686 | -0.077 | -0.120 to -0.041 | 0.000 | 0.006 | exploratory | B worse |
| stance_consistency | all | 86 | 0.984 | 0.965 | -0.019 | -0.052 to 0.004 | 0.248 | 0.248 | primary | no clear difference |
| stance_consistency | blog | 25 | 1.000 | 0.960 | -0.040 | -0.100 to 0.000 | 0.495 | 1.000 | exploratory | no clear difference |
| stance_consistency | linkedin | 21 | 0.937 | 1.000 | 0.063 | 0.000 to 0.175 | 0.500 | 1.000 | exploratory | no clear difference |
| stance_consistency | x | 16 | 1.000 | 1.000 | 0.000 | 0.000 to 0.000 | 1.000 | 1.000 | exploratory | no clear difference |
| stance_consistency | talk | 24 | 1.000 | 0.917 | -0.083 | -0.208 to 0.000 | 0.497 | 1.000 | exploratory | no clear difference |
| style_distance (lower is better) | all | 90 | 1.043 | 0.760 | -0.283 | -0.355 to -0.207 | 0.000 | 0.003 | exploratory | B better |
| style_distance (lower is better) | blog | 30 | 1.867 | 0.845 | -1.022 | -1.160 to -0.878 | 0.000 | 0.000 | primary | B better |
| style_distance (lower is better) | linkedin | 30 | 0.740 | 0.664 | -0.076 | -0.190 to 0.061 | 0.255 | 1.000 | exploratory | no clear difference |
| style_distance (lower is better) | x | 30 | 0.523 | 0.770 | 0.248 | 0.176 to 0.322 | 0.000 | 0.003 | exploratory | B worse |
| persona_utility | all | 15 | 0.619 | 0.686 | 0.068 | 0.042 to 0.092 | 0.000 | 0.006 | exploratory | B better |
| persona_utility | blog | 15 | 0.619 | 0.686 | 0.068 | 0.042 to 0.092 | 0.000 | 0.006 | exploratory | B better |
| evidence_support | all | 91 | 0.996 | 0.941 | -0.055 | -0.110 to -0.011 | 0.033 | 0.430 | exploratory | B worse (suggestive) |
| evidence_support | blog | 27 | 1.000 | 0.944 | -0.056 | -0.111 to 0.000 | 0.248 | 1.000 | exploratory | no clear difference |
| evidence_support | linkedin | 21 | 0.984 | 0.976 | -0.008 | -0.071 to 0.048 | 1.000 | 1.000 | exploratory | no clear difference |
| evidence_support | x | 17 | 1.000 | 0.941 | -0.059 | -0.176 to 0.000 | 1.000 | 1.000 | exploratory | no clear difference |
| evidence_support | talk | 26 | 1.000 | 0.910 | -0.090 | -0.205 to 0.000 | 0.252 | 1.000 | exploratory | no clear difference |
| word_count | all | 120 | 352.108 | 190.058 | -162.050 | -179.742 to -143.742 | 0.000 | 0.003 | exploratory | B lower |
| word_count | blog | 30 | 515.200 | 243.600 | -271.600 | -315.769 to -226.000 | 0.000 | 0.003 | exploratory | B lower |
| word_count | linkedin | 30 | 240.167 | 193.900 | -46.267 | -75.367 to -21.400 | 0.001 | 0.010 | exploratory | B lower |
| word_count | x | 30 | 116.700 | 42.933 | -73.767 | -86.967 to -59.400 | 0.000 | 0.003 | exploratory | B lower |
| word_count | talk | 30 | 536.367 | 279.800 | -256.567 | -304.733 to -208.067 | 0.000 | 0.003 | exploratory | B lower |

## Drafts that pass the common gates in both runs

Pairs in this view: 36

| Metric | Scope | n | Base Llama 3.1 (no RAG) mean | Relabel + RAG seed 43 mean | Diff | 95% interval | p | p adj | Family | Verdict |
|---|---|---|---|---|---|---|---|---|---|---|
| writing_quality | all | 36 | 3.731 | 3.352 | -0.380 | -0.611 to -0.137 | 0.010 | 0.021 | primary | B worse |
| writing_quality | blog | 18 | 3.630 | 3.463 | -0.167 | -0.518 to 0.194 | 0.399 | 1.000 | exploratory | no clear difference |
| writing_quality | linkedin | 1 | 4.000 | 4.000 | 0.000 | n/a | n/a | n/a | exploratory | too few pairs |
| writing_quality | x | 1 | 3.000 | 1.667 | -1.333 | n/a | n/a | n/a | exploratory | too few pairs |
| writing_quality | talk | 16 | 3.875 | 3.292 | -0.583 | -0.854 to -0.323 | 0.001 | 0.010 | exploratory | B worse |
| task_adherence | all | 36 | 0.754 | 0.707 | -0.047 | -0.075 to -0.020 | 0.002 | 0.005 | primary | B worse |
| task_adherence | blog | 18 | 0.762 | 0.694 | -0.069 | -0.100 to -0.037 | 0.001 | 0.010 | exploratory | B worse |
| task_adherence | linkedin | 1 | 0.750 | 0.778 | 0.028 | n/a | n/a | n/a | exploratory | too few pairs |
| task_adherence | x | 1 | 0.652 | 0.783 | 0.130 | n/a | n/a | n/a | exploratory | too few pairs |
| task_adherence | talk | 16 | 0.751 | 0.712 | -0.039 | -0.074 to -0.005 | 0.050 | 0.353 | exploratory | no clear difference |
| stance_consistency | all | 33 | 1.000 | 0.955 | -0.045 | -0.121 to 0.000 | 0.497 | 0.497 | primary | no clear difference |
| stance_consistency | blog | 17 | 1.000 | 0.971 | -0.029 | -0.088 to 0.000 | 1.000 | 1.000 | exploratory | no clear difference |
| stance_consistency | linkedin | 1 | 1.000 | 1.000 | 0.000 | n/a | n/a | n/a | exploratory | too few pairs |
| stance_consistency | x | 1 | 1.000 | 1.000 | 0.000 | n/a | n/a | n/a | exploratory | too few pairs |
| stance_consistency | talk | 14 | 1.000 | 0.929 | -0.071 | -0.214 to 0.000 | 1.000 | 1.000 | exploratory | no clear difference |
| style_distance (lower is better) | all | 20 | 1.710 | 0.804 | -0.906 | -1.075 to -0.727 | 0.000 | 0.002 | exploratory | B better |
| style_distance (lower is better) | blog | 18 | 1.808 | 0.808 | -1.001 | -1.138 to -0.878 | 0.000 | 0.000 | primary | B better |
| style_distance (lower is better) | linkedin | 1 | 0.917 | 0.717 | -0.200 | n/a | n/a | n/a | exploratory | too few pairs |
| style_distance (lower is better) | x | 1 | 0.727 | 0.826 | 0.099 | n/a | n/a | n/a | exploratory | too few pairs |
| persona_utility | all | 15 | 0.619 | 0.686 | 0.068 | 0.042 to 0.092 | 0.000 | 0.003 | exploratory | B better |
| persona_utility | blog | 15 | 0.619 | 0.686 | 0.068 | 0.042 to 0.092 | 0.000 | 0.003 | exploratory | B better |
| evidence_support | all | 34 | 1.000 | 0.941 | -0.059 | -0.141 to 0.000 | 0.247 | 1.000 | exploratory | no clear difference |
| evidence_support | blog | 18 | 1.000 | 0.944 | -0.056 | -0.139 to 0.000 | 0.506 | 1.000 | exploratory | no clear difference |
| evidence_support | linkedin | 1 | 1.000 | 1.000 | 0.000 | n/a | n/a | n/a | exploratory | too few pairs |
| evidence_support | x | 1 | 1.000 | 1.000 | 0.000 | n/a | n/a | n/a | exploratory | too few pairs |
| evidence_support | talk | 14 | 1.000 | 0.929 | -0.071 | -0.214 to 0.000 | 1.000 | 1.000 | exploratory | no clear difference |
| word_count | all | 36 | 499.111 | 252.667 | -246.444 | -285.298 to -211.428 | 0.000 | 0.002 | exploratory | B lower |
| word_count | blog | 18 | 525.056 | 233.778 | -291.278 | -340.779 to -247.778 | 0.000 | 0.002 | exploratory | B lower |
| word_count | linkedin | 1 | 227.000 | 172.000 | -55.000 | n/a | n/a | n/a | exploratory | too few pairs |
| word_count | x | 1 | 123.000 | 24.000 | -99.000 | n/a | n/a | n/a | exploratory | too few pairs |
| word_count | talk | 16 | 510.438 | 293.250 | -217.188 | -266.064 to -177.438 | 0.000 | 0.002 | exploratory | B lower |

Skipped (no paired values in the files): claim_support, grammar_rate, grammar_deviation

How to read the verdict: a difference is called clear only when the interval excludes zero and `p adj` is below 0.05. If it looks real on its own (interval excludes zero, p below 0.05) but `p adj` is not below 0.05, it is marked (suggestive): a lead worth a second run, not a finding. Rows by medium are exploratory unless listed as primary.

Limits: one run per arm unless more seeds are added, and the 30 gold topics are a development panel. A result that is not clear here is not proof of no difference.
