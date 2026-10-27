# Comparison: PersonaRAG (base + retrieval) against Base Llama 3.1 (no RAG)

Every difference is B minus A. Intervals are 95 percent, resampled by topic. p is a sign-flip test by topic. `p adj` corrects for how many tests were run (Holm). The four primary rows, which are the four parts of Persona Utility, are corrected together. Every other row is exploratory and is corrected as its own, larger family.

Paired drafts: 120 (only in Base Llama 3.1 (no RAG): 0, only in PersonaRAG (base + retrieval): 0).

## Gates

| Run | Drafts | Eligible (all gates) | Pass common gates | Completion fails | Copying fails | Format fails | Grounding fails |
|---|---|---|---|---|---|---|---|
| Base Llama 3.1 (no RAG) | 120 | 45 (38%) | 45 (38%) | 75 | 0 | 0 | 0 of 0 |
| PersonaRAG (base + retrieval) | 120 | 57 (48%) | 71 (59%) | 49 | 0 | 0 | 20 of 120 |

Grounding is counted only where it applies. A run with retrieval has one extra gate, so its eligible count is not comparable to a run without retrieval. Use the common gates column.

Style distance role by medium (selector means it may rank models): blog selector, linkedin diagnostic_only, x diagnostic_only, talk diagnostic_only

## All drafts (no draft dropped)

Pairs in this view: 120

| Metric | Scope | n | Base Llama 3.1 (no RAG) mean | PersonaRAG (base + retrieval) mean | Diff | 95% interval | p | p adj | Family | Verdict |
|---|---|---|---|---|---|---|---|---|---|---|
| writing_quality | all | 120 | 3.578 | 3.599 | 0.021 | -0.078 to 0.122 | 0.707 | 1.000 | primary | no clear difference |
| writing_quality | blog | 30 | 3.639 | 3.722 | 0.083 | -0.100 to 0.278 | 0.405 | 1.000 | exploratory | no clear difference |
| writing_quality | linkedin | 30 | 3.844 | 3.761 | -0.083 | -0.222 to 0.044 | 0.208 | 1.000 | exploratory | no clear difference |
| writing_quality | x | 30 | 3.061 | 3.122 | 0.061 | -0.195 to 0.317 | 0.674 | 1.000 | exploratory | no clear difference |
| writing_quality | talk | 30 | 3.767 | 3.789 | 0.022 | -0.100 to 0.150 | 0.794 | 1.000 | exploratory | no clear difference |
| task_adherence | all | 120 | 0.743 | 0.725 | -0.018 | -0.030 to -0.007 | 0.005 | 0.016 | primary | B worse |
| task_adherence | blog | 30 | 0.760 | 0.746 | -0.014 | -0.030 to 0.003 | 0.118 | 1.000 | exploratory | no clear difference |
| task_adherence | linkedin | 30 | 0.746 | 0.710 | -0.036 | -0.056 to -0.017 | 0.002 | 0.044 | exploratory | B worse |
| task_adherence | x | 30 | 0.704 | 0.712 | 0.008 | -0.015 to 0.030 | 0.512 | 1.000 | exploratory | no clear difference |
| task_adherence | talk | 30 | 0.763 | 0.732 | -0.031 | -0.054 to -0.009 | 0.014 | 0.294 | exploratory | B worse (suggestive) |
| stance_consistency | all | 92 | 0.967 | 0.989 | 0.022 | -0.021 to 0.066 | 0.624 | 1.000 | primary | no clear difference |
| stance_consistency | blog | 23 | 1.000 | 1.000 | 0.000 | 0.000 to 0.000 | 1.000 | 1.000 | exploratory | no clear difference |
| stance_consistency | linkedin | 21 | 0.952 | 1.000 | 0.048 | 0.000 to 0.143 | 1.000 | 1.000 | exploratory | no clear difference |
| stance_consistency | x | 22 | 0.909 | 1.000 | 0.091 | 0.000 to 0.227 | 0.503 | 1.000 | exploratory | no clear difference |
| stance_consistency | talk | 26 | 1.000 | 0.962 | -0.038 | -0.115 to 0.000 | 1.000 | 1.000 | exploratory | no clear difference |
| style_distance (lower is better) | all | 90 | 1.043 | 0.906 | -0.137 | -0.182 to -0.092 | 0.000 | 0.003 | exploratory | B better |
| style_distance (lower is better) | blog | 30 | 1.867 | 1.552 | -0.315 | -0.437 to -0.196 | 0.000 | 0.000 | primary | B better |
| style_distance (lower is better) | linkedin | 30 | 0.740 | 0.682 | -0.057 | -0.117 to -0.002 | 0.065 | 1.000 | exploratory | no clear difference |
| style_distance (lower is better) | x | 30 | 0.523 | 0.484 | -0.039 | -0.079 to -0.001 | 0.065 | 1.000 | exploratory | no clear difference |
| persona_utility | all | 14 | 0.614 | 0.633 | 0.018 | 0.003 to 0.033 | 0.040 | 0.801 | exploratory | B better (suggestive) |
| persona_utility | blog | 14 | 0.614 | 0.633 | 0.018 | 0.003 to 0.033 | 0.040 | 0.801 | exploratory | B better (suggestive) |
| evidence_support | all | 97 | 0.988 | 0.969 | -0.019 | -0.059 to 0.019 | 0.444 | 1.000 | exploratory | no clear difference |
| evidence_support | blog | 25 | 1.000 | 0.980 | -0.020 | -0.060 to 0.000 | 1.000 | 1.000 | exploratory | no clear difference |
| evidence_support | linkedin | 22 | 1.000 | 1.000 | 0.000 | 0.000 to 0.000 | 1.000 | 1.000 | exploratory | no clear difference |
| evidence_support | x | 24 | 0.950 | 0.938 | -0.013 | -0.138 to 0.113 | 1.000 | 1.000 | exploratory | no clear difference |
| evidence_support | talk | 26 | 1.000 | 0.962 | -0.038 | -0.115 to 0.000 | 1.000 | 1.000 | exploratory | no clear difference |
| word_count | all | 120 | 352.108 | 296.475 | -55.633 | -68.775 to -42.058 | 0.000 | 0.003 | exploratory | B lower |
| word_count | blog | 30 | 515.200 | 390.900 | -124.300 | -155.668 to -92.166 | 0.000 | 0.003 | exploratory | B lower |
| word_count | linkedin | 30 | 240.167 | 241.967 | 1.800 | -8.533 to 11.833 | 0.740 | 1.000 | exploratory | no clear difference |
| word_count | x | 30 | 116.700 | 124.567 | 7.867 | 4.333 to 10.967 | 0.000 | 0.003 | exploratory | B higher |
| word_count | talk | 30 | 536.367 | 428.467 | -107.900 | -137.268 to -76.632 | 0.000 | 0.003 | exploratory | B lower |

## Drafts that pass the common gates in both runs

Pairs in this view: 37

| Metric | Scope | n | Base Llama 3.1 (no RAG) mean | PersonaRAG (base + retrieval) mean | Diff | 95% interval | p | p adj | Family | Verdict |
|---|---|---|---|---|---|---|---|---|---|---|
| writing_quality | all | 37 | 3.730 | 3.775 | 0.045 | -0.073 to 0.162 | 0.511 | 1.000 | primary | no clear difference |
| writing_quality | blog | 19 | 3.623 | 3.702 | 0.079 | -0.140 to 0.307 | 0.534 | 1.000 | exploratory | no clear difference |
| writing_quality | linkedin | 2 | 3.833 | 3.917 | 0.083 | n/a | n/a | n/a | exploratory | too few pairs |
| writing_quality | talk | 16 | 3.844 | 3.844 | 0.000 | -0.083 to 0.073 | 1.000 | 1.000 | exploratory | no clear difference |
| task_adherence | all | 37 | 0.755 | 0.739 | -0.016 | -0.035 to 0.002 | 0.111 | 0.334 | primary | no clear difference |
| task_adherence | blog | 19 | 0.757 | 0.746 | -0.011 | -0.033 to 0.008 | 0.341 | 1.000 | exploratory | no clear difference |
| task_adherence | linkedin | 2 | 0.770 | 0.802 | 0.033 | n/a | n/a | n/a | exploratory | too few pairs |
| task_adherence | talk | 16 | 0.751 | 0.724 | -0.027 | -0.058 to 0.003 | 0.130 | 1.000 | exploratory | no clear difference |
| stance_consistency | all | 32 | 0.969 | 1.000 | 0.031 | 0.000 to 0.097 | 1.000 | 1.000 | primary | no clear difference |
| stance_consistency | blog | 16 | 1.000 | 1.000 | 0.000 | 0.000 to 0.000 | 1.000 | 1.000 | exploratory | no clear difference |
| stance_consistency | linkedin | 2 | 0.500 | 1.000 | 0.500 | n/a | n/a | n/a | exploratory | too few pairs |
| stance_consistency | talk | 14 | 1.000 | 1.000 | 0.000 | 0.000 to 0.000 | 1.000 | 1.000 | exploratory | no clear difference |
| style_distance (lower is better) | all | 21 | 1.725 | 1.458 | -0.267 | -0.398 to -0.137 | 0.002 | 0.022 | exploratory | B better |
| style_distance (lower is better) | blog | 19 | 1.819 | 1.539 | -0.280 | -0.419 to -0.134 | 0.003 | 0.010 | primary | B better |
| style_distance (lower is better) | linkedin | 2 | 0.833 | 0.689 | -0.144 | n/a | n/a | n/a | exploratory | too few pairs |
| persona_utility | all | 14 | 0.614 | 0.633 | 0.018 | 0.003 to 0.033 | 0.040 | 0.440 | exploratory | B better (suggestive) |
| persona_utility | blog | 14 | 0.614 | 0.633 | 0.018 | 0.003 to 0.033 | 0.040 | 0.440 | exploratory | B better (suggestive) |
| evidence_support | all | 31 | 1.000 | 0.984 | -0.016 | -0.056 to 0.000 | 1.000 | 1.000 | exploratory | no clear difference |
| evidence_support | blog | 16 | 1.000 | 0.969 | -0.031 | -0.094 to 0.000 | 1.000 | 1.000 | exploratory | no clear difference |
| evidence_support | linkedin | 2 | 1.000 | 1.000 | 0.000 | n/a | n/a | n/a | exploratory | too few pairs |
| evidence_support | talk | 13 | 1.000 | 1.000 | 0.000 | 0.000 to 0.000 | 1.000 | 1.000 | exploratory | no clear difference |
| word_count | all | 37 | 497.081 | 385.135 | -111.946 | -143.343 to -78.783 | 0.000 | 0.002 | exploratory | B lower |
| word_count | blog | 19 | 514.737 | 386.474 | -128.263 | -160.532 to -95.316 | 0.000 | 0.002 | exploratory | B lower |
| word_count | linkedin | 2 | 245.500 | 242.000 | -3.500 | n/a | n/a | n/a | exploratory | too few pairs |
| word_count | talk | 16 | 507.562 | 401.438 | -106.125 | -148.439 to -58.188 | 0.001 | 0.012 | exploratory | B lower |

Skipped (no paired values in the files): claim_support, grammar_rate, grammar_deviation

How to read the verdict: a difference is called clear only when the interval excludes zero and `p adj` is below 0.05. If it looks real on its own (interval excludes zero, p below 0.05) but `p adj` is not below 0.05, it is marked (suggestive): a lead worth a second run, not a finding. Rows by medium are exploratory unless listed as primary.

Limits: one run per arm unless more seeds are added, and the 30 gold topics are a development panel. A result that is not clear here is not proof of no difference.
