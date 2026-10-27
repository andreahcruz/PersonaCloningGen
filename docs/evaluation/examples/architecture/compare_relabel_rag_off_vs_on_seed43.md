# Comparison: Relabel RAG on seed 43 against Relabel RAG off seed 43

Every difference is B minus A. Intervals are 95 percent, resampled by topic. p is a sign-flip test by topic. `p adj` corrects for how many tests were run (Holm). The four primary rows, which are the four parts of Persona Utility, are corrected together. Every other row is exploratory and is corrected as its own, larger family.

Paired drafts: 120 (only in Relabel RAG off seed 43: 0, only in Relabel RAG on seed 43: 0).

## Gates

| Run | Drafts | Eligible (all gates) | Pass common gates | Completion fails | Copying fails | Format fails | Grounding fails |
|---|---|---|---|---|---|---|---|
| Relabel RAG off seed 43 | 120 | 81 (68%) | 81 (68%) | 36 | 1 | 7 | 0 of 0 |
| Relabel RAG on seed 43 | 120 | 72 (60%) | 82 (68%) | 36 | 1 | 3 | 16 of 120 |

Grounding is counted only where it applies. A run with retrieval has one extra gate, so its eligible count is not comparable to a run without retrieval. Use the common gates column.

Style distance role by medium (selector means it may rank models): blog selector, linkedin diagnostic_only, x diagnostic_only, talk diagnostic_only

## All drafts (no draft dropped)

Pairs in this view: 120

| Metric | Scope | n | Relabel RAG off seed 43 mean | Relabel RAG on seed 43 mean | Diff | 95% interval | p | p adj | Family | Verdict |
|---|---|---|---|---|---|---|---|---|---|---|
| writing_quality | all | 120 | 3.006 | 2.972 | -0.033 | -0.154 to 0.090 | 0.600 | 1.000 | primary | no clear difference |
| writing_quality | blog | 30 | 3.583 | 3.372 | -0.211 | -0.433 to 0.006 | 0.076 | 1.000 | exploratory | no clear difference |
| writing_quality | linkedin | 30 | 3.283 | 3.344 | 0.061 | -0.256 to 0.389 | 0.745 | 1.000 | exploratory | no clear difference |
| writing_quality | x | 30 | 1.822 | 2.000 | 0.178 | 0.006 to 0.372 | 0.060 | 1.000 | exploratory | no clear difference |
| writing_quality | talk | 30 | 3.333 | 3.172 | -0.161 | -0.444 to 0.144 | 0.302 | 1.000 | exploratory | no clear difference |
| task_adherence | all | 120 | 0.690 | 0.710 | 0.020 | -0.002 to 0.046 | 0.124 | 0.372 | primary | no clear difference |
| task_adherence | blog | 30 | 0.704 | 0.712 | 0.009 | -0.014 to 0.029 | 0.446 | 1.000 | exploratory | no clear difference |
| task_adherence | linkedin | 30 | 0.677 | 0.719 | 0.041 | -0.026 to 0.112 | 0.194 | 1.000 | exploratory | no clear difference |
| task_adherence | x | 30 | 0.720 | 0.722 | 0.002 | -0.003 to 0.007 | 0.486 | 1.000 | exploratory | no clear difference |
| task_adherence | talk | 30 | 0.658 | 0.686 | 0.028 | -0.014 to 0.080 | 0.293 | 1.000 | exploratory | no clear difference |
| stance_consistency | all | 85 | 0.978 | 0.965 | -0.014 | -0.062 to 0.030 | 0.621 | 1.000 | primary | no clear difference |
| stance_consistency | blog | 22 | 1.000 | 0.955 | -0.045 | -0.114 to 0.000 | 0.500 | 1.000 | exploratory | no clear difference |
| stance_consistency | linkedin | 21 | 0.976 | 1.000 | 0.024 | 0.000 to 0.071 | 1.000 | 1.000 | exploratory | no clear difference |
| stance_consistency | x | 18 | 0.954 | 1.000 | 0.046 | 0.000 to 0.120 | 0.508 | 1.000 | exploratory | no clear difference |
| stance_consistency | talk | 24 | 0.979 | 0.917 | -0.062 | -0.208 to 0.042 | 0.497 | 1.000 | exploratory | no clear difference |
| style_distance (lower is better) | all | 90 | 0.755 | 0.760 | 0.004 | -0.060 to 0.068 | 0.898 | 1.000 | exploratory | no clear difference |
| style_distance (lower is better) | blog | 30 | 0.746 | 0.845 | 0.099 | 0.010 to 0.191 | 0.040 | 0.161 | primary | B worse (suggestive) |
| style_distance (lower is better) | linkedin | 30 | 0.718 | 0.664 | -0.055 | -0.207 to 0.089 | 0.538 | 1.000 | exploratory | no clear difference |
| style_distance (lower is better) | x | 30 | 0.802 | 0.770 | -0.031 | -0.093 to 0.040 | 0.390 | 1.000 | exploratory | no clear difference |
| persona_utility | all | 15 | 0.714 | 0.681 | -0.033 | -0.060 to -0.007 | 0.035 | 0.915 | exploratory | B worse (suggestive) |
| persona_utility | blog | 15 | 0.714 | 0.681 | -0.033 | -0.060 to -0.007 | 0.035 | 0.915 | exploratory | B worse (suggestive) |
| evidence_support | all | 90 | 0.990 | 0.946 | -0.044 | -0.088 to -0.003 | 0.049 | 1.000 | exploratory | B worse (suggestive) |
| evidence_support | blog | 27 | 1.000 | 0.944 | -0.056 | -0.111 to 0.000 | 0.243 | 1.000 | exploratory | no clear difference |
| evidence_support | linkedin | 19 | 1.000 | 0.974 | -0.026 | -0.079 to 0.000 | 1.000 | 1.000 | exploratory | no clear difference |
| evidence_support | x | 18 | 0.981 | 0.972 | -0.009 | -0.083 to 0.056 | 1.000 | 1.000 | exploratory | no clear difference |
| evidence_support | talk | 26 | 0.978 | 0.910 | -0.067 | -0.192 to 0.026 | 0.372 | 1.000 | exploratory | no clear difference |
| word_count | all | 120 | 189.975 | 190.058 | 0.083 | -19.525 to 19.968 | 0.994 | 1.000 | exploratory | no clear difference |
| word_count | blog | 30 | 294.733 | 243.600 | -51.133 | -92.402 to -7.832 | 0.026 | 0.707 | exploratory | B lower (suggestive) |
| word_count | linkedin | 30 | 176.500 | 193.900 | 17.400 | -15.701 to 52.033 | 0.338 | 1.000 | exploratory | no clear difference |
| word_count | x | 30 | 32.633 | 42.933 | 10.300 | -3.333 to 25.134 | 0.182 | 1.000 | exploratory | no clear difference |
| word_count | talk | 30 | 256.033 | 279.800 | 23.767 | -23.872 to 73.635 | 0.360 | 1.000 | exploratory | no clear difference |

## Drafts that pass the common gates in both runs

Pairs in this view: 68

| Metric | Scope | n | Relabel RAG off seed 43 mean | Relabel RAG on seed 43 mean | Diff | 95% interval | p | p adj | Family | Verdict |
|---|---|---|---|---|---|---|---|---|---|---|
| writing_quality | all | 68 | 3.221 | 3.044 | -0.176 | -0.306 to -0.049 | 0.010 | 0.042 | primary | B worse |
| writing_quality | blog | 27 | 3.617 | 3.438 | -0.179 | -0.414 to 0.056 | 0.157 | 1.000 | exploratory | no clear difference |
| writing_quality | linkedin | 3 | 3.111 | 3.222 | 0.111 | n/a | n/a | n/a | exploratory | too few pairs |
| writing_quality | x | 12 | 1.750 | 1.736 | -0.014 | -0.083 to 0.042 | 1.000 | 1.000 | exploratory | no clear difference |
| writing_quality | talk | 26 | 3.500 | 3.218 | -0.282 | -0.538 to -0.019 | 0.043 | 0.739 | exploratory | B worse (suggestive) |
| task_adherence | all | 68 | 0.709 | 0.709 | 0.000 | -0.015 to 0.015 | 0.984 | 0.984 | primary | no clear difference |
| task_adherence | blog | 27 | 0.698 | 0.706 | 0.009 | -0.015 to 0.031 | 0.474 | 1.000 | exploratory | no clear difference |
| task_adherence | linkedin | 3 | 0.746 | 0.728 | -0.018 | n/a | n/a | n/a | exploratory | too few pairs |
| task_adherence | x | 12 | 0.732 | 0.734 | 0.002 | 0.000 to 0.006 | 1.000 | 1.000 | exploratory | no clear difference |
| task_adherence | talk | 26 | 0.706 | 0.698 | -0.007 | -0.032 to 0.017 | 0.555 | 1.000 | exploratory | no clear difference |
| stance_consistency | all | 48 | 0.993 | 0.948 | -0.045 | -0.120 to 0.007 | 0.248 | 0.496 | primary | no clear difference |
| stance_consistency | blog | 19 | 1.000 | 0.974 | -0.026 | -0.079 to 0.000 | 1.000 | 1.000 | exploratory | no clear difference |
| stance_consistency | linkedin | 2 | 1.000 | 1.000 | 0.000 | n/a | n/a | n/a | exploratory | too few pairs |
| stance_consistency | x | 6 | 0.944 | 1.000 | 0.056 | 0.000 to 0.167 | 1.000 | 1.000 | exploratory | no clear difference |
| stance_consistency | talk | 21 | 1.000 | 0.905 | -0.095 | -0.238 to 0.000 | 0.504 | 1.000 | exploratory | no clear difference |
| style_distance (lower is better) | all | 42 | 0.792 | 0.854 | 0.062 | -0.004 to 0.129 | 0.093 | 1.000 | exploratory | no clear difference |
| style_distance (lower is better) | blog | 27 | 0.742 | 0.819 | 0.077 | -0.012 to 0.165 | 0.099 | 0.297 | primary | no clear difference |
| style_distance (lower is better) | linkedin | 3 | 0.881 | 0.942 | 0.061 | n/a | n/a | n/a | exploratory | too few pairs |
| style_distance (lower is better) | x | 12 | 0.881 | 0.912 | 0.031 | -0.054 to 0.167 | 0.938 | 1.000 | exploratory | no clear difference |
| persona_utility | all | 15 | 0.714 | 0.681 | -0.033 | -0.060 to -0.007 | 0.035 | 0.669 | exploratory | B worse (suggestive) |
| persona_utility | blog | 15 | 0.714 | 0.681 | -0.033 | -0.060 to -0.007 | 0.035 | 0.669 | exploratory | B worse (suggestive) |
| evidence_support | all | 56 | 0.996 | 0.958 | -0.037 | -0.097 to 0.004 | 0.252 | 1.000 | exploratory | no clear difference |
| evidence_support | blog | 24 | 1.000 | 0.958 | -0.042 | -0.104 to 0.000 | 0.501 | 1.000 | exploratory | no clear difference |
| evidence_support | linkedin | 2 | 1.000 | 1.000 | 0.000 | n/a | n/a | n/a | exploratory | too few pairs |
| evidence_support | x | 7 | 1.000 | 1.000 | 0.000 | 0.000 to 0.000 | 1.000 | 1.000 | exploratory | no clear difference |
| evidence_support | talk | 23 | 0.989 | 0.942 | -0.047 | -0.152 to 0.022 | 0.491 | 1.000 | exploratory | no clear difference |
| word_count | all | 68 | 234.338 | 206.544 | -27.794 | -49.537 to -6.564 | 0.021 | 0.418 | exploratory | B lower (suggestive) |
| word_count | blog | 27 | 294.074 | 233.111 | -60.963 | -99.259 to -23.000 | 0.005 | 0.097 | exploratory | B lower (suggestive) |
| word_count | linkedin | 3 | 113.667 | 95.333 | -18.333 | n/a | n/a | n/a | exploratory | too few pairs |
| word_count | x | 12 | 26.333 | 27.583 | 1.250 | -7.333 to 10.417 | 0.750 | 1.000 | exploratory | no clear difference |
| word_count | talk | 26 | 282.231 | 274.385 | -7.846 | -48.423 to 32.423 | 0.719 | 1.000 | exploratory | no clear difference |

Skipped (no paired values in the files): claim_support, grammar_rate, grammar_deviation

How to read the verdict: a difference is called clear only when the interval excludes zero and `p adj` is below 0.05. If it looks real on its own (interval excludes zero, p below 0.05) but `p adj` is not below 0.05, it is marked (suggestive): a lead worth a second run, not a finding. Rows by medium are exploratory unless listed as primary.

Limits: one run per arm unless more seeds are added, and the 30 gold topics are a development panel. A result that is not clear here is not proof of no difference.
