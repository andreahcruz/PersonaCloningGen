# Comparison: Relabel RAG on seed 44 against Relabel RAG off seed 44

Every difference is B minus A. Intervals are 95 percent, resampled by topic. p is a sign-flip test by topic. `p adj` corrects for how many tests were run (Holm). The four primary rows, which are the four parts of Persona Utility, are corrected together. Every other row is exploratory and is corrected as its own, larger family.

Paired drafts: 120 (only in Relabel RAG off seed 44: 0, only in Relabel RAG on seed 44: 0).

## Gates

| Run | Drafts | Eligible (all gates) | Pass common gates | Completion fails | Copying fails | Format fails | Grounding fails |
|---|---|---|---|---|---|---|---|
| Relabel RAG off seed 44 | 120 | 80 (67%) | 80 (67%) | 39 | 1 | 3 | 0 of 0 |
| Relabel RAG on seed 44 | 120 | 70 (58%) | 83 (69%) | 34 | 1 | 2 | 14 of 120 |

Grounding is counted only where it applies. A run with retrieval has one extra gate, so its eligible count is not comparable to a run without retrieval. Use the common gates column.

Style distance role by medium (selector means it may rank models): blog selector, linkedin diagnostic_only, x diagnostic_only, talk diagnostic_only

## All drafts (no draft dropped)

Pairs in this view: 120

| Metric | Scope | n | Relabel RAG off seed 44 mean | Relabel RAG on seed 44 mean | Diff | 95% interval | p | p adj | Family | Verdict |
|---|---|---|---|---|---|---|---|---|---|---|
| writing_quality | all | 120 | 2.958 | 2.881 | -0.078 | -0.246 to 0.087 | 0.364 | 1.000 | primary | no clear difference |
| writing_quality | blog | 30 | 3.428 | 3.367 | -0.061 | -0.306 to 0.161 | 0.603 | 1.000 | exploratory | no clear difference |
| writing_quality | linkedin | 30 | 3.194 | 3.056 | -0.139 | -0.533 to 0.222 | 0.477 | 1.000 | exploratory | no clear difference |
| writing_quality | x | 30 | 1.950 | 1.844 | -0.106 | -0.333 to 0.111 | 0.392 | 1.000 | exploratory | no clear difference |
| writing_quality | talk | 30 | 3.261 | 3.256 | -0.006 | -0.350 to 0.339 | 0.993 | 1.000 | exploratory | no clear difference |
| task_adherence | all | 120 | 0.710 | 0.705 | -0.004 | -0.026 to 0.017 | 0.720 | 1.000 | primary | no clear difference |
| task_adherence | blog | 30 | 0.710 | 0.692 | -0.018 | -0.061 to 0.016 | 0.467 | 1.000 | exploratory | no clear difference |
| task_adherence | linkedin | 30 | 0.717 | 0.751 | 0.033 | -0.015 to 0.092 | 0.276 | 1.000 | exploratory | no clear difference |
| task_adherence | x | 30 | 0.720 | 0.719 | -0.001 | -0.006 to 0.004 | 0.875 | 1.000 | exploratory | no clear difference |
| task_adherence | talk | 30 | 0.691 | 0.659 | -0.032 | -0.085 to 0.019 | 0.279 | 1.000 | exploratory | no clear difference |
| stance_consistency | all | 82 | 0.978 | 0.980 | 0.002 | -0.031 to 0.039 | 0.983 | 1.000 | primary | no clear difference |
| stance_consistency | blog | 16 | 0.969 | 0.958 | -0.010 | -0.125 to 0.094 | 1.000 | 1.000 | exploratory | no clear difference |
| stance_consistency | linkedin | 22 | 1.000 | 0.985 | -0.015 | -0.045 to 0.000 | 1.000 | 1.000 | exploratory | no clear difference |
| stance_consistency | x | 21 | 0.960 | 0.968 | 0.008 | -0.063 to 0.079 | 1.000 | 1.000 | exploratory | no clear difference |
| stance_consistency | talk | 23 | 0.978 | 1.000 | 0.022 | 0.000 to 0.065 | 1.000 | 1.000 | exploratory | no clear difference |
| style_distance (lower is better) | all | 90 | 0.755 | 0.794 | 0.039 | -0.036 to 0.117 | 0.323 | 1.000 | exploratory | no clear difference |
| style_distance (lower is better) | blog | 30 | 0.742 | 0.862 | 0.121 | 0.008 to 0.263 | 0.078 | 0.313 | primary | no clear difference |
| style_distance (lower is better) | linkedin | 30 | 0.738 | 0.723 | -0.015 | -0.192 to 0.159 | 0.865 | 1.000 | exploratory | no clear difference |
| style_distance (lower is better) | x | 30 | 0.784 | 0.795 | 0.012 | -0.042 to 0.070 | 0.706 | 1.000 | exploratory | no clear difference |
| persona_utility | all | 8 | 0.701 | 0.716 | 0.015 | -0.014 to 0.048 | 0.445 | 1.000 | exploratory | no clear difference |
| persona_utility | blog | 8 | 0.701 | 0.716 | 0.015 | -0.014 to 0.048 | 0.445 | 1.000 | exploratory | no clear difference |
| evidence_support | all | 95 | 0.986 | 0.966 | -0.020 | -0.051 to 0.012 | 0.246 | 1.000 | exploratory | no clear difference |
| evidence_support | blog | 21 | 1.000 | 0.952 | -0.048 | -0.119 to 0.000 | 0.507 | 1.000 | exploratory | no clear difference |
| evidence_support | linkedin | 24 | 0.992 | 0.969 | -0.022 | -0.064 to 0.017 | 0.501 | 1.000 | exploratory | no clear difference |
| evidence_support | x | 22 | 0.962 | 0.955 | -0.008 | -0.091 to 0.076 | 1.000 | 1.000 | exploratory | no clear difference |
| evidence_support | talk | 28 | 0.988 | 0.982 | -0.006 | -0.054 to 0.036 | 1.000 | 1.000 | exploratory | no clear difference |
| word_count | all | 120 | 186.542 | 184.400 | -2.142 | -22.017 to 17.492 | 0.840 | 1.000 | exploratory | no clear difference |
| word_count | blog | 30 | 283.533 | 241.267 | -42.267 | -86.403 to -2.266 | 0.065 | 1.000 | exploratory | no clear difference |
| word_count | linkedin | 30 | 169.867 | 175.767 | 5.900 | -35.201 to 45.400 | 0.776 | 1.000 | exploratory | no clear difference |
| word_count | x | 30 | 35.967 | 35.000 | -0.967 | -17.234 to 14.367 | 0.894 | 1.000 | exploratory | no clear difference |
| word_count | talk | 30 | 256.800 | 285.567 | 28.767 | -12.268 to 72.302 | 0.211 | 1.000 | exploratory | no clear difference |

## Drafts that pass the common gates in both runs

Pairs in this view: 67

| Metric | Scope | n | Relabel RAG off seed 44 mean | Relabel RAG on seed 44 mean | Diff | 95% interval | p | p adj | Family | Verdict |
|---|---|---|---|---|---|---|---|---|---|---|
| writing_quality | all | 67 | 3.107 | 3.149 | 0.042 | -0.142 to 0.220 | 0.673 | 1.000 | primary | no clear difference |
| writing_quality | blog | 26 | 3.487 | 3.500 | 0.013 | -0.244 to 0.250 | 0.954 | 1.000 | exploratory | no clear difference |
| writing_quality | linkedin | 4 | 2.208 | 2.917 | 0.708 | n/a | n/a | n/a | exploratory | too few pairs |
| writing_quality | x | 9 | 1.870 | 1.796 | -0.074 | -0.241 to 0.037 | 0.750 | 1.000 | exploratory | no clear difference |
| writing_quality | talk | 28 | 3.280 | 3.292 | 0.012 | -0.292 to 0.333 | 0.944 | 1.000 | exploratory | no clear difference |
| task_adherence | all | 67 | 0.711 | 0.701 | -0.010 | -0.025 to 0.004 | 0.212 | 0.672 | primary | no clear difference |
| task_adherence | blog | 26 | 0.707 | 0.705 | -0.002 | -0.029 to 0.022 | 0.880 | 1.000 | exploratory | no clear difference |
| task_adherence | linkedin | 4 | 0.720 | 0.753 | 0.033 | n/a | n/a | n/a | exploratory | too few pairs |
| task_adherence | x | 9 | 0.723 | 0.723 | 0.000 | 0.000 to 0.000 | 1.000 | 1.000 | exploratory | no clear difference |
| task_adherence | talk | 28 | 0.709 | 0.683 | -0.026 | -0.052 to 0.000 | 0.069 | 1.000 | exploratory | no clear difference |
| stance_consistency | all | 46 | 0.982 | 0.971 | -0.011 | -0.056 to 0.029 | 0.746 | 1.000 | primary | no clear difference |
| stance_consistency | blog | 15 | 0.967 | 0.956 | -0.011 | -0.133 to 0.100 | 1.000 | 1.000 | exploratory | no clear difference |
| stance_consistency | linkedin | 3 | 1.000 | 1.000 | 0.000 | n/a | n/a | n/a | exploratory | too few pairs |
| stance_consistency | x | 7 | 0.952 | 0.905 | -0.048 | -0.190 to 0.095 | 1.000 | 1.000 | exploratory | no clear difference |
| stance_consistency | talk | 21 | 1.000 | 1.000 | 0.000 | 0.000 to 0.000 | 1.000 | 1.000 | exploratory | no clear difference |
| style_distance (lower is better) | all | 39 | 0.825 | 0.837 | 0.013 | -0.075 to 0.104 | 0.796 | 1.000 | exploratory | no clear difference |
| style_distance (lower is better) | blog | 26 | 0.707 | 0.791 | 0.084 | -0.021 to 0.198 | 0.168 | 0.672 | primary | no clear difference |
| style_distance (lower is better) | linkedin | 4 | 1.327 | 0.909 | -0.418 | n/a | n/a | n/a | exploratory | too few pairs |
| style_distance (lower is better) | x | 9 | 0.940 | 0.939 | -0.002 | -0.084 to 0.060 | 1.000 | 1.000 | exploratory | no clear difference |
| persona_utility | all | 8 | 0.701 | 0.716 | 0.015 | -0.014 to 0.048 | 0.445 | 1.000 | exploratory | no clear difference |
| persona_utility | blog | 8 | 0.701 | 0.716 | 0.015 | -0.014 to 0.048 | 0.445 | 1.000 | exploratory | no clear difference |
| evidence_support | all | 56 | 1.000 | 0.964 | -0.036 | -0.073 to -0.008 | 0.119 | 1.000 | exploratory | no clear difference |
| evidence_support | blog | 20 | 1.000 | 0.950 | -0.050 | -0.125 to 0.000 | 0.496 | 1.000 | exploratory | no clear difference |
| evidence_support | linkedin | 3 | 1.000 | 1.000 | 0.000 | n/a | n/a | n/a | exploratory | too few pairs |
| evidence_support | x | 7 | 1.000 | 0.929 | -0.071 | -0.214 to 0.000 | 1.000 | 1.000 | exploratory | no clear difference |
| evidence_support | talk | 26 | 1.000 | 0.981 | -0.019 | -0.058 to 0.000 | 1.000 | 1.000 | exploratory | no clear difference |
| word_count | all | 67 | 224.776 | 233.522 | 8.746 | -14.320 to 30.492 | 0.461 | 1.000 | exploratory | no clear difference |
| word_count | blog | 26 | 278.423 | 255.423 | -23.000 | -61.462 to 13.423 | 0.264 | 1.000 | exploratory | no clear difference |
| word_count | linkedin | 4 | 39.500 | 122.250 | 82.750 | n/a | n/a | n/a | exploratory | too few pairs |
| word_count | x | 9 | 27.333 | 26.556 | -0.778 | -8.111 to 5.333 | 1.000 | 1.000 | exploratory | no clear difference |
| word_count | talk | 28 | 264.893 | 295.607 | 30.714 | -6.857 to 71.107 | 0.139 | 1.000 | exploratory | no clear difference |

Skipped (no paired values in the files): claim_support, grammar_rate, grammar_deviation

How to read the verdict: a difference is called clear only when the interval excludes zero and `p adj` is below 0.05. If it looks real on its own (interval excludes zero, p below 0.05) but `p adj` is not below 0.05, it is marked (suggestive): a lead worth a second run, not a finding. Rows by medium are exploratory unless listed as primary.

Limits: one run per arm unless more seeds are added, and the 30 gold topics are a development panel. A result that is not clear here is not proof of no difference.
