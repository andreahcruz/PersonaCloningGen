# Example: Relabel QLoRA, RAG off against RAG on

This is an example output of `eval_tools.compare` and `eval_tools.grammar` (PUK-28), run on Kevin's saved results. It shows the tools on real data and gives PUK-34 a starting point. It is not a new model run: no draft was generated and no judge was called.

Inputs are copies of Kevin's saved results in `docs/evaluation/examples/data/`, with their source paths and SHA-256 hashes in `data/SOURCES.md`, so this can be rerun without his branch. The command is in that file too.

Check against Kevin's own comparison file: the mean differences here match his `metrics/comparison.json` (writing -0.104, task +0.008, stance +0.003 on 82 drafts, blog style distance +0.104 on 30, persona utility -0.026 on 15, word count -9.84). What is new is the interval, the p value and the correction for how many tests were run. His file marks significance as not computed.

What this shows:

- Gates. With RAG on, 83 of 120 drafts pass the common gates, against 77 with RAG off. The headline "eligible" counts (72 against 77) look worse for RAG only because RAG alone has the extra grounding gate.
- After correcting for the number of tests, nothing here is a clear difference. The nearest are blog style distance (a primary test, adjusted p 0.059), blog writing quality and Persona Utility on 15 blogs. They are marked "suggestive": the interval excludes zero on its own, but it does not survive the correction. They all point the same way as Kevin's finding that RAG does not improve voice. One run per arm cannot confirm it, so the next step is more seeds with RAG on (see `seed_stability_relabel_vs_repaired.md` for what a repeated result looks like). The blog voice lead is also mostly a length effect: RAG-on blog drafts are 25 words shorter, and the gap shrinks to nothing when every draft is cut to the same length (see `style_length_check.md`).
- Grammar. Both runs have almost no mechanical errors, so mechanics does not separate them. See `grammar_audit.md` for what that tool can and cannot see.

# Comparison: RAG on against RAG off

Every difference is B minus A. Intervals are 95 percent, resampled by topic. p is a sign-flip test by topic. `p adj` corrects for how many tests were run (Holm). The four primary rows, which are the four parts of Persona Utility, are corrected together. Every other row is exploratory and is corrected as its own, larger family.

Paired drafts: 120 (only in RAG off: 0, only in RAG on: 0).

## Gates

| Run | Drafts | Eligible (all gates) | Pass common gates | Completion fails | Copying fails | Format fails | Grounding fails |
|---|---|---|---|---|---|---|---|
| RAG off | 120 | 77 (64%) | 77 (64%) | 41 | 1 | 5 | 0 of 0 |
| RAG on | 120 | 72 (60%) | 83 (69%) | 36 | 2 | 3 | 14 of 120 |

Grounding is counted only where it applies. A run with retrieval has one extra gate, so its eligible count is not comparable to a run without retrieval. Use the common gates column.

Style distance role by medium (selector means it may rank models): blog selector, linkedin diagnostic_only, x diagnostic_only, talk diagnostic_only

## All drafts (no draft dropped)

Pairs in this view: 120

| Metric | Scope | n | RAG off mean | RAG on mean | Diff | 95% interval | p | p adj | Family | Verdict |
|---|---|---|---|---|---|---|---|---|---|---|
| writing_quality | all | 120 | 3.000 | 2.896 | -0.104 | -0.218 to 0.011 | 0.085 | 0.256 | primary | no clear difference |
| writing_quality | blog | 30 | 3.578 | 3.356 | -0.222 | -0.433 to -0.028 | 0.035 | 1.000 | exploratory | B worse (suggestive) |
| writing_quality | linkedin | 30 | 3.161 | 2.922 | -0.239 | -0.517 to 0.050 | 0.107 | 1.000 | exploratory | no clear difference |
| writing_quality | x | 30 | 1.917 | 2.006 | 0.089 | -0.161 to 0.317 | 0.535 | 1.000 | exploratory | no clear difference |
| writing_quality | talk | 30 | 3.344 | 3.300 | -0.044 | -0.306 to 0.222 | 0.778 | 1.000 | exploratory | no clear difference |
| task_adherence | all | 120 | 0.701 | 0.709 | 0.008 | -0.006 to 0.024 | 0.300 | 0.600 | primary | no clear difference |
| task_adherence | blog | 30 | 0.707 | 0.709 | 0.003 | -0.017 to 0.023 | 0.797 | 1.000 | exploratory | no clear difference |
| task_adherence | linkedin | 30 | 0.685 | 0.703 | 0.019 | -0.008 to 0.062 | 0.473 | 1.000 | exploratory | no clear difference |
| task_adherence | x | 30 | 0.719 | 0.719 | 0.000 | -0.007 to 0.008 | 0.985 | 1.000 | exploratory | no clear difference |
| task_adherence | talk | 30 | 0.693 | 0.706 | 0.012 | -0.025 to 0.061 | 0.733 | 1.000 | exploratory | no clear difference |
| stance_consistency | all | 82 | 0.972 | 0.975 | 0.003 | -0.037 to 0.045 | 1.000 | 1.000 | primary | no clear difference |
| stance_consistency | blog | 19 | 1.000 | 0.974 | -0.026 | -0.079 to 0.000 | 1.000 | 1.000 | exploratory | no clear difference |
| stance_consistency | linkedin | 21 | 0.952 | 0.952 | 0.000 | -0.143 to 0.143 | 1.000 | 1.000 | exploratory | no clear difference |
| stance_consistency | x | 18 | 0.954 | 0.981 | 0.028 | 0.000 to 0.083 | 1.000 | 1.000 | exploratory | no clear difference |
| stance_consistency | talk | 24 | 0.979 | 0.990 | 0.010 | -0.031 to 0.062 | 1.000 | 1.000 | exploratory | no clear difference |
| style_distance (lower is better) | all | 90 | 0.750 | 0.805 | 0.055 | 0.006 to 0.113 | 0.054 | 1.000 | exploratory | no clear difference |
| style_distance (lower is better) | blog | 30 | 0.709 | 0.814 | 0.104 | 0.023 to 0.213 | 0.015 | 0.059 | primary | B worse (suggestive) |
| style_distance (lower is better) | linkedin | 30 | 0.755 | 0.828 | 0.073 | -0.043 to 0.201 | 0.271 | 1.000 | exploratory | no clear difference |
| style_distance (lower is better) | x | 30 | 0.786 | 0.775 | -0.011 | -0.063 to 0.041 | 0.698 | 1.000 | exploratory | no clear difference |
| persona_utility | all | 15 | 0.717 | 0.692 | -0.026 | -0.051 to -0.005 | 0.036 | 1.000 | exploratory | B worse (suggestive) |
| persona_utility | blog | 15 | 0.717 | 0.692 | -0.026 | -0.051 to -0.005 | 0.036 | 1.000 | exploratory | B worse (suggestive) |
| evidence_support | all | 84 | 0.988 | 0.961 | -0.027 | -0.068 to 0.006 | 0.200 | 1.000 | exploratory | no clear difference |
| evidence_support | blog | 22 | 1.000 | 0.947 | -0.053 | -0.136 to 0.000 | 0.489 | 1.000 | exploratory | no clear difference |
| evidence_support | linkedin | 20 | 0.990 | 1.000 | 0.010 | 0.000 to 0.030 | 1.000 | 1.000 | exploratory | no clear difference |
| evidence_support | x | 19 | 0.982 | 0.921 | -0.061 | -0.193 to 0.035 | 0.493 | 1.000 | exploratory | no clear difference |
| evidence_support | talk | 23 | 0.980 | 0.975 | -0.006 | -0.039 to 0.027 | 1.000 | 1.000 | exploratory | no clear difference |
| word_count | all | 120 | 192.908 | 183.067 | -9.842 | -22.725 to 2.400 | 0.144 | 1.000 | exploratory | no clear difference |
| word_count | blog | 30 | 284.833 | 259.700 | -25.133 | -51.833 to -0.098 | 0.069 | 1.000 | exploratory | no clear difference |
| word_count | linkedin | 30 | 174.400 | 151.933 | -22.467 | -49.167 to 4.200 | 0.122 | 1.000 | exploratory | no clear difference |
| word_count | x | 30 | 33.900 | 39.467 | 5.567 | -7.567 to 20.333 | 0.477 | 1.000 | exploratory | no clear difference |
| word_count | talk | 30 | 278.500 | 281.167 | 2.667 | -38.300 to 42.001 | 0.898 | 1.000 | exploratory | no clear difference |
| grammar_rate (lower is better) | all | 120 | 0.005 | 0.011 | 0.006 | -0.007 to 0.021 | 0.369 | 1.000 | exploratory | no clear difference |
| grammar_rate (lower is better) | blog | 30 | 0.000 | 0.010 | 0.010 | 0.000 to 0.031 | 1.000 | 1.000 | exploratory | no clear difference |
| grammar_rate (lower is better) | linkedin | 30 | 0.000 | 0.032 | 0.032 | 0.000 to 0.084 | 0.490 | 1.000 | exploratory | no clear difference |
| grammar_rate (lower is better) | x | 30 | 0.000 | 0.000 | 0.000 | 0.000 to 0.000 | 1.000 | 1.000 | exploratory | no clear difference |
| grammar_rate (lower is better) | talk | 30 | 0.019 | 0.000 | -0.019 | -0.048 to 0.000 | 0.501 | 1.000 | exploratory | no clear difference |
| grammar_deviation (lower is better) | all | 90 | -0.033 | -0.019 | 0.014 | 0.000 to 0.031 | 0.244 | 1.000 | exploratory | no clear difference |
| grammar_deviation (lower is better) | blog | 30 | -0.010 | 0.001 | 0.010 | 0.000 to 0.031 | 1.000 | 1.000 | exploratory | no clear difference |
| grammar_deviation (lower is better) | linkedin | 30 | -0.009 | 0.023 | 0.032 | 0.000 to 0.084 | 0.490 | 1.000 | exploratory | no clear difference |
| grammar_deviation (lower is better) | x | 30 | -0.079 | -0.079 | 0.000 | 0.000 to 0.000 | 1.000 | 1.000 | exploratory | no clear difference |

## Drafts that pass the common gates in both runs

Pairs in this view: 72

| Metric | Scope | n | RAG off mean | RAG on mean | Diff | 95% interval | p | p adj | Family | Verdict |
|---|---|---|---|---|---|---|---|---|---|---|
| writing_quality | all | 72 | 3.153 | 3.039 | -0.113 | -0.239 to 0.012 | 0.095 | 0.286 | primary | no clear difference |
| writing_quality | blog | 27 | 3.586 | 3.426 | -0.161 | -0.364 to 0.031 | 0.129 | 1.000 | exploratory | no clear difference |
| writing_quality | linkedin | 4 | 2.958 | 2.333 | -0.625 | n/a | n/a | n/a | exploratory | too few pairs |
| writing_quality | x | 13 | 1.782 | 1.885 | 0.103 | -0.013 to 0.308 | 0.500 | 1.000 | exploratory | no clear difference |
| writing_quality | talk | 28 | 3.399 | 3.304 | -0.095 | -0.351 to 0.167 | 0.520 | 1.000 | exploratory | no clear difference |
| task_adherence | all | 72 | 0.715 | 0.712 | -0.002 | -0.016 to 0.012 | 0.761 | 1.000 | primary | no clear difference |
| task_adherence | blog | 27 | 0.703 | 0.705 | 0.002 | -0.020 to 0.024 | 0.894 | 1.000 | exploratory | no clear difference |
| task_adherence | linkedin | 4 | 0.732 | 0.721 | -0.011 | n/a | n/a | n/a | exploratory | too few pairs |
| task_adherence | x | 13 | 0.739 | 0.734 | -0.004 | -0.015 to 0.003 | 0.500 | 1.000 | exploratory | no clear difference |
| task_adherence | talk | 28 | 0.712 | 0.708 | -0.003 | -0.027 to 0.018 | 0.773 | 1.000 | exploratory | no clear difference |
| stance_consistency | all | 51 | 0.984 | 0.979 | -0.005 | -0.034 to 0.026 | 1.000 | 1.000 | primary | no clear difference |
| stance_consistency | blog | 18 | 1.000 | 0.972 | -0.028 | -0.083 to 0.000 | 1.000 | 1.000 | exploratory | no clear difference |
| stance_consistency | linkedin | 3 | 1.000 | 1.000 | 0.000 | n/a | n/a | n/a | exploratory | too few pairs |
| stance_consistency | x | 7 | 0.952 | 0.952 | 0.000 | 0.000 to 0.000 | 1.000 | 1.000 | exploratory | no clear difference |
| stance_consistency | talk | 23 | 0.978 | 0.989 | 0.011 | -0.033 to 0.065 | 1.000 | 1.000 | exploratory | no clear difference |
| style_distance (lower is better) | all | 44 | 0.785 | 0.843 | 0.058 | 0.001 to 0.134 | 0.068 | 1.000 | exploratory | no clear difference |
| style_distance (lower is better) | blog | 27 | 0.714 | 0.774 | 0.060 | 0.009 to 0.111 | 0.032 | 0.129 | primary | B worse (suggestive) |
| style_distance (lower is better) | linkedin | 4 | 0.912 | 1.227 | 0.315 | n/a | n/a | n/a | exploratory | too few pairs |
| style_distance (lower is better) | x | 13 | 0.895 | 0.868 | -0.026 | -0.087 to 0.012 | 0.594 | 1.000 | exploratory | no clear difference |
| persona_utility | all | 15 | 0.717 | 0.692 | -0.026 | -0.051 to -0.005 | 0.036 | 1.000 | exploratory | B worse (suggestive) |
| persona_utility | blog | 15 | 0.717 | 0.692 | -0.026 | -0.051 to -0.005 | 0.036 | 1.000 | exploratory | B worse (suggestive) |
| evidence_support | all | 52 | 0.991 | 0.947 | -0.044 | -0.098 to -0.002 | 0.128 | 1.000 | exploratory | no clear difference |
| evidence_support | blog | 19 | 1.000 | 0.939 | -0.061 | -0.158 to 0.000 | 0.502 | 1.000 | exploratory | no clear difference |
| evidence_support | linkedin | 3 | 1.000 | 1.000 | 0.000 | n/a | n/a | n/a | exploratory | too few pairs |
| evidence_support | x | 8 | 1.000 | 0.875 | -0.125 | -0.375 to 0.000 | 1.000 | 1.000 | exploratory | no clear difference |
| evidence_support | talk | 22 | 0.980 | 0.973 | -0.006 | -0.040 to 0.028 | 1.000 | 1.000 | exploratory | no clear difference |
| word_count | all | 72 | 230.069 | 217.569 | -12.500 | -28.476 to 3.652 | 0.160 | 1.000 | exploratory | no clear difference |
| word_count | blog | 27 | 285.444 | 264.074 | -21.370 | -43.519 to 1.002 | 0.080 | 1.000 | exploratory | no clear difference |
| word_count | linkedin | 4 | 108.500 | 75.000 | -33.500 | n/a | n/a | n/a | exploratory | too few pairs |
| word_count | x | 13 | 27.385 | 26.692 | -0.692 | -9.846 to 8.846 | 1.000 | 1.000 | exploratory | no clear difference |
| word_count | talk | 28 | 288.143 | 281.714 | -6.429 | -46.430 to 30.001 | 0.765 | 1.000 | exploratory | no clear difference |
| grammar_rate (lower is better) | all | 72 | 0.008 | 0.004 | -0.004 | -0.018 to 0.010 | 1.000 | 1.000 | exploratory | no clear difference |
| grammar_rate (lower is better) | blog | 27 | 0.000 | 0.012 | 0.012 | 0.000 to 0.035 | 1.000 | 1.000 | exploratory | no clear difference |
| grammar_rate (lower is better) | linkedin | 4 | 0.000 | 0.000 | 0.000 | n/a | n/a | n/a | exploratory | too few pairs |
| grammar_rate (lower is better) | x | 13 | 0.000 | 0.000 | 0.000 | 0.000 to 0.000 | 1.000 | 1.000 | exploratory | no clear difference |
| grammar_rate (lower is better) | talk | 28 | 0.020 | 0.000 | -0.020 | -0.051 to 0.000 | 0.499 | 1.000 | exploratory | no clear difference |
| grammar_deviation (lower is better) | all | 44 | -0.030 | -0.023 | 0.007 | 0.000 to 0.024 | 1.000 | 1.000 | exploratory | no clear difference |
| grammar_deviation (lower is better) | blog | 27 | -0.010 | 0.002 | 0.012 | 0.000 to 0.035 | 1.000 | 1.000 | exploratory | no clear difference |
| grammar_deviation (lower is better) | linkedin | 4 | -0.009 | -0.009 | 0.000 | n/a | n/a | n/a | exploratory | too few pairs |
| grammar_deviation (lower is better) | x | 13 | -0.079 | -0.079 | 0.000 | 0.000 to 0.000 | 1.000 | 1.000 | exploratory | no clear difference |

Skipped (no paired values in the files): claim_support

How to read the verdict: a difference is called clear only when the interval excludes zero and `p adj` is below 0.05. If it looks real on its own (interval excludes zero, p below 0.05) but `p adj` is not below 0.05, it is marked (suggestive): a lead worth a second run, not a finding. Rows by medium are exploratory unless listed as primary.

Limits: one run per arm unless more seeds are added, and the 30 gold topics are a development panel. A result that is not clear here is not proof of no difference.

