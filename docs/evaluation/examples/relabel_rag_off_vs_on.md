# Example: Relabel QLoRA, RAG off against RAG on

This is an example output of `eval_tools.compare` and `eval_tools.grammar` (PUK-28), run on Kevin's saved results. It is here to show the tools on real data and so PUK-34 can reuse it. It is not a new model run: no draft was generated and no judge was called.

Inputs (all on branch `kevin/298b-integration`):

- RAG off scores: `experiments/EXP-20261004-008-writing-evidence/metrics/rows.jsonl`, rows with source experiment `EXP-20261004-003-repaired-vs-relabel` and model `relabel` (120 rows).
- RAG on scores: `experiments/EXP-20261005-003-relabel-rag-final/metrics/rows.jsonl` (120 rows).
- Drafts for the grammar check: `raw/relabel.jsonl` of `EXP-20261004-003-repaired-vs-relabel` and of `EXP-20261005-003-relabel-rag-final`.
- Real Lemkin posts for the grammar reference: `data/*.jsonl` (400 posts per medium, seed 0).

Check against Kevin's own comparison file: the mean differences here match `metrics/comparison.json` (writing -0.104, task +0.008, stance +0.003 on 82 drafts, blog style distance +0.104 on 30, persona utility -0.026 on 15, word count -9.84). What is new is the interval and the p value, which that file marks as not computed.

Things worth noticing:

- Gates: with RAG on, 83 of 120 drafts pass the common gates, against 77 with RAG off. The headline "eligible" numbers (72 against 77) look worse for RAG only because RAG alone has the extra grounding gate.
- On blog drafts, RAG on has clearly worse writing quality and clearly worse style distance than RAG off (intervals exclude zero, p below 0.05). Other rows are not clear. Rows by medium are not corrected for many comparisons.
- Grammar mechanics is not where the runs differ. Both are near zero errors per 100 words, below the real posts. The rule based checker is coarse (see `eval_tools/README.md`).

# Comparison: RAG on against RAG off

Every difference is B minus A. Intervals are 95 percent, resampled by topic. p values are from a sign-flip test by topic; `p holm` corrects across metrics in the all-scope rows.

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

| Metric | Scope | n | RAG off mean | RAG on mean | Diff | 95% interval | p | p holm | Verdict |
|---|---|---|---|---|---|---|---|---|---|
| writing_quality | all | 120 | 3.000 | 2.896 | -0.104 | -0.218 to 0.011 | 0.085 | 0.598 | no clear difference |
| writing_quality | blog | 30 | 3.578 | 3.356 | -0.222 | -0.433 to -0.028 | 0.035 | n/a | B worse |
| writing_quality | linkedin | 30 | 3.161 | 2.922 | -0.239 | -0.517 to 0.050 | 0.107 | n/a | no clear difference |
| writing_quality | x | 30 | 1.917 | 2.006 | 0.089 | -0.161 to 0.317 | 0.535 | n/a | no clear difference |
| writing_quality | talk | 30 | 3.344 | 3.300 | -0.044 | -0.306 to 0.222 | 0.778 | n/a | no clear difference |
| task_adherence | all | 120 | 0.701 | 0.709 | 0.008 | -0.006 to 0.024 | 0.300 | 0.900 | no clear difference |
| task_adherence | blog | 30 | 0.707 | 0.709 | 0.003 | -0.017 to 0.023 | 0.797 | n/a | no clear difference |
| task_adherence | linkedin | 30 | 0.685 | 0.703 | 0.019 | -0.008 to 0.062 | 0.473 | n/a | no clear difference |
| task_adherence | x | 30 | 0.719 | 0.719 | 0.000 | -0.007 to 0.008 | 0.985 | n/a | no clear difference |
| task_adherence | talk | 30 | 0.693 | 0.706 | 0.012 | -0.025 to 0.061 | 0.733 | n/a | no clear difference |
| stance_consistency | all | 82 | 0.972 | 0.975 | 0.003 | -0.037 to 0.045 | 1.000 | 1.000 | no clear difference |
| stance_consistency | blog | 19 | 1.000 | 0.974 | -0.026 | -0.079 to 0.000 | 1.000 | n/a | no clear difference |
| stance_consistency | linkedin | 21 | 0.952 | 0.952 | 0.000 | -0.143 to 0.143 | 1.000 | n/a | no clear difference |
| stance_consistency | x | 18 | 0.954 | 0.981 | 0.028 | 0.000 to 0.083 | 1.000 | n/a | no clear difference |
| stance_consistency | talk | 24 | 0.979 | 0.990 | 0.010 | -0.031 to 0.062 | 1.000 | n/a | no clear difference |
| style_distance (lower is better) | all | 90 | 0.750 | 0.805 | 0.055 | 0.006 to 0.113 | 0.054 | 0.436 | no clear difference |
| style_distance (lower is better) | blog | 30 | 0.709 | 0.814 | 0.104 | 0.023 to 0.213 | 0.015 | n/a | B worse |
| style_distance (lower is better) | linkedin | 30 | 0.755 | 0.828 | 0.073 | -0.043 to 0.201 | 0.271 | n/a | no clear difference |
| style_distance (lower is better) | x | 30 | 0.786 | 0.775 | -0.011 | -0.063 to 0.041 | 0.698 | n/a | no clear difference |
| persona_utility | all | 15 | 0.717 | 0.692 | -0.026 | -0.051 to -0.005 | 0.036 | 0.322 | B worse |
| persona_utility | blog | 15 | 0.717 | 0.692 | -0.026 | -0.051 to -0.005 | 0.036 | n/a | B worse |
| evidence_support | all | 84 | 0.988 | 0.961 | -0.027 | -0.068 to 0.006 | 0.200 | 0.801 | no clear difference |
| evidence_support | blog | 22 | 1.000 | 0.947 | -0.053 | -0.136 to 0.000 | 0.489 | n/a | no clear difference |
| evidence_support | linkedin | 20 | 0.990 | 1.000 | 0.010 | 0.000 to 0.030 | 1.000 | n/a | no clear difference |
| evidence_support | x | 19 | 0.982 | 0.921 | -0.061 | -0.193 to 0.035 | 0.493 | n/a | no clear difference |
| evidence_support | talk | 23 | 0.980 | 0.975 | -0.006 | -0.039 to 0.027 | 1.000 | n/a | no clear difference |
| word_count | all | 120 | 192.908 | 183.067 | -9.842 | -22.725 to 2.400 | 0.144 | 0.743 | no clear difference |
| word_count | blog | 30 | 284.833 | 259.700 | -25.133 | -51.833 to -0.098 | 0.069 | n/a | no clear difference |
| word_count | linkedin | 30 | 174.400 | 151.933 | -22.467 | -49.167 to 4.200 | 0.122 | n/a | no clear difference |
| word_count | x | 30 | 33.900 | 39.467 | 5.567 | -7.567 to 20.333 | 0.477 | n/a | no clear difference |
| word_count | talk | 30 | 278.500 | 281.167 | 2.667 | -38.300 to 42.001 | 0.898 | n/a | no clear difference |
| grammar_rate (lower is better) | all | 120 | 0.007 | 0.014 | 0.006 | -0.009 to 0.023 | 0.407 | 0.900 | no clear difference |
| grammar_rate (lower is better) | blog | 30 | 0.000 | 0.022 | 0.022 | 0.000 to 0.056 | 0.504 | n/a | no clear difference |
| grammar_rate (lower is better) | linkedin | 30 | 0.000 | 0.032 | 0.032 | 0.000 to 0.084 | 0.490 | n/a | no clear difference |
| grammar_rate (lower is better) | x | 30 | 0.000 | 0.000 | 0.000 | 0.000 to 0.000 | 1.000 | n/a | no clear difference |
| grammar_rate (lower is better) | talk | 30 | 0.029 | 0.000 | -0.029 | -0.061 to 0.000 | 0.250 | n/a | no clear difference |
| grammar_deviation (lower is better) | all | 90 | -0.061 | -0.043 | 0.018 | 0.003 to 0.037 | 0.124 | 0.743 | no clear difference |
| grammar_deviation (lower is better) | blog | 30 | -0.015 | 0.007 | 0.022 | 0.000 to 0.056 | 0.504 | n/a | no clear difference |
| grammar_deviation (lower is better) | linkedin | 30 | -0.028 | 0.005 | 0.032 | 0.000 to 0.084 | 0.490 | n/a | no clear difference |
| grammar_deviation (lower is better) | x | 30 | -0.141 | -0.141 | 0.000 | 0.000 to 0.000 | 1.000 | n/a | no clear difference |

## Drafts that pass the common gates in both runs

Pairs in this view: 72

| Metric | Scope | n | RAG off mean | RAG on mean | Diff | 95% interval | p | p holm | Verdict |
|---|---|---|---|---|---|---|---|---|---|
| writing_quality | all | 72 | 3.153 | 3.039 | -0.113 | -0.239 to 0.012 | 0.095 | 0.668 | no clear difference |
| writing_quality | blog | 27 | 3.586 | 3.426 | -0.161 | -0.364 to 0.031 | 0.129 | n/a | no clear difference |
| writing_quality | linkedin | 4 | 2.958 | 2.333 | -0.625 | n/a | n/a | n/a | too few pairs |
| writing_quality | x | 13 | 1.782 | 1.885 | 0.103 | -0.013 to 0.308 | 0.500 | n/a | no clear difference |
| writing_quality | talk | 28 | 3.399 | 3.304 | -0.095 | -0.351 to 0.167 | 0.520 | n/a | no clear difference |
| task_adherence | all | 72 | 0.715 | 0.712 | -0.002 | -0.016 to 0.012 | 0.761 | 1.000 | no clear difference |
| task_adherence | blog | 27 | 0.703 | 0.705 | 0.002 | -0.020 to 0.024 | 0.894 | n/a | no clear difference |
| task_adherence | linkedin | 4 | 0.732 | 0.721 | -0.011 | n/a | n/a | n/a | too few pairs |
| task_adherence | x | 13 | 0.739 | 0.734 | -0.004 | -0.015 to 0.003 | 0.500 | n/a | no clear difference |
| task_adherence | talk | 28 | 0.712 | 0.708 | -0.003 | -0.027 to 0.018 | 0.773 | n/a | no clear difference |
| stance_consistency | all | 51 | 0.984 | 0.979 | -0.005 | -0.034 to 0.026 | 1.000 | 1.000 | no clear difference |
| stance_consistency | blog | 18 | 1.000 | 0.972 | -0.028 | -0.083 to 0.000 | 1.000 | n/a | no clear difference |
| stance_consistency | linkedin | 3 | 1.000 | 1.000 | 0.000 | n/a | n/a | n/a | too few pairs |
| stance_consistency | x | 7 | 0.952 | 0.952 | 0.000 | 0.000 to 0.000 | 1.000 | n/a | no clear difference |
| stance_consistency | talk | 23 | 0.978 | 0.989 | 0.011 | -0.033 to 0.065 | 1.000 | n/a | no clear difference |
| style_distance (lower is better) | all | 44 | 0.785 | 0.843 | 0.058 | 0.001 to 0.134 | 0.068 | 0.543 | no clear difference |
| style_distance (lower is better) | blog | 27 | 0.714 | 0.774 | 0.060 | 0.009 to 0.111 | 0.032 | n/a | B worse |
| style_distance (lower is better) | linkedin | 4 | 0.912 | 1.227 | 0.315 | n/a | n/a | n/a | too few pairs |
| style_distance (lower is better) | x | 13 | 0.895 | 0.868 | -0.026 | -0.087 to 0.012 | 0.594 | n/a | no clear difference |
| persona_utility | all | 15 | 0.717 | 0.692 | -0.026 | -0.051 to -0.005 | 0.036 | 0.322 | B worse |
| persona_utility | blog | 15 | 0.717 | 0.692 | -0.026 | -0.051 to -0.005 | 0.036 | n/a | B worse |
| evidence_support | all | 52 | 0.991 | 0.947 | -0.044 | -0.098 to -0.002 | 0.128 | 0.769 | no clear difference |
| evidence_support | blog | 19 | 1.000 | 0.939 | -0.061 | -0.158 to 0.000 | 0.502 | n/a | no clear difference |
| evidence_support | linkedin | 3 | 1.000 | 1.000 | 0.000 | n/a | n/a | n/a | too few pairs |
| evidence_support | x | 8 | 1.000 | 0.875 | -0.125 | -0.375 to 0.000 | 1.000 | n/a | no clear difference |
| evidence_support | talk | 22 | 0.980 | 0.973 | -0.006 | -0.040 to 0.028 | 1.000 | n/a | no clear difference |
| word_count | all | 72 | 230.069 | 217.569 | -12.500 | -28.476 to 3.652 | 0.160 | 0.801 | no clear difference |
| word_count | blog | 27 | 285.444 | 264.074 | -21.370 | -43.519 to 1.002 | 0.080 | n/a | no clear difference |
| word_count | linkedin | 4 | 108.500 | 75.000 | -33.500 | n/a | n/a | n/a | too few pairs |
| word_count | x | 13 | 27.385 | 26.692 | -0.692 | -9.846 to 8.846 | 1.000 | n/a | no clear difference |
| word_count | talk | 28 | 288.143 | 281.714 | -6.429 | -46.430 to 30.001 | 0.765 | n/a | no clear difference |
| grammar_rate (lower is better) | all | 72 | 0.012 | 0.009 | -0.003 | -0.021 to 0.017 | 1.000 | 1.000 | no clear difference |
| grammar_rate (lower is better) | blog | 27 | 0.000 | 0.025 | 0.025 | 0.000 to 0.063 | 0.500 | n/a | no clear difference |
| grammar_rate (lower is better) | linkedin | 4 | 0.000 | 0.000 | 0.000 | n/a | n/a | n/a | too few pairs |
| grammar_rate (lower is better) | x | 13 | 0.000 | 0.000 | 0.000 | 0.000 to 0.000 | 1.000 | n/a | no clear difference |
| grammar_rate (lower is better) | talk | 28 | 0.031 | 0.000 | -0.031 | -0.066 to 0.000 | 0.252 | n/a | no clear difference |
| grammar_deviation (lower is better) | all | 44 | -0.053 | -0.038 | 0.015 | 0.000 to 0.041 | 0.500 | 1.000 | no clear difference |
| grammar_deviation (lower is better) | blog | 27 | -0.015 | 0.010 | 0.025 | 0.000 to 0.063 | 0.500 | n/a | no clear difference |
| grammar_deviation (lower is better) | linkedin | 4 | -0.028 | -0.028 | 0.000 | n/a | n/a | n/a | too few pairs |
| grammar_deviation (lower is better) | x | 13 | -0.141 | -0.141 | 0.000 | 0.000 to 0.000 | 1.000 | n/a | no clear difference |

Skipped (no paired values in the files): claim_support

How to read the verdict: a difference is called clear only when the interval excludes zero and the p value is below 0.05. Rows by medium are not corrected for the many comparisons made, so treat a single medium result as a lead, not a finding. Use `p holm` for the all-scope rows.

Limits: one run per arm unless more seeds are added, and the 30 gold topics are a development panel. A result that is not clear here is not proof of no difference.

