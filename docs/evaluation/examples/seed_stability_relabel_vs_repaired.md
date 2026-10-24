# Example: does a result repeat across seeds? Relabel against repaired adapter

Output of `eval_tools.seeds` on Kevin's saved runs with no retrieval: the same 30 topics and four mediums generated with seeds 42, 43 and 44 (experiments `EXP-20261004-003`, `-005` and `-006`), scored rows from `EXP-20261004-008-writing-evidence/metrics/rows.jsonl` on branch `kevin/298b-integration`. A is the repaired adapter and B is the relabel adapter, so a positive difference means relabel is higher.

What it shows: relabel keeps its direction in all three seeds for writing quality (higher), blog voice distance (closer to Lemkin) and blog length (longer). So the adapter comparison replicates. The RAG against no RAG comparison has only one seed, so this check cannot be run on it yet.

To rerun:

```
python -m eval_tools.seeds --rows <rows.jsonl> --a-model repaired --b-model relabel \
  --experiments EXP-20261004-003-repaired-vs-relabel EXP-20261004-005-repaired-vs-relabel-seed43 EXP-20261004-006-repaired-vs-relabel-seed44 \
  --labels 42 43 44 [--scope blog]
```

## All mediums

Each cell is B minus A for that seed. Same sign: every seed moved the same way.

| Metric | Scope | seed 42 diff (95% interval) | seed 43 diff (95% interval) | seed 44 diff (95% interval) | Same sign | Intervals excluding 0 |
|---|---|---|---|---|---|---|
| writing_quality | all | +0.207 (+0.069 to +0.342) | +0.192 (+0.050 to +0.336) | +0.113 (-0.049 to +0.285) | yes | 2 of 3 |
| task_adherence | all | +0.008 (-0.016 to +0.030) | -0.003 (-0.036 to +0.027) | +0.017 (-0.003 to +0.035) | no | 0 of 3 |
| stance_consistency | all | -0.010 (-0.043 to +0.026) | -0.015 (-0.031 to +0.000) | -0.010 (-0.039 to +0.027) | yes | 0 of 3 |
| style_distance (lower is better) | all | +0.067 (-0.002 to +0.137) | +0.066 (+0.007 to +0.127) | +0.047 (-0.025 to +0.120) | yes | 1 of 3 |
| word_count | all | -11.808 (-26.310 to +4.525) | -7.058 (-24.984 to +11.784) | -12.158 (-27.292 to +1.125) | yes | 0 of 3 |

A result worth trusting keeps its sign in every seed and has an interval that excludes zero in most of them.
Three seeds is the minimum that shows a pattern; it is not a significance test.

## Blog only

Each cell is B minus A for that seed. Same sign: every seed moved the same way.

| Metric | Scope | seed 42 diff (95% interval) | seed 43 diff (95% interval) | seed 44 diff (95% interval) | Same sign | Intervals excluding 0 |
|---|---|---|---|---|---|---|
| writing_quality | blog | +0.222 (+0.006 to +0.456) | +0.200 (-0.044 to +0.444) | -0.050 (-0.289 to +0.206) | no | 1 of 3 |
| task_adherence | blog | +0.037 (+0.009 to +0.066) | +0.013 (-0.013 to +0.039) | +0.027 (+0.006 to +0.051) | yes | 2 of 3 |
| stance_consistency | blog | -0.053 (-0.158 to +0.000) | +0.000 (+0.000 to +0.000) | -0.038 (-0.115 to +0.000) | yes | 0 of 3 |
| style_distance (lower is better) | blog | -0.085 (-0.154 to -0.023) | -0.055 (-0.143 to +0.028) | -0.159 (-0.254 to -0.077) | yes | 2 of 3 |
| word_count | blog | +38.167 (+6.265 to +75.402) | +53.233 (+20.667 to +87.203) | +51.333 (+14.432 to +90.839) | yes | 3 of 3 |

A result worth trusting keeps its sign in every seed and has an interval that excludes zero in most of them.
Three seeds is the minimum that shows a pattern; it is not a significance test.

