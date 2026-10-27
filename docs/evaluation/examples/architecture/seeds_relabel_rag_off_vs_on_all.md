# Seed stability: rag_on against rag_off (all)

Each cell is B minus A for that seed. Same sign: every seed moved the same way.

| Metric | Scope | seed 42 diff (95% interval) | seed 43 diff (95% interval) | seed 44 diff (95% interval) | Same sign | Intervals excluding 0 |
|---|---|---|---|---|---|---|
| writing_quality | all | -0.104 (-0.221 to +0.008) | -0.033 (-0.162 to +0.089) | -0.078 (-0.250 to +0.096) | yes | 0 of 3 |
| task_adherence | all | +0.008 (-0.006 to +0.025) | +0.020 (-0.003 to +0.046) | -0.004 (-0.026 to +0.018) | no | 0 of 3 |
| stance_consistency | all | +0.003 (-0.038 to +0.044) | -0.014 (-0.062 to +0.028) | +0.002 (-0.032 to +0.040) | no | 0 of 3 |
| style_distance (lower is better) | all | +0.055 (+0.008 to +0.115) | +0.004 (-0.060 to +0.069) | +0.039 (-0.037 to +0.119) | yes | 1 of 3 |
| word_count | all | -9.842 (-23.109 to +1.992) | +0.083 (-19.525 to +20.278) | -2.142 (-21.459 to +18.634) | no | 0 of 3 |

A result worth trusting keeps its sign in every seed and has an interval that excludes zero in most of them.
Three seeds is the minimum that shows a pattern; it is not a significance test.
