# Seed stability: rag_on against personarag (all)

Each cell is B minus A for that seed. Same sign: every seed moved the same way.

| Metric | Scope | seed 42 diff (95% interval) | seed 43 diff (95% interval) | seed 44 diff (95% interval) | Same sign | Intervals excluding 0 |
|---|---|---|---|---|---|---|
| writing_quality | all | -0.703 (-0.849 to -0.557) | -0.626 (-0.753 to -0.515) | -0.718 (-0.864 to -0.569) | yes | 3 of 3 |
| task_adherence | all | -0.016 (-0.041 to +0.007) | -0.015 (-0.036 to +0.005) | -0.019 (-0.041 to +0.001) | yes | 0 of 3 |
| stance_consistency | all | -0.018 (-0.057 to +0.018) | -0.023 (-0.067 to +0.016) | -0.019 (-0.057 to +0.019) | yes | 0 of 3 |
| style_distance (lower is better) | all | -0.101 (-0.167 to -0.027) | -0.146 (-0.219 to -0.075) | -0.112 (-0.200 to -0.028) | yes | 3 of 3 |
| word_count | all | -113.408 (-130.346 to -96.515) | -106.417 (-127.184 to -86.225) | -112.075 (-131.686 to -91.808) | yes | 3 of 3 |

A result worth trusting keeps its sign in every seed and has an interval that excludes zero in most of them.
Three seeds is the minimum that shows a pattern; it is not a significance test.
