# Seed stability: rag_on against rag_off (blog)

Each cell is B minus A for that seed. Same sign: every seed moved the same way.

| Metric | Scope | seed 42 diff (95% interval) | seed 43 diff (95% interval) | seed 44 diff (95% interval) | Same sign | Intervals excluding 0 |
|---|---|---|---|---|---|---|
| writing_quality | blog | -0.222 (-0.439 to -0.022) | -0.211 (-0.433 to +0.006) | -0.061 (-0.306 to +0.178) | yes | 1 of 3 |
| task_adherence | blog | +0.003 (-0.018 to +0.022) | +0.009 (-0.013 to +0.029) | -0.018 (-0.061 to +0.017) | no | 0 of 3 |
| stance_consistency | blog | -0.026 (-0.079 to +0.000) | -0.045 (-0.114 to +0.000) | -0.010 (-0.125 to +0.094) | yes | 0 of 3 |
| style_distance (lower is better) | blog | +0.104 (+0.022 to +0.213) | +0.099 (+0.009 to +0.193) | +0.121 (+0.008 to +0.265) | yes | 3 of 3 |
| word_count | blog | -25.133 (-51.968 to -0.365) | -51.133 (-92.738 to -5.197) | -42.267 (-87.674 to -3.160) | yes | 3 of 3 |

A result worth trusting keeps its sign in every seed and has an interval that excludes zero in most of them.
Three seeds is the minimum that shows a pattern; it is not a significance test.
