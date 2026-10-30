# Seed stability: rag_on against personarag (blog)

Each cell is B minus A for that seed. Same sign: every seed moved the same way.

| Metric | Scope | seed 42 diff (95% interval) | seed 43 diff (95% interval) | seed 44 diff (95% interval) | Same sign | Intervals excluding 0 |
|---|---|---|---|---|---|---|
| writing_quality | blog | -0.367 (-0.589 to -0.167) | -0.350 (-0.561 to -0.144) | -0.356 (-0.606 to -0.133) | yes | 3 of 3 |
| task_adherence | blog | -0.037 (-0.063 to -0.009) | -0.034 (-0.059 to -0.008) | -0.053 (-0.094 to -0.020) | yes | 3 of 3 |
| stance_consistency | blog | -0.025 (-0.075 to +0.000) | -0.042 (-0.104 to +0.000) | -0.033 (-0.100 to +0.000) | yes | 0 of 3 |
| style_distance (lower is better) | blog | -0.739 (-0.882 to -0.592) | -0.707 (-0.852 to -0.573) | -0.690 (-0.849 to -0.518) | yes | 3 of 3 |
| word_count | blog | -131.200 (-170.233 to -92.749) | -147.300 (-195.938 to -97.892) | -149.633 (-190.936 to -109.830) | yes | 3 of 3 |

A result worth trusting keeps its sign in every seed and has an interval that excludes zero in most of them.
Three seeds is the minimum that shows a pattern; it is not a significance test.
