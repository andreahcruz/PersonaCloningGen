# Style distance and text length (blog)

Difference = RAG on s43 minus RAG off s43 in style distance (lower is closer to the Lemkin profile). Each row cuts every draft to the same number of words, so length cannot explain the gap.

Mean draft length: RAG off s43 294.7 words, RAG on s43 243.6 words.
Correlation of log(word count) with style distance over real posts and drafts: -0.55.

| Setting | Pairs | Difference | 95% interval | p |
|---|---|---|---|---|
| full length | 30 | +0.099 | +0.010 to +0.190 | 0.040 |
| first 100 words | 27 | +0.055 | -0.002 to +0.109 | 0.065 |
| first 150 words | 25 | +0.049 | -0.012 to +0.102 | 0.108 |
| first 200 words | 21 | +0.044 | -0.032 to +0.115 | 0.265 |

Reading it: if the gap at full length is clear but shrinks and its interval includes zero once drafts are cut to equal length, most of the gap was length. Cutting also drops the ends of drafts and some topics, so this is a check, not a replacement score.

## Drafts against real held-out Lemkin posts (blog)

40 real posts, median 222 words, mean distance 0.853. The numbers are the chance that a random draft is closer to the profile than a random real post (0.50 means equal, above 0.50 means the draft looks closer).

| Run | Drafts | Median words | Mean distance | Chance closer, full length (95% interval) | Chance closer, cut to real post lengths (range of 20 draws) |
|---|---|---|---|---|---|
| RAG off s43 | 30 | 293 | 0.746 | 0.63 (0.49 to 0.76) | 0.53 (0.47 to 0.56) |
| RAG on s43 | 30 | 262 | 0.845 | 0.52 (0.39 to 0.66) | 0.42 (0.36 to 0.47) |

If the full length value is above 0.50 and the length matched value is near 0.50, drafts only look closer than real posts because they are longer.
