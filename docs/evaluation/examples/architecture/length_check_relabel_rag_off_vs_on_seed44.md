# Style distance and text length (blog)

Difference = RAG on s44 minus RAG off s44 in style distance (lower is closer to the Lemkin profile). Each row cuts every draft to the same number of words, so length cannot explain the gap.

Mean draft length: RAG off s44 283.5 words, RAG on s44 241.3 words.
Correlation of log(word count) with style distance over real posts and drafts: -0.81.

| Setting | Pairs | Difference | 95% interval | p |
|---|---|---|---|---|
| full length | 30 | +0.121 | +0.007 to +0.264 | 0.078 |
| first 100 words | 26 | -0.071 | -0.146 to +0.003 | 0.079 |
| first 150 words | 25 | +0.006 | -0.066 to +0.085 | 0.886 |
| first 200 words | 22 | +0.043 | -0.024 to +0.111 | 0.258 |

Reading it: if the gap at full length is clear but shrinks and its interval includes zero once drafts are cut to equal length, most of the gap was length. Cutting also drops the ends of drafts and some topics, so this is a check, not a replacement score.

## Drafts against real held-out Lemkin posts (blog)

40 real posts, median 222 words, mean distance 0.853. The numbers are the chance that a random draft is closer to the profile than a random real post (0.50 means equal, above 0.50 means the draft looks closer).

| Run | Drafts | Median words | Mean distance | Chance closer, full length (95% interval) | Chance closer, cut to real post lengths (range of 20 draws) |
|---|---|---|---|---|---|
| RAG off s44 | 30 | 284 | 0.742 | 0.69 (0.57 to 0.81) | 0.53 (0.45 to 0.58) |
| RAG on s44 | 30 | 263 | 0.863 | 0.58 (0.44 to 0.72) | 0.46 (0.36 to 0.50) |

If the full length value is above 0.50 and the length matched value is near 0.50, drafts only look closer than real posts because they are longer.
