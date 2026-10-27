# Style distance and text length (blog)

Difference = Relabel+RAG s44 minus PersonaRAG in style distance (lower is closer to the Lemkin profile). Each row cuts every draft to the same number of words, so length cannot explain the gap.

Mean draft length: PersonaRAG 390.9 words, Relabel+RAG s44 241.3 words.
Correlation of log(word count) with style distance over real posts and drafts: -0.01.

| Setting | Pairs | Difference | 95% interval | p |
|---|---|---|---|---|
| full length | 30 | -0.690 | -0.848 to -0.522 | 0.000 |
| first 100 words | 26 | -0.678 | -0.886 to -0.485 | 0.000 |
| first 150 words | 26 | -0.632 | -0.772 to -0.497 | 0.000 |
| first 200 words | 25 | -0.549 | -0.669 to -0.450 | 0.000 |

Reading it: if the gap at full length is clear but shrinks and its interval includes zero once drafts are cut to equal length, most of the gap was length. Cutting also drops the ends of drafts and some topics, so this is a check, not a replacement score.

## Drafts against real held-out Lemkin posts (blog)

40 real posts, median 222 words, mean distance 0.853. The numbers are the chance that a random draft is closer to the profile than a random real post (0.50 means equal, above 0.50 means the draft looks closer).

| Run | Drafts | Median words | Mean distance | Chance closer, full length (95% interval) | Chance closer, cut to real post lengths (range of 20 draws) |
|---|---|---|---|---|---|
| PersonaRAG | 30 | 368 | 1.552 | 0.03 (0.00 to 0.08) | 0.10 (0.08 to 0.12) |
| Relabel+RAG s44 | 30 | 263 | 0.863 | 0.58 (0.44 to 0.72) | 0.46 (0.36 to 0.50) |

If the full length value is above 0.50 and the length matched value is near 0.50, drafts only look closer than real posts because they are longer.
