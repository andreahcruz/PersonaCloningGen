# Example: is the style distance a length effect? (PUK-28)

Output of `eval_tools.length_control` on Kevin's saved Relabel runs, RAG off against RAG on, using his own reference pack. The reference data (dataset, split assignments and gold dispositions in `EXP-20261003-004-relabel-continuations` and `EXP-20261003-003-complete-source-sft-v3`) is on GitHub, and the runs EXP-20261004-007, EXP-20261004-008 and EXP-20261005-003 all record the same hashes for it, so they share one reference pack. The tool reproduces the stored distances exactly (90 of 90 RAG-on drafts) and the stored real post mean for blog (0.8531). No draft was generated and no judge was called.

What it shows:

- The style distance depends on text length. Over real blog posts and drafts, log word count and distance correlate at -0.66.
- Drafts look closer to the profile than real posts do, but only because they are longer. A random RAG-off blog draft is closer than a random real post 70 percent of the time at full length, and 53 percent (equal) once drafts are cut to real post lengths. So the idea that the score rewards typical text was a length effect.
- The RAG blog voice gap is not established. RAG-on blog drafts are 25 words shorter. The gap at full length (+0.104) shrinks to +0.03 to +0.05, with intervals that include zero, when every draft is cut to the same length. LinkedIn shows no gap at any length. This does not show that RAG is harmless for voice. It shows this run cannot tell.

To rerun, make a checkout of Kevin's branch without the Git LFS adapter files (about 480 MB), then run the tool:

```
GIT_LFS_SKIP_SMUDGE=1 git worktree add ../kevin-ws origin/kevin/298b-integration
python -m eval_tools.length_control --kevin-root ../kevin-ws \
  --a-drafts <RAG off raw/relabel.jsonl> --b-drafts <RAG on raw/relabel.jsonl> \
  --label-a "RAG off" --label-b "RAG on" --medium blog
```

The drafts files are `experiments/EXP-20261004-003-repaired-vs-relabel/raw/relabel.jsonl` and `experiments/EXP-20261005-003-relabel-rag-final/raw/relabel.jsonl` on that branch (their hashes are in `data/SOURCES.md`).
## Style distance and text length (blog)

Difference = RAG on minus RAG off in style distance (lower is closer to the Lemkin profile). Each row cuts every draft to the same number of words, so length cannot explain the gap.

Mean draft length: RAG off 284.8 words, RAG on 259.7 words.
Correlation of log(word count) with style distance over real posts and drafts: -0.66.

| Setting | Pairs | Difference | 95% interval | p |
|---|---|---|---|---|
| full length | 30 | +0.104 | +0.023 to +0.214 | 0.014 |
| first 100 words | 29 | +0.049 | -0.026 to +0.126 | 0.225 |
| first 150 words | 28 | +0.034 | -0.037 to +0.113 | 0.391 |
| first 200 words | 25 | +0.042 | -0.034 to +0.121 | 0.312 |

Reading it: if the gap at full length is clear but shrinks and its interval includes zero once drafts are cut to equal length, most of the gap was length. Cutting also drops the ends of drafts and some topics, so this is a check, not a replacement score.

## Drafts against real held-out Lemkin posts (blog)

40 real posts, median 222 words, mean distance 0.853. The numbers are the chance that a random draft is closer to the profile than a random real post (0.50 means equal, above 0.50 means the draft looks closer).

| Run | Drafts | Median words | Mean distance | Chance closer, full length (95% interval) | Chance closer, cut to real post lengths (range of 20 draws) |
|---|---|---|---|---|---|
| RAG off | 30 | 294 | 0.710 | 0.70 (0.58 to 0.82) | 0.53 (0.46 to 0.59) |
| RAG on | 30 | 279 | 0.814 | 0.57 (0.45 to 0.71) | 0.46 (0.40 to 0.53) |

If the full length value is above 0.50 and the length matched value is near 0.50, drafts only look closer than real posts because they are longer.

## Style distance and text length (linkedin)

Difference = RAG on minus RAG off in style distance (lower is closer to the Lemkin profile). Each row cuts every draft to the same number of words, so length cannot explain the gap.

Mean draft length: RAG off 174.4 words, RAG on 151.9 words.
Correlation of log(word count) with style distance over real posts and drafts: -0.94.

| Setting | Pairs | Difference | 95% interval | p |
|---|---|---|---|---|
| full length | 30 | +0.073 | -0.046 to +0.204 | 0.274 |
| first 50 words | 22 | -0.041 | -0.094 to +0.011 | 0.151 |
| first 100 words | 19 | -0.021 | -0.069 to +0.027 | 0.409 |
| first 150 words | 16 | +0.014 | -0.039 to +0.069 | 0.612 |

Reading it: if the gap at full length is clear but shrinks and its interval includes zero once drafts are cut to equal length, most of the gap was length. Cutting also drops the ends of drafts and some topics, so this is a check, not a replacement score.

## Drafts against real held-out Lemkin posts (linkedin)

40 real posts, median 118 words, mean distance 0.814. The numbers are the chance that a random draft is closer to the profile than a random real post (0.50 means equal, above 0.50 means the draft looks closer).

| Run | Drafts | Median words | Mean distance | Chance closer, full length (95% interval) | Chance closer, cut to real post lengths (range of 20 draws) |
|---|---|---|---|---|---|
| RAG off | 30 | 212 | 0.754 | 0.73 (0.58 to 0.86) | 0.52 (0.42 to 0.63) |
| RAG on | 30 | 179 | 0.828 | 0.64 (0.49 to 0.79) | 0.50 (0.41 to 0.56) |

If the full length value is above 0.50 and the length matched value is near 0.50, drafts only look closer than real posts because they are longer.

