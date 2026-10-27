# What Relabel RAG on (seed 43) changed and what it cost, compared with Relabel RAG off (seed 43)

Differences are Relabel RAG on (seed 43) minus Relabel RAG off (seed 43), paired by topic and medium, with a 95 percent interval resampled by topic. Generation time is not saved with the drafts, so cost is in tokens and in the extra evidence seconds.

## Both runs

| Measure | Scope | n | Relabel RAG off (seed 43) | Relabel RAG on (seed 43) | Difference | 95% interval |
|---|---|---|---|---|---|---|
| Prompt tokens | all | 120 | 69.55 | 361.27 | +291.72 | +252.31 to +337.82 |
| Prompt tokens | blog | 30 | 69.80 | 361.77 | +291.97 | +252.56 to +338.07 |
| Prompt tokens | linkedin | 30 | 69.80 | 361.77 | +291.97 | +252.56 to +338.07 |
| Prompt tokens | x | 30 | 69.80 | 361.77 | +291.97 | +252.56 to +338.07 |
| Prompt tokens | talk | 30 | 68.80 | 359.77 | +290.97 | +251.56 to +337.07 |
| Generated tokens | all | 120 | 247.03 | 247.51 | +0.47 | -23.44 to +25.12 |
| Generated tokens | blog | 30 | 387.73 | 319.40 | -68.33 | -122.97 to -10.57 |
| Generated tokens | linkedin | 30 | 231.80 | 259.60 | +27.80 | -15.37 to +72.27 |
| Generated tokens | x | 30 | 46.50 | 61.87 | +15.37 | -4.00 to +35.53 |
| Generated tokens | talk | 30 | 322.10 | 349.17 | +27.07 | -33.87 to +89.97 |
| Words in the draft | all | 120 | 189.97 | 190.06 | +0.08 | -19.58 to +19.96 |
| Words in the draft | blog | 30 | 294.73 | 243.60 | -51.13 | -93.24 to -7.50 |
| Words in the draft | linkedin | 30 | 176.50 | 193.90 | +17.40 | -16.00 to +52.03 |
| Words in the draft | x | 30 | 32.63 | 42.93 | +10.30 | -3.27 to +25.10 |
| Words in the draft | talk | 30 | 256.03 | 279.80 | +23.77 | -24.30 to +74.14 |
| Ended at the length limit (share) | all | 120 | 0.15 | 0.20 | +0.05 | -0.01 to +0.11 |
| Ended at the length limit (share) | blog | 30 | 0.00 | 0.03 | +0.03 | +0.00 to +0.10 |
| Ended at the length limit (share) | linkedin | 30 | 0.50 | 0.60 | +0.10 | -0.10 to +0.30 |
| Ended at the length limit (share) | x | 30 | 0.10 | 0.13 | +0.03 | -0.13 to +0.20 |
| Ended at the length limit (share) | talk | 30 | 0.00 | 0.03 | +0.03 | +0.00 to +0.10 |
| Ended on its own using under half the token budget (share) | all | 120 | 0.55 | 0.59 | +0.04 | -0.07 to +0.14 |
| Ended on its own using under half the token budget (share) | blog | 30 | 0.50 | 0.87 | +0.37 | +0.13 to +0.60 |
| Ended on its own using under half the token budget (share) | linkedin | 30 | 0.23 | 0.17 | -0.07 | -0.23 to +0.10 |
| Ended on its own using under half the token budget (share) | x | 30 | 0.80 | 0.67 | -0.13 | -0.30 to +0.00 |
| Ended on its own using under half the token budget (share) | talk | 30 | 0.67 | 0.67 | +0.00 | -0.23 to +0.23 |

## Relabel RAG on (seed 43) only (retrieval on)

| Measure | All drafts | blog | linkedin | x | talk |
|---|---|---|---|---|---|
| Drafts | 120 | 30 | 30 | 30 | 30 |
| Topics | 30 | 30 | 30 | 30 | 30 |
| Evidence extraction seconds per topic (mean) | 33.1 | 33.1 | 33.1 | 33.1 | 33.1 |
| Total extra evidence seconds (counted once per topic) | 993 | n/a | n/a | n/a | n/a |
| Extra evidence seconds per draft (total spread over all drafts) | 8.3 | n/a | n/a | n/a | n/a |
| Drafts with unsupported numbers | 55% | 70% | 63% | 27% | 60% |
| Drafts flagged for copying a passage | 2% | 7% | 0% | 0% | 0% |
| Proposed claims that survive verification | 82% | 82% | 82% | 82% | 82% |
| Drafts that needed a second attempt | 3% | 3% | 7% | 3% | 0% |
| Retrieved passages that fit the requested medium | 25% | 57% | 9% | 33% | 0% |
| Mean passage distance (lower is closer) | 0.243 | 0.243 | 0.243 | 0.243 | 0.243 |

How to read the last two rows: retrieval here searches by topic only and ignores the requested medium, so every medium of a topic gets the same passages. A low fit means a LinkedIn or X draft is grounded mostly in blog text, and a talk draft gets no transcript passages at all.
