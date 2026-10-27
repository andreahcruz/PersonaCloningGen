# What Relabel RAG on (seed 44) changed and what it cost, compared with Relabel RAG off (seed 44)

Differences are Relabel RAG on (seed 44) minus Relabel RAG off (seed 44), paired by topic and medium, with a 95 percent interval resampled by topic. Generation time is not saved with the drafts, so cost is in tokens and in the extra evidence seconds.

## Both runs

| Measure | Scope | n | Relabel RAG off (seed 44) | Relabel RAG on (seed 44) | Difference | 95% interval |
|---|---|---|---|---|---|---|
| Prompt tokens | all | 120 | 69.55 | 361.27 | +291.72 | +252.31 to +337.82 |
| Prompt tokens | blog | 30 | 69.80 | 361.77 | +291.97 | +252.56 to +338.07 |
| Prompt tokens | linkedin | 30 | 69.80 | 361.77 | +291.97 | +252.56 to +338.07 |
| Prompt tokens | x | 30 | 69.80 | 361.77 | +291.97 | +252.56 to +338.07 |
| Prompt tokens | talk | 30 | 68.80 | 359.77 | +290.97 | +251.56 to +337.07 |
| Generated tokens | all | 120 | 243.06 | 237.84 | -5.22 | -30.32 to +19.89 |
| Generated tokens | blog | 30 | 372.33 | 315.57 | -56.77 | -114.20 to -6.27 |
| Generated tokens | linkedin | 30 | 223.33 | 231.93 | +8.60 | -43.67 to +60.10 |
| Generated tokens | x | 30 | 54.03 | 49.00 | -5.03 | -26.57 to +15.67 |
| Generated tokens | talk | 30 | 322.53 | 354.87 | +32.33 | -16.07 to +83.70 |
| Words in the draft | all | 120 | 186.54 | 184.40 | -2.14 | -22.00 to +17.35 |
| Words in the draft | blog | 30 | 283.53 | 241.27 | -42.27 | -86.77 to -3.30 |
| Words in the draft | linkedin | 30 | 169.87 | 175.77 | +5.90 | -34.47 to +45.47 |
| Words in the draft | x | 30 | 35.97 | 35.00 | -0.97 | -17.23 to +14.23 |
| Words in the draft | talk | 30 | 256.80 | 285.57 | +28.77 | -11.97 to +71.97 |
| Ended at the length limit (share) | all | 120 | 0.16 | 0.18 | +0.03 | -0.06 to +0.11 |
| Ended at the length limit (share) | blog | 30 | 0.03 | 0.00 | -0.03 | -0.10 to +0.00 |
| Ended at the length limit (share) | linkedin | 30 | 0.47 | 0.60 | +0.13 | -0.10 to +0.37 |
| Ended at the length limit (share) | x | 30 | 0.13 | 0.13 | +0.00 | -0.17 to +0.17 |
| Ended at the length limit (share) | talk | 30 | 0.00 | 0.00 | +0.00 | +0.00 to +0.00 |
| Ended on its own using under half the token budget (share) | all | 120 | 0.60 | 0.62 | +0.02 | -0.07 to +0.11 |
| Ended on its own using under half the token budget (share) | blog | 30 | 0.60 | 0.80 | +0.20 | -0.03 to +0.43 |
| Ended on its own using under half the token budget (share) | linkedin | 30 | 0.30 | 0.30 | +0.00 | -0.20 to +0.20 |
| Ended on its own using under half the token budget (share) | x | 30 | 0.77 | 0.80 | +0.03 | -0.13 to +0.20 |
| Ended on its own using under half the token budget (share) | talk | 30 | 0.73 | 0.57 | -0.17 | -0.40 to +0.07 |

## Relabel RAG on (seed 44) only (retrieval on)

| Measure | All drafts | blog | linkedin | x | talk |
|---|---|---|---|---|---|
| Drafts | 120 | 30 | 30 | 30 | 30 |
| Topics | 30 | 30 | 30 | 30 | 30 |
| Evidence extraction seconds per topic (mean) | 33.3 | 33.3 | 33.3 | 33.3 | 33.3 |
| Total extra evidence seconds (counted once per topic) | 1000 | n/a | n/a | n/a | n/a |
| Extra evidence seconds per draft (total spread over all drafts) | 8.3 | n/a | n/a | n/a | n/a |
| Drafts with unsupported numbers | 47% | 60% | 57% | 10% | 60% |
| Drafts flagged for copying a passage | 3% | 7% | 3% | 3% | 0% |
| Proposed claims that survive verification | 82% | 82% | 82% | 82% | 82% |
| Drafts that needed a second attempt | 2% | 0% | 0% | 0% | 7% |
| Retrieved passages that fit the requested medium | 25% | 57% | 9% | 33% | 0% |
| Mean passage distance (lower is closer) | 0.243 | 0.243 | 0.243 | 0.243 | 0.243 |

How to read the last two rows: retrieval here searches by topic only and ignores the requested medium, so every medium of a topic gets the same passages. A low fit means a LinkedIn or X draft is grounded mostly in blog text, and a talk draft gets no transcript passages at all.
