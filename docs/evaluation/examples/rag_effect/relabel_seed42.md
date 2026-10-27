# What Relabel RAG on (seed 42) changed and what it cost, compared with Relabel RAG off (seed 42)

Differences are Relabel RAG on (seed 42) minus Relabel RAG off (seed 42), paired by topic and medium, with a 95 percent interval resampled by topic. Generation time is not saved with the drafts, so cost is in tokens and in the extra evidence seconds.

## Both runs

| Measure | Scope | n | Relabel RAG off (seed 42) | Relabel RAG on (seed 42) | Difference | 95% interval |
|---|---|---|---|---|---|---|
| Prompt tokens | all | 120 | 69.55 | 359.90 | +290.35 | +251.35 to +334.25 |
| Prompt tokens | blog | 30 | 69.80 | 360.40 | +290.60 | +251.60 to +334.50 |
| Prompt tokens | linkedin | 30 | 69.80 | 360.40 | +290.60 | +251.60 to +334.50 |
| Prompt tokens | x | 30 | 69.80 | 360.40 | +290.60 | +251.60 to +334.50 |
| Prompt tokens | talk | 30 | 68.80 | 358.40 | +289.60 | +250.60 to +333.50 |
| Generated tokens | all | 120 | 250.77 | 237.15 | -13.62 | -31.67 to +3.08 |
| Generated tokens | blog | 30 | 370.83 | 341.90 | -28.93 | -64.87 to +3.93 |
| Generated tokens | linkedin | 30 | 229.50 | 199.47 | -30.03 | -64.87 to +3.73 |
| Generated tokens | x | 30 | 50.90 | 53.83 | +2.93 | -13.80 to +22.07 |
| Generated tokens | talk | 30 | 351.83 | 353.40 | +1.57 | -52.40 to +52.03 |
| Words in the draft | all | 120 | 192.91 | 183.07 | -9.84 | -22.78 to +2.27 |
| Words in the draft | blog | 30 | 284.83 | 259.70 | -25.13 | -51.47 to -0.33 |
| Words in the draft | linkedin | 30 | 174.40 | 151.93 | -22.47 | -49.07 to +4.13 |
| Words in the draft | x | 30 | 33.90 | 39.47 | +5.57 | -7.47 to +20.17 |
| Words in the draft | talk | 30 | 278.50 | 281.17 | +2.67 | -39.44 to +41.90 |
| Ended at the length limit (share) | all | 120 | 0.16 | 0.14 | -0.02 | -0.07 to +0.03 |
| Ended at the length limit (share) | blog | 30 | 0.00 | 0.00 | +0.00 | +0.00 to +0.00 |
| Ended at the length limit (share) | linkedin | 30 | 0.50 | 0.43 | -0.07 | -0.23 to +0.10 |
| Ended at the length limit (share) | x | 30 | 0.13 | 0.13 | +0.00 | -0.17 to +0.17 |
| Ended at the length limit (share) | talk | 30 | 0.00 | 0.00 | +0.00 | +0.00 to +0.00 |
| Ended on its own using under half the token budget (share) | all | 120 | 0.55 | 0.61 | +0.06 | -0.03 to +0.15 |
| Ended on its own using under half the token budget (share) | blog | 30 | 0.57 | 0.63 | +0.07 | -0.17 to +0.27 |
| Ended on its own using under half the token budget (share) | linkedin | 30 | 0.23 | 0.40 | +0.17 | +0.00 to +0.33 |
| Ended on its own using under half the token budget (share) | x | 30 | 0.77 | 0.73 | -0.03 | -0.20 to +0.10 |
| Ended on its own using under half the token budget (share) | talk | 30 | 0.63 | 0.67 | +0.03 | -0.23 to +0.30 |

## Relabel RAG on (seed 42) only (retrieval on)

| Measure | All drafts | blog | linkedin | x | talk |
|---|---|---|---|---|---|
| Drafts | 120 | 30 | 30 | 30 | 30 |
| Topics | 30 | 30 | 30 | 30 | 30 |
| Evidence extraction seconds per topic (mean) | 28.3 | 28.3 | 28.3 | 28.3 | 28.3 |
| Total extra evidence seconds (counted once per topic) | 848 | n/a | n/a | n/a | n/a |
| Extra evidence seconds per draft (total spread over all drafts) | 7.1 | n/a | n/a | n/a | n/a |
| Drafts with unsupported numbers | 50% | 77% | 50% | 10% | 63% |
| Drafts flagged for copying a passage | 2% | 7% | 0% | 0% | 0% |
| Proposed claims that survive verification | 80% | 80% | 80% | 80% | 80% |
| Drafts that needed a second attempt | 2% | 0% | 7% | 0% | 3% |
| Retrieved passages that fit the requested medium | 25% | 57% | 10% | 32% | 0% |
| Mean passage distance (lower is closer) | 0.243 | 0.243 | 0.243 | 0.243 | 0.243 |

How to read the last two rows: retrieval here searches by topic only and ignores the requested medium, so every medium of a topic gets the same passages. A low fit means a LinkedIn or X draft is grounded mostly in blog text, and a talk draft gets no transcript passages at all.
