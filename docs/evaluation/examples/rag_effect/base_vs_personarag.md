# What PersonaRAG (base + retrieval) changed and what it cost, compared with Base Llama 3.1 (no retrieval)

Differences are PersonaRAG (base + retrieval) minus Base Llama 3.1 (no retrieval), paired by topic and medium, with a 95 percent interval resampled by topic. Generation time is not saved with the drafts, so cost is in tokens and in the extra evidence seconds.

## Both runs

| Measure | Scope | n | Base Llama 3.1 (no retrieval) | PersonaRAG (base + retrieval) | Difference | 95% interval |
|---|---|---|---|---|---|---|
| Words in the draft | all | 120 | 352.11 | 296.48 | -55.63 | -68.95 to -42.27 |
| Words in the draft | blog | 30 | 515.20 | 390.90 | -124.30 | -156.03 to -92.80 |
| Words in the draft | linkedin | 30 | 240.17 | 241.97 | +1.80 | -8.63 to +11.73 |
| Words in the draft | x | 30 | 116.70 | 124.57 | +7.87 | +4.33 to +10.97 |
| Words in the draft | talk | 30 | 536.37 | 428.47 | -107.90 | -137.37 to -76.63 |

## PersonaRAG (base + retrieval) only (retrieval on)

| Measure | All drafts | blog | linkedin | x | talk |
|---|---|---|---|---|---|
| Drafts | 120 | 30 | 30 | 30 | 30 |
| Topics | 0 | 0 | 0 | 0 | 0 |
| Evidence extraction seconds per topic (mean) | n/a | n/a | n/a | n/a | n/a |
| Total extra evidence seconds (counted once per topic) | n/a | n/a | n/a | n/a | n/a |
| Extra evidence seconds per draft (total spread over all drafts) | n/a | n/a | n/a | n/a | n/a |
| Drafts with unsupported numbers | n/a | n/a | n/a | n/a | n/a |
| Drafts flagged for copying a passage | n/a | n/a | n/a | n/a | n/a |
| Proposed claims that survive verification | n/a | n/a | n/a | n/a | n/a |
| Drafts that needed a second attempt | 0% | 0% | 0% | 0% | 0% |
| Retrieved passages that fit the requested medium | 25% | 57% | 9% | 33% | 0% |
| Mean passage distance (lower is closer) | 0.243 | 0.243 | 0.243 | 0.243 | 0.243 |

How to read the last two rows: retrieval here searches by topic only and ignores the requested medium, so every medium of a topic gets the same passages. A low fit means a LinkedIn or X draft is grounded mostly in blog text, and a talk draft gets no transcript passages at all.
