# Evaluation report

Schema `eval-bench-v1`. Experiment `EXP-20261101-008-eval-bench-historical-replay`.

Retrieval, faithfulness, numeric checks, and latency are separate. There is no composite score.

## Arms

| Architecture | N | P@4 | R@4 | MRR | nDCG@4 | Faithfulness | Numeric pass | Citation leak | Words |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| historical_blog_token_pad | 5 | n/a | n/a | n/a | n/a | 0.8000 | 0.0000 | 0.0000 | 149.0000 |
| historical_hybrid_passages | 20 | n/a | n/a | n/a | n/a | 0.6500 | 0.2500 | 0.0500 | 151.5500 |
| historical_hybrid_plus_relationships | 20 | n/a | n/a | n/a | n/a | 0.6500 | 0.4286 | 0.3000 | 169.2000 |
| historical_matched_dense_preview | 20 | n/a | n/a | n/a | n/a | 0.7000 | 0.2500 | 0.1000 | 138.9500 |

Null ranked metrics mean the questions had no graded qrels. They are not zeros.

## Failures

None.

## Latency

- End-to-end p50: n/a
- End-to-end p95: n/a
- Throughput per second: n/a

Throughput counts scored rows only. It is not a quality metric.

## Paired faithfulness

historical_hybrid_passages minus historical_hybrid_plus_relationships: 0.0000
95 percent bootstrap interval: [-0.2, 0.2]
Questions: 20. Seed 298.

Word count is a covariate. A longer answer is not scored as better.
