# Evaluation report

Schema `eval-bench-v1`. Experiment `EXP-20261101-008-eval-bench-historical-replay`.

Retrieval, faithfulness, numeric checks, and latency are separate. There is no composite score.

## Arms

| Architecture | N | P@4 | R@4 | MRR | nDCG@4 | Faithfulness | Numeric pass | Citation leak | Words |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| historical_dense_retrieval | 12 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| historical_lightrag_hybrid | 12 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| historical_lightrag_mix | 12 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| historical_lightrag_naive | 12 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |

Null ranked metrics mean the questions had no graded qrels. They are not zeros.

## Failures

None.

## Latency

- End-to-end p50: n/a
- End-to-end p95: n/a
- Throughput per second: n/a

Throughput counts scored rows only. It is not a quality metric.

Word count is a covariate. A longer answer is not scored as better.
