# System testing and performance

## Tests this audit did not re-run

The request allowed a safe CPU pytest if needed to check a reported count of 81 passing tests. A search of the markdown in this checkout found no "81 passed" result. This audit did not start pytest, because several recorded counts already exist and a GPU-free run was not required to identify the gap. The counts below are historical records, not a result from this session.

| Date and source | Recorded result | Scope |
|---|---|---|
| 2026-10-30, `docs/integration/PUK28_PUK34_INTEGRATION_20261030.md` | 562 passed, 2 skipped, 243.57 s | `tests/`. Skips: `test_spark_job.py` (no pyspark) and `test_ui.py` (no streamlit) |
| 2026-10-30, same note | 266 passed, 2 skipped on a focused command of 268 items | Grammar, voice, eval tools, RAG effect, retrieval-by-medium, and related files named in that note |
| 2026-11-01, `docs/handoff/MODEL_F_HANDOFF_MANIFEST.md` | 156 passed, 2 SWIG warnings, 17.21 s | CPU handoff suite named in that manifest |
| 2026-11-01, `docs/implementation/MODEL_F_GRAPH_CONTEXT_IMPLEMENTATION.md` | 87 passed, 0.83 s | Graph-context tests. No GPU, Ollama, or live index |
| 2026-11-02, `docs/handoff/MODEL_F_DEPLOYMENT_TRANSFER_20261102.md` | 68 passed, 2 DeprecationWarning | `test_model_f_unified.py`, `test_model_f_graph_context.py`, `test_model_f_experiment.py`, `test_eval_bench.py`, `test_eval_bench_historical.py`, `test_lightrag_retrieval.py` |
| 2026-11-02, `docs/evaluation/UNIFIED_120_COMPARISON.md` | 20 passed | `test_model_f_unified.py` plus existing four-arm tests. At that writing, the Model F panel had no drafts |

`docs/teammate_handoff.md` names the current CPU command and does not paste a pass count:

```text
python -m pytest tests/test_model_f_experiment.py tests/test_model_f_unified.py tests/test_model_f_graph_context.py tests/test_model_f_pilot.py tests/test_model_f_desc_vs_rel.py tests/test_evidence_capsule.py -q
```

There is no CI dashboard score in the handoff (`docs/handoff/BRANCH_CATCHUP_FOR_TEAMMATE.md`).

## What the tests cover, from their file names and the notes

| Area | Test modules present in `tests/` |
|---|---|
| Model F plan, arms, pilot, stage-2 | `test_model_f_unified.py`, `test_model_f_experiment.py`, `test_model_f_pilot.py`, `test_model_f_desc_vs_rel.py`, `test_model_f_graph_context.py` |
| LightRAG retrieval and indexing | `test_lightrag_retrieval.py`, `test_indexing_checkpoint.py`, `test_indexing_quality.py`, `test_indexing_extraction_audit.py`, `test_graph_relationships.py` |
| Evidence, numbers, copying | `test_evidence_capsule.py`, `test_numeric_grounding.py`, `test_rag_source_copying.py`, `test_factual_reliability.py` |
| Unified metrics and comparisons | `test_unified_eval.py`, `test_eval_tools.py`, `test_rag_effect.py` |
| UI | `test_ui.py`, skipped when Streamlit is absent |
| End-to-end generation of the 120-cell panel | Not a unit test. The panel is the measured experiment |

The unified tests check plan refusal, the gold hash, and a synthetic cell with null writing quality. They do not re-judge the 120 drafts.

## Runtime

| Quantity | Value | What it includes |
|---|---|---|
| Model F generation seconds | mean 20.069, median 23.732, min 1.453, max 51.184, p95 32.311, n=120 | Audit field `generation_seconds`. Matches the judge-row latency mean |
| Tokens per second | mean 13.437, n=120 | Audit. Generation throughput, not retrieval |
| Prompt tokens | mean 832.8, min 537, max 1,385 | Audit |
| Output tokens | mean 261.0, max 548 | Audit |
| Context tokens | mean 232.4, max 559 | Audit |
| Retrieval latency | not stored | Do not add it into the 20-second mean |
| Latency for Base, PersonaRAG, Relabel off, Relabel + RAG | null in the comparison file | Missing, not zero |
| Documented response-time target | none found | `METRIC_MAP.md` has no target. Inventing one would fail the rubric's "against documented requirements" clause |

Hardware requirements for a regeneration are CUDA, the 4-bit 8B model, and the adapter. The development card named in requirements is an RTX 5070 Ti with 16 GB. Peak memory during the official run was not stored. Early EOS and GPU memory were not stored (`docs/model_f_results.md`).

Retries: 20 of 120 cells used an accepted attempt greater than 1. The runner allows one retry for a long source-copy span. The audit does not say all 20 retries were copy retries.

Failure recovery in the indexer is a separate subsystem: resumable status, attempt caps, and deferred timeouts (`docs/architecture/LIGHTRAG_INDEXING_AND_CHECKPOINTING.md`). The frozen snapshot still contains 153 interrupted and 5 permanent failures. Recovery did not finish the corpus.

## Reproducibility

Present for the official panel: experiment id, gold hash, generation hash, judge model, judge commit, dirty flag, seed formula, profile name, snapshot identity, and the command shape. The judge command stored a local absolute path; the handoff publishes a project-relative form.

Missing for a teammate reproduction: the snapshot and adapter in Git, a GPU name on the manifest, a clean worktree (the judge record says dirty), and confirmation that a second machine verified the zip hashes. `generation_manifest.json` still says `generated_not_judged` even though the judge outputs exist. Readers must use `unified_eval/evaluation_run.json`.

Loss: the relabel train's best eval loss is 1.3316 (`experiments/README.md`). That is a training loss, not a generation-quality score, and it is not reported by the pytest suite.

## Deployment readiness

Model F as a CLI experiment is measured. Model F as a web service is not implemented in this checkout. The older FastAPI file is a different stack. Calling the system deployment-ready would overclaim the branch state in `docs/teammate_handoff.md`.
