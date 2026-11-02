# Model F handoff manifest

Date: 2026-11-01. Branch: `kevin/lightrag-model-f`. Remote: `https://github.com/andreahcruz/298P.git`.

This manifest lists what the deployment handoff includes, what stays local, and how to use each piece. The partial LightRAG index is not in Git. The blog-only indexer was still writing `artifacts/lightrag/model_f/storage_expand_v1` when this manifest was prepared. No indexer module was edited for the handoff, and that directory was not copied or snapshotted.

Status words follow `AGENTS.md`: IMPLEMENTED, TESTED, MEASURED, VALIDATED.

## Audit before the handoff commit

| Item | Result |
|---|---|
| Current branch | `kevin/lightrag-model-f` |
| Upstream before this handoff | none |
| HEAD before the handoff commit | `95db4f54a518cb273ea9118b10b9175eeb9668be` |
| Remote | `origin` `https://github.com/andreahcruz/298P.git` |
| Deployment branch | `origin/andreah/deploy-with-react` (EC2/React demo line). `origin/andreah/deploy-test` is an older ancestor. `docs/deployment/DEPLOYMENT_HANDOFF.md` describes the earlier Relabel + Chroma handoff on `kevin/298b-integration`, which is already an ancestor of this branch. |
| Divergence from `andreah/deploy-with-react` | merge-base `ea1d58b30cafcb4cf18a03ded39963f64ce6aee7`; 35 commits on this branch and 34 on the deploy branch that the other does not have. An October 30 audit recorded content conflicts in `.dockerignore`, `.gitignore`, `README.md`, and `docker-compose.yml`. |
| Ready to hand off code | Yes. Live-index validation is still blocked until indexing exits. |

The branch was not switched, rebased, reset, or cleaned. The deployment branch was not checked out or modified.

## Test results

CPU only. `PYTHONDONTWRITEBYTECODE=1`. No GPU inference, no embedding job, no query of the live LightRAG store, and no change to corpus manifests or extraction caches.

Command, from the repository root, using `host_finetune\.venv`:

```text
host_finetune\.venv\Scripts\python.exe -X utf8 -m pytest tests/test_model_f_graph_context.py tests/test_model_f_experiment.py tests/test_eval_bench.py tests/test_eval_bench_historical.py tests/test_dev_benchmark.py tests/test_graph_relationships.py tests/test_relation_normalization.py tests/test_indexing_checkpoint.py tests/test_indexing_extraction_audit.py tests/test_indexing_quality.py tests/test_indexing_selection.py tests/test_indexing_timeout.py tests/test_evidence_capsule.py tests/test_factual_reliability.py tests/test_numeric_grounding.py tests/test_lightrag_retrieval.py -q --tb=line -p no:cacheprovider
```

Result on 2026-11-01: **156 passed**, 2 warnings, 17.21 seconds.

Warnings: `SwigPyPacked` and `SwigPyObject` deprecation warnings while importing during `test_context_preflight_fits_every_planned_cell`. They are not assertion failures.

`tests/test_eval_bench.py` and `tests/test_indexing_checkpoint.py` resolve protected path names and refuse them. They do not open `storage_expand_v1`.

## How to read the columns

- **Git** — included in the handoff commit.
- **Deploy** — required for the running service to load Model F. Tests, benchmarks, and experiment records are required for independent work and are marked no for process startup.
- Dependencies are Python 3.11, `host_finetune/requirements.txt`, and `host_finetune/requirements-lightrag.txt` unless a row says otherwise. GPU, Unsloth, and Ollama are required only when generating or querying. CPU tests do not need them.

## Graph-context implementation

Package: `host_finetune/model_f_context/`. Importing it does not open graph storage. Status: IMPLEMENTED and TESTED. Live hybrid query: not verified.

| Path | Purpose | Git | Deploy | Dependencies / setup |
|---|---|---|---|---|
| `host_finetune/model_f_context/__init__.py` | Package exports | yes | yes | none beyond the package |
| `host_finetune/model_f_context/schema.py` | Evidence schema `model-f-evidence-v1` | yes | yes | none |
| `host_finetune/model_f_context/normalize.py` | `normalize_graph_context` | yes | yes | saved or raw query payload |
| `host_finetune/model_f_context/multihop.py` | 1-hop and 2-hop `select_paths` | yes | yes | in-memory edges only |
| `host_finetune/model_f_context/ranking.py` | `rank_evidence` | yes | yes | none |
| `host_finetune/model_f_context/budget.py` | Per-medium token budgets | yes | yes | none |
| `host_finetune/model_f_context/prompt.py` | `render_context_prompt` | yes | yes | none |
| `host_finetune/model_f_context/leakage.py` | Leakage screen and paired-arm records | yes | yes | none |
| `host_finetune/model_f_context/pipeline.py` | `prepare_context`, `context_media_jobs`, `graph_context_callback` | yes | yes | none |
| `host_finetune/model_f_context/registry.py` | Model E and Model F arm registry | yes | yes | profiles from `pipeline.py` |
| `host_finetune/compare_adapters.py` | `--model-f-context` and `--lightrag-workdir`. Default context is `off`. Default backend is `dense`. | yes | yes | Relabel adapter and a LightRAG directory only when those flags are on |
| `host_finetune/evidence_capsule.py` | Claim extraction used by the same generation path | yes | yes | unchanged call path |
| `host_finetune/factual_reliability.py` | Factual and instruction-leak checks | yes | yes | none for the CPU checks |
| `host_finetune/numeric_grounding.py` | Unsupported-quantity checks | yes | yes | none |
| `host_finetune/lightrag_backend.py` | LightRAG open, normalize, and query. Already imported by the running indexer. Committed as it stood. Not edited during this handoff. | yes | yes | `lightrag-hku==1.5.7`, Ollama at query time |
| `host_finetune/graph_relationships.py` | Source-span check for relationship rows | yes | yes | imported by `lightrag_backend.py` |
| `host_finetune/graph_relationship_questions.json` | Pilot review questions and source ids. No passage text. | yes | no | read by the development benchmark |
| `host_finetune/requirements-lightrag.txt` | LightRAG pin and OpenTelemetry pin | yes | yes | install after the host requirements |

`compare_adapters` loads an externally supplied directory with `--lightrag-workdir`. The default path is `artifacts/lightrag/model_f/storage`, which is not the blog expansion directory.

## Four-arm experiment runner

Package: `host_finetune/model_f_experiment/`. Status: IMPLEMENTED and TESTED on synthetic fixtures. No snapshot has been copied. No four-arm generation has been run. Not VALIDATED.

| Path | Purpose | Git | Deploy | Dependencies / setup |
|---|---|---|---|---|
| `host_finetune/model_f_experiment/__init__.py` | Package marker | yes | no | none |
| `host_finetune/model_f_experiment/__main__.py` | `snapshot`, `verify`, `run`, `evaluate`, `audit` | yes | no | CLI |
| `host_finetune/model_f_experiment/arms.py` | F-SOURCE, F-REL, F-DESC, F-HOP channel sets | yes | no | none |
| `host_finetune/model_f_experiment/runner.py` | Resume-safe four-arm runner | yes | no | a snapshot path, questions, Relabel adapter at run time |
| `host_finetune/model_f_experiment/context_window.py` | Preflight against the 2048-token window | yes | no | none |
| `host_finetune/model_f_experiment/derive.py` | Trace derivation helpers | yes | no | none |
| `host_finetune/model_f_experiment/retrieval.py` | Frozen hybrid retrieval for the runner | yes | no | do not point it at the live store |
| `host_finetune/model_f_experiment/evaluate.py` | Convert arm traces for `eval_bench` | yes | no | saved traces |
| `host_finetune/model_f_experiment/snapshot.py` | Copy and verify a finished workdir. Refuses a lock and refuses a destination named `storage_expand_v1`. | yes | yes, after indexing exits | source directory supplied outside Git |
| `host_finetune/model_f_experiment/graph_audit.py` | Read-only audit of a snapshot | yes | no | snapshot, not the live store |
| `host_finetune/model_f_experiment/entity_descriptions.py` | Description handling for F-DESC | yes | no | raw entity rows in the frozen payload |

Commands and the plan id `EXP-20261101-009-model-f-four-arm` are in `docs/implementation/MODEL_F_FOUR_ARM_EXPERIMENT.md`. Do not run `snapshot`, `verify`, `run`, or `audit` while `indexing.lock` exists.

## Evaluation harness

Package: `host_finetune/eval_bench/`. Status: IMPLEMENTED and TESTED. Ranked retrieval metrics stay null until a person assigns grades.

| Path | Purpose | Git | Deploy | Dependencies / setup |
|---|---|---|---|---|
| `host_finetune/eval_bench/__init__.py` | Package marker | yes | no | none |
| `host_finetune/eval_bench/__main__.py` | Score a predictions file | yes | no | CPU |
| `host_finetune/eval_bench/schema.py` | `eval-bench-v1` | yes | no | none |
| `host_finetune/eval_bench/metrics.py` | Faithfulness, grounding, leak, copying, latency, tokens | yes | no | none |
| `host_finetune/eval_bench/stats.py` | Paired comparisons | yes | no | none |
| `host_finetune/eval_bench/coverage.py` | Coverage bookkeeping | yes | no | none |
| `host_finetune/eval_bench/report.py` | Report writer. Refuses `storage_expand_v1`. | yes | no | none |
| `host_finetune/eval_bench/runner.py` | Batch scoring | yes | no | none |
| `host_finetune/eval_bench/annotation.py` | Annotation sheet helpers | yes | no | none |
| `host_finetune/eval_bench/matrix.py` | Six-arm matrix, including E-DENSE and F-OFF | yes | no | none |
| `host_finetune/eval_bench/historical.py` | Replay of the October 31 pilot cells | yes | no | EXP-20261031-001 raw fixtures below |
| `host_finetune/eval_bench/dev_benchmark.py` | Builds the development question sheet from ids | yes | no | question files and frozen pilot retrieval files |

## Extraction normalization and indexing utilities

These modules are already loaded by the running indexer. They are committed as they stood. They were not edited for this handoff.

| Path | Purpose | Git | Deploy | Dependencies / setup |
|---|---|---|---|---|
| `host_finetune/relation_recovery.py` | `relation-normalization-v1`. Refuses to apply repairs to `storage_expand_v1`. | yes | no | a snapshot, not the live store |
| `host_finetune/indexing_session.py` | Resumable session CLI. Default workdir is the live expansion store. | yes | no | do not start a second session |
| `host_finetune/indexing_ledger.py` | SQLite progress ledger | yes | no | local to the workdir |
| `host_finetune/indexing_lightrag.py` | One-document index attempt | yes | no | Ollama when indexing |
| `host_finetune/indexing_schedule.py` | Blog-only queue | yes | no | graph-ready documents, which are local |
| `host_finetune/indexing_selection.py` | Selection policy | yes | no | none |
| `host_finetune/indexing_timeout.py` | Adaptive timeout policy | yes | no | none |
| `host_finetune/indexing_classify.py` | Failure classification | yes | no | none |
| `host_finetune/indexing_quality.py` | Format warnings and weak-entity checks | yes | no | none |
| `host_finetune/indexing_reports.py` | Status reports | yes | no | none |
| `host_finetune/indexing_reconcile.py` | Read-only reconciliation | yes | no | do not point it at the live store while indexing |
| `host_finetune/indexing_segments.py` | Omission-segment derivative. Must not be inserted into the expansion graph. | yes | no | none |
| `host_finetune/indexing_extraction_audit.py` | Census helpers for cached extraction records | yes | no | does not replace the cache |

## Tests

| Path | Purpose | Git | Deploy | Dependencies / setup |
|---|---|---|---|---|
| `tests/test_model_f_graph_context.py` | Normalizer, paths, ranking, budgets, prompt, saved hybrid payload | yes | no | `frozen_hybrid.jsonl` fixture |
| `tests/test_model_f_experiment.py` | Runner, snapshot refusal, audit, window preflight | yes | no | synthetic fixtures under `tmp_path` |
| `tests/test_eval_bench.py` | Scorer and live-store refusal | yes | no | synthetic records |
| `tests/test_eval_bench_historical.py` | October 31 replay | yes | no | EXP-20261031-001 generations and frozen retrieval files |
| `tests/test_dev_benchmark.py` | Ungraded development sheet | yes | no | benchmark builder and frozen retrieval ids |
| `tests/test_graph_relationships.py` | Relationship acceptance | yes | no | synthetic |
| `tests/test_relation_normalization.py` | Normalization policy. Writes its own ledger under `tmp_path`. | yes | no | synthetic |
| `tests/test_indexing_checkpoint.py` | Ledger resume and protected-path refusal | yes | no | `tmp_path` |
| `tests/test_indexing_extraction_audit.py` | Extraction census rules | yes | no | synthetic |
| `tests/test_indexing_quality.py` | Quality warnings | yes | no | synthetic |
| `tests/test_indexing_selection.py` | Blog-only selection | yes | no | synthetic |
| `tests/test_indexing_timeout.py` | Adaptive timeout | yes | no | synthetic |
| `tests/test_evidence_capsule.py` | Claim capsule | yes | no | synthetic |
| `tests/test_factual_reliability.py` | Factual checks | yes | no | synthetic |
| `tests/test_numeric_grounding.py` | Numeric gate | yes | no | synthetic |
| `tests/test_lightrag_retrieval.py` | Normalizer and fail-closed Ollama check. Ollama is monkeypatched. | yes | no | `lightrag` import skip if the package is missing |

## Development benchmark

| Path | Purpose | Git | Deploy | Dependencies / setup |
|---|---|---|---|---|
| `host_finetune/benchmarks/dev_retrieval_benchmark.jsonl` | 36 development questions, document ids, and null grades. Inspected: no passage text and no credentials. | yes | no | do not treat null grades as labels |
| `host_finetune/benchmarks/dev_retrieval_benchmark_withheld.jsonl` | Review notes withheld from an annotator. Short notes and ids only. | yes | no | not a grade file |

The corpus test split is not in these files. Do not open it to retune prompts.

## Documentation

| Path | Purpose | Git | Deploy | Dependencies / setup |
|---|---|---|---|---|
| `docs/implementation/MODEL_F_GRAPH_CONTEXT_IMPLEMENTATION.md` | Architecture, schema, ranking, budgets, limits | yes | no | none |
| `docs/implementation/MODEL_F_FOUR_ARM_EXPERIMENT.md` | Arm definitions and post-index commands | yes | no | run only after the indexer exits |
| `docs/implementation/MODEL_F_POST_INDEX_CHECKLIST.md` | Snapshot, entity-description follow-up, scoring | yes | no | blocked on the indexer |
| `docs/implementation/LIGHTRAG_RELATIONSHIP_REPAIR.md` | Normalization policy. Apply only on a clone. | yes | no | none |
| `docs/architecture/LIGHTRAG_MODEL_F.md` | Model F beside Model E | yes | no | none |
| `docs/architecture/LIGHTRAG_INDEXING_AND_CHECKPOINTING.md` | Resumable indexing | yes | no | none |
| `docs/evaluation/MODEL_F_EXPERIMENT_SPEC.md` | Broader six-arm spec. E-DENSE and F-OFF are not in the four-arm inference run. | yes | no | none |
| `docs/evaluation/MODEL_F_EVALUATION_INFRASTRUCTURE.md` | Metric and harness notes | yes | no | none |
| `docs/evaluation/RETRIEVAL_ANNOTATION_GUIDE.md` | How to grade the development sheet | yes | no | none |
| `docs/evaluation/LIGHTRAG_EXTRACTION_AND_GRAPH_QUALITY_AUDIT.md` | Audit write-up | yes | no | none |
| `docs/deployment/DEPLOYMENT_HANDOFF.md` | Earlier Relabel + Chroma v1 deployment note. Already in the branch. | already tracked | no for Model F graph context | Chroma and the Relabel adapter, which are not in this commit |
| `docs/handoff/TEAMMATE_DEPLOYMENT_HANDOFF.md` | Deployment instructions for this branch | yes | no | none |
| `experiments/README.md` | Experiment layout and the new run notes | yes | no | none |

## Experiment records included

These are registered run records. They are not required to start the service. Raw pilot retrieval files are included only because the CPU tests read them. They contain retrieved passage excerpts and saved drafts from the October 31 pilot. They are not the corpus, not an index, and not an LLM cache.

| Path | Purpose | Git | Deploy | Dependencies / setup |
|---|---|---|---|---|
| `experiments/EXP-20261031-001-graph-relationship-conditioning/` except `raw/pilot_dense_matrix.npy` | Pilot relationship run. Tests read `raw/generations/*.json` and `raw/frozen_{pilot_dense,naive,hybrid,mix}.jsonl`. | yes, except the matrix | no | passage text is inside the frozen hit records |
| `experiments/EXP-20261031-002-lightrag-resumable-index/` | Resumable indexer record. Small status files. | yes | no | none |
| `experiments/EXP-20261031-003-lightrag-index-hardening/` | Hardening and reconciliation summaries | yes | no | none |
| `experiments/EXP-20261031-004-lightrag-adaptive-timeout/` | Adaptive timeout record | yes | no | none |
| `experiments/EXP-20261031-005-lightrag-blog-only-selection/` | Blog-only policy counts | yes | no | none |
| `experiments/EXP-20261101-006-lightrag-extraction-audit/` except `raw/cache_census.json` | Audit counts and manifest | yes, except the census | no | census stays local because it samples extraction fields |
| `experiments/EXP-20261101-007-lightrag-relation-recovery/` except `raw/recovery_ledger.jsonl` | Dry-run summary and manifest. The ledger stays local. | yes, except the ledger | no | summary is counts only |
| `experiments/EXP-20261101-008-eval-bench-historical-replay/` except both `results.jsonl` files | Replay metrics, reports, and short analysis. The results files repeat passage text. | yes, except those two files | no | tests replay from EXP-20261031-001 into a temp directory |
| `experiments/EXP-20261030-012-fixed-style-ablation/manifest.yaml` | Style-ablation registration | yes | no | none |
| `experiments/EXP-20261030-012-fixed-style-ablation/notes.md` | What that run held fixed | yes | no | none |
| `experiments/EXP-20261030-012-fixed-style-ablation/metrics/paired_decision.json` | Small decision summary | yes | no | none |
| `experiments/EXP-20261030-012-fixed-style-ablation/metrics/sft_readiness.json` | Small readiness summary | yes | no | none |

## Intentionally excluded

| Path | Why it stays out of Git |
|---|---|
| `artifacts/lightrag/model_f/storage_expand_v1/` | Live partial index. Already ignored. Not copied. |
| Any `storage_*` snapshot or `kv_store_llm_response_cache.json` | Index and LLM response cache |
| `data/cleaned/` and `data/cleaned.zip` | Source datasets. Already tracked in older commits that are already on GitHub. Not added again. |
| `deployment_handoff/` | Local adapter and Chroma bundles |
| `host_finetune/output/` | Local weights and Chroma builds. Two historical `adapter_model.safetensors` files and `chroma_db/chroma.sqlite3` are already tracked from earlier commits. They were not removed and were not re-added. |
| `experiments/EXP-20261031-001-graph-relationship-conditioning/raw/pilot_dense_matrix.npy` | Embedding matrix |
| `experiments/EXP-20261101-006-lightrag-extraction-audit/raw/cache_census.json` | Samples taken from the extraction cache |
| `experiments/EXP-20261101-007-lightrag-relation-recovery/raw/recovery_ledger.jsonl` | Extraction-candidate ledger, about 1.4 MB |
| `experiments/EXP-20261101-008-eval-bench-historical-replay/generation/results.jsonl` | Replay dump containing source passages |
| `experiments/EXP-20261101-008-eval-bench-historical-replay/retrieval/results.jsonl` | Replay dump |
| `experiments/EXP-20261030-012-fixed-style-ablation/raw/` | Generated cell dumps |
| `experiments/EXP-20261030-012-fixed-style-ablation/metrics/judge_cache.jsonl` | LLM judge cache |
| `experiments/EXP-20261030-012-fixed-style-ablation/metrics/comparison.json` | Large generated comparison, about 1.2 MB |
| `.env` | Secrets. Only `.env.example` is appropriate to share. No `.env` was present to stage. |

Already tracked, reported, and left in place: `data/cleaned/*.jsonl`, `data/cleaned.zip`, `chroma_db/chroma.sqlite3`, `host_finetune/output/lemkin_lora_r32/adapter_model.safetensors`, `host_finetune/output/lemkin_lora_repaired/adapter_model.safetensors`, and `unsloth_compiled_cache/`. Removing them from the branch tip would not remove the blobs from history, and history was not rewritten. `.gitignore` now ignores new files of those kinds. Ignore rules do not untrack files Git already follows.

## Index replacement

The finished index is supplied outside Git. After the indexer has exited and `indexing.lock` is gone:

```text
host_finetune\.venv\Scripts\python.exe -X utf8 -m host_finetune.model_f_experiment snapshot --source artifacts\lightrag\model_f\storage_expand_v1 --dest artifacts\lightrag\model_f\storage_expand_v1_snapshot --allow-expansion-source
host_finetune\.venv\Scripts\python.exe -X utf8 -m host_finetune.model_f_experiment verify --snapshot artifacts\lightrag\model_f\storage_expand_v1_snapshot
```

Point the service at the verified directory with `--lightrag-workdir`. Do not commit that directory. Replacing a partial copy means changing the path the service loads. Do not merge a partial directory into the finished one.
