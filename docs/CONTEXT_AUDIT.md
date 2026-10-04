# Context audit delivery — 2026-10-03

## Scope and evidence

Read-first audit of `kevin/298b-integration` at ea1d58b. No runtime, source-data, split, prompt, model parameter, or experiment-output changes. New docs are a proposed research plan and evidence map, not approval to begin implementation.

Major conclusions and their evidence:

- **QLoRA is already implemented:** host config/trainer, assistant-only masking call, adapter configs and LFS-tracked weights; training logs under EXP-003/004/005/009. Quality not independently reproduced.
- **Recent comparisons are not RAG experiments:** `compare_adapters.py` generation functions and EXP-007/010 run parameters/traces. Root `generate.py` has a different retrieval/prompt call path.
- **Fragments are a plausible defect:** upstream builder, `sft_chunk_utils.py`, fitter fallback to words and full-post prefixes. Causal certainty in pasted summary exceeds available controlled evidence.
- **Gold overall is not voice:** explicit formula in `eval_gold_rag.score_row`; format/prompt mismatch in comparison mediums; validity concerns in `eval_rag.fit_style_classifier` and reference loaders.
- **Frozen files match assignments:** current `assert_manifest_matches` checks on EXP-002/008; identities in data/artifact_hashes.csv. Stored duplicate diagnostics were inspected, not independently rerun.
- **Run registration incomplete:** filesystem inventory finds eight of 13 EXP folders lack manifest.yaml. Existing manifests also contain unknowns/n/a values. No values were invented to repair history.
- **Past materials describe other states:** report's branch references, presentation slide 7 metricsft, separate persona code/Compose mounts, master tree containing Paul Graham, outdated notebook preparation narrative.
- **Rubric breadth is real:** local workbook rubric PDF p1 visually confirms five-plus proposals and an improved model; demo rubric text identifies preprocessing/splits/analytics/ML/presentation categories.

## Verification and limitations

103 tests passed, four skipped in `.venv-test`; Compose parses with obsolete-version warning. Services/GPU generation/training/export were not exercised. Current source and manifest hashes were captured. Bibliographic inventory covers all 126 local PDF/PPTX inputs; substantive synthesis is selective. External metadata, live LMS and remote branch freshness were not checked.

## Files created/updated

Updated: README.md, docs/project_state.md, docs/rubric_matrix.md, experiments/README.md.

Created: docs/PROJECT_CONTEXT.md, docs/PLAN.md, docs/DECISION_LOG.md, docs/CONTEXT_AUDIT.md, docs/experiment_inventory.csv, docs/architecture/ARCHITECTURE.md, docs/architecture/EXPERIMENT_MATRIX.md, docs/data/DATA_PROVENANCE.md, docs/data/DATA_SPLITS.md, docs/data/artifact_hashes.csv, docs/evaluation/EVALUATION_PLAN.md, docs/research/INDEX.md, docs/research/manifest.csv, docs/research/model_catalog.md.

## Approval boundary

Approve the audit before implementing the exact next task in PLAN.md: source-faithful SFT preparation with frozen document-family ownership and recorded validation. DEC-003 through DEC-005 are proposed architecture/training/evaluation decisions. Existing source assignments remain fixed unless separately approved. A full retrain or index replacement is not part of this delivery.
