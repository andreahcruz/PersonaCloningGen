# Agent rules — DATA 298B

These rules apply to all work on this repository.

## Source of truth

- Current repository code, committed configuration, dataset/split manifests, and artifacts produced inside a registered `experiments/EXP-*` run are authoritative evidence of what was executed.
- Historical workbooks, presentations, READMEs, logs, and result files outside a registered experiment remain historical evidence until reproduced.

## Status definitions

- **PROPOSED** — described or planned, but not present in executable code.
- **IMPLEMENTED** — executable code or configuration exists.
- **TESTED** — relevant automated tests or controlled smoke tests have passed.
- **MEASURED** — a registered experiment produced metrics or results under a recorded manifest.
- **VALIDATED** — the result was independently checked or reproduced and satisfies the stated acceptance criteria.

Do not call historical outputs current results unless they have been reproduced. Never fabricate metrics or results.

## Before modifying code

- Inspect the relevant implementation, tests, configs, and call sites.
- State the intended change and the acceptance criteria.
- Make the smallest coherent change.
- Do not silently alter datasets, splits, prompts, model parameters, evaluation methodology, or experiment configuration.

## Experiments

Every experiment must record:

- experiment ID
- timestamp
- Git commit and whether the worktree was dirty
- dataset/split version, including source, processed-dataset, split, and gold-eval identities or hashes
- model/configuration, including adapter or checkpoint identity
- persona profile hash, Chroma collection or index identity, and embedding model, when retrieval is involved
- random seed where applicable
- evaluator version or commit, and LLM judge model and configuration, when scoring is involved
- execution command
- environment, package snapshot, and hardware
- raw outputs
- metrics
- failures and warnings

Store each run under `experiments/` using the layout in `experiments/README.md`. A metric from a file outside a registered `experiments/EXP-*` run is historical evidence.

## Data

- Never modify raw source data in place.
- Do not modify train/validation/test assignments without explicit approval.
- Check for exact and near-duplicate leakage.

## Quality

- Run the relevant tests.
- Do not silently fix unrelated problems.
- Preserve reproducibility.
- Keep work traceable to the assignment rubric.
