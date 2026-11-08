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

## Shared Cursor and Codex procedure

Cursor and Codex both start here. DEC-011 records the operational snapshot. Do not treat a chat summary as project state.

Load only what the task size requires (`docs/project/README.md`):

- Small: this file, `docs/project/HANDOFF.md`, the active task in `docs/project/tasks.json`, and the code to change.
- Medium: also `docs/project/state.json` and the architecture note or DEC entry the task names.
- Planning, experiments, or report claims: also `docs/DECISION_LOG.md`, `experiments/README.md`, and the cited `EXP-*` or `docs/final_report_support/` page.

Do not read `docs/research/` for an ordinary code change. `docs/project_state.md` and `docs/PLAN.md` below the 2026-11-07 banners are dated records.

Before editing, name the task id, the files it owns, and the acceptance check. One `in_progress` task may own a path. Two active tasks must not own the same path or a parent of it. Substantial work uses a branch or worktree. Do not commit, push, merge, or reset unless the user asks. Do not discard uncommitted experiment artifacts.

Do not edit frozen `experiments/EXP-*` outputs, raw source data, model weights, or the LightRAG snapshot. A new run gets a new `experiments/EXP-*` directory and a manifest. Check it with `python -m project_pm check-manifest`. Experiments dated on or after the cutoff in `state.json` fail `python -m project_pm check` until that manifest has the required fields. `status: incomplete` keeps a missing-field result visible as a warning and does not make the run a result.

The 30-topic file `data/openai_ft/lemkin_gold_eval.jsonl` is the existing comparison benchmark. It is not an untouched final test. Do not regenerate it or reshuffle splits in routine work. TASK-002 is the separate leakage audit.

When a task changes architecture or evaluation policy, add a new DEC entry. Do not renumber or rewrite accepted decisions. When the phase, next task, or worktree identity changes, update `docs/project/state.json`, `docs/project/tasks.json`, and the field list at the top of `docs/project/HANDOFF.md` in the same session. A task with status `done` needs a `verification` string. Leave unfinished work as `proposed`, `in_progress`, or `blocked`.

Before finishing substantial work, run `python -m project_pm check`. Warnings may remain. Errors may not. The checker does not mark tasks complete.

To hand a task to the other agent, set `agent`, `branch`, `status`, `owns_files`, and `acceptance` in `tasks.json`, point `HANDOFF.md` at that task, and stop. The next agent reads the handoff and the task. Do not restate the task in chat.
