# Experiments

Each experiment is one directory:

```text
experiments/EXP-YYYYMMDD-NNN-short-name/
  manifest.yaml
  config/
  raw/
  metrics/
  figures/
  notes.md
```

- `EXP-YYYYMMDD-NNN-short-name` is the experiment ID. `YYYYMMDD` is the run date. `NNN` is a three-digit sequence for that date, starting at `001`. `short-name` is a lowercase hyphenated label.
- `manifest.yaml` records what was executed. A metric is not interpretable without the identities below: Git commit, dataset and split, gold set, persona profile, Chroma or index, embedding model, generation or base model, adapter or checkpoint, and judge model when a judge was used.
- `config/` holds the configuration used for that run.
- `raw/` holds machine-generated outputs (traces, logs, model responses).
- `metrics/` holds computed metric files.
- `figures/` holds plots or other figures for that run.
- `notes.md` states the question, what was held fixed, and any failures or warnings.

Artifacts inside a registered `experiments/EXP-*` directory are evidence of what that run executed. Historical JSON, logs, workbooks, and presentations outside that layout are not a registered run.

Do not modify raw source data in place. Do not change train, validation, or test assignments inside an experiment unless that change was explicitly approved and is recorded in `manifest.yaml`.

No experiment run has been created yet.

## Required manifest

Every `manifest.yaml` must include these fields. Use an empty value or `n/a` when a field does not apply, and say why in `notes.md`.

```yaml
experiment_id:
timestamp:
execution_commands: []

git:
  commit:
  dirty: false

data:
  source_manifest:
  source_hashes:
  processed_dataset_hash:
  split_manifest_hash:
  gold_eval_hash:

rag:
  persona_profile_hash:
  chroma_collection:
  index_version_or_snapshot:
  embedding_model:

model:
  model_name:
  model_or_adapter_hash:
  config:
  random_seed:

evaluation:
  evaluator_commit:
  judge_model:
  judge_config:

environment:
  python_version:
  package_snapshot:
  hardware:

outputs:
  raw:
  metrics:
  warnings_and_failures:
```

Record, when applicable:

- Git commit and dirty or clean state
- raw or source data identities or hashes
- processed dataset hash
- split manifest or hash
- gold evaluation set hash
- `persona_profile.json` hash
- Chroma collection and index identity
- embedding model
- generation or base model
- adapter or checkpoint identity or hash
- model configuration
- random seeds
- evaluator version or commit
- LLM judge model and configuration
- Python, environment, and package information
- hardware
- exact execution commands
- raw outputs
- metrics
- warnings and failures
