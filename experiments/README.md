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

As of the 2026-10-03 audit, 13 EXP directories exist; five have manifests and eight
have incomplete registration. See [inventory](../docs/experiment_inventory.csv) and
[state](../docs/project_state.md). Preserve existing artifacts: do not invent missing
run metadata or retroactively treat recorded logs as independently validated results.

After the approved preparation task, three additional registered runs exist:
EXP-20261003-001/002/003. EXP-20261003-004 records the relabeled continuation
file. EXP-20261003-005 is the finished rank-16 train on that file (exit 0,
best eval loss 1.3316). Generation scores have not been recorded. EXP-20261004-001 is a failed
audit attempt; EXP-20261004-002 is the successful CPU audit of that relabel
file. The `lemkin_train_only` embed finished on 2026-10-04 at 03:55 with
21,194 documents. It is a local Chroma build, not a registered
`experiments/EXP-*` run. The latest whole-source
candidate passes structural verification but fails the content-quality gate;
see [preparation evidence](../docs/data/COMPLETE_SOURCE_SFT.md).
The relabel file is a different dataset and is not that candidate.

EXP-20261027-001 records the derived graph-ready audit. It does not replace the Relabel file or the Chroma indexes. EXP-20261027-002 records a three-document LightRAG index and one five-mode query. It is not a generation comparison. EXP-20261029-003 records the graph-ready v2 derivative and a three-document extraction comparison. It is not the 160-document pilot. EXP-20261029-004 records the finished 160-document pilot index: 152 processed documents and 8 extraction failures. EXP-20261029-005 records the eight-question retrieval comparison across matched dense search, frozen Chroma v2, and the five LightRAG modes. Precision and recall were not scored. EXP-20261029-006 records the same pilot retrieval sheet plus a three-topic Relabel generation. Accepted claims dropped the quoted speaker. EXP-20261029-007 records full-corpus Chroma v2 generation on those topics and the unified gates. The Brex Chroma topic produced no draft because the evidence JSON was unparseable. No arm in that run named Anneka Gupta. EXP-20261029-008 records a read-only integrity audit of the finished pilot graph. EXP-20261029-009 records the hardened four-arm generation on the same three topics, including the Qwen 2.5 14B writing judge. Style distances in that run are not ranked across arms because the eligible medium sets differ. EXP-20261030-010 records the same frozen retrieval under prompt contract factual-reliability-v1 and rescores EXP-009 with the strict factual checks. It does not replace EXP-009. EXP-20261030-011 records span repairs of the EXP-009 drafts. It does not replace EXP-009 or EXP-010. EXP-20261031-001 records source-verified relationship conditioning on the existing 152-document pilot. It does not expand the graph. Designated-source overlap in that run is not precision or recall. EXP-20261031-002 records the resumable LightRAG indexer and a bounded expansion in `storage_expand_v1`. It does not replace the pilot graph or the active retriever. EXP-20261031-003 records the read-only store audit, omission-segment derivative, and a further bounded expansion of that same store to 25 graph commits. It does not replace EXP-002 or mix segmented text into the current graph. EXP-20261031-004 records the adaptive timeout policy and an isolated four-document benchmark. It does not replace the expansion graph. EXP-20261031-005 records the blog-only scheduling policy on the same expansion store. It does not delete the mixed-media documents already indexed. EXP-20261101-006 records a read-only census of cached extraction records and the expansion graph. It does not insert edges or change the running indexer. EXP-20261101-007 records relation-normalization-v1 and a dry-run recovery ledger. It does not insert edges or change the fingerprint.

The Model F official panel is `EXP-20261102-001-model-f-unified-panel`. Its raw generations and embedded LightRAG store are local and gitignored. Aggregate metrics, the arm-selection history, and the judge provenance that are safe to publish are in `results/model-f-handoff-v1/`. Narrative: `docs/model_f_experiments.md` and `docs/model_f_results.md`. Those pages do not replace a `manifest.yaml` inside the experiment directory. The generation manifest on that run still says `generated_not_judged`; the judge record is `unified_eval/evaluation_run.json`.

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

New experiment directories whose id date is on or after `validate_id_date_on_or_after` in `docs/project/state.json` are checked by `python -m project_pm check`. `python -m project_pm check-manifest <path>` checks one manifest. Neither command edits the directory. A manifest may set `status: incomplete`; missing fields then stay visible as warnings and the run is not a finished result. Do not add that status to a frozen experiment to silence the checker.
