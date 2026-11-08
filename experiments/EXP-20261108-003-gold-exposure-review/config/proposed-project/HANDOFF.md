# Session handoff

- updated: 2026-11-08
- git_commit: b40ff1738d57e30a06ad2ebd7ba9b4cf6b060f5e
- git_branch: codex/task-002-gold-partition
- active_task: TASK-003
- session: active
- next_action: Begin TASK-003 planning only in docs/architecture/ONTOLOGY_PLAN.md on its assigned branch when the owner starts it. Do not implement ontology extraction or alter frozen data or indexes.

## Current context

Phase is `ready-for-ontology-planning`. Official Model F is F-REL in `results/model-f-handoff-v1`. Persisted `lemkin_train_only` has 21,193 documents. The 21,194 figure is the 2026-10-04 exit log. Two DEC-007 headings remain on purpose.

## Latest completed work

Documentation reconciliation on 2026-11-07. No experiment measurement, dataset, weight, or graph snapshot was changed. F-SOURCE and F-DESC stay uncommitted and unregistered.

## Agent boundaries

Codex, TASK-002, status done after owner applies this closure snapshot and verifies the live checker: `host_finetune/gold_partition_audit.py`, `tests/test_gold_partition_audit.py`, and `experiments/EXP-20261108-001-gold-partition-audit/`. Also owns `experiments/EXP-20261108-002-gold-exposure-review/` (incomplete UTF-16 parsing attempt) and `experiments/EXP-20261108-003-gold-exposure-review/` (corrected follow-up). See that run's notes and verification. Read-only against existing indexes and splits. Do not rerun historical experiments.

Cursor, TASK-003, status proposed: `docs/architecture/ONTOLOGY_PLAN.md` only. Planning does not wait for TASK-002. Do not implement extraction in this task.

Shared files, including `docs/project/`, `AGENTS.md`, and `docs/DECISION_LOG.md`, stay with the user until one agent has finished and the other is idle.

## Remaining issues

The DEC-007 identifier collision is documented and not renumbered. Eligible versus all-draft Model F writing quality still needs both numbers in the final report prose. The demo UI is still the older `generate.py` path.

## References

- `docs/project/tasks.json`
- `docs/evaluation/UNIFIED_120_COMPARISON.md`
- `results/model-f-handoff-v1/architecture_comparison.json`
- `experiments/EXP-20261004-009-dense-rag-v1/metrics/corpus.json`

## TASK-002 closure evidence

Corrected follow-up: `experiments/EXP-20261108-003-gold-exposure-review/notes.md`. 184 gold-family train rows were selected by the historical trainer; completed two-epoch consumption is strongly supported, without per-row proof. Literal topic exposure limits independent-question claims; it is not automatic invalid leakage. The 24 development questions are proposed train-source material. Retrieval parity is unchanged. No retraining or split change is authorized. TASK-003 remains planning only.
