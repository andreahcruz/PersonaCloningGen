# Session handoff

- updated: 2026-11-08
- git_commit: b40ff1738d57e30a06ad2ebd7ba9b4cf6b060f5e
- git_branch: codex/ont-01-schema-admission
- active_task: ONT-HANDOFF
- session: awaiting_owner_coordination
- next_action: Owner reviews docs/architecture/ONTOLOGY_HANDOFF.md, confirms teammate/checkpoint/reviewers and coordinates ONT-02 activation alongside fine-tuning. ONT-02 through ONT-04 ownership is proposed, not activated; all five sources remain pending.

## Current context

Phase is `ontology-handoff-review`. Official Model F is F-REL in `results/model-f-handoff-v1`. Persisted `lemkin_train_only` has 21,193 documents. The 21,194 figure is the 2026-10-04 exit log. Two DEC-007 headings remain on purpose. The unchanged 30-topic gold set is the primary fixed architecture benchmark; the 24 TASK-002 questions are development material. An independent final question set is optional and nonblocking (DEC-012).

## Latest completed work

ONT-HANDOFF completed: docs/architecture/ONTOLOGY_HANDOFF.md is the self-contained teammate guide for ONT-02 through ONT-04. It records executable invariants, exact references/commands, extraction/review specification, store/retrieval exits, generator interface, evaluation rules and proposed disjoint ownership. Only the documentation task was registered; no teammate or fine-tuning implementation task was activated/reassigned. Checkpoint transfer and human source admission remain prerequisites.

ONT-01 contracts completed and tested on 2026-11-08: 53 ontology tests and 169 relevant tests passed. EXP-20261108-004 records five pending candidates, zero admitted documents and zero real assertions. Manifest and project-state checks pass with only the two known historical warnings. DEC-013 records the user-approved architecture constraints. See docs/architecture/ONT_01_REPORT.md and the new run verification artifacts. Stop for user review; ONT-02 has not started.

TASK-003 design completed on 2026-11-08 in docs/architecture/ONTOLOGY_PLAN.md, with the user-approved evaluation policy appended as DEC-012. Project-state checks pass with two known warnings; all 16 project-management tests pass. TASK-002 closure and all prior experiment outputs remain intact. No model inference, experiment measurement, dataset, weight, or graph snapshot was changed by TASK-003. F-SOURCE and F-DESC stay uncommitted and unregistered.

## Agent boundaries

Codex, TASK-002, status done after owner applies this closure snapshot and verifies the live checker: `host_finetune/gold_partition_audit.py`, `tests/test_gold_partition_audit.py`, and `experiments/EXP-20261108-001-gold-partition-audit/`. Also owns `experiments/EXP-20261108-002-gold-exposure-review/` (incomplete UTF-16 parsing attempt) and `experiments/EXP-20261108-003-gold-exposure-review/` (corrected follow-up). See that run's notes and verification. Read-only against existing indexes and splits. Do not rerun historical experiments.

Codex, TASK-003, status done, branch `codex/task-003-ontology-plan`: `docs/architecture/ONTOLOGY_PLAN.md` and append-only `docs/DECISION_LOG.md` for DEC-012. The user explicitly approved reassignment from Cursor and synchronization of the three shared project-state files. The general direction was approved subject to DEC-013. ONT-01 is now registered and complete; ONT-02 through ONT-06 remain proposed backlog labels, not authorized implementation tasks. No extraction, model inference, training, indexing, or frozen-data edits occurred.

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

## ONT-01 review boundary

ONT-01 is done on codex/ont-01-schema-admission; its owned paths and verification are listed in tasks.json. User review is pending. User approved existing train-only Chroma with DuckDB-backed claim expansion; a separate claim-vector index is optional only if justified later. Five candidate blogs remain pending where manual attribution review is absent. Historical data, experiments, weights and indexes remain frozen.
