# TASK-002 gold-partition audit

Lexical audit uses existing exact/title/8-word containment and token Jaccard >=0.80 rules; semantic paraphrases and unrecorded source-family links can remain undetected.

Gold is the reused 30-topic comparison benchmark, not an untouched final test.

Family exclusions apply to retrieval selection, not removal from the frozen fine-tuning dataset. Overlap rows by split remain present there; no split changes are authorized.

Questions are manually paraphrased proposals anchored to unchanged train blog rows. They are neither independent test items nor verified answerable benchmarks; human review is still needed.

No source bodies, answers, or expected facts exported. No training, model generation, scoring, embedding, vector search, or graph access performed.

SQLite metadata/content parity does not validate HNSW vectors or retrieval quality. No historical experiment was rerun.

Shared project state intentionally unchanged under user instructions; task status and branch fields require owner reconciliation.

Audit check failures: none. All eight checks in metrics/checks.json passed.

Recorded dispositions identify 34 blocked families. Replaying those exclusions removes 183 train rows, leaving 21,194. The existing headline check additionally removes train_26148 (gold_001 topic, normalized Jaccard 1.0), leaving 21,193. The persisted collection matches every selected ID, body, and metadata field. No remaining lexical overlap was detected in it. These are measured audit results, not model-quality results or semantic decontamination guarantees.

The frozen dataset still contains 184 train, 16 validation, and 4 test rows belonging to the detected gold families. Retrieval filtering does not erase fine-tuning exposure. No change is proposed or made to those assignments. Any future data exclusion or split change requires explicit approval and separate provenance.

Twelve development questions were proposed from distinct eligible unchanged train blog families, with no rejected seed. They are outside the detected gold families and use dev_candidate_001 through dev_candidate_012. They are development material, not independent final evaluation. Review question relevance and answerability before using them.

Verification: 7 new tests and 29 existing split/train-selection tests pass. Combined with project_pm tests, 51 pass and 1 fails (test_live_repository_check). Manifest validation passes with no warnings. project_pm check --no-git passes with the two existing unregistered-experiment warnings. The full project_pm check exits 1: stale_branch plus three unlisted_worktree_change errors for the TASK-002 owned paths. Exact commands, outputs, and exit codes are in raw/verification.json.

Owner action required: reconcile the current branch and TASK-002 status/ownership in docs/project/state.json, tasks.json, and HANDOFF.md, then rerun the full checker and live-state test. The user prohibited edits to those files, so this task is not marked done in shared state. No approval is needed to preserve the audit; changing protected data or promoting these candidates is outside this task.

An initial CPU-only audit attempt was interrupted before artifacts were written because the single-field matcher repeated normalization. The registered execution uses the existing batched headline matcher for persisted bodies. No source or historical experiment was changed by either attempt.
