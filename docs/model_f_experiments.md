# Model F experiments

Numbers below are copied from saved JSON. Directories that hold raw generations and LightRAG stores are local. Their shareable derivatives are in [`results/model-f-handoff-v1`](../results/model-f-handoff-v1). Re-running these commands needs the snapshot and the Relabel adapter, which are not in Git.

## What each run was allowed to decide

| Run | ID recorded on the artifact | What it selected | What it did not do |
|---|---|---|---|
| Frozen pilot | `EXP-20261102-002-model-f-quick-pilot` | **F-SOURCE** | Persona, stance, or writing quality |
| Stage-2 recommendation | `EXP-20261102-003-model-f-desc-vs-rel` | **F-REL** over F-DESC | Replace the frozen pilot |
| Official 120-prompt panel | `EXP-20261102-001-model-f-unified-panel` | Nothing. Profile was already F-REL | Use the 120 scores to pick an arm |

`posthoc_arm_selection.json` on the official panel says `uses_120_prompt_panel_for_selection: false` and `frozen_pilot_arm: F-SOURCE`.

## Pilot

Artifact paths (local): `experiments/EXP-model-f-quick-pilot/arm_selection.json` and `arm_comparison.json`.

Eight questions were complete for F-SOURCE, F-REL, F-DESC, and F-HOP. The written rule selects a graph arm only when it strictly beats F-SOURCE on deterministic faithfulness, or ties that rate and strictly beats F-SOURCE on source-copy or prompt-leak failures. No graph arm did. Status: `fallback`. Frozen: true.

Factual pass rate on those 8 cells (`arm_comparison.json`):

| Arm | n | Factual pass rate | Copy or leak failure rate |
|---|---:|---:|---:|
| F-SOURCE | 8 | 0.500 | 0.000 |
| F-DESC | 8 | 0.500 | 0.000 |
| F-HOP | 8 | 0.500 | 0.125 |
| F-REL | 8 | 0.375 | 0.125 |

F-REL was worse than F-SOURCE on this sheet. That is why the pilot stayed F-SOURCE.

## Stage-2, F-DESC versus F-REL

Artifact paths (local): `experiments/EXP-model-f-desc-vs-rel-32/desc_vs_rel_recommendation.json` and `desc_vs_rel_comparison.json`.

26 questions were complete for both arms. Two question ids are listed as incomplete. The primary key is echo-adjusted faithfulness: a cell passes only when the deterministic faithfulness check passes and the answer is not a question echo. Copy or leak rate is the secondary key. Context-token count is not a score.

| Arm | n | Echo-adjusted pass rate | Scorer faithfulness | Copy or leak failure | Numeric failure |
|---|---:|---:|---:|---:|---:|
| F-DESC | 26 | 0.3077 | 0.5385 | 0.0385 | 0.2308 |
| F-REL | 26 | 0.3462 | 0.5385 | 0.0385 | 0.1538 |

Question wins were 4 for F-DESC, 4 for F-REL, and 18 ties. The recommendation file still selects F-REL: "F-REL leads on echo-adjusted faithfulness or, given a tie, on copy and leakage." Status: `selected`. The file says the frozen pilot is unchanged.

The official plan calls this profile source a "36-question development experiment." The comparison file's scored n is 26 complete questions, plus 2 incomplete ids. This page does not invent the missing 8.

## Official 120-prompt evaluation

Panel: `gold_001`–`gold_030` × blog, LinkedIn, X, talk. 120 cells. Profile F-REL.

| Identity | Recorded value |
|---|---|
| Gold file | `data/openai_ft/lemkin_gold_eval.jsonl` |
| Gold SHA-256 | `d7c4f997899e28f28eb3ea9aa3417857da6f0666acbf72fd8ac31422034e520e` |
| Generation SHA-256 | `6ed2af4f47c3f592aa833c643dc6354956b0ea7f96b7634b98a902b2a746724c` |
| Judge | `qwen2.5:14b`, blinded, temperature 0, family `qwen`, candidate family `llama` |
| Schema | `unified_eval.v1` |
| Judge time | `2026-11-03T00:43:46.431318+00:00` |
| Git commit at judge time | `31b27c03eb04faf88f00a7c7930c3e9f5a32e7ed` |
| Worktree at judge time | dirty: true |
| Python on the judge process | 3.11.9 |
| Seed | base 42, then `42 + gold_index * 4 + medium_index` |

The generation manifest status is still `generated_not_judged`, and `judge_overlay` is `not_applied`. `unified_eval/summary.json` and `evaluation_run.json` are the judge record. Do not treat the manifest status as "the judge never ran."

The only command string stored on the judge record used a local absolute path. The same invocation with a project-relative output directory is:

```powershell
python -m host_finetune.model_f_experiment unified-judge --profile F-REL --out experiments/EXP-20261102-001-model-f-unified-panel
```

That command scores saved drafts. It does not rebuild the graph and it does not regenerate. A fresh generation, which this handoff does not ask anyone to run, is:

```powershell
python -m host_finetune.model_f_experiment unified-plan --profile F-REL --snapshot artifacts/lightrag/model_f/storage_expand_v1_snapshot --out experiments/EXP-YYYYMMDD-NNN-model-f-rerun
python -m host_finetune.model_f_experiment unified-run --profile F-REL --snapshot artifacts/lightrag/model_f/storage_expand_v1_snapshot --out experiments/EXP-YYYYMMDD-NNN-model-f-rerun
```

Use a new output directory. Do not overwrite `EXP-20261102-001`.

Utility weights stored on the judge summary: persona style 0.45, perspective fidelity 0.20, writing quality 0.25, task adherence 0.10. Persona utility itself is null for Model F, so those weights did not produce a utility number on this panel.

Writing-quality calibration on the summary is `calibrated: true`, role `selector`, reason `writing_order_holds`. Stance calibration is `calibrated: false`, role `diagnostic_only`, reason `stance_controls_missing`.

## Earlier graph work

The branch history before this panel is described in [`docs/architecture/LIGHTRAG_MODEL_F.md`](architecture/LIGHTRAG_MODEL_F.md). Those experiments (pilot index EXP-20261029-004, four-arm notes, and the expansion audits) are registered separately. They are not the 120-prompt F-REL result.
