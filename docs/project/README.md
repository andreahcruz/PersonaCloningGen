# What to load

| Task size | Read |
|---|---|
| Small edit | `AGENTS.md`, this handoff, the active task in `tasks.json`, and the code you will change |
| Medium change | Also `state.json` and the one architecture or decision section the task names |
| Planning, experiments, or report claims | `state.json`, `docs/DECISION_LOG.md`, `experiments/README.md`, and the specific `EXP-*` or `docs/final_report_support/` page cited by the task |

Do not load `docs/research/` for an ordinary code change. `docs/project_state.md` and `docs/PLAN.md` below their 2026-11-07 banners are dated records, not the current next step.

Commands:

```text
python -m project_pm check
python -m project_pm check-manifest experiments/EXP-YYYYMMDD-NNN-name/manifest.yaml
```

Warnings do not fail the command. Errors do. The checker never marks a task done.
