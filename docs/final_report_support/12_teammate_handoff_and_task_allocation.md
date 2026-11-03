# Teammate handoff and task allocation

Roles below are workstreams. This note does not assign a person. Branch history already records who has worked on the React deploy line and on the Model F line; those facts are in `docs/teammate_handoff.md` and `docs/integration/RECENT_BRANCH_AUDIT_20261030.md`. Use them only as ownership context the repository already states.

## Modeling and evaluation

| Task | Depends on | Acceptance |
|---|---|---|
| Freeze the narrative that there is no single winner | This package | Report text matches `05_results_and_interpretation.md` |
| Footnote training-copy 104 versus 120 and the 16 blank flags | Flag CSV | Both denominators appear |
| Decide on persona utility | A new experiment directory if it is computed | Official panel files unchanged |
| Choose redacted example topics | Private zip and a publication decision | No 12-word copy span and no URL in the public example |
| Optional F-SOURCE 120-cell run | Snapshot copy, GPU, new `EXP-*` id | Not written into `EXP-20261102-001` |

## Data and graph

| Task | Depends on | Acceptance |
|---|---|---|
| Deliver snapshot and adapter zips on a private channel and confirm hashes | Files listed in `docs/teammate_handoff.md` | Receiver recomputes SHA-256 |
| Read-only subgraph, if the report needs one | A copied snapshot | Frozen directory byte-identical afterward |
| Leave the 413 relationship repairs unapplied | — | Snapshot hash unchanged |
| Any further indexing | A new store | New experiment id and a statement that counts will differ |

## Backend and deployment

| Task | Depends on | Acceptance |
|---|---|---|
| Add a Model F path beside the existing deploy route | Deploy branch plus Model F modules | Response names profile F-REL, hybrid, top_k 20, and the snapshot id |
| Keep Model E on its current route until both smokes pass | Transfer note section | Dense route unchanged |
| Set `num_ctx` 16384 on the demo path | `lightrag_backend.py` default is 32768 when the CLI does not pass it | Demo request records 16384 |
| Refuse to open the live `storage_expand_v1` directory | `verify` behavior | Process uses a runtime copy |

## Frontend

| Task | Depends on | Acceptance |
|---|---|---|
| Topic form submits to the route above | Backend route | UI shows the architecture name returned by the API |
| Do not imply a model switch that the API does not have | Current `backend/main.py` has no switch | Either the control exists or the UI does not show it |
| Screenshots of submit, result, and error | Running UI | Appendix A files |

## Testing

| Task | Depends on | Acceptance |
|---|---|---|
| Run the CPU pytest command in `docs/teammate_handoff.md` | venv without GPU | Log stored outside the official experiment directory |
| Time one end-to-end topic if the stack is up | Snapshot and adapter | Report retrieval time and generation time as separate numbers |
| Do not treat historical 562, 156, or 68 as this week's run | — | Date on the log |

## Report

| Task | Depends on | Acceptance |
|---|---|---|
| Draft from `13_final_report_outline.md` | This directory | Every metric has a denominator and a path |
| Replace stale "Model F not measured" sentences | `model_f_results.md` | No dash for the measured panel |
| State the classifier-metric gap | `04_evaluation_methodology.md` | No fabricated confusion matrix |

## Presentation

| Task | Depends on | Acceptance |
|---|---|---|
| Build slides from `09_demo_2_plan.md` | Figures | Each results slide shows n |
| Rehearse the offline path | Figures only | Talk still works if Ollama is down |
| Keep private drafts off the public deck | Export safety note | No URL and no long copied span |

## Suggested order

UI route and screenshots first, because that category is 4 of 12.5 demo points and the code gap is already documented. Then the report's results section, which can be written from files that already exist. Then the CPU test log and the latency target decision. Graph completion and a second official arm are research follow-ups, not demo blockers.
