# Repaired versus relabel, retrieval off, seed base 44

Same protocol as EXP-20261004-003 with `--seed-base 44`. Retrieval is off.

## Result

Exit 0. 120 answers each. `run_parameters.json` records seed base 44 and retrieval off.

| Adapter | Early stop | Mid-sentence |
|---|---:|---:|
| repaired | 0.617 | 0.642 |
| relabel | 0.600 | 0.325 |

By medium, early stop then mid-sentence: repaired blog 0.900 / 0.167, LinkedIn 0.067 / 0.633, X 0.500 / 0.833, talk 1.000 / 0.933. Relabel blog 0.600 / 0.067, LinkedIn 0.300 / 0.567, X 0.767 / 0.633, talk 0.733 / 0.033.
