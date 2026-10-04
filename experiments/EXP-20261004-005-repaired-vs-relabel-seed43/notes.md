# Repaired versus relabel, retrieval off, seed base 43

Same protocol as EXP-20261004-003 with `--seed-base 43`. Retrieval is off. Used to see whether the seed-42 stop rates hold.

## Result

Exit 0. 120 answers each. `run_parameters.json` records seed base 43 and retrieval off.

| Adapter | Early stop | Mid-sentence |
|---|---:|---:|
| repaired | 0.608 | 0.600 |
| relabel | 0.550 | 0.300 |

By medium, early stop then mid-sentence: repaired blog 0.867 / 0.133, LinkedIn 0.067 / 0.733, X 0.533 / 0.700, talk 0.967 / 0.833. Relabel blog 0.500 / 0.033, LinkedIn 0.233 / 0.567, X 0.800 / 0.533, talk 0.667 / 0.067.
