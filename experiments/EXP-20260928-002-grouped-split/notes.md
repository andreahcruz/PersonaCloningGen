# EXP-20260928-002 — Grouped 80/10/10 split

Question: can the frozen 19,321-row training file be assigned to train, validation, and test so that chunks of one post, normalized-exact copies, and Jaccard ≥ 0.8 near-duplicates stay inside one split?

Held fixed: `host_finetune/data/dataset.jsonl` (SHA-256 `a4bfe637b9c178a29bb885ef8502ff45d348fe743d8d2938ba56f8ea3f71e4f5`). EXP-20260927-001 was not edited. No model was loaded and `finetune.py` was not executed.

## Method

`host_finetune/split_groups.py` parses the instruction prefix `Write in the style of Jason Lemkin about:`, strips ` (part i of n)`, and groups on NFKC + casefold + collapsed whitespace. Union-find then merges groups when any output pair has token-set Jaccard ≥ 0.8 (rare-token prefix filter, same cutoff as EXP-001) or when normalized output text matches. Whole groups are sorted by size, then by group key, and given to the split with the most remaining row quota. Tie-break is the split name. There is no `datasets.train_test_split` call and no random seed.

`source`, `url`, `date`, and `essay_id` are not in the file. Every assignment records `source_platform: unavailable`.

## Result

19,321 assignment rows, 4,379 groups. Counts are 15,457 train / 1,932 validation / 1,932 test. Rounded shares are 0.8000 / 0.1000 / 0.1000. Against the exact quotas 15,456.8 / 1,932.1 / 1,932.1 the remainder is +0.2 / −0.1 / −0.1 rows, inside one percentage point.

No `group_id` appears in more than one split. No base title appears in more than one split (2,541 + 1,030 + 1,026 = 4,597 titles). Near-duplicate pairs inside the splits are 3,549 + 11 + 8 = 3,568, matching the EXP-001 pair count, and zero of those pairs cross a split. Normalized-exact groups also stay inside one split. The four EXP-001 normalized-exact index pairs and the three recorded Jaccard-1.0 pairs were checked against this file and each pair shares a split.

Most near-duplicate mass sits in train because the largest groups are assigned first, while the train quota is still the largest. Validation and test max out at 4 rows per base title. Train includes the 67-row document.

## What a later training run must do

`SPLIT_MANIFEST` is optional. Unset, `finetune.py` still uses `train_test_split(test_size=0.1, seed=42)` and prints that this is not the leakage-safe manifest. Set, it checks the manifest dataset hash against `DATASET_LOCAL`, skips dataset prep so the frozen file is not rewritten, trains on the train indices, evaluates on the validation indices, and leaves the test indices out of both. That training was not started here.

Checkpoints and scores for `lemkin-clone` and `lemkin-clone-v4` belong to the old 90/10 shuffle. They are not comparable to a model trained on this assignment. The 30-row gold file remains a separate author set.

## Checks

Assignment command: Python 3.11.9 from `host_finetune/.venv`. The script imports only the standard library. Tests: `.venv-test` Python 3.10.11, `tests/test_split_groups.py`, 7 passed, no model loaded.
