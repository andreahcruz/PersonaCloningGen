# EXP-20260927-001 baseline reproduction

Status: MEASURED for the offline gold scorer and the leakage audit. Not VALIDATED as a regeneration of model answers.

## What was held fixed

- Raw JSONL under `data/`, `host_finetune/data/dataset.jsonl` and its sibling copies, `data/openai_ft/lemkin_gold_eval.jsonl`, and `data/persona_profile.json` were hashed and not rewritten.
- Persona on disk is the 1000-document snapshot (`meta.generated_at` `2026-05-12T02:51:08.348792`). It was not rebuilt.
- Chroma was not started, loaded, or rebuilt. Nothing was listening on `127.0.0.1:8000`, so `index_version_or_snapshot` is `unknown`.
- The fine-tune was not run. The watcher was not started.
- Historical files under `host_finetune/output/eval/` were not overwritten. The 30 gold traces were copied into this experiment and scored here.

## What this run did

`host_finetune.eval_gold_rag` scored the copied traces with `data/openai_ft/lemkin_gold_eval.jsonl` and `formats/format_specs.yaml`. The parsed result in `metrics/gold_rag_rescore.json` is equal to the historical file `host_finetune/output/eval/gold_eval_baseline_scores_30.json`. That shows the current scorer emits the same document from those inputs. It does not show that the answers in the traces can be regenerated. Those traces do not record a model name. `generate_gold_eval_outputs.py` was not run. Its default model is `llama3.2:latest`, and that default was not invoked.

`eval_rag.py` was not run. Historical `traces.jsonl` uses `host_finetune/eval_set.example.jsonl` and models `llama3.1` and `lemkin-clone-v4`. Reproducing those traces needs the same Chroma collection. This run does not claim that.

## Leakage

The trainer input is `host_finetune/data/dataset.jsonl` (19321 rows). There is no saved train/validation/test file. `metrics/split_indices.jsonl` is a `datasets==4.3.0` `train_test_split(test_size=0.1, seed=42)` on row indices before the chat-template map in `finetune.py`. That map does not drop rows. The 10% side is the in-trainer validation split, not a held-out test set.

Counts are in `metrics/leakage_report.json`. At Jaccard 0.8 there is no gold-topic, gold-reference, or trace-context match to a training output. Four normalized-exact duplicate groups exist inside the training file. Token-set near duplicates also exist inside that file, including pairs that cross the approximated 90/10 index split.

## Git

Commit `ebe5d8ebb4a9a8b5d700b45c7d50d40930c55a36`. The worktree was dirty because of untracked files, including `docs/research/`.
