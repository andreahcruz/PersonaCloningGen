# Provenance of the LanguageTool grammar data

Every `<run>.grammar_lt.jsonl` file is the per-draft result of `python -m eval_tools.grammar_lt` on the full draft file of that run
(one line per draft: errors, words, rate per 100 words, deviation from real posts, which rules fired). `summaries.json` has the pooled
numbers per run and medium and the rate of real Lemkin posts. No model or judge was run. The full draft files (4 MB each for the Relabel runs)
are not in the repository. They are on the team Google Drive (`298 B/puk28_results/`) and, for Kevin's runs, on branch
`kevin/298b-integration` (commit `0fac41485ec3655ea1439d6d2484115f243fdab1`). These are the SHA-256 hashes of the draft files that were scored,
the same ones recorded for PUK-28 and PUK-34:

| Run | Where the drafts come from | SHA-256 of the draft file |
|---|---|---|
| `relabel_rag_off_seed42` | Kevin's branch `kevin/298b-integration`: `experiments/EXP-20261004-003-repaired-vs-relabel/raw/relabel.jsonl` | `5d7ebfb57e93150288602e6523f83de300cd79a600c3ee119a48ca1a97a62f67` |
| `relabel_rag_off_seed43` | same branch: `experiments/EXP-20261004-005-repaired-vs-relabel-seed43/raw/relabel.jsonl` | `8c5da8462c9a14733057d07a657ef6fca66cba79f8b8674daba618baa5b56e99` |
| `relabel_rag_off_seed44` | same branch: `experiments/EXP-20261004-006-repaired-vs-relabel-seed44/raw/relabel.jsonl` | `50ed43f3d9f01e4091685292de0eb0632ee2ed990b13f7f5bf56cac5b03107f8` |
| `relabel_rag_on_seed42` | same branch: `experiments/EXP-20261005-003-relabel-rag-final/raw/relabel.jsonl` | `1dca8e3afd4ca3cdb21225afc58bb111a007cac7e58db2ebaca8d3da7086f2d7` |
| `relabel_rag_on_seed43` | Colab run `EXP-COLAB-relabel-rag-seed43`, `raw/relabel.jsonl` | `32abc777fa8a96c6275a761e6895dc2e9c1eaf5e13200ef22fd7f6fe0f9c6ac5` |
| `relabel_rag_on_seed44` | Colab run `EXP-COLAB-relabel-rag-seed44`, `raw/relabel.jsonl` | `12981eb4075116e713eb0887296fff02bdde6a81a3ba20cca969c0bba3ab942e` |
| `personarag_seed42` | Colab run `EXP-COLAB-personarag-newdata-seed42`, `raw/llama3.1.jsonl` | `381fa68c5d04a7576eeb887e869d2687260bb99f0474102802514a232afe013f` |
| `base_llama31` | Kevin's branch: `experiments/EXP-20261001-010-repaired-and-baselines/raw/llama3.1.jsonl` | `fa460a01e5bd92694b6fccbefe2d00d4d669fa60ecb2f1c3a1536c6dd75c57c9` |

Reference rates come from 400 real posts per medium (blog, LinkedIn, X) sampled with seed 0 from `data/*.jsonl`. The same posts give the vocabulary.

LanguageTool 6.8 (Java 17, `language_tool_python`), language en-US. `coverage_audit.json` comes from `audit_languagetool.py`.
Hand labels of every flag: `../../grammar_lt_flag_labels.csv`.

## Rebuild

```
python -m eval_tools.grammar_lt --traces <full draft file> --data-dir data --out-jsonl <run>.grammar_lt.jsonl --out-json <run>_summary.json
python docs/evaluation/examples/data/grammar_lt/audit_languagetool.py
```
