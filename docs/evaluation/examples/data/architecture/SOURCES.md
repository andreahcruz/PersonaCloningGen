# Provenance of the architecture comparison data

Everything here is a saved result. No draft was generated and no judge was run to make these copies. The three Colab files are
new runs (see `docs/evaluation/colab/PUK28_comparison_runs.ipynb`). The three Kevin files are filtered copies of results he saved on
branch `kevin/298b-integration` (commit `0fac41485ec3655ea1439d6d2484115f243fdab1`), so the comparison can be rerun without that branch.
Rows are unedited; only the rows of one run are kept.

| File here | Where it comes from | Rows | SHA-256 of this file |
|---|---|---|---|
| `colab_relabel_rag_seed43_rows.jsonl` | Colab run `EXP-COLAB-relabel-rag-seed43`, `metrics/rows.jsonl` (Relabel adapter + RAG, seed 43) | 120 | `e574e0e64fcf473dd9a44753a8bf96e32da6e435cfdfc587a49958a11f2cf27b` |
| `colab_relabel_rag_seed44_rows.jsonl` | Colab run `EXP-COLAB-relabel-rag-seed44`, `metrics/rows.jsonl` (seed 44) | 120 | `c3d5a7801a239121a676e6f6c33f357912a39c4b1aea0b0a66b16e321e585716` |
| `colab_personarag_seed42_rows.jsonl` | Colab run `EXP-COLAB-personarag-newdata-seed42`, `metrics/rows.jsonl` (base Llama 3.1 + train-only retrieval, seed 42) | 120 | `3abe988ffb77c6102519022c154a839a72fedb10961c1056ee56874fb564b144` |
| `colab_*_summary.json` (3 files) | `metrics/summary.json` of each Colab run: gates, calibration, means | - | see the files |
| `colab_environment.json` | Written by the last notebook cell: commit, GPU, Python, `pip freeze`, the scoring patch note | - | `976e7b7015335e27ebbaa27d91e3911aa0256544f154be06b21f45c9f612bf7c` |
| `kevin_relabel_rag_off_seed43_rows.jsonl` | `experiments/EXP-20261004-008-writing-evidence/metrics/rows.jsonl`, rows of source experiment `EXP-20261004-005-repaired-vs-relabel-seed43`, model `relabel` | 120 | `30bf88dbe7a7a72044f9327eb02958e77c1680764cbda98ee452fca451ba1093` |
| `kevin_relabel_rag_off_seed44_rows.jsonl` | same file, source experiment `EXP-20261004-006-repaired-vs-relabel-seed44`, model `relabel` | 120 | `e65b97f610f13e6950b95420f981790274b11ff35b939a5fed2dd003096f1984` |
| `kevin_base_llama31_rows.jsonl` | same file, source experiment `EXP-20261001-010-repaired-and-baselines`, model `llama3.1` (base, no RAG) | 120 | `1be383ed7ad6e9503da005ffadf6aef35c1a3a7c2693aafa93847cba3bbb6e6f` |

Seed 42 of Relabel (RAG off and RAG on) uses the copies already in `docs/evaluation/examples/data/`
(`relabel_rag_off_rows.jsonl`, `relabel_rag_on_rows.jsonl`); their provenance is in `../SOURCES.md`.

## The raw drafts (not in the repository)

The drafts themselves (4 MB per Relabel run) are on the team Google Drive in `298 B/puk28_results/` and on Vrushabh's laptop. Hashes of the
files the length check read, so they can be matched:

| File | SHA-256 |
|---|---|
| `EXP-COLAB-relabel-rag-seed43/raw/relabel.jsonl` | `32abc777fa8a96c6275a761e6895dc2e9c1eaf5e13200ef22fd7f6fe0f9c6ac5` |
| `EXP-COLAB-relabel-rag-seed44/raw/relabel.jsonl` | `12981eb4075116e713eb0887296fff02bdde6a81a3ba20cca969c0bba3ab942e` |
| `EXP-COLAB-personarag-newdata-seed42/raw/llama3.1.jsonl` | `381fa68c5d04a7576eeb887e869d2687260bb99f0474102802514a232afe013f` |

## Facts about the Colab runs

- Code: commit `89fe55e` of `andreahcruz/PersonaCloningGen` (Kevin's final v1 pipeline). The branch tip is not used because it defaults to RAG v2.
- Machine: NVIDIA A100-SXM4-40GB, Python 3.13.16, torch 2.10.0+cu128, unsloth 2026.5.2, peft 0.19.1, chromadb 1.5.5, Ollama 0.40.11.
  Kevin's runs: RTX 5070 Ti, Python 3.11.
- Index: `lemkin_train_only`, 21,193 documents, top-4, `nomic-embed-text`. Judge: `qwen2.5:14b`.
- Kevin's scorer refuses any traces file except his seed-42 one (a hard-coded SHA check). In the Colab copy only, that check was turned off.
- Retrieved passages equal Kevin's seed-42 run for 29 of 30 topics; the extracted evidence differs on 15 of 30 topics (hardware and library differences). Seeds 43 and 44 have identical evidence to each other.

## Rebuild

```
python docs/evaluation/examples/data/architecture/rebuild_reports.py
python docs/evaluation/figures/make_figures.py
```
