# Teammate handoff — Model F

Read this before the longer notes. The GitHub repository is public (`andreahcruz/PersonaCloningGen`). This handoff publishes code, tests, aggregate metrics, and the 120-row judge and audit tables. It does not publish the graph, the adapter, prompts, retrieved passages, or the 120 draft texts.

## Quickstart

From a clone of branch `model-f-handoff` (or `kevin/lightrag-model-f` plus this commit):

```powershell
py -3.11 -m venv host_finetune\.venv
host_finetune\.venv\Scripts\activate
python -m pip install --upgrade pip
pip install "torch>=2.10.0,<2.11.0" "torchvision>=0.25.0,<0.26.0" --index-url https://download.pytorch.org/whl/cu128
pip install -r host_finetune\requirements.txt
pip install -r host_finetune\requirements-lightrag.txt
ollama pull nomic-embed-text
ollama pull llama3.1
ollama pull qwen2.5:14b
```

`qwen2.5:14b` is the judge only. Generation is the Relabel adapter on `unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit`, requested as 4-bit, context 2048. Graph queries use Ollama `num_ctx` 16384. The requirements file documents an RTX 5070 Ti (16 GB) as the development GPU. The official manifest does not store a GPU name.

CPU checks that do not load weights:

```powershell
python -m host_finetune.model_f_experiment -h
python -m pytest tests/test_model_f_experiment.py tests/test_model_f_unified.py tests/test_model_f_graph_context.py tests/test_model_f_pilot.py tests/test_model_f_desc_vs_rel.py tests/test_evidence_capsule.py -q
```

Example paths, not loaded automatically, are in [`config/model_f.example.json`](../config/model_f.example.json).

## Files to review first

1. [`docs/model_f_architecture.md`](model_f_architecture.md) — flow, graph counts, F-REL versus F-SOURCE and F-DESC.
2. [`docs/model_f_experiments.md`](model_f_experiments.md) — why the pilot froze F-SOURCE and the panel still ran F-REL.
3. [`docs/model_f_results.md`](model_f_results.md) — judge table beside the 52/120 unsupported-number count.
4. [`results/model-f-handoff-v1/`](../results/model-f-handoff-v1/) — the JSON behind those pages.
5. `host_finetune/model_f_experiment/` — CLI, arms, unified panel.
6. `host_finetune/model_f_context/` — what each arm is allowed to render.
7. [`docs/handoff/MODEL_F_DEPLOYMENT_TRANSFER_20261102.md`](handoff/MODEL_F_DEPLOYMENT_TRANSFER_20261102.md) — zip hashes for the snapshot and adapter. That note still contains a local Windows path for the transfer folder. The hashes are the part to verify.

## Required external artifacts

Get these from the person who holds `DATA298_model_f_transfer`. Do not rebuild the graph to "match" the paper.

| Artifact | SHA-256 recorded on 2026-11-02 | Bytes |
|---|---|---:|
| `lightrag_storage_expand_v1_snapshot.zip` | `4e7a00a977efe0c4ab121b0356704ecd4472cb17d0a96533f8c80a5ad500b9ea` | 481,019,443 |
| `relabel_qlora_adapter.zip` | `644041cbe47e0a1ce69997e790be799b5ed42862be2854b792e94a24dd93777f` | 158,836,161 |
| Adapter weights inside that zip | `850e9aa88d04fa4e7c9490d5d974715c8a1018c47c7ada340d642b4b2afea530` | 167,832,240 |

Extract the snapshot outside the repo, then:

```powershell
python -m host_finetune.model_f_experiment verify --snapshot PATH\TO\storage_expand_v1_snapshot
```

`verify` refuses a directory whose path contains the live store name `storage_expand_v1`. Copy the verified extract to a runtime directory before any query. LightRAG can rewrite the working directory it opens.

Also needed, and not in the zips: the 4-bit Llama base (Hugging Face `unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit`), Ollama `nomic-embed-text`, and Ollama `llama3.1:latest`.

## Row-level scores on GitHub

[`results/model-f-panel-rows-v1/`](../results/model-f-panel-rows-v1/) has the 120 Qwen rows and the 120 deterministic audit flags.

| File | What it is |
|---|---|
| `judge_scores.csv` | One judge row per `question_id` and `medium` |
| `deterministic_flags.csv` | Unsupported numbers, source copy, topic echo, mid-sentence stop, token-cap stop |
| `manifest.json` | Generation hash, gold hash, export hashes, private-zip hash |
| `README.md` | Column definitions and the judge-versus-number-check limit |

Join on `question_id` and `medium`. Empty cells are missing values.

The 120 response texts are not in Git. Seven drafts fail the saved 12-word source-copy check, ten share 8 or more words with a retrieved passage, and two contain a URL. This repository is public, and publication permission for those responses was not established. The private archive `model-f-frel-120-evaluation.zip` keeps all 120 responses unchanged. It also keeps the judge CSV, the flag CSV, `unified_eval/rows.jsonl`, and the audit `cells.jsonl`. It does not contain prompts, retrieved passages, the LightRAG snapshot, or the LoRA adapter.

Ask the person who built the panel for that zip. Check SHA-256 `324d44f941b93396f62147095e12bb959e34ff726978c608936d25decd9b4813` and size 101,769 bytes. Receipt of this zip, and receipt of the earlier snapshot and adapter zips, has not been confirmed from a teammate machine.

## Inspect results without a GPU

The public package is enough to read the official scores:

- `results/model-f-handoff-v1/aggregate_summary.json`
- `results/model-f-handoff-v1/architecture_comparison.json`
- `results/model-f-handoff-v1/per_medium.json`
- `results/model-f-handoff-v1/deterministic_audit_summary.json`
- `results/model-f-handoff-v1/provenance.json`
- `results/model-f-handoff-v1/excluded_artifacts.json`

If you have the local experiment directory, do not edit it. Regenerate the package with:

```powershell
python scripts/export_model_f_shareable.py
```

The local audit narrative, including which cells failed and why, is `audits/AUDIT-20261102-001-model-f-frel-prejudge/REPORT.md`. That report quotes draft spans and one tracking URL. It is gitignored in part (`sample_answers.json`, `cells.jsonl`, `questions.json`). Confirm with the team before putting the report on the public remote.

`generation_manifest.json` still says `generated_not_judged`. Trust `unified_eval/evaluation_run.json` for the judge.

## Responsibilities still open

- Transfer the snapshot zip and adapter zip on a private channel. GitHub is the wrong place for either file.
- Decide whether F-REL remains the engineering profile after the official panel. The panel was not allowed to change the frozen F-SOURCE pilot, and its deterministic audit does not support calling F-REL factually reliable.
- Persona utility is null for Model F. Someone has to decide whether that metric should be computed before any architecture ranking that uses it.
- The training-copy denominator is 104 in the audit `summary.json` and 120 in the audit report and in `comparison.json`. Reconcile that file before citing a single fraction.
- The graph covers 2,713 of 13,140 eligible documents. Finishing the index is a new experiment, not a resume that preserves these hashes.
- Deployment integration with `andreah/deploy-with-react` was not done. The transfer note lists six conflicting paths. Do not copy this branch's `backend/main.py` over the React service.
- Historical LoRA weight files are already in Git history (`host_finetune/output/lemkin_lora_r32` and `lemkin_lora_repaired`). This handoff does not rewrite history to remove them. Do not add `lemkin_lora_relabel`.
