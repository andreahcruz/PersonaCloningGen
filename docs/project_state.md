# DATA 298B project state

Status labels are defined in `AGENTS.md`: PROPOSED, IMPLEMENTED, TESTED, MEASURED, VALIDATED.
Files outside a registered `experiments/EXP-*` run are historical evidence until reproduced.

## Canonical branch

- Branch: `kevin/298b-integration`
- Last verified runtime baseline: `298b-baseline-pre-rerun`
- Baseline commit: `6a01d45adea309d11a11849ed454e98e6e9e0ed1`
- Documentation and workflow commits may advance the branch beyond this baseline without changing runtime behavior. Use `git rev-parse HEAD` to identify the current repository commit.

## Baseline tag

- Tag: `298b-baseline-pre-rerun`
- Tagged commit: `6a01d45adea309d11a11849ed454e98e6e9e0ed1`
- HEAD is one commit after that tag. That commit does not change runtime code.

## Current integration lineage

1. `0990c938a43eeb03eb937dbe3bfbcbd5bada4f8b` — branch created from the existing QLoRA fine-tune commit (`FT`).
2. `d71ce485c2965b616b8c2a7490d3d6d4b841155c` — cherry-pick `Add gold eval metrics tooling and results`.
3. `86dd5c62cec0c0651be7cd41821ba12083c8a9e2` — `PUK-17: Add error handling and logging for pipeline, retrieval, model and deployment failures` (brought in by fast-forward of `kevin/puk17-eval`).
4. `6a01d45adea309d11a11849ed454e98e6e9e0ed1` — `PUK-17: fix false "malformed rows" warning in mark_training_ready`. This is `298b-baseline-pre-rerun`.
5. `eaae79400cc21c00ccc67d8d9a75ace804e29469` — gitignore-only commit, current HEAD.

## Known model implementations

Implemented in the current tree:

- RAG generation in `generate.py` and `user_interface.py`. Query embeddings use Ollama `nomic-embed-text`. The default generation model is Ollama `llama3.1`. The UI can also select a registered fine-tune such as `lemkin-clone`.
- Host QLoRA training in `host_finetune/finetune.py`, configured by `host_finetune/config.py`. Default base model: `unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit`. Default LoRA settings in code: `LORA_R=16`, `LORA_ALPHA=32`, `LORA_DROPOUT=0`, `NUM_EPOCHS=1`, `PER_DEVICE_BATCH=2`, `GRAD_ACCUM=8`, `LEARNING_RATE=1e-4`, `MAX_SEQ_LENGTH=512`. The committed code is the authority for these defaults. The README table still says `NUM_EPOCHS` defaults to 3.
- Merge and GGUF export in `host_finetune/merge_and_export.py`. Default quant is `q4_k_m`. The merge base is the non-bnb model name derived from `HF_MODEL_NAME`.
- Ollama registration in `host_finetune/register_ollama.py`. Default served name: `lemkin-clone`.
- A separate evaluator lives at `persona_pipeline/evaluation/evaluate.py` (BERTScore, RAGAS, TF-IDF authorship classifier, BLEU/ROUGE-L). It is not the gold-eval harness.

There is no canonical DATA 298B Model 1–5 numbering in this tree. See known technical risks. No model weights or metric values on this baseline are marked MEASURED or VALIDATED.

## Data pipeline

Airflow DAG `lemkin_content_pipeline` in `dags/lemkin_pipeline_dag.py`:

1. `extract_to_minio` reads local JSONL from `./data`: `jasonlemkin_blog.jsonl`, `jasonlemkinlinkedin.jsonl`, `jasonmlemkinyoutubetranscripts.jsonl`, `saastryoutubetranscripts.jsonl`, `jasonlk_originals.jsonl`. Normalized objects go to MinIO bucket `lemkin-raw` under `raw/`.
2. `trigger_spark_clean` runs `spark_jobs/clean_and_embed.py`: HTML cleanup, chunking, `nomic-embed-text` embeddings, chunk JSON under `lemkin-processed/chunks/`, and SFT JSONL under `lemkin-processed/training/dataset_jsonl/`.
3. `load_to_chroma` upserts embedded chunks into Chroma collection `lemkin_content`.
4. `mark_training_ready` concatenates Spark part files to `training/dataset.jsonl` and writes `training/_READY` when at least one valid `instruction`/`output` row is present. Row counting splits on `\n` only.
5. `extract_persona` builds `persona_profile.json` from every paged `raw/` object and writes it to MinIO and the host data directory.

`host_finetune/watcher.py` polls the sentinel, downloads the dataset, runs `dataset_prepare`, then `finetune`, `merge_and_export`, and `register_ollama`.

## Infrastructure

`docker-compose.yml` defines Postgres, MinIO plus bucket setup, Airflow webserver/scheduler/init, Spark master and worker, Chroma, and Streamlit. Published endpoints documented in `README.md`: Airflow `localhost:8080`, Streamlit `localhost:8501`, MinIO console `localhost:9001`, Spark UI `localhost:8081`, Chroma `localhost:8000`. Spark and RAG call host Ollama at `host.docker.internal:11434`.

`docker compose config -q` succeeded. The only reported warning was the obsolete Compose `version` key (`version: '3.8'` at the top of `docker-compose.yml`). That check parses the compose file. It does not start the stack.

Streamlit image dependencies are `requirements-streamlit.txt` (`chromadb==1.5.5`). The root `requirements.txt` still pins `chromadb==0.4.22` and Airflow 2.8.0.

## Evaluation tooling

Implemented, not re-run on this baseline:

- `host_finetune/eval_rag.py` — RAG traces, G-Eval via an Ollama judge, BERTScore, BLEU, ROUGE-L, TF-IDF style classifier, optional RAGAS. Default outputs: `host_finetune/output/eval/traces.jsonl`, `geval.jsonl`, `summary.json`.
- `host_finetune/generate_gold_eval_outputs.py` — writes gold-eval traces.
- `host_finetune/eval_gold_rag.py` — offline format/brief/fact rubric and retrieval-grounding scores. Default summary path: `host_finetune/output/eval/gold_rag_summary.json`. Optional gold input includes `data/openai_ft/lemkin_gold_eval.jsonl`.
- `generate.py` `validate_draft` and `spark_jobs/output_checks.py` `looks_degenerate` reject empty or collapsed UI/CLI output. `eval_rag.py` calls `ollama_generate` directly and does not use `validate_draft`.

## Testing status

TESTED on the lightweight `.venv-test` environment:

- Unit tests: **67 passed, 4 skipped**.
- The skips are the optional-dependency gates in that environment: PySpark (`tests/test_spark_job.py` imports `pyspark` and `bs4`), Streamlit (`tests/test_ui.py` imports `streamlit.testing.v1`), and NLTK (`tests/test_dag.py` calls `pytest.importorskip("nltk")` in the persona tests).
- `docker compose config -q`: succeeded, with only the obsolete Compose version warning.

These tests mock Chroma, MinIO, Ollama, and Airflow. They do not measure model quality. No fine-tune, retrieval rebuild, or gold-eval rerun is marked MEASURED on `298b-baseline-pre-rerun`.

## Known technical risks

- `load_to_chroma` still splits chunk files with `str.splitlines()` (`dags/lemkin_pipeline_dag.py`). U+2028 inside a JSON string can be treated as a line break. `mark_training_ready` already avoids that by splitting on `\n` only.
- `extract_persona` now pages the entire `raw/` prefix. Rebuilding `persona_profile.json` can change later RAG prompts and evaluation relative to historical results.
- `load_to_chroma` now pages every chunk key and refuses to load when the missing-embedding fraction exceeds `MAX_MISSING_EMBEDDING_FRACTION` (default 0.10). Rebuilding Chroma can change retrieval relative to historical gold-eval artifacts.
- `register_ollama.py` raises when the post-create smoke answer is empty or degenerate. `host_finetune/watcher.py` treats that failure as a pipeline error and retries the whole fine-tune workflow, with backoff up to one hour.
- Committed `NUM_EPOCHS` default is 1. The README fine-tune table still says 3.
- Dependency and version divergence: `requirements-streamlit.txt` pins `chromadb==1.5.5`, while root `requirements.txt` pins `chromadb==0.4.22`.
- The repository does not yet have a canonical DATA 298B model-numbering scheme. Historical documents refer to Models 1–5. Current executable implementations are the RAG path (`generate.py` and Streamlit), QLoRA training, export, and registration (`host_finetune/`), and the separate evaluator (`persona_pipeline/evaluation/evaluate.py`). Those implementations have not been mapped to Models 1–5. A historical workbook or presentation label is not evidence that the current tree can run that model.

## Unverified historical results

These files are on disk. Their numbers are historical evidence, not current results:

- `host_finetune/output/eval/gold_rag_summary.json`
- `host_finetune/output/eval/gold_rag_summary_with_gold.json`
- `host_finetune/output/eval/gold_eval_baseline_scores_30.json`
- `host_finetune/output/eval/gold_eval_baseline_traces_30.jsonl`
- `host_finetune/output/eval/traces.jsonl`
- `host_finetune/output/eval/geval.jsonl`
- `host_finetune/output/eval/summary.json`
- `docs/DATA298_PRESENTATION_NOTEBOOK.ipynb`
- Local adapter and GGUF directories under `host_finetune/output/` (`lemkin_lora`, `lemkin_lora_stash`, `lemkin-clone`, `lemkin-clone_gguf`). The large checkpoint paths are gitignored.

`data/openai_ft/lemkin_gold_eval.jsonl` is a gold-input file used by the offline scorer. It has not been re-scored on this baseline.

## Next engineering tasks

PROPOSED, not started:

1. Reproduce the gold evaluation from this baseline and store it as a registered `experiments/EXP-*` run. Do not cite the historical JSON as the result of that run.
2. Decide whether a persona rebuild and a Chroma rebuild are in scope before any new RAG measurement. Record dataset, persona, and index identities in the experiment manifest.
3. Review the `load_to_chroma` `splitlines()` path before relying on a rebuilt index.
4. Review watcher behavior when an Ollama registration smoke check fails, before an unattended fine-tune.
5. Map executable implementations to the historical Model 1–5 labels before a workbook treats those names as current models.
6. Map completed work to the assignment rubric when `docs/rubric_matrix.md` is in scope. That file is intentionally untouched.
