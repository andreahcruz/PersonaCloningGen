# Model F deployment handoff

The partial LightRAG index is not in this handoff. Indexing was still running when the branch was published. The finished directory will be delivered separately, outside Git.

## 1. Branch and commit

- Branch: `kevin/lightrag-model-f`
- Remote: `origin` `https://github.com/andreahcruz/298P.git`
- Implementation commit: `IMPLEMENTATION_COMMIT_PENDING`
- Parent before this handoff: `95db4f54a518cb273ea9118b10b9175eeb9668be`
- This file records the implementation commit above. `git rev-parse HEAD` is the pushed tip if a later commit only fills this hash or adds the pull-request link.

## 2. Pull request

PR_LINK_PENDING

Target, if the pull request exists: `andreah/deploy-with-react`. That is the EC2/React deployment branch. This handoff does not merge it. The histories diverged at `ea1d58b30cafcb4cf18a03ded39963f64ce6aee7` (35 commits on this branch, 34 on the deploy branch). An October 30 audit recorded conflicts in `.dockerignore`, `.gitignore`, `README.md`, and `docker-compose.yml`. Resolve those on the deploy branch when merging. Do not merge `kevin/direct-qlora` as well. Its direct-QLoRA patch is already on the deploy line.

`andreah/deploy-test` is an ancestor of that line, not a second target. `docs/deployment/DEPLOYMENT_HANDOFF.md` is the older Relabel + Chroma note. Its branch, `kevin/298b-integration`, is already contained in `kevin/lightrag-model-f`.

## 3. What Model F does

Model F is the existing Relabel generator with optional LightRAG graph context. It is not a second adapter and not a new index format.

`host_finetune/model_f_context/` turns a saved or raw LightRAG payload into three channels:

- Channel A: verbatim source passages.
- Channel B: a relationship only after a source passage supports that edge.
- Channel C: generated descriptions. They stay derived context and are never promoted into Channel A.

The pipeline ranks evidence, applies a per-medium token budget, and renders `model-f-graph-context-v1`. `compare_adapters.generate` accepts `graph_context_for`. The default is off, so Model E dense retrieval and the current passage-only prompt stay as they were.

`--model-f-context` accepts `off`, `source`, `relationships`, `descriptions`, and `multihop`. It requires `--retrieval --retrieval-backend lightrag`. It cannot be combined with `--graph-relationships` or a context pad.

The four-arm runner in `host_finetune/model_f_experiment/` asks once per question, freezes that payload, and renders F-SOURCE, F-REL, F-DESC, and F-HOP. E-DENSE and F-OFF are in the evaluation registry and are not part of that inference run. The runner, snapshot tool, verifier, and graph audit are tested on synthetic fixtures. They have not been run on the expansion graph.

`host_finetune/eval_bench/` scores faithfulness, numeric grounding, citation-marker leak, source copying, latency, and token counts. Precision, recall, MRR, and nDCG stay empty until a person fills grades. There is no composite score.

`relation-normalization-v1` can dry-run a recovery ledger. It refuses to apply repairs to a directory named `storage_expand_v1`.

## 4. Repository setup

From a fresh clone of `kevin/lightrag-model-f`, at the repository root:

```powershell
py -3.11 -m venv host_finetune\.venv
host_finetune\.venv\Scripts\activate
python -m pip install --upgrade pip
pip install "torch>=2.10.0,<2.11.0" "torchvision>=0.25.0,<0.26.0" --index-url https://download.pytorch.org/whl/cu128
pip install -r host_finetune\requirements.txt
pip install -r host_finetune\requirements-lightrag.txt
```

CPU tests do not need the CUDA torch install. Use the existing `host_finetune\.venv` on this machine if it is already present. Do not create a second environment in a way that replaces that venv while indexing is running. Indexing uses `host_finetune\.venv`.

Ollama, only when querying or indexing:

```powershell
ollama pull llama3.1:latest
ollama pull nomic-embed-text
```

The Relabel adapter and the Chroma directories are local artifacts. Paths and the older deployment checklist are in `docs/deployment/DEPLOYMENT_HANDOFF.md` and `docs/deployment/ARTIFACTS.md`. They are not produced by this branch's new commit.

## 5. Dependencies and environment

| Requirement | Use |
|---|---|
| Python 3.11 | Host venv. The recovery manifest recorded 3.11.9. |
| `host_finetune/requirements.txt` | Relabel generation, Unsloth, Chroma, tests that import torch |
| `host_finetune/requirements-lightrag.txt` | `lightrag-hku==1.5.7`, Ollama client, OpenTelemetry pin `1.41.1` |
| Ollama `llama3.1:latest` | LightRAG keyword and extraction calls. Not used by the CPU tests. |
| Ollama `nomic-embed-text` | Query and index embeddings, 768 dimensions. Not used by the CPU tests. |
| CUDA torch and the Relabel adapter | Generation only |
| A LightRAG working directory supplied outside Git | Required before any live Model F query |

No API key is required for the local Ollama path. Do not commit a `.env`. `.env.example` is the placeholder file.

## 6. Tests

Run this from the repository root. It is CPU only.

```powershell
$env:PYTHONDONTWRITEBYTECODE = "1"
host_finetune\.venv\Scripts\python.exe -X utf8 -m pytest tests/test_model_f_graph_context.py tests/test_model_f_experiment.py tests/test_eval_bench.py tests/test_eval_bench_historical.py tests/test_dev_benchmark.py tests/test_graph_relationships.py tests/test_relation_normalization.py tests/test_indexing_checkpoint.py tests/test_indexing_extraction_audit.py tests/test_indexing_quality.py tests/test_indexing_selection.py tests/test_indexing_timeout.py tests/test_evidence_capsule.py tests/test_factual_reliability.py tests/test_numeric_grounding.py tests/test_lightrag_retrieval.py -q --tb=line -p no:cacheprovider
```

Handoff result: **156 passed**, 2 Swig deprecation warnings, 17.21 seconds, on 2026-11-01.

Do not run GPU generation, embedding, or a LightRAG query against `artifacts/lightrag/model_f/storage_expand_v1` while the indexer is running. Do not start a second `indexing_session` on that directory.

## 7. Point the service at an external LightRAG directory

`backend/` does not load LightRAG. The integration point is `compare_adapters`.

```powershell
host_finetune\.venv\Scripts\python.exe -X utf8 -m host_finetune.compare_adapters --adapters relabel=host_finetune/output/lemkin_lora_relabel --retrieval --retrieval-backend lightrag --lightrag-workdir D:\path\to\verified-snapshot --lightrag-mode hybrid --model-f-context off --skip-score --out-dir experiments\EXP-YYYYMMDD-NNN-name
```

`--lightrag-workdir` is the directory to load. Use the verified snapshot, or a later finished copy, not the live indexing directory. `--model-f-context off` keeps the current prompt. Set it to `source`, `relationships`, `descriptions`, or `multihop` only for an evaluation configuration. The default of `--retrieval-backend` is `dense`, so a deploy merge that does not pass `lightrag` keeps Model E.

There is no environment variable for this path. Wire the flag, or the `lightrag_workdir` argument inside `compare_adapters`, to the directory you place on disk.

The four-arm runner takes the same kind of directory:

```powershell
host_finetune\.venv\Scripts\python.exe -X utf8 -m host_finetune.model_f_experiment run --questions host_finetune\benchmarks\dev_retrieval_benchmark.jsonl --out experiments\EXP-20261101-009-model-f-four-arm --snapshot D:\path\to\verified-snapshot
```

`run` refuses a path whose name is `storage_expand_v1`.

## 8. Ready for deployment integration

- Graph-context package, schemas, ranking, budgets, and the default-off generation flag.
- `--lightrag-workdir` for an externally supplied directory.
- Four-arm runner, snapshot, verify, and audit commands.
- Evaluation harness and metric code.
- Relation-normalization code, including the refusal to edit the live store.
- Indexing session code, for later resumes. Do not start it against the directory that is already indexing.
- CPU tests, development questions, and the implementation docs.
- Existing dense Model E path, left as the default.

## 9. Still needs the finished index

- A live hybrid query.
- End-to-end Relabel generation with `--model-f-context` other than `off`.
- The four-arm run `EXP-20261101-009-model-f-four-arm`.
- Snapshot and `verify` of the finished blog graph.
- Entity descriptions on the `query_documents` payload. `lightrag_backend.py` still drops that text from hits. The follow-up is in `docs/implementation/MODEL_F_POST_INDEX_CHECKLIST.md` and waits because the indexer imports that file.
- Human grades before precision, recall, MRR, or nDCG.
- Wiring `backend/` to the LightRAG workdir. That service code is not on this branch's new path yet.

## 10. Limitations

- The blog-only expansion index is incomplete and is not in Git.
- October 31 cells A–D are not F-SOURCE, F-REL, F-DESC, or F-HOP. The historical replay records that mismatch. Do not treat those cells as the new profiles.
- F-DESC reads entity descriptions from a raw payload. `query_documents` does not yet return them as a separate list.
- A two-hop path is not a claim that one source connected the endpoints.
- Inferred, rejected, and unsourced edges stay out of the prompt.
- Relationship recovery was a dry run. Nothing was inserted into the graph.
- Style distance is not a Model F score. The style-ablation raw dumps were not included.
- `deployment_handoff/` remains local and untracked. It is the older Chroma and adapter bundle, not this graph.
- Cleaned source JSONL files, `chroma_db/chroma.sqlite3`, and two historical LoRA `safetensors` files are already in Git history from earlier commits. This handoff does not delete them and does not rewrite history. New copies of those kinds are ignored.

## 11. Replace the partial index later

The partial index is not included. Do not commit it, and do not copy it into the repository.

When indexing has exited, `indexing.lock` is absent, and no document is still `processing`:

1. Snapshot the finished workdir with `python -m host_finetune.model_f_experiment snapshot`. The command refuses a held lock. Keep the copy outside Git. `artifacts/` is ignored.
2. Run `verify` on the copy and keep the catalog hash it prints.
3. Send that directory by disk or object storage, not by Git.
4. Change the service's `--lightrag-workdir` to the new directory.
5. Stop using any earlier partial copy. Do not merge the partial directory into the finished one, and do not query the live directory while a writer still has it open.
6. The same replacement applies when a later completed index supersedes this snapshot: point `--lightrag-workdir` at the new verified directory and leave the old path unused.

Details and the exact flags are in `docs/implementation/MODEL_F_FOUR_ARM_EXPERIMENT.md` and `docs/handoff/MODEL_F_HANDOFF_MANIFEST.md`.
