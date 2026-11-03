# System architecture, deployment, and demo

## What is in this checkout

Several generations of the system coexist. They are not one deployed application.

| Path | What it does | Model F |
|---|---|---|
| Compose pipeline in `docs/architecture/ARCHITECTURE.md` | Airflow, MinIO, Spark, Chroma `lemkin_content`, Ollama, Streamlit | Not this path. `generate.py` and Streamlit were not repointed (`LIGHTRAG_MODEL_F.md`) |
| `backend/main.py` | FastAPI `GET /health`, `GET /formats`, `POST /generate`. Calls `generate.py`, persona profile, and `data/index` | No LightRAG import |
| `host_finetune/compare_adapters.py` | Relabel generation, optional dense retrieval, optional LightRAG | Official generation entry for Model E and Model F |
| `host_finetune/model_f_experiment/` | Plan, run, score, and judge the F-REL panel | Measured path. Writes a new experiment directory |
| `host_finetune/lightrag_backend.py` | LightRAG query wrapper | Used by the experiment runner. CLI default workdir is the old small store, not the frozen snapshot (`MODEL_F_DEPLOYMENT_TRANSFER_20261102.md`) |
| React demo | Documented on branch `andreah/deploy-with-react` | Not present in this tree. Transfer note says that backend has no LightRAG route |

External artifacts required for a Model F regeneration, and not in Git (`docs/teammate_handoff.md`):

| Artifact | SHA-256 recorded 2026-11-02 | Bytes |
|---|---|---:|
| `lightrag_storage_expand_v1_snapshot.zip` | `4e7a00a977efe0c4ab121b0356704ecd4472cb17d0a96533f8c80a5ad500b9ea` | 481,019,443 |
| `relabel_qlora_adapter.zip` | `644041cbe47e0a1ce69997e790be799b5ed42862be2854b792e94a24dd93777f` | 158,836,161 |

Also required: the 4-bit Llama base from Hugging Face, Ollama `nomic-embed-text`, Ollama `llama3.1:latest` for graph operations, and `qwen2.5:14b` only if rescoring. Python 3.11.9 was the judge process. Development GPU named in the requirements file: RTX 5070 Ti, 16 GB. Not stored on the official manifest.

`verify` refuses a path that contains the live store name `storage_expand_v1`, because LightRAG can rewrite the directory it opens. Queries belong on a runtime copy.

## Diagrams

| Diagram | File |
|---|---|
| Repository paths | `figures/25_system_architecture.png` |
| F-REL inference sequence | `figures/26_inference_sequence.png` |
| Deployment dependencies | `figures/27_deployment_dependencies.png` |
| Graph construction | `figures/21_graph_construction.png` |
| Retrieval and generation | `figures/22_retrieval_generation.png` |

Component view, from code rather than from a new drawing:

```text
Gold topic
  -> LightRAG hybrid on a runtime copy of the snapshot
  -> claim extraction, adapter off
  -> F-REL relationship filter
  -> 2048-token window
  -> Relabel QLoRA, adapter on
  -> four media
  -> deterministic audit
  -> optional Qwen 14B unified_eval.v1
```

The dense Model E path replaces the first step with Chroma `lemkin_train_only`, top 4, and does not render F-REL lines.

The older API path replaces all of that with `generate.py`: embed the request, retrieve from `data/index`, build a prompt with the persona profile, and call an Ollama generator (`backend/main.py`).

## Integration gap

`docs/teammate_handoff.md` says deployment integration with `andreah/deploy-with-react` was not done. The transfer note lists merge conflicts in `.dockerignore`, `.gitignore`, `README.md`, `backend/Dockerfile`, `backend/main.py`, and `docker-compose.yml`. It says the deploy service uses an Ollama GGUF tag and temperature 0, while Model F uses the PEFT adapter and temperature 0.7. Pointing `QLORA_MODEL` at a GGUF tag does not serve Model F.

`docs/VRUSHABH_CONTRIBUTIONS.md` records a Next.js frontend on branch `vb-puk-5`. `docs/integration/RECENT_BRANCH_AUDIT_20261030.md` says that pull request was closed and not merged, and that later demo work is the React line. This checkout has no `web/` or `frontend/` tree. `tests/test_ui.py` is skipped when Streamlit is not installed (`docs/integration/PUK28_PUK34_INTEGRATION_20261030.md`).

## What can be demonstrated from this checkout

| Demo action | Can it be shown now | Evidence |
|---|---|---|
| Explain the five scored architectures and the metric tradeoff | Yes, from the handoff JSON and these figures | `results/model-f-handoff-v1/` |
| Show aggregate Model F gates and the 52/120 number audit | Yes | `docs/model_f_results.md` |
| Show a live Model F draft from the GPU | Only on a machine that has the snapshot, adapter, CUDA, and Ollama. This audit did not run it | `README.md` inference section |
| Show the 120 draft texts | Only from the private zip. Do not put them in a public slide | `results/model-f-panel-rows-v1/README.md` |
| Select a model in a web UI | Not in this checkout | `backend/main.py` has format, topic, audience, goal, CTA, and k. No architecture switch |
| Show LightRAG inside the React app | Not implemented on this branch | transfer note |
| Show evaluation charts | Yes, the figures in this directory | `figures/` |
| Record a backup video of a live UI | The UI to record is on the deploy branch, which is not here | branch audit |

## Demo checklist

1. Confirm the machine has the snapshot copy, the adapter, Ollama models, and the 4-bit base before promising a live generation.
2. If those files are absent, demonstrate from saved scores and figures, and say generation is not running in the room.
3. If generation is available, use a new experiment directory. Do not write into `EXP-20261102-001`.
4. State the profile as F-REL and state that the frozen pilot selected F-SOURCE.
5. Show one topic producing four media, or show the saved per-medium table if live generation is too slow. Mean recorded generation time is about 20 seconds per draft before retrieval time, which was not stored. A four-medium topic can exceed a minute of generation time alone.
6. Put the grounding number beside the 52/120 unsupported-number count on the same slide.
7. Do not describe the React screens as part of this branch. If a teammate runs the deploy branch, label which model that screen actually calls.
8. Capture GUI screenshots on the branch that owns the GUI. This audit found none in `docs/final_report_support/`.
9. Keep prompts, retrieved passages, and failed copy spans out of public slides. The private zip exists for the team. Its SHA-256 is `324d44f941b93396f62147095e12bb959e34ff726978c608936d25decd9b4813` (101,769 bytes).
