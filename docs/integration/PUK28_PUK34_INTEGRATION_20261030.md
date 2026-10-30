# PUK-28 and PUK-34 integration — 2026-10-30

Local integration of `origin/vb-puk-28` and `origin/vb-puk-34` into `kevin/lightrag-model-f`. Nothing was pushed. `rag_active.json` was not modified. Frozen experiment outputs were not rewritten. Deployment and frontend branches were not merged.

The prior review is [RECENT_BRANCH_AUDIT_20261030.md](RECENT_BRANCH_AUDIT_20261030.md).

## 1. Final local commits

Branch: `kevin/lightrag-model-f`

| Role | SHA | Parents |
| --- | --- | --- |
| Starting HEAD | `0fac41485ec3655ea1439d6d2484115f243fdab1` | already on `origin/kevin/298b-integration` |
| Model F record | `2031704cf604c1699acb440c1cc7320c95f48513` | 121 files, +119,422 / −39 |
| PUK-28 merge (`--no-ff`) | `b06560b9965a833f8d3a39dcf94afaab8149c0b6` | `2031704` + `eb790118e7f4bcb58881289e169da865f079600f` |
| PUK-34 merge (`--no-ff`) | `94282a1eceb18cc90b4d74967e7243cf64fbf731` | `b06560b` + `97444828725c50e479ea2c81e0dfe98316e9c928` |
| This record and the audit | committed directly on `94282a1` | no code or experiment files |

The code merges end at `94282a1eceb18cc90b4d74967e7243cf64fbf731`. This file does not name its own commit.

The Model F commit keeps the LightRAG retrieval path, its tests, and the registered experiment records through EXP-20261030-011. Graph storage, model weights, GGUF files, Chroma copies, and the multi-gigabyte `deployment_handoff/` tree stayed untracked. That handoff directory is still untracked after these merges.

## 2. Merged branch SHAs

- `origin/vb-puk-28` tip: `eb790118e7f4bcb58881289e169da865f079600f` (LanguageTool grammar and fixed-length voice).
- `origin/vb-puk-34` tip: `97444828725c50e479ea2c81e0dfe98316e9c928` (RAG cost report and the medium-filter retrieval check).

PUK-28 also brought in the shared eval-tool history and the older QLoRAFT FastAPI commits (`7b2caeb`, `efde946`, `3b6eb08`, `6050284`). Those commits are ancestors of both PUK branches. `origin/QLoRAFT` was not merged a second time.

## 3. Conflicts and resolutions

PUK-28 merged cleanly onto the Model F commit.

PUK-34 conflicted only in the three files the audit predicted. Both sides were kept.

- `docs/evaluation/README.md`. The landing page now lists the grammar and fixed-length voice writeups next to the RAG-cost and medium-filter writeups, and it lists `grammar_lt.py`, `voice_fixed.py`, and `rag_effect.py` together. A short note under the diagram says grammar flags are not an eligibility gate, factual search stays unfiltered, and `--fixed-style` is unchanged.
- `eval_tools/README.md`. Section 5b (fixed-length voice) is followed by section 6 (`rag_effect`) and one tests section. The test count is the combined 141 cases: 76 eval tools, 26 LanguageTool, 13 fixed-length voice, 20 `rag_effect`, and 6 medium-filter summary tests. The older "115" and "96" counts described one branch each.
- `docs/evaluation/figures/make_figures.py`. `voice_fixed_figure`, `passage_fit_figure`, and `retrieval_filter_figure` are each complete, and `main()` still draws fig4 and fig6 only when the PUK-34 result files exist and fig5 only when the voice summaries exist. The blog chart title now says the comparison is blog openings and that Relabel is near real posts. The previous title said Relabel matches real posts on voice, which the LinkedIn prefixes do not support.

## 4. Changed files

Against the Model F commit, the PUK-28 merge added 162 files (+36,338). That includes `eval_tools/` (compare, grammar, length control, `grammar_lt`, `voice_fixed`), `tests/test_eval_tools.py`, `tests/test_grammar_lt.py`, `tests/test_voice_fixed.py`, the evaluation writeups and examples, figures 1, 2, 3, and 5, and the inherited `backend/` service (`Dockerfile`, `main.py`, `requirements.txt`) plus `docs/VRUSHABH_CONTRIBUTIONS.md`.

The PUK-34 merge then changed 32 files (+6,402 / −10). New code is `eval_tools/rag_effect.py`, `tests/test_rag_effect.py`, and `tests/test_retrieval_by_medium.py`. New evidence is `docs/evaluation/RAG_BEFORE_AFTER.md`, `docs/evaluation/TYPE_SPECIFIC_RETRIEVAL_CHECK.md`, the slim RAG-effect rows, `examples/rag_effect/`, `examples/data/retrieval_by_medium/`, and figures 4 and 6. The three conflicted docs were the only modifications.

`git diff 2031704 HEAD -- experiments` is empty. `git diff 0fac414 HEAD -- rag_active.json` is empty. `host_finetune/compare_adapters.py` still calls `query_excerpts(..., where=None)` for factual retrieval, and `--fixed-style` still loads style exemplars only when that flag is set.

## 5. Tests executed and outcomes

Interpreter: `host_finetune\.venv` (Python 3.11.9, pytest 9.1.1). `PYTHONPATH` was the repository root. No generation, no LightRAG indexing, and no writes to a Chroma store.

Imports of `eval_tools` (`grammar_lt`, `voice_fixed`, `rag_effect`, `compare`) and `host_finetune` (`lightrag_backend`, `factual_reliability`, `unified_eval`, `compare_adapters`) succeeded in that environment.

Focused set, 1.63s:

- `tests/test_grammar_lt.py` — 26 passed
- `tests/test_voice_fixed.py` — 13 passed
- `tests/test_eval_tools.py` — 76 passed
- `tests/test_rag_effect.py` — 20 passed
- `tests/test_retrieval_by_medium.py` — 6 passed
- Model E / Model F and reliability: `test_evidence_capsule.py` (19), `test_lightrag_retrieval.py` (14, including the installed LightRAG 1.5.7 mode check), `test_graph_ready.py` (12), `test_factual_reliability.py` (5), `test_span_repair.py` (6), `test_quote_and_pilot.py` (14), `test_numeric_grounding.py` (11), `test_unified_eval.py` (12), `test_retrieval_eval.py` (5), `test_rag_source_copying.py` (10), `test_train_only_index.py` (19)

That focused command collected 268 items: 266 passed and 2 skipped. The skips are the whole of `tests/test_spark_job.py` (`pyspark` is not installed) and the whole of `tests/test_ui.py` (`streamlit` is not installed). They are environment skips, not assertion failures.

The broader `tests/` suite then passed **562 tests, with the same 2 skipped**, in 243.57s. Two warnings are SWIG deprecation notices from a native extension. No test was edited to obtain this result.

`language_tool_python` is not installed, so the live LanguageTool server was not started and the ~260 MB download was not made. The grammar tests use a fake tool.

## 6. Dependency changes

No package was installed or uninstalled for this integration.

Inherited with PUK-28, and isolated in `backend/requirements.txt`: `fastapi==0.115.0`, `uvicorn[standard]==0.30.6`, `pydantic==2.9.2`, `chromadb==1.5.5`, `requests==2.31.0`, `pyyaml==6.0.1`. The chromadb pin matches the host fine-tune pin. `fastapi` is not installed in the host virtualenv. `backend/main.py` wraps the root `generate.py` Ollama path. It is not called by `compare_adapters.py` or by the unified-eval gates. CORS is `allow_origins=["*"]`. That was left as inherited and was not executed.

Already recorded in the Model F commit, `host_finetune/requirements-lightrag.txt`: `lightrag-hku==1.5.7`, `ollama>=0.5.4,<1.0.0`, `opentelemetry-api==1.41.1`. The installed package reports 1.5.7, which is what `test_installed_lightrag_modes_match_the_pinned_release` checks.

`language_tool_python` is not pinned in any requirements file and is not installed. Java was already present on this machine; the grammar code calls a local `LanguageTool("en-US")` server when someone runs it, not the public API.

## 7. Model E and Model F compatibility

Default retrieval is still off, and the default backend is still dense Chroma through `rag_active.json`. Factual queries still use `where=None`. LightRAG remains an optional backend (`naive`, `local`, `global`, `hybrid`, `mix`) that returns verbatim chunks into the existing atomic-evidence path. It is not the ontology cell and it is not a new generator.

`rag_effect` reads fields the comparison traces already save (`prompt_tokens`, `numeric_grounding.pass`, `rag_source_copying.flagged`, `extraction_seconds`, hit mediums and distances). LightRAG hits that carry `medium` fit that reader. The report does not change retrieval, prompts, or eligibility.

`unified_eval.py` gates are still completion, copying, format, and grounding. There is no grammar gate.

EXP-20261027-001 through EXP-20261030-011 are unchanged from the Model F commit. EXP-20261030-010 and EXP-20261030-011 do not replace EXP-009. Style distances in EXP-009 are not ranked across arms, because the eligible medium sets differ.

## 8. Updated evaluation capabilities

- Fixed-length voice (`eval_tools.voice_fixed`). First 100 words of blog and LinkedIn drafts, scored with the existing voice distance. Drafts shorter than that are counted and not scored. X and talk are reported as unsupported.
- LanguageTool diagnostics (`eval_tools.grammar_lt`). Optional, filtered grammar flags. They are a diagnostic beside the older rule-based checker. They do not decide whether a draft may be ranked.
- Eval harness already on PUK-28: paired comparisons with intervals, multiple-test correction, seed checks, and length control for the style distance.
- RAG cost report (`eval_tools.rag_effect`). Paired prompt tokens, generated tokens, length, truncation, and early stops, plus retrieval-side extraction seconds (once per topic), unsupported-number flags, copying flags, claim survival, retries, and whether a retrieved passage matches the requested medium. This is not precision, recall, entailment, generation latency, or dollar cost.
- Medium-specific retrieval study. A separate script and `results.json` for what `medium_where` would return on the v1 index. It is not wired into `build_runtime_retriever`.

## 9. Outstanding limitations

**Fixed-length voice, and what it does change.** The score compares the first 100 words of a draft with a profile fit on full held-out validation posts. Truncation is the first N whitespace tokens joined with single spaces, so newlines collapse and paragraph rate becomes a shared offset. It is a surface-style distance, not an authorship test.

On blog prefixes, Relabel sits next to held-out Lemkin prefixes, and base Llama 3.1 does not. Recomputed means that match the saved summaries: real posts 1.1096; Relabel RAG-off seeds 42 / 43 / 44 = 1.0843 / 1.0486 / 1.1273 (30 / 30 / 29 scored); Relabel RAG-on seed 42 = 1.1312 (29 scored); base Llama = 2.0276. The teammate's paired intervals also put those Relabel blog prefixes closer than PersonaRAG. The PersonaRAG blog mean 1.7086 is their Colab figure and was not recomputed in this integration. PersonaRAG has one seed.

LinkedIn prefixes do not separate the models. Saved summaries: real 0.792 (24 of 40 long enough), base 0.710, PersonaRAG 0.680, Relabel RAG-off seed 42 = 0.668. X is unsupported because Relabel X drafts have a median of about 16 words. Talk has no held-out reference.

The length control qualifies an older voice result. The full-length Relabel RAG-on blog penalty of about +0.10, whose interval excluded 0, shrinks to about +0.05 and the interval includes 0 once every draft is cut to the same prefix. The claim that drafts were more typical than real posts (p_closer 0.70, then 0.53 on length-matched text) is a length effect. Prefix means are not comparable to the older full-text means. Seeds 43 and 44 also retrieved different evidence from seed 42 on 15 of 30 topics, so they are not pure generation replicates. None of this replaces the EXP-009 or EXP-010 factual findings, and it does not turn the small Model F generation comparison into a quality win for the graph.

**Grammar.** On the reported annotation of 103 draft flags: 67 false alarms, 19 colloquial, 14 real mistakes, 3 not grammar. One annotator. The older rule-based checker's reported precision is a different tool. Neither number is a gate.

**Retrieval study.** Medium-filtered factual search was measured on v1 and was not adopted. Blog filtering barely moved the best-passage distance. Talk filtering returned poor matches and kept none of the short-passage cuts that mattered. Strategy B, unfiltered facts plus train-owned style exemplars, is the existing `--fixed-style` flag. It is still off unless a run asks for it, and this merge does not measure it.

**Other.** The FastAPI sidecar was not run. Live LanguageTool was not run. `deployment_handoff/` remains local and untracked, including weights and Chroma copies.

## 10. Remaining tasks before the final demo

- One registered generation experiment for `--fixed-style`, if that comparison is part of the demo. Do not change the flag's behavior in order to run it.
- Optional: pin and install `language_tool_python`, then re-run the diagnostic on a sample. Keep the flags out of the gates.
- Deployment UI is a separate merge (`andreah/deploy-with-react`), not this branch. `kevin/direct-qlora` matches that deploy patch and should not be merged as well. `vb`, `vb-puk-5`, and the open vanilla-frontend pull requests stay out.
- Do not merge `origin/master`. It has no merge base with this branch.
- Do not re-index LightRAG, retrain QLoRA, or rewrite EXP-009, EXP-010, or EXP-011 to absorb these metrics.
