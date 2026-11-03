# Model inventory and comparison

## Letter names

`docs/evaluation/UNIFIED_120_COMPARISON.md` states that this repository does not label the saved architectures A–E. Letters A–D inside `EXP-20261031-001` are pilot arms on a different question set. In the six-arm registry, Model E is dense Relabel + RAG. Model F is the later LightRAG profile. This document does not assign Base, PersonaRAG, or Relabel RAG off to letters A–D.

`docs/research/model_catalog.md` (2026-10-03) records two incompatible historical schemes: slides used M1 base, M2 prompt, M3 PersonaRAG, M4 QLoRA, and a proposed M5 LightRAG; another note used M0–M6. Those schemes were not approved as canonical. The comparison below uses the names in `architecture_comparison.json`.

## Five scored systems, plus the proposals around them

The workbook criterion is five or more proposals, with at least one improved model (`docs/PROJECT_CONTEXT.md`). The repository contains more proposals than five completed systems. The five systems that have a 120-draft `unified_eval.v1` row are the ones in the table. Two of them change the method rather than only swapping a prompt: the relabel adapter, and LightRAG relationship conditioning.

### Base

| Item | Evidence |
|---|---|
| Purpose | Unadapted instruct generation on the same gold topics |
| Family | Llama 3.1 8B, no adapter |
| Retrieval | None. Grounding is not applicable |
| Data | Gold topics only; no train-index lookup |
| Decoding | Where a manifest records it: temperature 0.7, top_p 0.9, repetition penalty 1.15, seed base 42 (`UNIFIED_120_COMPARISON.md`) |
| Result | 45/120 eligible, completion 0.375, writing quality 3.578 on 120 drafts, style distance 1.043 on 90 |
| Role | Baseline. Highest task adherence (0.743). Lowest completion |

The protocol calls this row exact on the shared scorer and gold file, one seed. Source rows: `docs/evaluation/examples/data/architecture/kevin_base_llama31_rows.jsonl`.

### PersonaRAG

| Item | Evidence |
|---|---|
| Purpose | Retrieval without the Relabel adapter |
| Family | Base Llama 3.1, no adapter |
| Retrieval | Chroma `lemkin_train_only`, 21,193 documents, top 4, `nomic-embed-text` |
| Prompt | The protocol says this is the v1 script, not the semester-1 `generate.py` pipeline |
| Result | 57/120 eligible, completion 0.592, grounding 0.833, writing quality 3.599, style distance 0.906 |
| Role | Approximate retrieval baseline |

This is the highest all-draft writing quality. The protocol marks the row approximate. Source: `docs/evaluation/examples/data/architecture/colab_personarag_seed42_rows.jsonl`.

### Relabel RAG off

| Item | Evidence |
|---|---|
| Purpose | Test the relabel adapter without retrieval |
| Family | QLoRA on `unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit` |
| Fine-tune | Rank 16, alpha 32, trained scale 1.0 (`README.md`). Train `EXP-20261003-005`, best eval loss 1.3316 |
| Training data | Relabeled 512-token continuation file `EXP-20261003-004` |
| Retrieval | None |
| Result | 77/120 eligible, completion 0.658, format 0.958, one training-copy failure, writing quality 3.000, style distance 0.750, persona utility 0.705 on 23 eligible blogs |
| Role | Strongest eligibility and the only populated utility that leads the small blog-only utility sample |

The training change is an improvement over the previous supervision bug: slices that were not finished posts are no longer labeled as finished (`DEC-006`). Completion rises from the Base rate of 0.375 to 0.658. Writing quality falls from 3.578 to 3.000. The improvement is specific to the cutoff problem, and it costs judge-rated prose quality on this panel.

### Relabel + RAG (Model E)

| Item | Evidence |
|---|---|
| Purpose | Ground the same adapter with train-only dense evidence |
| Retrieval | Frozen factual dense RAG v1. Topic string only, cosine, top 4, no reranker, no medium filter (`README.md`) |
| Evidence | Atomic claims extracted with the adapter off. Raw passages stay out of the generator prompt |
| Result | 72/120 eligible, completion 0.700, grounding 0.883, writing quality 2.896, style distance 0.805, claim support 0.994 on 108 drafts |
| Role | Deployment candidate in `docs/deployment/DEPLOYMENT_HANDOFF.md` |

This is the second methodological change relative to a plain prompt: retrieval is reduced to span-checked claims. On this panel it has the best grounding pass rate of the three retrieval systems and the lowest writing quality. The deployment note already says RAG does not improve blog voice or writing quality.

### Model F (LightRAG, official profile F-REL)

| Item | Evidence |
|---|---|
| Purpose | Add a knowledge-graph channel beside dense RAG without a new generator |
| Family | Same Relabel adapter and 4-bit Llama 3.1 8B |
| Retrieval | LightRAG 1.5.7, mode `hybrid`, top_k 20, evidence budget 4, `num_ctx` 16384 |
| Embedding | `nomic-embed-text`, 768 dimensions |
| Graph extraction model | Ollama `llama3.1:latest` |
| Context | Generator window 2048. Blog and talk evidence caps are 557 source tokens and 83 graph tokens |
| Prompt family | `model-f-graph-context-v1` |
| Result | 67/120 eligible, completion 0.700, format 0.967, training-copy pass 1.000 on the comparison file, grounding 0.817, all-draft writing quality 3.165, eligible writing quality 3.2662, style distance 0.739 on 90, task adherence 0.695, persona utility null, mean generation latency 20.069 s |
| Role | Implemented and measured graph-augmented generator. Not the leader on eligibility, grounding, writing quality, or task adherence |

F-REL is different from conventional dense RAG because the generator can see relationship lines that survived a passage check, and because retrieval ranks are LightRAG hybrid ranks rather than Chroma cosine distances (`docs/model_f_architecture.md`). The measured panel does not show that this difference improved grounding or eligibility relative to Relabel + RAG.

Hardware for development is documented as an RTX 5070 Ti, 16 GB, in the requirements file. The official manifest does not store a GPU name.

## What "innovative" means here

Two changes are architectural, and both were measured:

1. **Relabel QLoRA.** The training labels were rewritten so incomplete slices are continuations. That is a change to the learning signal, not a larger model. Evidence: `DEC-006`, `EXP-20261003-005`, and the completion gap versus Base on the shared panel.
2. **Source-checked graph context (Model F).** LightRAG builds entities and edges with a local LLM, then F-REL keeps a relationship only when its endpoints and predicate words occur in a retrieved passage. Descriptions are not treated as source spans. Multi-hop paths are a separate arm and were not the official panel. Evidence: `docs/model_f_architecture.md`, `provenance.json`.

PersonaRAG and dense top-4 RAG are conventional retrieval baselines in this comparison. The atomic-claim filter used by Model E is a real control on copying, and it is part of why Relabel + RAG can be discussed as an improved retrieval pipeline even though its writing-quality mean is the lowest of the five.

## Selection justification

A single "best model" is not supported.

- If the objective is eligibility under the project's gates, Relabel RAG off leads (77/120).
- If the objective is judge grounding on a retrieval system, Relabel + RAG leads (0.883 versus 0.817 for Model F and 0.833 for PersonaRAG).
- If the objective is all-draft writing quality, PersonaRAG leads (3.599) and Base is close (3.578). Both complete fewer drafts.
- If the objective is surface style distance on blog, LinkedIn, and X, Model F has the lowest mean (0.739). The paired interval versus Relabel RAG off includes zero (`tables/paired_model_f_differences.csv`). Style distance is not authorship accuracy.
- Persona utility cannot decide. It is missing for every Model F draft and, on the other systems, exists only for a small set of eligible blogs (n=19 to 23).

The deployment document's earlier choice, Relabel + RAG, remains the strongest measured answer to "ground claims with dense train-only retrieval." Model F is the measured answer to "does a partial knowledge graph add a useful channel?" On this panel the graph channel did not clearly beat dense RAG. Keeping both results is the justified report. Promoting Model F because it is the newest run is not.

Paired topic-cluster differences, Model F minus the other architecture, are descriptive intervals from this reporting script. They are not a new registered experiment. Writing quality: Model F is lower than Base by 0.41 and lower than PersonaRAG by 0.43, and higher than Relabel RAG off by 0.17 and higher than Relabel + RAG by 0.27. Those four writing-quality intervals exclude zero. Task adherence versus Relabel RAG off and versus Relabel + RAG includes zero.

Eligible-only writing quality, recomputed from the same rows, is Base 3.733 (n=45), PersonaRAG 3.708 (n=57), Model F 3.266 (n=67), Relabel RAG off 3.154 (n=77), Relabel + RAG 3.060 (n=72). Do not rank these against the all-draft means.

## Comparison matrix

Full numeric export: `tables/architecture_metrics_by_medium.csv`. Figures: `figures/01_eligible_drafts.png` through `figures/12_format_compliance_by_medium.png`.

| Question | Base | PersonaRAG | Relabel off | Relabel + RAG | Model F |
|---|---|---|---|---|---|
| Generator | Llama 3.1 8B | Llama 3.1 8B | Relabel QLoRA | Relabel QLoRA | Relabel QLoRA |
| Retrieval | none | dense top 4 | none | dense top 4 + claims | LightRAG hybrid + F-REL |
| 120-draft row | yes | yes, approximate | yes, seed 42 | yes, seed 42 | yes, F-REL |
| Innovation relative to a plain prompt | no | conventional RAG | label repair | claim-level RAG | graph relations with a passage check |
