# Final report outline

Aligned to the criteria in `10_rubric_compliance_matrix.md`. Each section names the evidence to paste. Do not add a metric that is not in those files.

## 1. Problem and scope (5.1)

- Actor: a user who supplies a topic and a medium. Systems: generator, retriever, judge. Out of scope: author endorsement and publishing rights (`docs/PROJECT_CONTEXT.md`).
- Target behavior: blog, LinkedIn, X, and talk in a Lemkin / SaaStr register, with claims that can be checked against train-owned sources.
- Evidence: `docs/architecture/ARCHITECTURE.md` boundary paragraph. Add the use-case page that is still missing.

## 2. Data (Appendix B)

- Sources and the relabel file `EXP-20261003-004`.
- Gold file hash `d7c4f997…e520e`.
- Chroma `lemkin_train_only`, 21,193 documents, and the older 21,194 note left unresolved.
- Graph-ready 13,140 eligible documents; frozen snapshot 2,713 completed.
- What is public and what stays in the private zip.
- Evidence: `03_lightrag_graph_construction.md`, `figures/20_indexed_vs_eligible_coverage.png`.

## 3. Proposed models (4.1, 4.2)

- Historical naming conflict, then the five scored names.
- Relabel QLoRA and F-REL as the two methodological improvements.
- Support stack: Unsloth, QLoRA, Ollama, Chroma, LightRAG 1.5.7, Qwen judge.
- Figures: `25_system_architecture.png`, `21_graph_construction.png`, `27_deployment_dependencies.png`.

## 4. Architectures in detail (4.3, 5.3)

One subsection each for Base, PersonaRAG, Relabel RAG off, Relabel + RAG, and Model F, using the matrix in `02_model_inventory_and_comparison.md`.

Model F subsection must include:

- Hybrid, top_k 20, evidence budget 4, `num_ctx` 16384, window 2048.
- F-SOURCE pilot versus F-REL engineering choice.
- Partial coverage.
- Figure: `22_retrieval_generation.png`, `26_inference_sequence.png`.

## 5. Evaluation methods (4.4)

- Gates, writing quality, task adherence, style distance, stance, utility weights.
- Why F1, accuracy, and ROC are not used.
- Judge versus deterministic audit.
- Evidence: `04_evaluation_methodology.md`.

## 6. Results (4.5, 6.1, 6.4)

- Table of all-draft metrics and a second table of eligible-only writing quality.
- Figures 01–12.
- Conditional leader table from `05_results_and_interpretation.md`.
- Paired intervals labeled as descriptive bootstrap, not a registered test.
- Representative outputs: only after the redaction task. Until then, cite topic ids and scores without draft text.

## 7. LightRAG quality (6.1)

- Coverage, node and edge totals, omission counts, evidence-status chart.
- Explicit statement that graph size is not a quality metric.
- Figures 16, 17, 19, 20.

## 8. Failures, achievements, constraints (6.2)

- Paste the verified-findings list from `06_failure_analysis.md`.
- Separate the hypothesis list.
- Resource constraint: partial index, adapter not in Git, dirty judge worktree, no stored GPU name.

## 9. System quality (6.3, Appendix A)

- Historical test table from `08_system_testing_and_performance.md`, dated and not described as this week's run.
- Generation latency histogram `figures/23_generation_latency.png`.
- Statement that no response-time target is documented and retrieval latency was not stored.
- GUI screenshots: insert when they exist. Until then the section is incomplete.

## 10. System design and demo (5.2, 5.4, Appendix C)

- Three stacks: Compose/Streamlit, FastAPI `generate.py`, host_finetune Model E/F.
- Integration gap.
- Artifact hashes.
- Pointers to code directories and to this support package.
- Video link when it exists.

## Numbers that must appear exactly once with their denominator

| Number | Denominator | File |
|---|---|---|
| Eligible 67 | 120 | `aggregate_summary.json` |
| Eligible writing quality 3.2662 | 67 | same |
| All-draft writing quality 3.165278 | 120 | `architecture_comparison.json` |
| Completion 0.700 | 120 | same |
| Grounding 0.8167 | 120 | same |
| Training-copy comparison pass 1.0 | 120 | same |
| Unsupported-number cells 52 | 120 | audit summary |
| Cap and mid-sentence intersection 31 | 120 | `tables/model_f_cap_and_completion.csv` |
| Source-copy failures 7 | 120 | audit summary |
| Zero-claim drafts 4 | 120 | audit summary and flag CSV |
| Indexed documents 2,713 | 13,140 eligible | architecture note |

Persona utility for Model F: write "null", not 0.
