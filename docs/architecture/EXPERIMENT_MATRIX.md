# Proposed experiment matrix

2026-10-03. Descriptive keys below are PROPOSED study cells, not renamings of historical M1–M5 or claims of completed runs.

- **BASE:** unadapted instruct model and task without author conditioning. Current one-line Lemkin comparator is not this control.
- **PROMPT:** same base plus structured persona/style instructions, no retrieval/training.
- **FIXED-STYLE:** PROMPT plus fixed train-owned, medium-matched exemplars.
- **STYLE-RETRIEVAL:** same budget, dynamically selected style exemplars; compare fixed/random/topic selection.
- **VECTOR-FACTS:** train-owned factual vector retrieval, fixed style conditioning; isolate grounding.
- **PERSONA-RETRIEVAL:** same corpus/model with persona-aware query/rerank vs ordinary vector retrieval. Existing separate implementation is a candidate, not evidence of benefit.
- **QLORA:** complete-target adapter vs its unadapted generator under identical prompts. EXP-20261003-005 is a narrower cell: the same rank-16 setup as the repaired adapter, with only the 512-token labels changed. It is running and is not this complete-target comparison.
- **QLORA+VECTOR:** adapter plus factual retrieval; style examples as a separate toggle.
- **ONTOLOGY:** typed, provenance-backed factual retrieval vs vector retrieval with generator/style held fixed.
- **QLORA+ONTOLOGY:** final hybrid if simpler comparisons justify it. RAFT/CharLoRA remain optional extensions.

Freeze task IDs, source ownership, gold set, model/tokenizer revision, prompts, output/context budgets, seed list, backend/quantization and evaluator. Vary one factor or preregister a factorial design. Report paired task-level differences and uncertainty by medium. Rank-32 history also changes learning rate; it is not a clean rank ablation.

Cleaning views (raw-derived, organic strict/broad, promotional) are proposed separate corpus ablations, not existing labels. Test material must not supply training, profiles, examples, tuning or retrieval evidence unless explicitly approved under a defensible protocol.

Record the required experiment manifest plus cleaning, ontology, prompt and retriever versions. Rubric: WB-4.1 five-plus proposals and an improved model; WB-4.3 comparison; WB-4.4 metrics; WB-5.2/5.3 architecture/interfaces. No result is copied from slides. See [historical naming](../research/model_catalog.md).
