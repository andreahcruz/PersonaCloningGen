# Model naming and proposed lineup

2026-10-03. No new canonical M numbering has been accepted.

Historical presentation slide 8/report XVIII–XIX use M1 base Llama, M2 prompt/persona, M3 PersonaRAG and M4 QLoRA; presentation slides 1/4 and report conclusion propose M5 LightRAG. Derived Canonical Context instead uses M0 base, M1 prompting, M2 vector PersonaRAG, M3 ontology, M4 QLoRA, M5 hybrid, M6 advanced adaptation. These schemes are incompatible; do not silently merge them.

Executable candidates include root vector RAG, host QLoRA, the separate persona-aware retrieval/inference pipeline, and one-line comparison paths. The unadapted comparator still receives an author-style prompt. Ollama selection supports adapter-plus-RAG integration but does not establish measured gains.

Use descriptive [experiment keys](../architecture/EXPERIMENT_MATRIX.md) pending approval. More than five proposals address the verified workbook criterion without claiming five completed systems. Proposed core contribution: controlled QLoRA style adaptation plus separately evaluated factual/style retrieval, with an ontology-aware extension.
