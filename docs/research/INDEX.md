# Research index and decision synthesis

Audited 2026-10-03. [manifest.csv](manifest.csv) is the complete filename/hash inventory of local PDF/PPTX research inputs. It contains 126 artifacts: 119 paper-folder candidates, three derived reports, two actual rubric exports, and two historical project artifacts. The PDFs remain local/untracked; this audit does not upload or commit them.

## Inventory completeness and limits

All 125 PDFs and the 13-slide PPTX were text-extracted successfully. Extraction emitted some PDF object/font warnings; text extraction is not visual validation. The workbook and demo first pages were rendered and visually inspected. Rubric text was reviewed in full; historical/derived architecture sections and eight primary-paper abstracts/intros were selectively reviewed, with RAGs to Style results also inspected. **Inventory complete for these local formats; literature synthesis and bibliographic verification incomplete.** Remaining 111 paper candidates are extracted/inventoried, not claim-reviewed. Filename/PDF metadata is not trusted bibliography; blank metadata is unknown.

Numbers in filenames are not unique citation IDs. Labels 113–116 each occur twice; labels 019 and 020 are absent. Two GraphRAG files, [030] and [114] Edge, have identical SHA256 bytes despite different names/years. Manifest IDs use relative-path digests to distinguish files; SHA256 detects identical content. Past report reference numbers are not reliably the same as current filenames: report [111] is Unsloth, whereas local [111] is the Llama 3 paper. Do not cite by number alone.

No explicitly titled OG-RAG PDF was identified in this local inventory. Derived claims about OG-RAG, RAFT, CharLoRA and other unreviewed sources require primary-source verification before implementation decisions. No external bibliography/link validation was performed in this pass.

## Authority classification

- **PROJECT_EVIDENCE:** `derived/workbook1rubric(1).pdf` (ART-420604f96a), `derived/demo1rubric(1).pdf` (ART-07b9537083). Folder location does not make these ChatGPT reports. Workbook PDF p1 confirms five-plus proposals; demo p1 lists Oct 6 noon and slides/video; pp2–3 specify scoring categories. Saved exports do not prove current LMS state.
- **DERIVED_RESEARCH:** Agentic Operating Plan (ART-7b091bf371), Canonical Project Context (ART-0a3211cf95), Research/Implementation Blueprint (ART-46c650b2e0). The blueprint p1 explicitly says it had not inspected local repository/PDFs. Their model ladders are proposals, not accepted architecture or evidence of execution.
- **HISTORICAL_PROJECT_MATERIAL:** Semester 1 report (ART-5a7833c947) and Final Presentation (ART-8410fe2967). They describe multiple branches; slide 7 explicitly refers to metricsft, report refers to FTv2/QLoRAFT. Their scores are historical, not new DATA 298B results.
- **PRIMARY_RESEARCH_CANDIDATE:** paper-directory files. Eight have selective content review in the manifest. Some may be surveys or nonacademic sources; directory membership is not peer-review certification.
- **CURRENT_IMPLEMENTATION_EVIDENCE:** repository source/config/tests and recorded experiments, indexed in [state](../project_state.md), not mixed into this paper inventory.

## Research connected to decisions

### What should be retrieved?

RAGs to Style (ART-b75178ae48, PDF pp1,3–4, Table 1) studies style embeddings on LaMP with Flan-T5. Gains over BM25 are modest and metric/task-dependent; one Table 1 ROUGE-L comparison is tied. It motivates separate style-example selection and a fixed/random/topic/style ablation, not a claim that style retrieval always beats topical retrieval for Lemkin blogs. Factual evidence still needs topical relevance and source support.

PersonaRAG (ART-8a4bbb66dd, p1) describes adaptive user-centric agents and QA experiments. The local copy is arXiv v2 dated January 15, 2026 despite the filename's 2024. Local `persona_rag.py` uses static vocabulary-based rewrite/rerank; describe that actual algorithm instead of claiming a reproduction of the paper.

### What does QLoRA contribute?

QLoRA (ART-7d4200908e, p1) supports training adapters through a frozen four-bit model, with NF4/double quantization. This justifies a resource-conscious candidate, not its style quality or 2048-token fit on this GPU.

PEFT style customization (ART-c151b90e3a, p1) reports lexical/syntactic/surface alignment for author adaptation and warns about content memorization. Pair style measurements with copying diagnostics and document-level isolation.

StyleAdaptedLM (ART-2486aa95a5, p1/Figure 1 description) trains adapters on a base model using unstructured style material and merges into a separate instruction-following model. Current host code trains instruction/answer pairs directly on an instruct model. README's “StyleAdaptedLM-style” description should not be read as experimental reproduction. Keep that paper's method a distinct optional branch.

### How should voice be evaluated?

Wang et al., Catch Me If You Can? (ART-36b1ec5ed8, p1), combines authorship attribution, verification and other measures across domains, reporting difficulty with implicit blog/forum style. It supports multiple measures and medium controls; it does not validate the current TF-IDF classifier or gold-overlap rubric.

Chen/Moscholios (ART-9c6b44a652, p1) compares prompting strategies for real-person language style. Use structured prompting as a baseline, without assuming its favored strategy transfers unchanged to this task.

Narcissistic Evaluators (ART-4f69d4e20d, p1) studies model-based summarization metrics and self-favoring effects, especially without references. It motivates human calibration and independent judging; it does not establish the amount of bias in this project's specific Llama judge.

### What contaminates the corpus, and what makes RAG ontological?

Derived Blueprint pp1–2 and Canonical Context pp1–2 distinguish knowledge, expression and author attribution, and propose provenance-aware curation plus ontology retrieval. Treat these as useful design hypotheses. Actual current filters are regexes and chunking; they do not establish speaker identity or agency authorship. Proposed organic/promotional labels need annotation criteria and a review sample. Source rights and provenance remain open.

A typed schema with validated relations and evidence-backed retrieval is the proposed ontology criterion. A generic entity graph or installing LightRAG is not by itself evidence of ontology-grounded retrieval. Primary ontology-method review remains outstanding; no graph implementation is approved by this audit.

## Historical contradictions to preserve

Past slides/report mix DuckDB/persona pipeline with root Airflow/MinIO/Unsloth paths. The deployed root code does not traverse DuckDB. The report's branch-specific language and local master Paul Graham tree support genuine branch drift, not merely missing documentation. The presentation shows M1–M5, while derived Canonical Context proposes M0–M6; preserve [model mapping](model_catalog.md) without arbitrary renaming. Historical BERTScore/style/latency values are not reproduced results. The operating report's “roughly two weeks” is relative to its old date, not the current October 3 audit.

## Next research work

Verify bibliography from full title pages/primary publishers; reconcile duplicate IDs and content; inspect ontology sources and evaluation protocols fully; audit the origins/support of gold facts and author examples. Extend manifest claim/location/limitation fields as each paper is actually reviewed. Do not confuse extraction completion with reading 119 papers. Keep all research and project-management material outside persona data/indexes.
