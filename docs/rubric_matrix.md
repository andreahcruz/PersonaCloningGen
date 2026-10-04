# DATA 298B Rubric Traceability Matrix

Post-audit evidence: [complete-source preparation](data/COMPLETE_SOURCE_SFT.md)
now contributes MEASURED preprocessing, split and eligibility evidence to D1-1,
D1-2, D1-3 and WB-5.3. EXP-20261003-003 independently verifies structural
split/token/source invariants, but its content-quality gate failed. This does not
complete the demo, validate authorship or establish any new model-quality result.
The matrix below retains the broader outstanding rubric requirements.

Audit update (2026-10-03): local workbook rubric PDF page 1 (printed page 2 of 6)
confirms five or more proposals and at least one improved/innovative model. Demo rubric
pages 1–3 confirms the five demonstration categories below; its saved deadline is Oct 6
at noon, not live LMS verification. See [context](PROJECT_CONTEXT.md) for provenance.
The table below is the earlier planning baseline: its statements that no registered
rescoring/splits exist are superseded by [current state](project_state.md). Source/split
records and incomplete training/comparison registrations now exist, but they do not
establish independently validated model quality. [Model catalog](research/model_catalog.md)
and [proposed matrix](architecture/EXPERIMENT_MATRIX.md) now provide naming and proposal
coverage; descriptive keys have not been approved as canonical historical M numbering.
Immediate next work maps to D1-1, D1-2, D1-3, WB-4.4 and WB-5.3 in [PLAN.md](PLAN.md).

Status definitions come from `AGENTS.md`:
PROPOSED → IMPLEMENTED → TESTED → MEASURED → VALIDATED

Historical workbook/presentation material is evidence of prior work only.
Current claims must be supported by the current repository or a registered experiment.

| ID | Rubric requirement | Excellent acceptance criteria | Historical evidence | Current repo evidence | Required 298B work | Evidence artifact | Status |
|---|---|---|---|---|---|---|---|
| WB-4.1 | Model Proposals | Propose 5+ models; adequately address targeted problems in detailed terms of concepts, features, architectures and algorithms; at least one model is innovative/improved | Previous workbook/presentation described multiple baseline, persona, RAG, QLoRA and future-model concepts | Current tree has RAG and QLoRA-related executable paths; canonical Model 1–5 mapping not yet established | Define canonical five-model lineup; map each model to executable code or explicitly mark as proposed; identify improved/innovative model | `docs/research/model_catalog.md`, code refs | PROPOSED |
| WB-4.2 | Model Supports | Clearly describe platform/environment/tools supporting each model and provide sufficient high-quality architecture/component/data-flow diagrams | Prior architecture diagrams and software-stack tables exist | Docker, Airflow, Spark, MinIO, ChromaDB, Ollama, Streamlit, QLoRA tooling are documented in `project_state.md` | Verify support environment for each canonical model and regenerate diagrams from current architecture | `docs/architecture/`, architecture figures | IMPLEMENTED |
| WB-4.3 | Model Comparison & Justification | Thoroughly compare targeted problems, features, statistical/ML/DL approaches, strengths and limitations; sufficiently justify every model | Previous workbook includes a four-model comparison | Current canonical 5-model scheme does not exist yet | Produce updated 5+ model comparison after canonical mapping and measurements | comparison table + citations | PROPOSED |
| WB-4.4 | Model Evaluation Methods | Correctly, completely and clearly present evaluation methods/metrics for each model | Prior workbook describes BERTScore, RAGAS, Style Consistency, BLEU, ROUGE-L and human evaluation | Gold-eval and other evaluator code exists but has not been re-run as a registered experiment | Audit metric validity by model; define primary/secondary metrics; reproduce baseline measurements | registered experiments + evaluation table | IMPLEMENTED |
| WB-5.1 | System Requirements Analysis | Completely and clearly identify system boundary, actors and use cases; describe high-level analytics/ML functions and capabilities | Prior workbook/presentation includes stakeholders and functionality | Current pipeline/application components are known, but no verified 298B use-case/system-boundary artifact exists | Define actors, use cases, boundary and ML/analytics capabilities | use-case/system-context diagram | PROPOSED |
| WB-5.2 | System Design | Complete architecture with AI components; platform/framework/cloud integration; data-management/database solution; excellent UI/data visualization | Prior materials contain architecture, Docker stack, DuckDB/Chroma/MinIO and Streamlit descriptions | Current stack includes Docker/Airflow/Spark/MinIO/Chroma/Ollama/Streamlit; exact current database/data-management design must be verified | Produce current architecture; verify database/storage roles; document any cloud component; improve/demo UI and visualization | architecture diagrams, deployment diagram, UI screenshots | IMPLEMENTED |
| WB-5.3 | Intelligent Solution | Present developed AI/ML solutions for each targeted problem including proposed/ensembled/developed/applied models; clearly describe input datasets, outputs, system contexts and solution APIs | Previous workbook describes proposed models and input/output flow | RAG, QLoRA, evaluation and UI code exist; canonical complete model suite not yet established | Map each target problem → model → input → output → API/interface; implement/measure missing solution(s) | model/API specification + experiments | IMPLEMENTED |
| WB-5.4 | System Support Environment | Complete, correct, clear description of technologies, platforms and frameworks | Prior software stack exists | Current stack has been partly verified; dependency-version divergence remains | Freeze reproducible environment/version documentation and resolve/record version conflicts | environment manifest + stack table | IMPLEMENTED |
| D1-1 | Data Pre-Processing | Complete, correct and clear demonstration of preprocessing results for targeted problems and requirements | Prior workbook/demo includes cleaning operations and before/after examples | Current preprocessing code exists; current results have not yet been captured as a registered experiment | Re-run or safely reproduce preprocessing; capture before/after, record counts, quality checks | Demo figures/tables + experiment | IMPLEMENTED |
| D1-2 | Training & Test Data Preparation | Complete, correct and clear demonstration of training/test preparation tied to project problems and requirements | Prior workbook describes splits | Current split/leakage state has not yet been independently validated | Freeze dataset/splits; hash them; test exact and near-duplicate leakage; demonstrate train/validation/test composition | split manifest + leakage report | PROPOSED |
| D1-3 | Data Analytics Results | Complete, correct and clear demonstration of project data analytics | Prior workbook contains corpus/style analytics and charts | Analytics logic/data exists historically but current plots are not registered results | Regenerate current descriptive/stylometric analytics from frozen dataset | figures + CSV/JSON tables | PROPOSED |
| D1-4 | Ongoing ML Results | Complete, correct and clear demonstration of high-quality ongoing ML results based on selected models | Historical gold-eval JSON/results exist | Evaluation tooling exists but historical results are explicitly unverified | Reproduce baseline; measure selected models on frozen evaluation set; retain traces and configs | `experiments/EXP-*` | PROPOSED |
| D1-5 | Presentation | Clear, coherent project-progress presentation with effective communication, visuals and logical flow | Prior presentation provides structure/style | No current Demo 1 deck | Build deck around reproducible evidence and live/demo screenshots; prepare video flow | `reports/demo1/` slides/video assets | PROPOSED |
