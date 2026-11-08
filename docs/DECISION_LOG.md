# Decision log

## DEC-009 — Writing judge, claim grounding, and stance versus held-out evidence

Status: ACCEPTED from the user's 2026-10-04 second evaluation pass. Problem: EXP-20261004-007 left writing quality and Persona Utility empty, called token overlap grounding, and measured perspective as framing-cue frequency. Decision: score saved drafts with a blinded Qwen judge (`qwen2.5:14b`), a different family from the Llama candidates. The local Qwen 7B GGUF is a base model and is not the judge. Writing quality is a selector only after real prose and ordinary drafts both outscore deliberately degraded text on coherence. RAG grounding is claim support against saved `retrieval_hits` when retrieval was on, and stays empty when it was not. Lexical overlap moves to `diagnostics.retrieved_lexical_overlap`. `lemkin_evidence_support` retrieves a frozen held-out validation corpus after generation and is not called RAG grounding. Perspective selection uses stance consistency of advice claims against that evidence; framing distance stays diagnostic. Stance is a selector only when a real held-out post, with that post excluded from its own evidence, is more consistent than an opposed draft. Persona Utility is filled only when writing quality and stance are both selectors. No adapter is retrained and no answer is regenerated. Affected: `host_finetune/writing_quality.py`, `host_finetune/claim_judge.py`, `host_finetune/judge_client.py`, `host_finetune/rescore_evidence.py`, EXP-20261004-008.

## DEC-008 — Unified evaluation axes; stop rates are a gate

Status: ACCEPTED from the user's 2026-10-04 evaluation specification. Problem: gold-rubric overall, BLEU, and ROUGE were readable as persona quality, and DEC-007 let completion stop rates stand in for whether a draft sounds like Lemkin. Decision: one harness, `host_finetune.unified_eval`, reports completion, copying, persona style, perspective fidelity, writing quality, task adherence, grounding, and diagnostics separately. Completion, copying, requested-medium format, and applicable grounding are eligibility gates and are not averaged with style. Persona style uses held-out unchanged Lemkin posts, medium by medium, and may rank models only after it places real held-out Lemkin closer than generic, degraded, and available base-model controls. Perspective fidelity is a separate framing-rate distance, not RAG grounding. Writing quality reuses a G-Eval-style judge schema and stays diagnostic until blinded calibration passes. BLEU, ROUGE, BERTScore, and gold-rubric overall remain diagnostics; gold overall means brief/task agreement. A project-specific Persona Utility score (45% style, 20% perspective, 25% writing quality, 10% task adherence) is computed only after gates and only when those four components are selection-valid. Those weights are a project decision, not a universal scientific law. This narrows the DEC-007 clause that made early-`<|eot_id|>` and mid-sentence rates the practical model selector. Human voice preference stays out. Affected: `host_finetune/unified_eval.py`, `host_finetune/voice_distance.py`, `host_finetune/writing_quality.py`, `host_finetune/compare_adapters.py`, `docs/evaluation/EVALUATION_PLAN.md`.

## DEC-007 — Drop human voice preference from selection

Identifier note, 2026-11-07: this is the voice-preference decision. A later heading reuses `DEC-007` for the train-only index. The two records are not one decision, and this note does not renumber either of them. Voice-preference and stop-rate citations mean this record.

Status: ACCEPTED from the user's 2026-10-04 instruction. Problem: the user cannot judge whether one draft sounds more like Lemkin than another, so a blinded preference sheet cannot choose an adapter. Decision: human voice review is not part of metric testing. Comparisons no longer write a preference sheet. Selection uses the early-`<|eot_id|>` rate and the mid-sentence stop rate. BLEU, ROUGE, and the gold lexical overall stay logged and do not select. Finished experiment sheets remain historical records. This supersedes the voice-preference clause in DEC-006.

## DEC-006 — Preserve active relabel run and evaluate before another FT iteration

Status: ACCEPTED from the user's 2026-10-04 instruction. EXP-20261003-004/005
introduce an alternative to whole-source-only preparation: scope-aligned section
tasks at 512 tokens. Leave current training, inputs and outputs untouched. Audit
the frozen artifacts using CPU-only code; add future comparison telemetry and
record remaining quality concerns. Completion is scored by the stop-rate metrics.
DEC-007 removed blinded voice preference from selection.
See evaluation/RELABEL_FT_NEXT_STEPS.md. The running model is not yet MEASURED
for generation quality; proposed promotion thresholds must be fixed before
comparison outputs are reviewed.

2026-10-03. These are the first explicit records in this file. Do not erase old decisions or rename historical models retroactively. All records currently supersede none.

## DEC-001 — Evidence and immutable history

Status: ACCEPTED, from explicit user instructions and AGENTS.md. Problem: artifacts disagree. Decision: use PROJECT_CONTEXT.md precedence; preserve raw sources, frozen splits and prior run artifacts; distinguish implementation/testing/measurement/validation. Alternative: trust latest chat or slides, rejected by user instructions. Consequence: unknown provenance stays unknown. Affected: context and future experiments.

## DEC-002 — Audit before implementation

Status: ACCEPTED and fulfilled. The attached FIRST task required an audit before implementation. The user subsequently approved the next task and requested resumption; preparation implementation followed. The original documentation-only restriction no longer describes current work. Affected: PLAN.md.

## DEC-003 — QLoRA-centered controlled hybrid

Status: PROPOSED. Problem: recent adapter comparisons do not test retrieval or isolate voice. Proposal: retain Llama 3.1 8B QLoRA initially; separate source curation, style exemplars/conditioning, and factual retrieval. Evaluate ontology-aware retrieval as its own extension. Evidence: host trainer, generate.py, user priority, local PEFT/style-retrieval papers. Alternatives: prompting only; standalone adapter; graph-first build. Consequence: controls precede hybrid benefit claims. Affected: architecture/EXPERIMENT_MATRIX.md and future harness.

## DEC-004 — Complete-target preparation before capacity changes

Status: ACCEPTED for preparation following user approval; training pilot remains PROPOSED. Problem: 1600-character chunking and 512-token fitting permit incomplete answers under full-post instructions. Decision: retain complete cleaned-source bodies, preserve frozen EXP-008 family ownership, quarantine conflicts rather than reassign, use title-derived tasks, measure full-chat budgets with the pinned tokenizer. Defer long bodies, untitled social material, transcripts without speaker attribution and detected contamination. Three registered candidates record conservative policy iterations; the latest has 1,011 rows but remains unsuitable for automatic training promotion. Alternatives: rank sweep; generation-only completion prompt; re-fit already-cut fragments. Consequence: source-quality recovery precedes any 2048-token/rank16/alpha32 pilot; no training configuration changed. Affected: preparation code, data specs, EXP-20261003-001/002/003.

## DEC-005 — Independent evaluation dimensions

Status: PROPOSED. Problem: lexical/format score is mistaken for voice; existing style proxies have leakage/confounding risks. Proposal: completion, style, factuality, format and copying outcomes; human-calibrated judging; split-aware references; corrected classifier CV and controls. Evidence: scorer implementations and selected local evaluation papers. Alternatives: gold overall alone; current TF-IDF alone; one same-family judge. Consequence: evaluator validity precedes declaring a best adapter. Affected: evaluation/EVALUATION_PLAN.md and future scoring.

## DEC-006 — Relabel existing 512-token rows, then one controlled train

Status: ACCEPTED. The user confirmed the unified correction plan and this train was started. Problem: `<|eot_id|>` was supervised on slices whose prompt said the whole post was finished, and generation then stopped early. Decision: keep whole X, blog, and LinkedIn posts, including posts with no period. Relabel sentence-complete fragments, colon or heading lead-ins, and sentence-final last slices as an opening or a continuation that quotes the previous ending. Drop mid-sentence cuts and transcript fragments. Do not refit, do not edit raw files, and do not change rank, learning rate, epochs, batch, or sequence length. One new adapter, `lemkin_lora_relabel`, is the comparison against `lemkin_lora_repaired`. Evidence: EXP-20261003-004 (MEASURED file), EXP-20261003-005 (train finished, eval loss 1.3316), and EXP-20261004-003 (relabel cutoff baseline: early stop 0.550 vs 0.650, mid-sentence 0.342 vs 0.633). This does not supersede DEC-004. The whole-source candidate is still not training-ready. Affected: `host_finetune/relabel_continuations.py`, EXP-004, EXP-005.

## DEC-010 — LightRAG is a selectable retrieval backend, not an ontology and not a new generator

Status: ACCEPTED for implementation from the 2026-10-27 Model F assignment. Problem: the project needs graph-enhanced retrieval without replacing Relabel QLoRA, the frozen Chroma indexes, or the train-only split. Decision: add HKUDS LightRAG 1.5.7 as an optional backend (`naive`, `local`, `global`, `hybrid`, `mix`) that returns verbatim source chunks into the existing atomic-evidence path. Build a derived graph-ready corpus downstream of EXP-20261003-004. Do not retrain the adapter, do not edit raw or cleaned sources, and do not treat LightRAG as the ontology requirement. Default retrieval remains off, and the default backend remains dense Chroma selected by `rag_active.json`. Consequence: a working graph is a retrieval option. It is not evidence that voice or factual completeness improved. Affected: `host_finetune/graph_ready.py`, `host_finetune/lightrag_backend.py`, `host_finetune/compare_adapters.py`, `docs/architecture/LIGHTRAG_MODEL_F.md`.

## DEC-011 — Shared operational project state

Status: ACCEPTED from the 2026-11-07 request to give Cursor and Codex one project-state procedure. Problem: `docs/project_state.md` and `docs/PLAN.md` are dated narratives, and agents were reconstructing status from chat. Decision: keep those narratives and this decision log. Add `docs/project/state.json` as the operational snapshot, `docs/project/tasks.json` for task status, and `docs/project/HANDOFF.md` for the session pointer. Validate them with `python -m project_pm check`. Do not create a second decision history or a second experiment layout. Alternatives: rewrite the October state file in place, or leave coordination only in chat. Both were rejected. The first destroys the dated record. The second does not survive a new session. Consequence: a task is not done without a verification string. An experiment directory dated on or after the cutoff in `state.json` needs a manifest. This does not change DEC-001 through DEC-010, split assignments, or any measured result. Affected: `AGENTS.md`, `docs/project/`, `project_pm/`, `.cursor/hooks.json`.

## DEC-007 — Separate train-only index, no live-collection replacement

Identifier note, 2026-11-07: this is the train-only index decision. It collides with the earlier voice-preference `DEC-007`. Gold-overlap and `lemkin_train_only` citations mean this record. The 21,194 figure in the paragraph below is the 2026-10-04 build-exit count. The persisted index used by later retrieval has 21,193 documents, with `train_26148` absent (`experiments/EXP-20261004-009-dense-rag-v1/metrics/corpus.json`). This note does not rewrite or renumber the decision.

Status: ACCEPTED by the 2026-10-04 implementation request. Problem: the Compose collection `lemkin_content` is not split-aware, so using it as context can retrieve validation, test, and gold-overlapping text. Decision: index only the EXP-004 train split, excluding gold-overlap families from the EXP-003 dispositions, into a new collection `lemkin_train_only`. Refuse to delete or upsert `lemkin_content`. Embed with `nomic-embed-text` and `num_gpu: 0` while EXP-005 holds the GPU. Keep `compare_adapters.py` retrieval off for the repaired-versus-relabel run. Consequence: a finished index is a retrieval resource, not a measured RAG gain and not a Streamlit deployment. The build later exited with collection count 21,194. Affected: `host_finetune/train_only_index.py`, `host_finetune/rebuild_chroma.py`, `host_finetune/compare_adapters.py`, `host_finetune/relabel_compare.py`.


## DEC-012 — Fixed architecture benchmark and separate development questions

Status: ACCEPTED from the user's 2026-11-08 evaluation decision and explicit TASK-003 coordination approval. This entry records that policy; the ontology architecture in `docs/architecture/ONTOLOGY_PLAN.md` remains PROPOSED pending design review.

Problem: TASK-002 confirmed training-family exposure and prior development reuse of the established 30-topic gold set. Treating an independent new final set as mandatory would prevent direct comparison with measured Models E and F and unnecessarily block ontology development.

Decision: preserve `data/openai_ft/lemkin_gold_eval.jsonl` unchanged (SHA-256 `d7c4f997899e28f28eb3ea9aa3417857da6f0666acbf72fd8ac31422034e520e`) as the primary fixed architecture-comparison benchmark. Preserve its historical experiment outputs and scores. Use the 24 proposed questions in `experiments/EXP-20261108-003-gold-exposure-review/raw/development_questions.jsonl` for implementation debugging, architecture tuning, and preliminary studies. They are development material, not an independent final test.

Report known training-family exposure and previous benchmark development reuse. Comparisons on the gold set are benchmark-specific, not independent generalization results. Source-family overlap is not automatically invalid leakage in author-corpus learning. Apply consistent recorded configurations and report corpus coverage, prompts, budgets, evidence preparation, failures, uncertainty, cost, and other architecture tradeoffs transparently. Distinguish controlled matched comparisons from descriptive comparisons with historical systems. Do not change existing metric-validity or gate semantics silently.

An additional independent question evaluation may be created later under a separately reviewed protocol. It is optional and must not block ontology implementation or comparisons on the established benchmark. Do not regenerate the gold set, reopen or reshuffle the corpus splits, modify frozen experiments, or recommend retraining historical models solely to eliminate this benchmark limitation.

Evidence: TASK-002 `EXP-20261108-003-gold-exposure-review/notes.md`, `raw/training_exposure.json`, and `closure.md`. The 184 train-family rows are actual EXP-005 training-dataset members; completed consumption is strongly supported without per-row proof. The 16 validation-family and four test-family rows have different trainer roles. This policy does not erase those distinctions or establish semantic independence.

Alternatives: replace the established gold file, require an untouched final set before architecture work, or retrain every historical model to remove family exposure. Rejected for the present comparison objective. A later independent evaluation remains useful when feasible, without becoming a dependency.

Consequences and scope: TASK-003 is design only. Creating a new ontology store, extraction implementation, inference, model evaluation, training, index changes, and deployment require subsequent implementation tasks and registered runs as applicable. Existing accepted decisions and duplicate DEC-007 identifiers remain unchanged. This policy supersedes only older planning language that makes an independent final question set a prerequisite. Affected: `docs/architecture/ONTOLOGY_PLAN.md`, the operational evaluation fields in `docs/project/state.json`, TASK-003 registration in `tasks.json`, and `HANDOFF.md`.



## DEC-013 — Pilot ontology contracts and existing-index claim expansion

Status: ACCEPTED from the user's ONT-01 instruction and explicit architecture decisions on 2026-11-08. This approves the bounded contract/admission task, not automatic continuation to extraction or full ontology deployment.

Initial retrieval will reuse the existing train-only Chroma index with DuckDB-backed structured claim expansion. A separate claim-vector collection is an optional later ablation only when development evaluation establishes a meaningful need. Do not build a graph database or new vector index in ONT-01.

Generation outputs, verification results, and benchmark eligibility are distinct records. Apply consistent validation/scoring rules to matched architectures, retain every benchmark cell (including unsupported claims, truncations and abstentions), and do not add an ontology-specific output-revision mechanism in the initial retrieval comparison. Preserve historical output and metric semantics.

Begin with five candidate blog documents and explicit source lineage. Admit only author-attributed material with recorded human provenance/attribution review. Initially exclude ambiguous authorship, guest/interview transcripts, reposts and third-party writing. If human approval has not occurred, preserve pending decisions and publish no retrievable claims. Claims require source-supported entity, quantity, condition and time bindings; span occurrence alone does not establish semantic support. Admission and review decisions are immutable content-bound records; changes append a new decision rather than overwriting old evidence.

ONT-01 implements data contracts, validation, a pending-review pilot manifest and synthetic tests only. It does not extract real claims, execute model inference, train, build DuckDB/Chroma indexes, change datasets/splits, or revise generation outputs. Capsule/numeric integration is documented for subsequent review. Stop after ONT-01 and request review before ONT-02.

DEC-012 remains unchanged: the original 30-topic gold benchmark is the primary fixed comparison benchmark; the 24 questions are development material; additional independent questions are optional and nonblocking. Existing decisions, frozen experiments, models, indexes and source data remain intact. This entry supersedes only the earlier ONTOLOGY_PLAN.md proposal to make a separate claim index mandatory initially.

Affected: host_finetune/ontology_contract.py, tests/test_ontology_contract.py, experiments/EXP-20261108-004-ont-01-schema-admission/, docs/architecture/ONT_01_REPORT.md, docs/architecture/ONTOLOGY_PLAN.md and approved ONT-01 operational state/handoff.

