# Decision log

## DEC-009 — Writing judge, claim grounding, and stance versus held-out evidence

Status: ACCEPTED from the user's 2026-10-04 second evaluation pass. Problem: EXP-20261004-007 left writing quality and Persona Utility empty, called token overlap grounding, and measured perspective as framing-cue frequency. Decision: score saved drafts with a blinded Qwen judge (`qwen2.5:14b`), a different family from the Llama candidates. The local Qwen 7B GGUF is a base model and is not the judge. Writing quality is a selector only after real prose and ordinary drafts both outscore deliberately degraded text on coherence. RAG grounding is claim support against saved `retrieval_hits` when retrieval was on, and stays empty when it was not. Lexical overlap moves to `diagnostics.retrieved_lexical_overlap`. `lemkin_evidence_support` retrieves a frozen held-out validation corpus after generation and is not called RAG grounding. Perspective selection uses stance consistency of advice claims against that evidence; framing distance stays diagnostic. Stance is a selector only when a real held-out post, with that post excluded from its own evidence, is more consistent than an opposed draft. Persona Utility is filled only when writing quality and stance are both selectors. No adapter is retrained and no answer is regenerated. Affected: `host_finetune/writing_quality.py`, `host_finetune/claim_judge.py`, `host_finetune/judge_client.py`, `host_finetune/rescore_evidence.py`, EXP-20261004-008.

## DEC-008 — Unified evaluation axes; stop rates are a gate

Status: ACCEPTED from the user's 2026-10-04 evaluation specification. Problem: gold-rubric overall, BLEU, and ROUGE were readable as persona quality, and DEC-007 let completion stop rates stand in for whether a draft sounds like Lemkin. Decision: one harness, `host_finetune.unified_eval`, reports completion, copying, persona style, perspective fidelity, writing quality, task adherence, grounding, and diagnostics separately. Completion, copying, requested-medium format, and applicable grounding are eligibility gates and are not averaged with style. Persona style uses held-out unchanged Lemkin posts, medium by medium, and may rank models only after it places real held-out Lemkin closer than generic, degraded, and available base-model controls. Perspective fidelity is a separate framing-rate distance, not RAG grounding. Writing quality reuses a G-Eval-style judge schema and stays diagnostic until blinded calibration passes. BLEU, ROUGE, BERTScore, and gold-rubric overall remain diagnostics; gold overall means brief/task agreement. A project-specific Persona Utility score (45% style, 20% perspective, 25% writing quality, 10% task adherence) is computed only after gates and only when those four components are selection-valid. Those weights are a project decision, not a universal scientific law. This narrows the DEC-007 clause that made early-`<|eot_id|>` and mid-sentence rates the practical model selector. Human voice preference stays out. Affected: `host_finetune/unified_eval.py`, `host_finetune/voice_distance.py`, `host_finetune/writing_quality.py`, `host_finetune/compare_adapters.py`, `docs/evaluation/EVALUATION_PLAN.md`.

## DEC-007 — Drop human voice preference from selection

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

## DEC-007 — Separate train-only index, no live-collection replacement

Status: ACCEPTED by the 2026-10-04 implementation request. Problem: the Compose collection `lemkin_content` is not split-aware, so using it as context can retrieve validation, test, and gold-overlapping text. Decision: index only the EXP-004 train split, excluding gold-overlap families from the EXP-003 dispositions, into a new collection `lemkin_train_only`. Refuse to delete or upsert `lemkin_content`. Embed with `nomic-embed-text` and `num_gpu: 0` while EXP-005 holds the GPU. Keep `compare_adapters.py` retrieval off for the repaired-versus-relabel run. Consequence: a finished index is a retrieval resource, not a measured RAG gain and not a Streamlit deployment. The build later exited with collection count 21,194. Affected: `host_finetune/train_only_index.py`, `host_finetune/rebuild_chroma.py`, `host_finetune/compare_adapters.py`, `host_finetune/relabel_compare.py`.
