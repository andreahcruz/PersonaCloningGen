# Evaluation plan

2026-10-04. One harness, separate axes. DEC-008. Historical gold-rubric files are unchanged.

## What the project is trying to answer

Given a requested topic, did the model produce a complete, coherent, useful response that plausibly sounds like Jason Lemkin, reflects his characteristic perspective, and does so without copying his writing?

## Harness

`host_finetune.unified_eval` scores a saved trace into separate families. They are not averaged into `gold_rubric.overall`.

- `completion`: early `<|eot_id|>` when the trace recorded it, mid-sentence stop, and sentence-final completion. A non-empty sentence-final answer passes. Early EOS does not fail a finished answer. This is a gate, not persona fidelity.
- `copying_memorization`: longest shared word span against training targets, excluding spans that also appear in the prompt. A span of 12 or more words fails the gate.
- `persona_style`: mean absolute z-score of surface features against held-out unchanged validation posts, matched by medium. Train rows and gold-overlap families are excluded. `gold_reference` is not the voice target. Features include sentence length, questions, contractions, first person, numbers, function-word rates, punctuation, type-token ratio, and paragraphing. Closer surface counts are not authorship identity.
- `perspective_fidelity`: for advice claims, the share that match held-out Lemkin evidence (`stance_consistency_rate`). Framing-rate distance (second person, advice cues, SaaS terms, tradeoff cues) stays a diagnostic and is not the selector. This is not RAG grounding.
- `writing_quality`: coherence, fluency, logical progression, relevance, specificity, and usefulness, each 1–5. The production prompt is blinded to model and adapter identity. It stays `diagnostic_only` until a sanity check scores real prose and ordinary model drafts above deliberately degraded text. The judge must be a different family from the Llama candidates.
- `task_adherence`: topic-token coverage plus the requested-medium length band. The legacy X-thread heading rubric is not this score.
- `grounding`: claim-level, and only when generation retrieved passages. Each substantive claim is `supported`, `unsupported`, `contradicted`, or `not_verifiable` against the saved `retrieval_hits`. Retrieval-off runs keep `rag_grounding` empty. Token overlap with those passages is `diagnostics.retrieved_lexical_overlap`, not faithfulness.
- `lemkin_evidence_support`: optional evaluation-time check of the same claims against a frozen held-out Lemkin corpus. It can support, contradict, or not establish a claim. It is not RAG grounding and does not use the train-only Chroma collection.
- `diagnostics.reference_similarity`: BERTScore, BLEU, and ROUGE-L against the constructed reference.
- `diagnostics.brief_overlap.gold_rubric_overall`: the old five-part lexical average. Its role is brief/task agreement, not persona quality.

## Gates, then ranking

A draft is eligible only when completion, copying, and requested-medium format pass, and grounding passes when it applies. Style cannot average away a truncated or memorized draft.

Among eligible drafts, rank with calibrated `persona_style` (lower distance), then calibrated `perspective_fidelity`, then calibrated `writing_quality`, then `task_adherence`. Persona Utility, weighted 45/20/25/10, is computed only when all four of those components are selection-valid. The weights are a project decision. BLEU, ROUGE, BERTScore, gold overlap, and stop rates are refused as selectors.

A persona metric is selection-valid only when real held-out Lemkin is closer to the reference profile than frozen generic prose, degraded prose, and base-model drafts when those drafts are available. Otherwise the distance stays diagnostic. Talk has no held-out references until attributed transcripts exist.

## What remains unvalidated

`eval_rag.py` and `persona_pipeline/evaluation/evaluate.py` still contain the older G-Eval, TF-IDF authorship, BERTScore, and RAGAS paths. They are not the selector. The TF-IDF vectorizer is still fit before cross-validation, and the persona-pipeline RAGAS call still uses a dummy context. EXP-20261004-007 did not run the writing judge. EXP-20261004-008 rescores the same saved answers with `qwen2.5:14b` and does not retrain or regenerate.

Comparison prompts still ask for an X post while the legacy format spec describes a thread. That legacy score remains inside brief overlap. The format gate uses a medium length band instead.

Freeze model, prompt, partitions, gold, evaluator, and seeds. Do not read a small single-seed gap as a settled model ranking. The 30 gold topics are a development panel.

Research support and limitations are in [INDEX.md](../research/INDEX.md). Rubric: WB-4.4 evaluation, WB-4.3 comparison, D1-4 ongoing ML evidence. Preserve original scores; never invent missing metadata during registration repair.
