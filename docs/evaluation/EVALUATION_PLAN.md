# Evaluation plan

2026-10-03. Proposed protocol, with existing code distinguished from validated measurement.

## Current score meanings

`eval_gold_rag.score_row` averages format, brief token coverage, expected-fact token coverage, gold-reference token coverage and must-have coverage. It is a lexical/task-format proxy, not voice measurement or factual entailment. Empty-context scores in EXP-007/010 do not measure RAG quality.

`eval_rag.py` implements G-Eval-inspired voice/rhythm dimensions, TF-IDF/logistic-regression authorship, BERTScore, BLEU/ROUGE and optional RAGAS. Its references/anchors are not split-aware. TF-IDF is fit before classifier cross-validation; negatives can fall back to repeated synthetic text. Topic/medium can drive author scores. Best-of-K BERTScore against sampled corpus passages is semantic similarity, not independent voice evidence. Existing tooling needs validation, not merely invocation.

Comparison prompts request X posts but use the x_thread rubric, omit scored structural requirements, and use different backends for adapter vs Ollama generation. Sampling seeds and stop reasons are not recorded. Account for these before ranking architectures.

EXP-20261003-005 has a saved adapter. EXP-20261004-003 compared it with the repaired adapter on the EXP-010 protocol, retrieval off. The selection measures are early `<|eot_id|>` (before half of `max_new_tokens`) and mid-sentence stops. Human voice preference is not a metric. BLEU and ROUGE are logged only.

## Independent outcomes to implement

- Task integrity: requested-unit completeness, abrupt endings, prompt/title echo, repetition, length, token count and stop reason. `.?!` checks are triage; assess lists and short posts appropriately.
- Style: not scored by a human read. DEC-007 removed that review. Automatic surface counts in `voice_distance.py` stay a diagnostic and do not select an adapter.
- Authorship diagnostics: split-aware, family-grouped evaluation with topic/medium-matched negatives; fit vectorization within CV folds; report calibration/discrimination. Disable synthetic-negative fallback for research results.
- Grounding: retrieval relevance/coverage, claim-level support/contradiction and source spans, stance/date correctness. Test ontology retrieval on structured questions independently of style.
- Copying: shared spans and near-duplicates against training/retrieved content, interpreted alongside quality. Do not reward memorized paragraphs as style success.
- Usability/cost: instruction/format adherence, human usefulness, latency, peak VRAM and context consumption.

Freeze model/tokenizer/adapter, prompt, source partitions, gold, profile/index, evaluator/judge, seeds and backend settings. Use paired differences and uncertainty by medium; do not interpret small single-sample gaps as model improvements. Set acceptance thresholds through calibration before final scoring, not from historical slide targets.

Research support and limitations are in [INDEX.md](../research/INDEX.md). Rubric: WB-4.4 evaluation, WB-4.3 comparison, D1-4 ongoing ML evidence. Preserve original scores; never invent missing metadata during registration repair.
