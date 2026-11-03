# Project retrospective

Status labels follow `AGENTS.md`: PROPOSED, IMPLEMENTED, TESTED, MEASURED, VALIDATED. A commit message is not an experimental result. Scores below are taken from registered notes or from `results/model-f-handoff-v1/`.

## Problem and objective

`docs/PROJECT_CONTEXT.md` (audited 2026-10-03) states the project as AI-driven persona cloning for B2B content, target Jason Lemkin / SaaStr. The accepted evaluation split, from `DEC-008` and `DEC-009`, keeps completion, copying, format, grounding, style distance, writing quality, and task adherence as separate measurements. Generated text is not evidence that Jason Lemkin endorsed it.

The gold task file used for the shared panel is `data/openai_ft/lemkin_gold_eval.jsonl`: 30 topics, four media, 120 cells. The reference text in that file is an evaluator reference. Model F does not put it in the prompt (`docs/evaluation/UNIFIED_120_COMPARISON.md`).

## Chronology that the repository can support

| Period | What the artifacts show | Status |
|---|---|---|
| Through 2026-10-03 | Audit of historical workbooks, model-name conflict, and proposed experiment matrix. `docs/research/model_catalog.md` refuses to merge two incompatible M-number schemes. | Historical proposals; not a completed five-model lineup |
| 2026-10-03 | Whole-source SFT candidates `EXP-20261003-001/002/003`. Latest candidate fails the content-quality gate (`experiments/README.md`). | MEASURED preparation, not promoted |
| 2026-10-03 to 2026-10-04 | Relabel file `EXP-20261003-004` and rank-16 train `EXP-20261003-005`. Best eval loss 1.3316. Adapter `host_finetune/output/lemkin_lora_relabel`. Decision `DEC-006`. | Train MEASURED; adapter not in Git |
| 2026-10-04 | Train-only Chroma embed. Context records 21,194 documents. Later parity records 21,193 with `train_26148` absent (`docs/architecture/LIGHTRAG_MODEL_F.md`). | MEASURED index; count conflict left in place |
| 2026-10-04 to 2026-10-05 | Unified evaluator and Qwen 14B judge (`DEC-008`, `DEC-009`). Relabel plus dense RAG scored as `EXP-20261005-003`. Deployment note selects that path (`docs/deployment/DEPLOYMENT_HANDOFF.md`). | MEASURED |
| 2026-10-27 to 2026-10-31 | LightRAG 1.5.7 added as an optional backend (`DEC-010`). Pilot index, five query modes, relationship conditioning, resumable expansion. | MEASURED development runs; not the 120-cell panel |
| 2026-11-01 | Four-arm runner and graph-context code handed off. CPU tests recorded as 156 passed in `docs/handoff/MODEL_F_HANDOFF_MANIFEST.md`. The 36-question run `EXP-20261101-009` is described as not executed in the Nov 2 transfer note. | IMPLEMENTED and TESTED as code; that named run is absent |
| 2026-11-02 | Official panel `EXP-20261102-001-model-f-unified-panel`, profile F-REL. Judge timestamp `2026-11-03T00:43:46Z` in `provenance.json`. Shareable aggregates published in `results/model-f-handoff-v1/`. | MEASURED |
| This checkout, Nov 2 handoff commits | `9a1bc34` publishes aggregates. `3942501` publishes judge rows and audit flags and keeps drafts private. | Documentation of the measured panel |

Git subjects were read with `git log --oneline -25` on 2026-11-02. They are a sequence of commits, not proof of metric values.

## Architectures that reached a 120-draft score

| Name in the comparison file | What it is | Experiment cited by the protocol note |
|---|---|---|
| Base | Llama 3.1 8B, no adapter, no retrieval, seed 42 | `EXP-20261001-010`, rows scored in `EXP-20261004-008` |
| PersonaRAG | Documented as a stand-in: base Llama plus train-only Chroma, top 4, not the semester-1 `generate.py` pipeline | `EXP-COLAB-personarag-newdata-seed42` |
| Relabel RAG off | Relabel adapter, no retrieval | Seeds 42/43/44 exist; the comparison row is seed 42 |
| Relabel + RAG | Same adapter, frozen dense RAG v1, atomic claims | `EXP-20261005-003` seed 42 |
| Model F | Same adapter, LightRAG hybrid, profile F-REL | `EXP-20261102-001-model-f-unified-panel` |

The protocol note says PersonaRAG is approximate. Relabel seeds 43 and 44 retrieved the same passages as seed 42 on 29 of 30 topics, while extracted evidence differed on 15 of 30 topics because machines and Python versions differed. The comparison JSON uses one row per architecture, not a three-seed pool.

## Selection and rejection

- `DEC-006` keeps the relabel adapter and does not replace it with another fine-tune before evaluation.
- `docs/deployment/DEPLOYMENT_HANDOFF.md` selects Relabel plus dense RAG for deployment because grounding was a required property, and it records that RAG did not improve blog voice or writing quality.
- `DEC-010` says LightRAG is not the ontology requirement and is not a new generator.
- The frozen pilot rule kept F-SOURCE because no graph arm strictly beat it on 8 questions. F-REL's factual pass rate there was 0.375 against 0.500 for F-SOURCE (`docs/model_f_experiments.md`).
- Stage-2 on 26 questions recommended F-REL on echo-adjusted faithfulness (0.3462 versus 0.3077 for F-DESC) with equal copy-or-leak rates. Question wins were 4, 4, and 18 ties. The official panel was not allowed to change that choice.
- Nothing in the handoff declares Model F the winner of the 120-draft panel.

## Failures recorded during construction

- Whole-source SFT failed its content-quality gate and was not trained (`experiments/README.md`).
- Early relabel generations were built to stop the mid-sentence cutoff caused by supervising `<|eot_id|>` on incomplete slices (`DEC-006`).
- The 160-document pilot processed 152 documents and had 8 extraction failures (`docs/architecture/LIGHTRAG_MODEL_F.md`). It was not copied into the expansion snapshot.
- Expansion coverage stopped at 2,713 of 13,140 eligible documents, with 10,269 pending, 153 interrupted, and 5 permanent failures (`docs/model_f_architecture.md`).
- A direct chunk-vector count was 9,266 against 9,269 successful chunks. The 3-row gap was not repaired (`docs/handoff/MODEL_F_DEPLOYMENT_TRANSFER_20261102.md`).
- Relationship-normalization repairs were a dry run and were not applied (same transfer note).
- `gold_030` evidence status is `extraction_failed` with 0 accepted claims, and all four drafts stayed in the panel (`diagnostics/descriptive_omissions.json`).
- The generation manifest still says `generated_not_judged`. The judge record is `unified_eval/evaluation_run.json` (`experiments/README.md`).
- Requested quantization is 4-bit, and the manifest also stores `is_loaded_in_4bit: true` together with `is_loaded_in_8bit: true`. `docs/model_f_architecture.md` leaves that pair unresolved.
- The official generation manifest does not store a GPU name.

## Stale documents

`docs/evaluation/UNIFIED_120_COMPARISON.md` says Model F on the 120-cell panel is not measured and prints dashes for Model F. `docs/model_f_results.md` and `results/model-f-handoff-v1/architecture_comparison.json` contain the measured Model F row. The later files are the score source. The protocol note is still the source for how Base, PersonaRAG, and the Relabel rows were produced, and for the statement that this repository does not label those rows A–E.

`docs/rubric_matrix.md` is a 2026-10-03 planning matrix. It predates the measured Relabel and Model F panels.

## Limitations still open

- Partial graph. Re-indexing will not reproduce the frozen node and edge counts (`README.md`).
- Persona utility is null for Model F, so the project's composite cannot rank it.
- Style distance is missing for all 30 talk drafts on every architecture.
- Latency is stored for Model F generation seconds and is null for the other four architectures.
- Deterministic unsupported numbers and judge grounding disagree often enough that neither can be quoted as the other's result.
- The React demo and this branch are not integrated.
- No architecture has been marked VALIDATED against an independent reproduction on a teammate machine. The handoff says receipt of the snapshot, adapter, and private zip has not been confirmed (`docs/teammate_handoff.md`).
