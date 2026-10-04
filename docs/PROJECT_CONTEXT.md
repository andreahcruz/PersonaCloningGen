# Project context — DATA 298B

Audited 2026-10-03 (America/Los_Angeles). Read this, [state](project_state.md), [plan](PLAN.md), and [decisions](DECISION_LOG.md) before new work.

## Identity and accepted scope

AI-Driven Persona Cloning for B2B Content Generation, SJSU DATA 298B, repository 298P. Target: Jason Lemkin / SaaStr. Current user priority: a defensible QLoRA-based generative model, potentially improved with retrieval, better training examples, cleaning, and prompts. Generated drafts are not evidence of the author's endorsement.

The initial request required a read-first audit. The user subsequently approved implementation (“Approved, do what you need to do”) and requested resumption. Whole-source SFT preparation is implemented, tested and measured in three registered runs. The latest candidate passes structural checks but fails the content-quality gate. See [preparation evidence](data/COMPLETE_SOURCE_SFT.md).

The user then accepted a separate label fix on the existing 512-token rows (DEC-006). EXP-20261003-004 is that file. EXP-20261003-005 finished training it to `host_finetune/output/lemkin_lora_relabel`. Trainer defaults and the repaired adapter weights were left in place. Generation results for the new adapter do not exist yet. On 2026-10-04 a separate CPU job embedded that file's train rows into Chroma collection `lemkin_train_only` (21,194 documents). It does not replace `lemkin_content` and was not part of the training step. See [state](project_state.md).

Separate three questions: what evidence grounds an answer; how the author expresses it; which source material reliably represents that author. Measure grounding, style, task completion, and copying separately. Compare complexity against simpler baselines.

## Source of truth and statuses

Professor instructions and actual rubrics determine academic constraints. Accepted scope lives here; accepted designs belong in DECISION_LOG.md. `project_state.md` records the latest audit; subsystem specifications describe behavior and proposals. Code, configuration, hashes, tests, and registered run artifacts establish what exists or ran. Documentation cannot override contrary execution evidence.

Primary papers inform decisions; derived ChatGPT reports synthesize advice; previous workbooks/slides describe historical states; chat summaries provide leads. Record contradictions explicitly. An EXP folder name alone does not establish complete registration or reproducibility.

Preserve AGENTS.md lifecycle labels: **PROPOSED, IMPLEMENTED, TESTED, MEASURED, VALIDATED**. Decision statuses **ACCEPTED, SUPERSEDED, REJECTED** are separate. Evidence qualifiers **VERIFIED, HISTORICAL, CONFLICTING, UNKNOWN** qualify claims. Historical “EVALUATED” means scoring was reported, not automatically independently validated.

## Academic constraints

- `research/derived/workbook1rubric(1).pdf`, PDF page 1 (printed page 2 of 6): Excellent requires five or more detailed model proposals, including at least one innovative/improved model. It does not require five completed training runs. Pages 2–5 cover comparison, evaluation, system requirements/design, intelligent solution, and support environment.
- `research/derived/demo1rubric(1).pdf`, pages 1–3: preprocessing, train/test preparation, analytics, ongoing ML evidence, slides and video. The saved page lists October 6 at noon; capture footer is September 21, 2026. Current LMS deadline/timezone has not been verified.
- Ontological RAG is a user-reported professor requirement. Exact acceptance language was not found in these rubric pages. Preserve the requirement without equating it with generic GraphRAG or assuming LightRAG is mandatory.

## Navigation and maintenance

- [State and conflicts](project_state.md), [next task](PLAN.md), [decisions](DECISION_LOG.md)
- [Architecture](architecture/ARCHITECTURE.md), [proposed experiments](architecture/EXPERIMENT_MATRIX.md)
- [Data dictionary, provenance, cleaning](data/DATA_PROVENANCE.md), [splits](data/DATA_SPLITS.md), [current hashes](data/artifact_hashes.csv)
- [Evaluation](evaluation/EVALUATION_PLAN.md), [experiment inventory](experiment_inventory.csv)
- [Research index](research/INDEX.md), [model naming](research/model_catalog.md), [rubric mapping](rubric_matrix.md)

Keep existing lowercase state/rubric paths canonical; do not create competing state files. Keep research/project documents out of the persona corpus. Raw sources and historical experiments stay immutable. Never change split ownership without explicit approval, fabricate metrics, reuse old slide scores as current results, or silently change accepted decisions. Update state, plan, decisions and run records as work progresses.

## Terms and open assumptions

QLoRA trains low-rank adapters through a frozen quantized base. SFT uses instruction/answer examples. Style exemplars demonstrate expression; factual evidence supports claims. Ontology-aware retrieval must actually use an explicit schema and typed relationships. A grouped split keeps related source records together. Gold generation tasks are distinct from the held-out source partition.

Unknowns: current LMS changes, exact ontology deliverable, source rights/attribution, speaker identity in transcripts, clean-corpus lineage, live Chroma contents, export parity, and validated voice metrics. Old slide performance targets are not accepted thresholds by default.
