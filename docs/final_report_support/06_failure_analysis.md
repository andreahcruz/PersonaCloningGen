# Failure analysis and findings

Achievements and failures are separated. A measured system that does not rank first is still a completed measurement.

## Verified findings

1. **Fine-tuning changed completion more than it raised judge writing quality.** Base completes 37.5% of drafts and scores 3.578 writing quality. Relabel RAG off completes 65.8% and scores 3.000. The training decision `DEC-006` targeted early stops caused by bad end-of-text labels. The panel is consistent with that target and also shows a writing-quality cost.

2. **Dense claim-level RAG is the strongest grounding configuration that was scored.** Relabel + RAG grounding pass rate is 0.883, above PersonaRAG (0.833) and Model F (0.817). The same system has the lowest all-draft writing quality (2.896). The deployment note already recorded this tradeoff for an earlier eligible cut. The shared comparison file shows the same direction.

3. **The partial knowledge graph did not clearly beat dense RAG.** Model F matches Relabel + RAG on completion (0.700) and has fewer training-copy failures in the comparison file (0 versus 2), a higher all-draft writing mean (3.165 versus 2.896; paired interval excludes zero), and a lower grounding pass rate. Style distance is not clearly better than Relabel RAG off. Persona utility was not computed, so the project's composite cannot be used.

4. **Graph relationships were not shown to add usable evidence at generation time.** The pilot preferred F-SOURCE. Stage 2 preferred F-REL on a 26-question echo-adjusted rate with a 4–4–18 win/tie split. The official panel did not re-decide. On that panel, both endpoint names of a relationship line appear in the answer for 40 of 475 measurable lines. That is a presence diagnostic, not a usefulness proof. Rejected graph edges averaged 75.1 per question against 33.7 accepted (`docs/model_f_architecture.md`).

5. **Evidence can be retrieved and still not control the draft.** `gold_030` has zero accepted claims and `extraction_failed`, and the four drafts remained in the panel. The results note says the blog and talk still state revenues. Draft text is not copied here. Mean accepted claims on the panel were 10.7 per question, so the failure is not "retrieval always empty." It is a utilization and abstention failure on at least this topic.

6. **Unsupported quantities are common and are not the same as judge grounding failures.** 52/120 cells and 185 quantities (`deterministic_audit_summary.json`). 43 of the 52 still passed judge grounding. Blog has 19 of the 30 number-flagged cells, more than LinkedIn or X. The results note attributes the sharp cases to generation, including topics where retrieval had returned on-topic claims. That attribution is the audit's diagnosis. This package does not re-read the drafts.

7. **Short outputs hit the token cap.** 36 cells have output tokens equal to the medium cap, and 36 fail mid-sentence completion. The intersection is 31 (`tables/model_f_cap_and_completion.csv`). By medium, at-cap counts are blog 0, LinkedIn 17, X 19, talk 0. Mid-sentence counts are 0, 17, 18, and 1. X's budget is 160 new tokens and LinkedIn's is 320 (`docs/model_f_architecture.md`). The cap is a verified correlate of the short-form completion failures. Five mid-sentence failures are not at the cap, so the cap is not the only completion mechanism.

8. **Source copying and instruction leakage are smaller than the number problem and are real.** 7/120 fail the 12-word retrieved-passage threshold. 10/120 have a span of at least 8 words. The audit cites `gold_012` blog and talk at 19 words and says the span remains in the local generation files. Instruction or marker leak is 2/120. Citation-marker leak is 2/120. The judge training-copy gate does not catch retrieved-passage copying; its Model F pass rate is 1.0 on the comparison file.

9. **Context fitting drops evidence.** Audit passage-omitted cells: 20/120. Graph-item-omitted cells: 2/120, identified in the descriptive file as `gold_011` blog and talk with 9 omitted graph items. The generator window is 2048 tokens. Graph queries use 16,384. The results note says those `gold_011` drafts are short title echoes. Whole passages are dropped rather than character-cut, which is a design choice recorded in `docs/model_f_architecture.md`.

10. **The index is partial.** 2,713 of 13,140 eligible documents were inserted. Pending, interrupted, and failed documents are absent. A graph-quality finding count of 25,184 was recorded without a severity pass (`MODEL_F_DEPLOYMENT_TRANSFER_20261102.md`). Coverage limits what the retriever can return. Coverage does not, by itself, explain a grounding gap versus the dense index, because the dense index is a different corpus representation (21,193 chunk documents) and a different ranker.

11. **Latency was measured for generation, not for the full pipeline.** Model F mean generation time is 20.069 seconds (median 23.732, minimum 1.453, maximum 51.184, n=120). By medium the means are blog 24.0 s, LinkedIn 19.7 s, X 9.6 s, talk 27.0 s. No retrieval-time component is in the judge row. No response-time target is written in `docs/evaluation/METRIC_MAP.md` or the Model F handoff. Other architectures have null latency, so a speed ranking is impossible.

12. **Speaker attribution failed where it applied.** The results note says 8 of 8 applicable cells failed, with 112 not applicable. Role-unsupported cells: 1/120.

## Hypotheses, labeled as hypotheses

- F-REL may add tokens that crowd the 2048 window and contribute to the 20 passage-omission cells. The audit records omissions. It does not isolate F-REL as the cause against an F-SOURCE rerun of the same 120 topics. That rerun was intentionally not done on the official panel.
- Number errors may come from the Relabel prior rather than from bad edges, because the audit names cases with on-topic claims. A controlled ablation that removes only the relationship lines on the same retrieval cache was not the official panel.
- X and LinkedIn completion may move if the new-token budget rises, if the prompt asks for a shorter ending, or if the cap is treated as a hard format the model is trained to finish inside. None of those changes was measured here.
- Finishing the remaining 10,269 documents might change retrieval. It might also add noise. Either outcome is unmeasured. Re-indexing would be a new experiment and would not preserve the frozen hashes.

## What each technique was worth on this evidence

| Technique | Supported conclusion |
|---|---|
| Fine-tuning (relabel) | Raises completion relative to Base and moves style distance down. Lowers writing quality. |
| Standard dense RAG plus claim checks | Best grounding pass rate. Does not raise writing quality above the same adapter with RAG off. |
| Graph-augmented RAG | Implemented, partially indexed, and scored. Does not clearly beat dense RAG on grounding or eligibility. Writes at a higher all-draft quality than dense RAG and a lower quality than the unadapted models. |
| Evidence verification | Removes unsupported spans from the prompt and still lets generation add numbers. Zero-claim topics still produced drafts. |
| Prompt and context budgets | The 2048 window, medium caps, and whole-item omission are visible in the failure counts. They are part of the system, not a post-hoc excuse. |
| Decoding | Temperature 0.7 and the listed penalties are recorded. No decoding ablation is in the official comparison file. |

## Achievements to keep in the report

- A shared 120-cell schema across five architectures, with a blinded judge from a different family than the generator.
- A reproducible relabel adapter and a documented reason for it.
- A train-only dense index that refuses the older mixed collection.
- A frozen LightRAG snapshot with published node, edge, and coverage counts, and an arm-selection history that does not pretend the test panel chose the arm.
- Public score tables that keep drafts, prompts, and passages out of the public repository.
- CPU tests recorded at several dates (see the testing note). This audit did not re-run them.

## Limits to keep next to those achievements

- No single winning model.
- Judge grounding is not numeric factuality.
- Talk style and Model F persona utility are missing.
- The demo UI on the deploy branch is not connected to Model F.
- The graph is a fifth of the eligible corpus.
- The panel is not an untouched held-out corpus split.
