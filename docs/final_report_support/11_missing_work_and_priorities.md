# Missing work and priorities

Ordered by rubric points, then by whether the evidence already exists.

## P0 — blocks Demo 2's largest category

1. **Connect a demonstrable UI to a named model.** This checkout's `backend/main.py` is the older `generate.py` service. The React app is on `andreah/deploy-with-react` and has no LightRAG route. Acceptance: a reviewer can submit a topic in the UI, see which architecture ran, and read a draft that the same build can regenerate. If the UI stays on the GGUF path, the demo script must say it is not Model F.
2. **Record a backup video** on that UI, plus a fallback recording of the figure deck if the GPU is unavailable. Acceptance: the file is playable without the private zip's raw drafts on screen.
3. **GUI screenshots for the major use cases** (submit topic, see draft, see which model). Acceptance: images live with the report appendix and match the branch that was running.

## P1 — the report can be wrong if these are skipped

4. **Put the measured table, not the stale dashes, into the report.** `UNIFIED_120_COMPARISON.md` still says Model F was not measured. Acceptance: the report cites `architecture_comparison.json` and shows eligible versus all-draft writing quality as two numbers.
5. **Keep judge grounding and the 52/120 number audit on the same page.** Acceptance: the 2×2 table is in the results section.
6. **Do not declare a single best model.** Acceptance: the selection section is conditional on the objective. Model E remains the dense-RAG deployment candidate. Model F remains the measured graph system.
7. **Explain why accuracy, F1, and ROC are not computed.** Acceptance: one paragraph from `04_evaluation_methodology.md` is in section 6.1.
8. **State that there is no documented response-time target, and report generation latency separately from end-to-end latency.** Acceptance: 20.069 seconds is labeled generation-only, n=120, Model F only.

## P2 — correctness and honesty

9. **Resolve or footnote the training-copy denominator.** Comparison file: 0/120 failures. Audit summary: denominator 104. Flag CSV: 104 true, 16 blank, ids `gold_027`–`gold_030`. Acceptance: the report quotes both and does not pick a silent winner.
10. **Decide whether to compute Model F persona utility before any composite ranking.** Acceptance: either a new experiment directory with the overlay rules, or an explicit statement that utility is null and will not be imputed. Do not edit `EXP-20261102-001`.
11. **Re-run the CPU pytest command in `docs/teammate_handoff.md` and save the log.** The "81 passed" figure is not in this tree. Acceptance: a log with a count, a date, and the command.
12. **Representative outputs under a publication rule.** Acceptance: two or three topic ids, media labeled, copy spans and URLs removed, private zip not committed.

## P3 — stronger science, not required to stop misleading claims

13. Read-only subgraph from a runtime copy of the snapshot, with a written selection rule. Do not write to the frozen directory.
14. Severity pass on the 25,184 graph-audit findings. Do not apply the dry-run relationship repairs to the frozen snapshot.
15. Finish indexing only as a new experiment. The current hashes would no longer describe the index.
16. An F-SOURCE rerun of the 120 topics, in a new directory, if the team wants the panel itself to compare arms. The official directory must stay F-REL.
17. Store retrieval latency if a future run is timed.
18. Package freeze from the judge environment, and a GPU name if a teammate still has the run log.

## Explicitly out of scope for a cleanup pass

- Retraining the adapter.
- Rescoring by editing `unified_eval/rows.jsonl`.
- Rebuilding the graph to chase the published node count.
- Filling talk style or persona utility with zero.
- Using development 8-question or 26-question rates as if they were the 120-draft result.
