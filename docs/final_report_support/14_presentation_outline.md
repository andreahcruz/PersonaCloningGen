# Presentation outline

Demo 2 deck. Timings assume about 12 minutes of slides and 3 minutes of demo. If the GPU is down, slides 7–12 carry the result and slide 14 uses the saved table.

| Slide | Main message | Suggested visual | Evidence |
|---|---|---|---|
| 1. Problem | Generate four media in a Lemkin register and measure completion, grounding, and writing separately | Title | `docs/PROJECT_CONTEXT.md` |
| 2. Data | 30 gold topics, four media, train-only retrieval, gold text stays out of the prompt | Gold-hash callout | `provenance.json` |
| 3. Evolution | From an unadapted baseline, through a label repair, to dense claims, then a partial graph | Five-box timeline | `01_project_retrospective.md` |
| 4. Models | Five scored systems. Two changes are methodological: relabel supervision, and passage-checked graph lines | Comparison matrix | `02_model_inventory_and_comparison.md` |
| 5. Graph construction | 2,713 of 13,140 documents entered the frozen graph | `figures/20_indexed_vs_eligible_coverage.png` | `03_lightrag_graph_construction.md` |
| 6. Arm history | Pilot froze F-SOURCE. Stage 2 recommended F-REL. The test panel did not choose | Small table of n=8 and n=26 | `docs/model_f_experiments.md` |
| 7. Eligibility and completion | Relabel systems finish more drafts than Base. Model F ties dense RAG at 70% | `figures/01_eligible_drafts.png`, `figures/02_completion_pass_rate.png` | comparison JSON |
| 8. Grounding | Dense Relabel + RAG leads the retrieval systems. Base is omitted on purpose | `figures/03_grounding_pass_rate.png` | comparison JSON |
| 9. Writing quality | Unadapted drafts score higher and finish less often. Show the all-draft chart, and say eligible Model F is 3.27 on 67 drafts | `figures/04_writing_quality_all_drafts.png` | comparison JSON and summary JSON |
| 10. Style | Lower distance is closer. Model F is lowest. The gap versus Relabel off is not a clear separation. Talk is missing | `figures/06_style_distance.png` | paired CSV |
| 11. Where it breaks | LinkedIn and X hit the token cap | `figures/14_token_cap_by_medium.png`, `figures/09_completion_failures_by_medium.png` | flag CSV |
| 12. Two factuality checks | 43 drafts pass the judge and fail the number check | `figures/13_grounding_vs_unsupported_numbers.png` | crosstab CSV |
| 13. Pipeline | One retrieval, claim check, F-REL lines, 2048-token fit, Relabel decode | `figures/22_retrieval_generation.png` | architecture note |
| 14. Live demo | Submit one topic only if the running service is the model being claimed | Running UI, or the per-medium table if not | `09_demo_2_plan.md` |
| 15. Achievements and limits | Measured five-way panel, frozen partial graph, no single winner, UI not wired in this branch | Three columns | `06_failure_analysis.md` |
| 16. Next | UI route, screenshots, redacted examples, do not rebuild the frozen graph for the demo | Checklist | `11_missing_work_and_priorities.md` |

Speaker rule: if a number is on the slide, the denominator is on the slide. If a bar is missing, say "not scored" rather than leaving a zero-height bar unexplained. Figures 10 and 12 already omit non-applicable grounding architectures or mark talk style unavailable.
