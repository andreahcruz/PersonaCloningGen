# Final report support package

This directory is a reporting layer added on 2026-11-02. It does not modify `experiments/EXP-20261102-001-model-f-unified-panel`, `results/model-f-handoff-v1/`, or `results/model-f-panel-rows-v1/`.

Start with `00_executive_summary.md`.

| File | Contents |
|---|---|
| `00_executive_summary.md` | Project, five systems, headline tradeoff |
| `01_project_retrospective.md` | Chronology and status labels |
| `02_model_inventory_and_comparison.md` | Architectures, innovation, selection |
| `03_lightrag_graph_construction.md` | Snapshot construction, arms, omissions |
| `04_evaluation_methodology.md` | Gates, judge, why F1 is not used |
| `05_results_and_interpretation.md` | Leaders by metric, both denominators |
| `06_failure_analysis.md` | Verified findings and hypotheses |
| `07_system_architecture_and_deployment.md` | Stacks and demo limits |
| `08_system_testing_and_performance.md` | Historical tests and generation latency |
| `09_demo_2_plan.md` | Demo script |
| `10_rubric_compliance_matrix.md` | Evidence against the stated rubric |
| `11_missing_work_and_priorities.md` | Ordered gaps |
| `12_teammate_handoff_and_task_allocation.md` | Workstreams and acceptance checks |
| `13_final_report_outline.md` | Section order for the course report |
| `14_presentation_outline.md` | Slide messages |
| `figures/` | PNG and SVG |
| `tables/` | CSV recomputed from saved rows |
| `scripts/build_report_figures.py` | Regenerates figures and the metric CSV |
| `evidence_manifest.csv` | Paths the narrative depends on |

Regenerate figures from the repository root:

```powershell
python docs/final_report_support/scripts/build_report_figures.py
```

The script exits without writing figures if a recomputed overall metric disagrees with `results/model-f-handoff-v1/architecture_comparison.json` by more than 0.0005. The run on 2026-11-02 matched.

## Figures

| Stem | Question it answers |
|---|---|
| 01–06 | Eligibility, completion, grounding, writing quality, task adherence, style distance |
| 04b | Eligible-only writing quality |
| 07–12 | The same family split by medium |
| 13–18 | Model F judge-versus-numbers, caps, source copy, evidence status, omissions, number flags |
| 19–20 | Graph size and document coverage |
| 21–22, 25–27 | Construction, retrieval, system, inference, deployment |
| 23–24 | Model F generation latency |

Retrieval-versus-generation bottleneck slices were not stored, so there is no stacked timing chart. Talk style is marked unavailable. Base and Relabel RAG off are omitted from the grounding chart.

## Integrity notes

- Model F persona utility is null.
- Judge grounding is not the unsupported-number audit.
- The frozen pilot selected F-SOURCE. The scored panel is F-REL. The panel did not make that choice.
- `docs/evaluation/UNIFIED_120_COMPARISON.md` is stale on the sentence that says Model F was not measured.
- A reported count of 81 passing CPU tests was not found. Recorded counts are 20, 68, 87, 156, and 562 in the notes cited in `08_system_testing_and_performance.md`.
- The LightRAG snapshot was not opened. No subgraph image was invented.
- Draft text, prompts, and retrieved passages are not copied into these files.
