# Demo 2 plan

Demo 2 is scored here from the criteria in the task statement: Machine Learning Modeling (3), Machine Learning Evaluation Results (3), Web-based System Design and Demo UI (4), Presentation (2.5). Total 12.5. This checkout can support the modeling and evaluation talks. It cannot support a claim that the web demo already runs Model F.

## Recommended story

The live message is the tradeoff, not a winner announcement. Relabel training fixed a completion failure mode and lowered writing quality. Dense RAG has the best grounding pass rate and the lowest writing quality. LightRAG was built, partially indexed, and scored, and it did not clearly beat dense RAG.

## Slide outline

Each slide has one message, one visual, and a source. Do not paste draft text or retrieved passages.

| # | Message | Visual | Source |
|---|---|---|---|
| 1 | The system writes Lemkin-style blog, LinkedIn, X, and talk from a topic, and the scorecard separates voice, grounding, and completion | Title only | `docs/PROJECT_CONTEXT.md` |
| 2 | The gold panel is 30 topics by 4 media, 120 drafts, judge Qwen 14B at temperature 0 | Small schema diagram | `provenance.json` |
| 3 | Training data is a relabeled continuation set because truncated posts had been labeled as finished | One sentence from `DEC-006`, no raw text | `docs/DECISION_LOG.md` |
| 4 | Five scored systems, two of them methodological changes: relabel QLoRA, and source-checked graph context | Matrix from `02_model_inventory_and_comparison.md` | comparison JSON |
| 5 | The graph is real and partial: 2,713 of 13,140 documents, 27,797 nodes, 47,897 edges | `figures/20_indexed_vs_eligible_coverage.png` | `provenance.json` |
| 6 | F-SOURCE won the frozen pilot. F-REL was an engineering choice afterward. The 120-prompt panel did not vote | Arm table | `docs/model_f_experiments.md` |
| 7 | Eligibility and completion | `figures/01_eligible_drafts.png`, `figures/02_completion_pass_rate.png` | comparison JSON |
| 8 | Grounding only where retrieval exists. Relabel + RAG leads. Model F is third among the three | `figures/03_grounding_pass_rate.png` | comparison JSON |
| 9 | Writing quality on all 120 drafts. Unadapted models lead. Say the eligible mean is a different chart | `figures/04_writing_quality_all_drafts.png` | comparison JSON |
| 10 | Style distance, lower is closer, talk excluded, not authorship | `figures/06_style_distance.png` | comparison JSON |
| 11 | Short posts hit the cap. 31 of 36 mid-sentence stops are also at the token cap | `figures/14_token_cap_by_medium.png` | flag CSV |
| 12 | Judge grounding and the number audit disagree. 43 drafts pass the judge and fail the number check | `figures/13_grounding_vs_unsupported_numbers.png` | joined CSVs |
| 13 | Pipeline the demo can narrate even if the GPU is offline | `figures/22_retrieval_generation.png` | architecture note |
| 14 | Live generation or an explicit fallback | If the snapshot is mounted, one new-directory smoke. Otherwise show the saved per-medium table and say the service is not wired | `README.md`, `07_system_architecture_and_deployment.md` |
| 15 | What is achieved, what is limited, what is next | Three columns from `06_failure_analysis.md` | this package |

## Live demo script

1. State which machine and which artifact hashes are in the room.
2. If the web app from `andreah/deploy-with-react` is running, say which backend it calls before typing. If it calls the GGUF route, do not call the output Model F.
3. If Model F CLI is running, generate into a new directory, one topic, four media, and show latency from the run log beside the historical 20-second mean.
4. Open `figures/03_grounding_pass_rate.png` and `figures/13_grounding_vs_unsupported_numbers.png` after the draft, so a fluent draft is not left unqualified.
5. If generation fails, switch to the saved panel. Do not repair a draft on stage.

## Backup

Record the deploy-branch UI on the branch that contains it, and record a screen capture of the figure deck. This checkout does not contain that UI, so the video is a teammate task. Keep the private zip off the public recording if the drafts contain copied spans or URLs. The export manifest says 7 source-copy failures, 2 cells with URLs, and 2 cells with prompt markers.
