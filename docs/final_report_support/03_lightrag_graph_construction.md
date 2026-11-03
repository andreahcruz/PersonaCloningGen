# LightRAG graph construction and use

The counts in this note belong to the frozen snapshot named in `results/model-f-handoff-v1/provenance.json`: `artifacts/lightrag/model_f/storage_expand_v1_snapshot`. They are not the 160-document pilot and not the early expansion census in `experiments/EXP-20261031-003-lightrag-index-hardening/metrics/indexing_summary.json` (that file records 25 completed documents, 266 nodes, and 217 edges).

The snapshot directory is not in Git. This audit did not open it. Node-type and edge-type breakdowns, and a representative subgraph drawing, are unavailable until a read-only copy is mounted. Figure `figures/19_graph_size.png` shows only the two published totals.

## Data preparation

`host_finetune.graph_ready` writes a derived corpus. It does not edit raw files, `data/cleaned/`, the Relabel JSONL, or the Chroma directories (`docs/architecture/LIGHTRAG_MODEL_F.md`).

Measured on 2026-10-27 from `EXP-20261003-004`:

| Quantity | Count |
|---|---:|
| Eligible train rows | 21,193 |
| Included graph documents | 13,140 |
| Eligible rows inside included documents | 20,572 |
| Quarantined documents / rows | 573 / 621 |
| Cleaned-document reconstructions | 11,917 |
| Ordered eligible-chunk reconstructions | 948 |
| Single chunks | 275 |

Eligibility for the index follows `graph_ready.eligible_train_ids`: gold-family blocks, then headline overlap with the gold rows, then exact-duplicate collapse (`UNIFIED_120_COMPARISON.md`). A validation or test sibling blocks stitching. Rows whose split is not `train` are not written into the graph.

Chunk token size on the expansion configuration is 400. Overlap is recorded as `library_default_not_overridden` in the early expansion summary (`EXP-20261031-003`). This audit did not find a numeric overlap override for the frozen snapshot. Entity guidance is `project`. Embedding prefixes recorded on that early configuration are `search_query: ` and `search_document: `.

The frozen snapshot's completed set is a subset of the 13,140 eligible documents:

| State | Documents | Source |
|---|---:|---|
| Completed graph insert | 2,713 | `completed_graph_indexed` |
| Pending | 10,269 | `docs/model_f_architecture.md` |
| Interrupted | 153 | same |
| Permanent failures | 5 | same |
| Sum | 13,140 | 2,713 + 10,269 + 153 + 5 |

Successful chunks: 9,269. A direct count of chunk-vector rows was 9,266. The gap was not repaired. The transfer note says the completed documents include 1,890 blogs and 823 LinkedIn/X posts. The pilot store was not copied in. Re-indexing will not reproduce these counts (`README.md`).

Figure: `figures/20_indexed_vs_eligible_coverage.png`.

## Graph construction

| Setting | Value | Source |
|---|---|---|
| Package | `lightrag-hku==1.5.7` | `provenance.json`, `host_finetune/requirements-lightrag.txt` |
| Extraction LLM | Ollama `llama3.1:latest` | `provenance.json` |
| Embedding | `nomic-embed-text`, 768 dimensions | same |
| Query `num_ctx` | 16384 | same |
| Nodes | 27,797 | `graph_count_source`: measured after shutdown flush |
| Edges | 47,897 | same |
| Temperature / gleaning | 0.0 and `entity_extract_max_gleaning` 1 on the early expansion config | `EXP-20261031-003` indexing summary. Confirm against the snapshot file before treating gleaning as frozen-snapshot metadata |

Entity and relationship descriptions are generated summaries. They are not source spans (`docs/model_f_architecture.md`). Duplicate-entity policy beyond LightRAG's own merge is not restated as a measured audit in the public handoff. The transfer note says a graph audit listed 25,184 findings and still needed a severity pass, and that 413 relationship repairs were a dry run and were not applied. Those findings are not a typed error rate.

No indexing wall-clock for the full 2,713-document snapshot is in the public handoff JSON. The early 25-document run recorded 1,170.8 seconds and an approximate remaining-time formula. That formula is not a schedule for the frozen snapshot.

Figure: `figures/21_graph_construction.png`.

## Retrieval modes

LightRAG 1.5.7 modes implemented in this project, from `docs/architecture/LIGHTRAG_MODEL_F.md`: `naive`, `local`, `global`, `hybrid`, `mix`. `bypass` exists in the library and is rejected. Five-mode retrieval was measured on the pilot in `EXP-20261029-005`. Precision and recall were not scored.

The official panel used **hybrid**, top_k 20, evidence budget 4, `live_store_queried: false` (`provenance.json`). Hybrid merges local entity results and global relationship results. Ranks are not cosine distances.

## Arms

| Arm | What is rendered | Where it was selected |
|---|---|---|
| F-SOURCE | Primary source claims only | Frozen pilot, 8 complete questions, status `fallback` |
| F-DESC | Claims plus derived entity descriptions | Compared in stage 2; not selected |
| F-REL | Claims plus relationship lines whose endpoints and predicate words occur in a retrieved passage | Stage-2 recommendation; official panel profile |
| F-HOP | Claims plus multi-hop paths | Pilot only; not the panel |

Pilot factual pass rates on n=8: F-SOURCE 0.500, F-DESC 0.500, F-HOP 0.500, F-REL 0.375 (`docs/model_f_experiments.md`). F-REL was worse on that sheet. Stage 2 compared 26 complete questions. Echo-adjusted pass rate was 0.3462 for F-REL and 0.3077 for F-DESC. Numeric failure rate was 0.154 for F-REL and 0.231 for F-DESC. The recommendation file selects F-REL and says the frozen pilot is unchanged. The official plan's "36-question" label does not match the scored n of 26 plus 2 incomplete ids. This note does not invent the missing 8.

The 120-prompt evaluation did not independently select F-REL. `posthoc_selection.uses_120_prompt_panel_for_selection` is false.

## How evidence reaches the generator

One topic is retrieved once and reused for four media. Claim extraction runs with the Relabel adapter off. Generation runs with the adapter on. A source-grounded claim outranks a verified relationship. A verified relationship outranks a generated description. A multi-hop path is not written as if one source stated the whole path.

Passage text is not character-truncated. If the extraction prompt does not fit 2048 tokens, whole trailing passages are omitted. Graph items that do not fit the chat window are omitted as whole items and recorded on `window_omitted`.

On the official panel the pre-judge audit recorded, per question, mean 10.7 accepted claims and 3.8 rejected claims, and mean 33.7 accepted graph edges and 75.1 rejected edges (`docs/model_f_architecture.md`).

The public flag file shows `evidence_status=verified` on 116 drafts and `extraction_failed` on 4 drafts, which are the four media of `gold_030` (`tables/model_f_failures_by_medium.csv` counts one zero-claim draft per medium). Figure: `figures/16_evidence_extraction_status.png`.

## Omissions

Two omission series exist and use different units (`figures/17_retrieval_omissions.png`):

| Series | Count | Unit |
|---|---:|---|
| Audit `passage_omitted` | 20 | cells out of 120 |
| Audit `graph_items_omitted` | 2 | cells out of 120 |
| Descriptive whole-passage events | 5 | events in `descriptive_omissions.json` |
| Descriptive graph-item events | 2 | `gold_011` blog and talk, 9 items each |

The descriptive file is not a recount of the 20 audit cells. `gold_030` is both an omitted hit and an extraction failure. `gold_011` blog and talk are the graph-window omissions. The results note says those two drafts are short title echoes.

## Quality limits

- The graph covers about 20.6% of eligible documents (2,713 / 13,140). Pending documents cannot support edges.
- Relationship descriptions can be noisy because they are generated summaries. F-REL tries to refuse edges that are not lexically supported by a retrieved passage. The audit's relationship-endpoint diagnostic found both endpoint names in the answer for 40 of 475 measurable lines (rate 0.084). That diagnostic is not proof the relationship was used correctly (`deterministic_audit_summary.json`).
- A 4-token claim-overlap diagnostic found 48 of 1,284 claims present in the answer (rate 0.037). The audit says this is not entailment.
- Lexical grounding passed 8 of 120. The audit says that gate fights the anti-copy instruction, so the low rate is a weak factuality signal.
- Graph size is not evidence that retrieval got better. Grounding pass rate on the panel is lower for Model F (0.817) than for dense Relabel + RAG (0.883).

Figure: `figures/22_retrieval_generation.png`.
