# Model F offline evaluation infrastructure

Status of this harness: **IMPLEMENTED** and **TESTED** on synthetic fixtures. No architecture comparison has been **MEASURED**. This page does not replace `docs/evaluation/METRIC_MAP.md` (PUK-28, 2026-10-24). It adds a scorer that can sit on saved traces from Model E and the existing Model F profiles.

The live LightRAG expansion in `artifacts/lightrag/model_f/storage_expand_v1` was not queried. No generator, embedding job, or GPU measurement was run.

## 1. Audit of what already exists

| Area | Location | What it does | Decision |
|---|---|---|---|
| Graded retrieval | `host_finetune/retrieval_eval.py` `evaluate_query` | Precision@k, recall@k, MRR, nDCG@k. Recall only when `relevance_scope` is `corpus`. Unjudged documents are not treated as relevant. Precision and nDCG stay unset when fewer than k hits were returned. | Reuse. This is the ranked-retrieval implementation. |
| LightRAG observation | `host_finetune/lightrag_eval.py` | Hit counts, unique sources, duplicate sources, train attribution, mean latency. The module says these are not precision or recall. | Reuse as a diagnostic observation. Do not treat its hit counts as retrieval quality. |
| LightRAG mode cells | `host_finetune/retrieval_contract.py` | E0 dense plus F1–F5 for naive, local, global, hybrid, and mix. | Keep for mode studies. The context-profile matrix in this page is a different comparison. |
| Factual checks | `host_finetune/factual_reliability.py` `assess_factual` | Quantity, role, speaker, instruction leak, and source-copying checks. The method string says this is not lexical entailment. An empty claim list fails closed with `no_accepted_factual_evidence`. | Reuse for answers that have claims. Do not use that fail-closed result as the faithfulness score of an abstention. |
| Numbers | `host_finetune/numeric_grounding.py` | Deterministic quantities against allowed numbers from claims and extra texts. | Reuse. Report it separately. A rate that includes answers with no quantity would look better than it is, so the aggregate counts only answers that state a quantity. |
| Source copying | `host_finetune/rag_source_copying.py` | Called from `assess_factual`. | Reuse. |
| Unified harness | `host_finetune/unified_eval.py` | Completion, format, grounding, copying, persona utility, BLEU, ROUGE. DEC-009 and METRIC_MAP keep BLEU, ROUGE, BERTScore, and the gold-rubric overall as diagnostics. Style distance is length-confounded. Stance and claim-support scores sit near the ceiling. Length bands punish a short abstention. | Do not call this from the Model F comparison scorer. Do not rank profiles by persona utility or n-gram overlap. |
| LLM judges | `host_finetune/claim_judge.py`, `writing_quality.py` | One writing judge and one claim judge. | Not the sole factual measure. This harness does not call them. |
| Gold overlap | `host_finetune/eval_gold_rag.py` | Token overlap and F1 against a reference blog. | Diagnostic only. Overlap with a reference is not faithfulness. |
| Older RAG scores | `host_finetune/eval_rag.py` | BLEU, ROUGE, BERTScore, RAGAS, G-Eval. | Do not rerun. METRIC_MAP already retired G-Eval with a Llama-family judge. |
| Voice distance | `host_finetune/voice_distance.py` | Surface-style distance to held-out posts. | Out of this factual harness. Length confounds it. |
| Generation leakage screen | `host_finetune/model_f_context/leakage.py` | Configurations A–D and a held-out split screen. D is multi-hop and already includes descriptions. | Leave that module as the generation-time screen. The experiment matrix below has five arms so descriptions can be scored on their own. |
| Gold development file | `data/openai_ft/lemkin_gold_eval.jsonl` | 30 rows, all `blog_draft`. Fields are audience, cta, expected_facts, format, goal, gold_reference, id, k, rubric_must_haves, topic. | Generation development set already used for prompt work. Not the corpus test split. Not a graded qrel. Not modified. |
| Review questions | `host_finetune/pilot_questions.json`, `graph_relationship_questions.json` | Kinds include simple fact, author position, relational, multi-document, conflict, conditional advice, attribution, and unsupported. Review notes say a preselected source id is not precision or recall. `pilot_retrieval.py` must not send those ids to the retriever. | Development questions only. |

## 2. Metric specification

A metric is omitted from the scorer when it repeats another metric, rewards a longer answer, or rewards retrieving more text. Null means the input was missing. Null is not stored as zero.

### Retrieval

| Metric | Purpose | Needs | Automated? | In the scorer | Limitation |
|---|---|---|---|---|---|
| Precision@k | Share of the top k that a graded qrel marks relevant | Graded qrels, k hits | Yes | Yes, via `retrieval_eval`, k of 1 and 4 | Unset when fewer than k hits exist. Useless without grades. |
| Recall@k | Share of known relevant documents found in the top k | A corpus-scope qrel, not a judged pool | Yes | Yes, only when qrels are present | A pool of preselected ids is not the relevant set. |
| MRR | How high the first relevant hit sits | Same grades as precision | Yes | Yes | Ignores every relevant hit after the first. |
| nDCG@k | Whether higher grades sit above lower grades | Graded qrels | Yes | Yes, at 4 | Redundant with precision when every relevant grade is the same. Kept because grades can differ. |
| MAP | Average precision over the list | Grades | Yes | No | On a list of four, it repeats MRR and precision. |
| Designated-source coverage | Overlap with a reviewer's preselected ids | That id list | Yes | Yes, labeled `overlap_with_preselected_ids_not_recall` | This is not recall. |
| Evidence coverage as "more hits" | Count of returned hits | Hits | Yes | Count only | More hits are not a better retrieval. |
| Source diversity | Unique source ids divided by hit count | Source ids | Yes | Yes, with the unique count | One hit scores 1.0. It is not a ranking metric. |
| Provenance completeness | Share of hits that have an id, passage text, and train split or a provenance string | Hit records | Yes | Yes | A complete label does not make the passage relevant. |
| Relevant-document coverage | Same question as recall@k | Corpus qrels | Yes | Folded into recall@k | A second name would be a duplicate. |

### Generation

| Metric | Purpose | Needs | Automated? | In the scorer | Limitation |
|---|---|---|---|---|---|
| Faithfulness to retrieved evidence | Whether the draft survives the existing quantity, role, speaker, and copying checks | Claims or retrieved text | Automated, not an entailment model | Yes, when evidence or claims exist | Not an LLM judgment and not proof the world is as the draft says. |
| Factual correctness against the world | Agreement with a source outside the corpus | Human or external evidence | No | No | METRIC_MAP gap G10. An LLM judge is not a substitute. |
| Unsupported-claim rate from a judge | Share of claims a model calls unsupported | Judge output | No, and not run here | No | The saved judge scores are high and flat. |
| Unsupported numeric count | Quantities in the answer that the claims do not allow | Answer plus claims | Yes | Yes, separate from faithfulness | Answers with no quantity are left out of the rate. |
| Citation precision and recall | Whether an explicit citation list points at supporting ids | A citation list the generator emits | Yes, only then | Status `not_emitted` otherwise | The prompt forbids evidence tags. Scoring tags in the prose would reward a leak. |
| Citation completeness | Same as citation recall | Same list | Yes | Folded into citation recall | A second name would be a duplicate. |
| Citation-marker leak | Prose contains `[E1]`, `[R1]`, `[P1]`, `[D1]`, or a `gd_` id | Answer text | Yes | Yes | A leak is a failure, not a citation. |
| Contradiction rate | Both sides of an annotated forbidden pair appear | `contradicted_pairs` | Yes, when annotated | Otherwise `not_annotated` | Cannot be inferred from fluent prose. |
| Abstention | No forbidden assertion, no invented quantity, no marker leak | `abstention_expected` and optional forbidden strings | Yes, when the question is marked | Yes | A short abstention is a pass on this check. The unified length band is not applied. |
| BLEU, ROUGE, BERTScore | N-gram or embedding overlap with one reference blog | A reference | Yes | No | They can favor length and copying. A missing reference stays unset. |
| Word count | Covariate | Answer | Yes | Recorded, not scored | A longer answer is not better. |

### Efficiency

| Metric | Purpose | Needs | In the scorer | Limitation |
|---|---|---|---|---|
| Retrieval, generation, and end-to-end seconds | Cost of the recorded run | Timestamps already on the trace | Yes | Missing clocks stay null. |
| P50 and P95 | The middle and the slow tail of those clocks | The scored rows | Yes | Describes the recorded questions only. |
| Prompt and output tokens | Size of the recorded call | A token count on the trace | Yes | The indexer estimates tokens as characters divided by four. A recorded tokenizer count should set `tokens.source`. |
| Context utilization | Prompt tokens divided by the stated budget | Both numbers | Yes | Filling the budget is not a retrieval win. |
| Throughput | Scored rows divided by summed end-to-end seconds | Those clocks | Yes | Excludes failed rows. Not a quality metric. |
| GPU memory | Bytes already written on the trace | A prior measurement | Passed through | This harness does not query a GPU. Absent values stay absent. |

No composite score is computed. Architecture-only fields, including graph path counts, stay on the raw record and are not averaged into the cross-architecture table.

## 3. Result schema

Schema version `eval-bench-v1`, validated by `host_finetune.eval_bench.schema.validate_record`.

Required: `experiment_id`, `architecture`, `question_id`, `retrieval_config`, `generation_config`, `retrieved_source_ids`, `evidence`, `answer`, `latency`.

Optional, and kept when present: `model_version`, `medium`, `uses_graph`, `provenance`, `reference_answer`, `qrels`, `review_source_ids`, `claims`, `grounding_texts`, `citations`, `supporting_ids`, `abstention_expected`, `forbidden_assertions`, `contradicted_pairs`, `tokens`, `context_budget_tokens`, `gpu_memory_bytes`, `error`.

A dense system sets `uses_graph` false and omits relationship fields. Extra keys are preserved so a later architecture can add its own trace fields without a second schema. Raw records are written to `results.jsonl` before any aggregate, so a metric can be recomputed.

Duplicate key is `(experiment_id, architecture, question_id, medium)`. The second row is `duplicate_prediction` and is left out of the aggregate. A recorded `error` is classified and not scored as a success. Evidence whose `split` is `validation` or `test` is `held_out_evidence` and is not scored.

## 4. Experiment matrix

Same question ids, the frozen Relabel adapter, and the decoding already recorded by `compare_adapters`: temperature 0.7, top_p 0.9, repetition penalty 1.15, seed base 42. Evidence budget is four passages on every arm. Model F channel budgets stay the ones in `model_f_context.budget`. Dense `k` counts passages. LightRAG `top_k` counts entities or relationships, and `chunk_top_k` counts chunks. Both facts are recorded. They are not the same score.

| Arm | Architecture | Profile | Retrieval | Prompt family |
|---|---|---|---|---|
| E-DENSE | `model_e` | none | dense | `factual-reliability-v1` |
| F-OFF | `model_f_off` | off | LightRAG hybrid | `factual-reliability-v1` |
| F-SOURCE | `model_f_source` | source | LightRAG hybrid | `model-f-graph-context-v1` |
| F-REL | `model_f_relationships` | relationships | LightRAG hybrid | `model-f-graph-context-v1` |
| F-DESC | `model_f_descriptions` | descriptions | LightRAG hybrid | `model-f-graph-context-v1` |
| F-HOP | `model_f_multihop` | multihop | LightRAG hybrid | `model-f-graph-context-v1` |

Paired comparison uses the same `question_id` and the same medium, then a bootstrap of the faithfulness difference with seed 298. The interval describes those questions. It is not a winner score.

The ids live in `host_finetune/model_f_context/registry.py`. `leakage.EVALUATION_CONFIGURATIONS` and `eval_bench.EXPERIMENT_MATRIX` both read that tuple. E-DENSE is not the `off` profile. Both leave the graph-context callback empty, and E-DENSE uses the dense index while F-OFF uses LightRAG. F-DESC is descriptions. F-HOP is multi-hop.

EXP-20261031-001 is a real saved run, and its letters are a different design. Its A is hybrid passages, its B adds the older relationship block, its C is a 400-character matched dense preview, and its D is a blog token pad. None of those cells used these six ids. `eval_bench.historical` scores them under their own architecture names.

The corpus validation and test splits stay closed for profile choice, prompt edits, and budget choice.

## 5. Benchmark coverage

The 30-topic gold file can support a blog-draft generation check. It has no label for entity linking, a two-hop chain, a comparison of two dates, or abstention. `expected_facts` is a rubric hint, not a qrel. `gold_reference` must not be used to pick an architecture.

The development retrieval sheet is `host_finetune/benchmarks/dev_retrieval_benchmark.jsonl`. Its grades are null. The annotation rules are in `docs/evaluation/RETRIEVAL_ANNOTATION_GUIDE.md`. Preselected ids are candidates. The withheld review notes are not part of the annotator file.

The two older review files cover simple fact, relational questions, more than one document, conflict, conditional advice, attribution, and unsupported questions. Their source ids were chosen by a reviewer. Overlap with that list is the designated-source field, not recall. None of them is a graded two-hop item where the endpoint claim is unsupported by any single document.

`host_finetune/eval_bench/coverage.py` records those gaps and three synthetic question ids (`syn_hop`, `syn_abstain`, `syn_temporal`). Each is marked `ground_truth: not_human_validated`. They exercise the scorer. They are not reference answers.

## 6. How to run it later

From the repository root, after indexing has finished and a separate job has written a predictions file. This command still does not open the live store. A path containing `storage_expand_v1` is refused.

```text
host_finetune\.venv\Scripts\python.exe -X utf8 -m host_finetune.eval_bench ^
    --predictions path\to\predictions.jsonl ^
    --out experiments\EXP-YYYYMMDD-NNN-name\metrics ^
    --experiment-id EXP-YYYYMMDD-NNN ^
    --pair-left model_e ^
    --pair-right model_f_source
```

The predictions file is JSONL in the schema above. The command writes `results.jsonl`, `metrics.json`, `per_question_analysis.jsonl`, `failures.jsonl`, `experiment_manifest.json`, and `report.md`. Register a real `experiments/EXP-*` directory only for a run that actually executed, and record the manifest fields required by `experiments/README.md`.

Tests, which do not load a model:

```text
host_finetune\.venv\Scripts\python.exe -X utf8 -m pytest tests/test_eval_bench.py -q --tb=short
```

## 7. What waits for the indexer to exit

- Any new query against `storage_expand_v1`.
- Passing entity descriptions through `lightrag_backend.query_documents`. That file is on the indexer import path, so arm D cannot be filled from a live payload until the indexer exits.
- Generation with `--model-f-context` against the finished blog graph.
- A paired generation run, confidence intervals on real questions, and a registered experiment directory.
- Building graded qrels. Until those exist, precision, recall, MRR, and nDCG stay null on real questions. That is the correct output.
