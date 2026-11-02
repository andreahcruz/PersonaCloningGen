# 120-prompt architecture comparison

**Status of this protocol: IMPLEMENTED and TESTED on CPU fixtures. Model F on this panel is not MEASURED.** No draft was generated, no judge was run, and the live LightRAG store was not opened.

The common comparison is the existing 30-topic gold file crossed with blog, LinkedIn, X, and talk: 120 cells, schema `unified_eval.v1`. The 36-question LightRAG run stays a development ablation. It chooses the Model F profile. It is not the architecture comparison, and it is not an untouched final test.

This repository does not label the saved architectures A–E. The letters A–D inside `EXP-20261031-001` are pilot arms on a different question set. Model E, in the six-arm registry, is the dense Relabel + RAG system already scored as "Relabel + RAG". Model F is the LightRAG profile selected later.

## 1. Historical audit

Gold file for every 120-cell run below: `data/openai_ft/lemkin_gold_eval.jsonl`, SHA-256 `d7c4f997899e28f28eb3ea9aa3417857da6f0666acbf72fd8ac31422034e520e`. Question ids are `gold_001` through `gold_030`. The file stores one row per topic. The 120 cells are those topics times four media. The reference text in that file is an evaluator reference. It is not a corpus document and it is not passed into the Model F prompt.

| Architecture | Experiment | Saved cells | Generator | Retrieval | Comparability |
|---|---|---|---|---|---|
| Base | `EXP-20261001-010`, scored in `EXP-20261004-008`. Rows: `docs/evaluation/examples/data/architecture/kevin_base_llama31_rows.jsonl` | 120, seed 42 | Llama 3.1 8B, no adapter | none | Exact on the shared scorer and gold file. One seed. Grounding is not applicable. |
| PersonaRAG stand-in | `EXP-COLAB-personarag-newdata-seed42`. Rows: `docs/evaluation/examples/data/architecture/colab_personarag_seed42_rows.jsonl` | 120, seed 42 | Base Llama 3.1, no adapter | Chroma `lemkin_train_only`, 21,193 docs, top-4, `nomic-embed-text` | Approximate. This is base Llama plus train-only retrieval through the v1 script, not the semester-1 `generate.py` pipeline. One seed. |
| Relabel, RAG off | Seeds 42/43/44 from `EXP-20261004-003`, `EXP-20261004-005`, `EXP-20261004-006`, scored in `EXP-20261004-008` | 120 per seed | Relabel adapter `host_finetune/output/lemkin_lora_relabel` on `unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit`, 4-bit | none | Exact within the Relabel family and the shared scorer. Seed 42 rows: `docs/evaluation/examples/data/relabel_rag_off_rows.jsonl`. |
| Relabel + RAG (Model E) | `EXP-20261005-003` seed 42. Colab seeds 43 and 44 | 120 per seed. Seed 42 has `metrics/rows.jsonl` | Same Relabel adapter | Frozen factual dense RAG v1, same 21,193-doc index, top-4, atomic evidence | Exact for the saved seed-42 rows. Seeds 43 and 44 retrieved the same passages as seed 42 on 29 of 30 topics. Evidence extracted from those passages differs on 15 of 30 topics because the machines and Python versions differ. |
| Fixed-style ablation | `EXP-20261030-012` | Style-off 120 drafts. Style-on 90 drafts plus 30 unsupported talk records | Same Relabel adapter. Prompt contract `factual-reliability-v1` | Chroma v2 shadow `lemkin_train_only_v2_shadow`, 21,193 docs, k=4. Claims frozen once per topic | A style on/off ablation, not a new architecture. No `metrics/rows.jsonl`. Talk style-on is unsupported because `youtube_jason` exemplars are empty. Approximate if converted later. |
| Pilot graph | `EXP-20261031-001` | 5 topics, arms A/B/C, plus blog-only D. Not 120 cells | Same Relabel adapter. Prompt contract `factual-reliability-v1` | LightRAG `storage_pilot160`, 152 processed documents. Arm C uses a different dense preview | Not comparable to the 120-prompt panel. Different questions, corpus, and prompt. Do not rescore it as if it were that panel. |

Shared decoding where a manifest records it: temperature 0.7, top_p 0.9, repetition penalty 1.15, seed base 42, seed `42 + gold_index * 4 + medium_index`. Output budgets are blog 768, LinkedIn 320, X 160, talk 768. The generation window used by the current generator is 2048 tokens.

Judge on the scored 120-row files: `qwen2.5:14b`, temperature 0, blinded, candidate family llama. `score_saved_experiment.py` still refuses every trace file except the frozen `EXP-20261005-003` generation file (`266507aca9...`). That gate stays. Model F is not scored by disabling it.

`evaluate_row` writes completion, copying, format, and lexical grounding gates, style and perspective distances, task adherence, and diagnostic BLEU/ROUGE. Writing quality is unavailable until `apply_judge_overlay`. Persona utility stays null unless the draft is eligible and style, perspective, and writing are all selector metrics with values present. `eval_tools.compare` pairs rows on `(id, medium)`. Missing numbers are skipped. They are not treated as zero. The common-gate view is completion, copying, and format. Grounding is excluded from that view because retrieval-off runs do not have it. Persona utility is not the conclusion. The paired primary family is writing quality, blog style distance, stance consistency, and task adherence.

What the saved rows can and cannot do:

- Base, PersonaRAG, Relabel off, and Relabel + RAG already have `unified_eval.v1` rows. They can be compared again with `eval_tools.compare`. They do not need to be regenerated for that comparison.
- Latency was not stored on those rows. The latency mean is null, not zero.
- Style distance is present for blog, LinkedIn, and X, and missing for all 30 talk rows. Talk style is null, not zero.
- Claim support is present for the retrieval runs and missing for Base and Relabel off.
- Persona utility is populated only for a small set of eligible blog drafts. It is null on LinkedIn, X, and talk in these saved files.
- `EXP-20261030-012` can be converted later into a new directory. The conversion was not run. Talk style-on remains an unsupported cell.
- `EXP-20261031-001` cannot be turned into the 120-cell panel by rescoring. Those drafts answer different questions.

## 2. Frozen comparison contract

- Panel: `gold_001`–`gold_030` × `blog`, `linkedin`, `x`, `talk`.
- Gold hash: `d7c4f997899e28f28eb3ea9aa3417857da6f0666acbf72fd8ac31422034e520e`. `write_plan` refuses a different hash.
- Schema: `unified_eval.v1`, one row per `(id, medium)`.
- Corpus eligibility for the LightRAG index is the existing rule in `graph_ready.eligible_train_ids`: gold-family blocks, then headline overlap with the gold rows, then exact-duplicate collapse. Query time uses a verified snapshot of that index. The gold reference string is not an indexed document and is not inserted into the prompt.
- Model F retrieval, once the snapshot exists: LightRAG hybrid, top_k 20, evidence budget 4, `num_ctx` 16384, Relabel adapter, 4-bit, the decoding and seed formula above, prompt family `model-f-graph-context-v1`.
- The profile argument is one of `F-SOURCE`, `F-REL`, `F-DESC`, `F-HOP`. The plan records that this panel did not select it.
- Claims are frozen from the retrieved evidence before the draft, and reused across media, by the same evidence cache the four-arm runner uses.
- Metrics come from `evaluate_row` and, later, the same `apply_judge_overlay` configuration. A metric that was not computed stays null.
- Historical scores stay in their original experiment directories. Newly scored Model F rows go to `experiments/EXP-20261102-001-model-f-unified-panel`.
- Style distance applies to blog, LinkedIn, and X. Talk has no style value in the saved rows. Fixed-length style is a separate side file and is not imputed into `style_distance`.
- Numeric correctness remains the existing unsupported-number diagnostic. It is not a field in `eval_tools.rows.METRIC_PATHS`, so the comparison matrix leaves it null instead of inventing a rate.
- Latency and token counts for Model F come from the cell record (`latency.generation_seconds`, prompt tokens, output tokens). They are not copied from older runs.

A CPU token count of the empty graph prompt plus the fitted evidence allowance put the longest gold topic, `gold_016` blog on a graph arm, at 2025 tokens against the 2048 window. The four-arm budgets were not changed. `unified-run` still calls the preflight and refuses a cell that does not fit.

## 3. Model F compatibility

`host_finetune/model_f_experiment/unified.py` reuses the four-arm retrieval freeze, evidence freeze, arm renderer, and `compare_adapters.generate`. It does not add a second retrieval stack. The four-arm experiment id and arm list are unchanged.

| Command | What it does | Run now? |
|---|---|---|
| `unified-plan` | Writes `unified_plan.json` for an explicitly named profile and a snapshot path. Refuses the live store and any historical experiment directory. | Safe. It does not retrieve. |
| `unified-run` | Verifies the snapshot, retrieves once per topic, freezes claims, and generates four media. | Not run. The verified partial snapshot exists. This panel is still pending local evaluation. |
| `unified-score` | Runs `evaluate_row` over saved cells with the same reference pack and copy index. Writing quality and persona utility stay null. | No. There are no Model F panel cells yet. |

`unified-score` writes `metrics/rows.jsonl` in the schema `eval_tools.compare` already reads. The judge overlay is a later step in that new directory. It is not `score_saved_experiment`, because that script is pinned to the October 5 trace file.

## 4. Leakage and overlap

The 120 topics are not an input to profile selection. `require_profile` accepts only an arm the caller names, and the plan field `profile_selected_by_this_panel` is false.

Overlap between `host_finetune/benchmarks/dev_retrieval_benchmark.jsonl` (36 questions) and the 30 gold topics:

- Shared ids: none. Development ids look like `q_brex_price`. Panel ids are `gold_001`–`gold_030`.
- Pairs that share two or more content words of length 5 or more: none.
- Pairs that share one such word: 30. The words are generic (`sales` 12, `acquisition` 5, `capital` 3, `person` 3, `director` 2, and one each of `companies`, `month`, `managing`, `leave`, `customers`).

| Development question | Gold id | Shared word |
|---|---|---|
| `q_brex_price` | `gold_004` | capital |
| `q_dev_fact_acquirer` | `gold_005` | acquisition |
| `q_dev_fact_acquirer_target` | `gold_004` | capital |
| `q_dev_entity_acquisition_parties` | `gold_005` | acquisition |
| `q_dev_entity_acquisition_parties` | `gold_024` | companies |
| `q_dev_entity_speaker` | `gold_001` | person |
| `q_dev_entity_sales_role` | `gold_007` | sales |
| `q_dev_entity_sales_role` | `gold_017` | sales |
| `q_dev_entity_sales_role` | `gold_025` | sales |
| `q_cvc_second_check` | `gold_004` | capital |
| `q_investor_update` | `gold_011` | month |
| `q_dev_rel_director_promotion` | `gold_004` | director |
| `q_first_vp_sales` | `gold_004` | managing |
| `q_first_vp_sales` | `gold_007` | sales |
| `q_first_vp_sales` | `gold_017` | sales |
| `q_first_vp_sales` | `gold_025` | sales |
| `q_dev_hop_valuation_and_price` | `gold_005` | acquisition |
| `q_dev_hop_vp_and_craft` | `gold_007` | sales |
| `q_dev_hop_vp_and_craft` | `gold_017` | sales |
| `q_dev_hop_vp_and_craft` | `gold_020` | leave |
| `q_dev_hop_vp_and_craft` | `gold_025` | sales |
| `q_boom_promotions` | `gold_004` | director |
| `q_dev_time_valuation_vs_price` | `gold_005` | acquisition |
| `q_vp_sales_tension` | `gold_007` | sales |
| `q_vp_sales_tension` | `gold_017` | sales |
| `q_vp_sales_tension` | `gold_025` | sales |
| `q_dev_attr_speaker_role` | `gold_001` | person |
| `q_dev_attr_lemkin_or_speaker` | `gold_001` | person |
| `q_dev_unans_headcount` | `gold_005` | acquisition |
| `q_dev_fact_niche_boundary` | `gold_018` | customers |

A development question asks for a retrieved fact. A gold topic asks for a Lemkin-style draft. A shared word does not make them the same task. Both sets have already been used in development. Neither is the untouched corpus test split, and this protocol does not open that split.

## 5. Rescoring plan

| Saved output | Rescore? | Where a new score would go |
|---|---|---|
| `EXP-20261005-003` rows | No. They are already `unified_eval.v1`. Leave the SHA gate in place. | Nowhere. The existing `metrics/rows.jsonl` remains the seed-42 Relabel + RAG result. |
| Base, Relabel-off, and Colab rows under `docs/evaluation/examples/data/` | No. Compare the copies that already exist. | A new judge run would be a new experiment, not an edit of these files. |
| `EXP-20261030-012` cells | Possible later, into a new directory. Talk style-on stays unsupported and is not filled with zero. The prompt contract and Chroma v2 index differ from `EXP-20261005-003`, so the result would be labeled approximate. | Not this task. `metrics/comparison.json` stays as measured. |
| `EXP-20261031-001` generations | No. Five pilot topics cannot be rescored into the 30-topic panel. | Not applicable. |
| Model F | Generate first, then `unified-score`, then the judge overlay in the new directory. | `experiments/EXP-20261102-001-model-f-unified-panel` |

## 6. Metric compatibility

| Quantity | Base | PersonaRAG | Relabel off | Relabel + RAG | Fixed-style | Pilot graph | Model F panel |
|---|---|---|---|---|---|---|---|
| 120 cells | yes | yes | yes | yes | style-off yes; style-on talk unsupported | no | prepared, not run |
| `unified_eval.v1` rows | yes | yes | yes | yes | no | no | after `unified-score` |
| Completion, format, copying | scored | scored | scored | scored | in cell metrics, not row files | different scorer | same gates after scoring |
| Lexical grounding gate | not applicable | scored | not applicable | scored | scored in the ablation | different scorer | same gate |
| Claim support | null | scored | null | scored | ablation metric | diagnostic | null until the same judge overlay |
| Writing quality | Qwen 14B, temperature 0 | same | same | same | same judge, different prompt | diagnostic on 5 topics | null until that overlay |
| Style distance | blog, LinkedIn, X | same | same | same | fixed-length voice, separate from this column | not the same reference setup | same function after the reference pack is loaded |
| Talk style | null | null | null | null | unsupported for style-on | not comparable | null unless a talk profile exists; do not zero it |
| Persona utility | rare eligible blogs only | same | same | same | not the decision metric | not comparable | null until selector components exist |
| Numeric diagnostic | in the generation notes | same family | same family | 60 of 120 flagged in the manifest; not a validity gate | counted in the ablation notes | not the same panel | not imputed |
| Latency | null in the row files | null | null | null | recorded for that ablation | recorded for the pilot | stored on the future cell |
| Paired intervals | `eval_tools.compare` | same | same | same | the ablation's own interval | not paired to this panel | only after both sides have rows |

## 7. Saved-row matrix

These means were read from the saved row files on 2026-11-02. Table entries are rounded to three decimals. `saved_architecture_matrix()` keeps six. They are not a new experiment, and they are not a rank. `comparison_matrix` sets `composite_score` and `rank` to null. Paired differences and their intervals stay in `ARCHITECTURE_COMPARISON.md`. A missing value is omitted from the mean. The `n` column is how many rows had a number.

Overall, 120 drafts:

| Architecture | Eligible | Completion pass | Format pass | Copy failures | Writing | Task | Stance (n) | Style (n) | Utility (n) | Claim support (n) | Words |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Base | 45 | 0.375 | 1.000 | 0 | 3.578 | 0.743 | 0.969 (108) | 1.043 (90) | 0.620 (20) | null | 352.1 |
| PersonaRAG | 57 | 0.592 | 1.000 | 0 | 3.599 | 0.725 | 0.990 (103) | 0.906 (90) | 0.631 (19) | 1.000 (100) | 296.5 |
| Relabel, RAG off | 77 | 0.658 | 0.958 | 1 | 3.000 | 0.701 | 0.958 (102) | 0.750 (90) | 0.705 (23) | null | 192.9 |
| Relabel + RAG | 72 | 0.700 | 0.975 | 2 | 2.896 | 0.709 | 0.973 (94) | 0.805 (90) | 0.688 (19) | 0.994 (108) | 183.1 |
| Model F | — | — | — | — | — | — | — | — | — | — | — |

The style `n` of 90 is blog, LinkedIn, and X. Talk contributes the 30 missing style values. Grounding pass rate is null for Base and Relabel off (120 missing) and 0.833 / 0.883 for PersonaRAG and Relabel + RAG. Latency is null for every saved architecture. Persona utility is a blog-only quantity in these files.

Per medium, writing quality / style distance / words. Style is null for talk.

| Architecture | Blog | LinkedIn | X | Talk |
|---|---|---|---|---|
| Base | 3.639 / 1.867 / 515.2 | 3.844 / 0.740 / 240.2 | 3.061 / 0.523 / 116.7 | 3.767 / null / 536.4 |
| PersonaRAG | 3.722 / 1.552 / 390.9 | 3.761 / 0.682 / 242.0 | 3.122 / 0.484 / 124.6 | 3.789 / null / 428.5 |
| Relabel, RAG off | 3.578 / 0.709 / 284.8 | 3.161 / 0.755 / 174.4 | 1.917 / 0.786 / 33.9 | 3.344 / null / 278.5 |
| Relabel + RAG | 3.356 / 0.814 / 259.7 | 2.922 / 0.828 / 151.9 | 2.006 / 0.775 / 39.5 | 3.300 / null / 281.2 |
| Model F | — | — | — | — |

`saved_architecture_matrix()` reprints this from the row files, including task, stance, claim support, and gate counts for every medium. Use `eval_tools.compare` when the question is a paired difference and an interval.

## 8. Tests

`tests/test_model_f_unified.py` and the existing four-arm tests passed together: 20 tests, CPU only. The new tests check the 120-cell plan, the gold hash, the profile lock, the empty id overlap, refusal of the live store and of historical directories, a synthetic cell scored to `unified_eval.v1` with null writing quality and null persona utility, and a saved-row matrix whose Model F entry is null.

## 9. Commands after the indexer exits

Do not run these while `storage_expand_v1` is being written. Snapshot first. The profile name is filled in only after the 36-question run has been read. `SELECTED` is one of `F-SOURCE`, `F-REL`, `F-DESC`, or `F-HOP`.

```powershell
host_finetune\.venv\Scripts\python.exe -X utf8 -m host_finetune.model_f_experiment snapshot --source artifacts\lightrag\model_f\storage_expand_v1 --dest artifacts\lightrag\model_f\snapshots\expand_v1_YYYYMMDD --allow-expansion-source
host_finetune\.venv\Scripts\python.exe -X utf8 -m host_finetune.model_f_experiment verify --snapshot artifacts\lightrag\model_f\snapshots\expand_v1_YYYYMMDD
host_finetune\.venv\Scripts\python.exe -X utf8 -m host_finetune.model_f_experiment audit --snapshot artifacts\lightrag\model_f\snapshots\expand_v1_YYYYMMDD
host_finetune\.venv\Scripts\python.exe -X utf8 -m host_finetune.model_f_experiment plan --questions host_finetune\benchmarks\dev_retrieval_benchmark.jsonl --out experiments\EXP-20261101-009-model-f-four-arm --snapshot artifacts\lightrag\model_f\snapshots\expand_v1_YYYYMMDD
host_finetune\.venv\Scripts\python.exe -X utf8 -m host_finetune.model_f_experiment run --questions host_finetune\benchmarks\dev_retrieval_benchmark.jsonl --out experiments\EXP-20261101-009-model-f-four-arm --snapshot artifacts\lightrag\model_f\snapshots\expand_v1_YYYYMMDD
host_finetune\.venv\Scripts\python.exe -X utf8 -m host_finetune.model_f_experiment evaluate --out experiments\EXP-20261101-009-model-f-four-arm
host_finetune\.venv\Scripts\python.exe -X utf8 -m host_finetune.model_f_experiment unified-plan --profile SELECTED --snapshot artifacts\lightrag\model_f\snapshots\expand_v1_YYYYMMDD --out experiments\EXP-20261102-001-model-f-unified-panel
host_finetune\.venv\Scripts\python.exe -X utf8 -m host_finetune.model_f_experiment unified-run --profile SELECTED --snapshot artifacts\lightrag\model_f\snapshots\expand_v1_YYYYMMDD --out experiments\EXP-20261102-001-model-f-unified-panel
host_finetune\.venv\Scripts\python.exe -X utf8 -m host_finetune.model_f_experiment unified-score --profile SELECTED --out experiments\EXP-20261102-001-model-f-unified-panel
```

The judge overlay uses `qwen2.5:14b` at temperature 0, blinded, candidate family llama, through `apply_judge_overlay`, and writes only into the new experiment directory. There is no command for it in this change. `score_saved_experiment` must not be pointed at the Model F traces.

After both row files exist, the comparison command is the existing one:

```powershell
host_finetune\.venv\Scripts\python.exe -X utf8 -m eval_tools.compare --a docs\evaluation\examples\data\relabel_rag_on_rows.jsonl --label-a "Relabel + RAG" --b experiments\EXP-20261102-001-model-f-unified-panel\metrics\rows.jsonl --label-b "Model F" --out-md experiments\EXP-20261102-001-model-f-unified-panel\metrics\comparison.md
```

That command is also not for today. It needs the Model F rows, including the judge overlay, before a writing-quality comparison means anything.

## 10. What still prevents a fair comparison

- The blog indexer is still running. There is no verified snapshot, so Model F has no 120 drafts.
- The profile has not been selected. Selecting it from the 120 topics would leak the comparison set into the choice.
- Writing quality, claim support, and persona utility are null until the same Qwen overlay is applied. The deterministic scorer does not fill them with zero.
- Persona utility is already sparse on the historical blog rows and null on the other media. It cannot be the comparison by itself.
- Historical row files have no latency. A latency comparison waits until Model F cells exist, and it still will not have a historical latency column unless those older traces are read in a new scoring directory.
- PersonaRAG is a one-seed approximation of the semester-1 pipeline.
- Colab seeds 43 and 44 do not share Kevin's extracted evidence on 15 of 30 topics.
- The fixed-style run and the October 31 pilot are different protocols. Putting them in this table as if they were the same 120-cell measurement would change the comparison.
- The corpus test split stays closed.
