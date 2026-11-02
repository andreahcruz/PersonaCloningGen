# Four-arm Model F inference experiment

Status of the runner, the channel restriction, the snapshot utility, and the graph audit: **IMPLEMENTED** and **TESTED** on synthetic fixtures. No snapshot has been copied. No query has been sent. No draft has been generated. Nothing here is **VALIDATED** against the expansion graph.

This experiment is not a model comparison. One Relabel generator sees four renderings of one frozen LightRAG payload per question.

## What the profiles actually do

`source` renders source claims only. `relationships` adds source-verified direct edges. `descriptions` adds entity descriptions and also keeps direct edges. `multihop` adds paths and also keeps direct edges and descriptions.

The four experiment arms do not use those extra sections:

| Arm | Rendered sections | Built with profile |
|---|---|---|
| F-SOURCE | source claims | `source` |
| F-REL | source claims and verified direct relationships | `relationships` |
| F-DESC | source claims and derived descriptions | `descriptions`, then relationship and path sections are removed |
| F-HOP | source claims and multi-hop paths | `multihop`, then relationship and description sections are removed |

Paths are built only from direct, accepted, sourced edges. Cycles, duplicate paths, and a two-hop path presented as one source claim are rejected by the existing selector. An inferred edge such as an unsupported cross-document endorsement stays out of F-REL. A description stays in the supplementary section and is not added to grounding texts, so a number inside it is not treated as an allowed fact.

A relationship whose predicate is already in a verified claim is omitted as redundant. F-REL can therefore match F-SOURCE on a question whose edges only repeat the claims. That is recorded per trace, not treated as a retrieval difference.

## Controls

Retrieval runs once per question. The raw `aquery_data` payload is stored. All four arms read that file. The mode is hybrid, the passage budget is 4, and `top_k` is 20. Query `num_ctx` is 16384, matching the expansion index configuration rather than the library default of 32768.

The generator is the frozen Relabel adapter. Decoding is temperature 0.7, top_p 0.9, repetition penalty 1.15, seed base 42, with the existing per-topic and per-medium offset. The prompt contract is `model-f-graph-context-v1` on every arm. Only the graph-context sections change.

Source claims use the same primary cap on every arm. The graph cap is the same number on every arm. Blog and talk use 557 source tokens and 83 graph tokens so the chat prompt, a retry line, and 768 new tokens stay inside the 2048-token generation window. LinkedIn keeps 480 and 80. X keeps 320 and 40. Default `BUDGETS` are unchanged. Arms can still consume different amounts of context when a section is empty or its evidence is short. `run` checks all 576 cells against that window before loading the model, and a sample that would still overflow is refused instead of truncated. A one-hop edge is not rendered as an F-HOP path. A two-hop path still shows each edge. Each trace records tokens by channel and `token_source`. A loaded generator tokenizer is passed through when generation runs. Until then the counter is characters divided by 4.

Questions are the 36 rows in `host_finetune/benchmarks/dev_retrieval_benchmark.jsonl`. The corpus test split is not read. Grades stay null. Precision, recall, MRR, and nDCG are not computed. Persona style distance is not a score. Relationship accuracy and multi-hop correctness stay `not_annotated` until a person labels them. Automated scores are faithfulness, numeric grounding, citation-marker leak, source copying, latency, and token counts. Paired comparisons are F-SOURCE against each other arm on question id and medium. There is no composite score.

## Entity descriptions

`query_documents` still drops description text. That function lives in `lightrag_backend.py`, which the indexer imports, so it was not edited. `generation_payload` runs the existing normalizer for verbatim hits and then copies `data.entities` back onto `raw_entities` and `data`. F-DESC reads those rows. Descriptions are not hits. After the indexer exits, `query_documents` should return the same `raw_entities` list. The experiment does not depend on that edit if it freezes `aquery_data` output.

## Graph audit

`graph_audit.audit_records` classifies weak names, the pronouns `I` and `You`, alias groups, duplicate edges, missing edge or description provenance, orphan vectors, and malformed 4-field, 6-field, and 7-field extraction lines. Cache-only malformed rows are cosmetic. Weak names, alias splits, duplicate edges, and orphan vectors affect retrieval. Missing provenance affects generation because an unsourced description must not be shown as a fact. Recovery is not applied. The live store is refused.

## Commands after the indexer has exited

Do not run these while `indexing.lock` exists or any document is `processing`.

```text
host_finetune\.venv\Scripts\python.exe -X utf8 -m host_finetune.model_f_experiment snapshot ^
    --source artifacts\lightrag\model_f\storage_expand_v1 ^
    --dest artifacts\lightrag\model_f\storage_expand_v1_snapshot ^
    --allow-expansion-source
```

```text
host_finetune\.venv\Scripts\python.exe -X utf8 -m host_finetune.model_f_experiment verify ^
    --snapshot artifacts\lightrag\model_f\storage_expand_v1_snapshot
```

`verify` checks the required graph, catalog, and vector files, requires 768-dimensional chunk vectors, and recomputes the SHA-256 file hashes. It refuses a path that still contains `storage_expand_v1`.

The same `run` command starts the experiment and resumes it. A cell whose file already has `"status": "complete"` is skipped. Retrieval is frozen once per question. The four arms then generate in order, one medium at a time, on the loaded Relabel adapter. The plan id is `EXP-20261101-009-model-f-four-arm`. Its fingerprint covers the arm list, channel sets, decoding, retrieval settings, adapter path, budgets, and question ids.

```text
host_finetune\.venv\Scripts\python.exe -X utf8 -m host_finetune.model_f_experiment run ^
    --questions host_finetune\benchmarks\dev_retrieval_benchmark.jsonl ^
    --out experiments\EXP-20261101-009-model-f-four-arm ^
    --snapshot artifacts\lightrag\model_f\storage_expand_v1_snapshot
```

```text
host_finetune\.venv\Scripts\python.exe -X utf8 -m host_finetune.model_f_experiment evaluate ^
    --out experiments\EXP-20261101-009-model-f-four-arm
```

Audit reads the snapshot copy and writes its report beside that directory. It does not write into the snapshot and it does not apply relationship recovery. Do not point the audit at the live store.

```text
host_finetune\.venv\Scripts\python.exe -X utf8 -m host_finetune.model_f_experiment audit ^
    --snapshot artifacts\lightrag\model_f\storage_expand_v1_snapshot
```

## What execution will require

No new embedding model. The snapshot already contains `nomic-embed-text` vectors at 768 dimensions. Query time still embeds the question with that model, and LightRAG's keyword step calls `llama3.1`. Generation loads the Relabel adapter on the GPU. Run one generation at a time.

The pilot's saved drafts took roughly 9 to 27 seconds each, and one four-medium file took about 78 seconds of wall time. This run is 36 questions, 4 arms, and 4 media: 576 drafts. At about 20 seconds each, generation is on the order of 3 hours, plus one hybrid query per question and model load. That is an estimate from the October 31 pilot, not a measurement of this run.

## Tests

```text
host_finetune\.venv\Scripts\python.exe -X utf8 -m pytest tests\test_model_f_experiment.py tests\test_model_f_graph_context.py -q --tb=short
```
