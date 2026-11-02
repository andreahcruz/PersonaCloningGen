# Post-index checklist for Model F

Do this only after the blog indexer has exited and nothing else is writing `artifacts/lightrag/model_f/storage_expand_v1`. Do not start a second indexing process. Do not query that directory while another job has it open. `query_documents` calls Ollama for both the extraction model and `nomic-embed-text`, so it also waits until the indexer has released those models.

The October 31 pilot run is not a substitute for these steps. Its cells A–D did not use the five context profiles.

## 1. Snapshot first

Copy the finished workdir with `python -m host_finetune.model_f_experiment snapshot` before the first query. The command refuses a held `indexing.lock` and any document still marked processing, parsing, analyzing, or pending. Then run `verify` on the copy. Record the catalog hash it prints. Keep validation and test rows out of the catalog. Do not retune prompts or budgets on the corpus test split. The exact commands are in `docs/implementation/MODEL_F_FOUR_ARM_EXPERIMENT.md`.

## 2. Keep entity descriptions in the query payload

`lightrag_backend.normalize_lightrag_data` is imported by the indexer, which is why this edit waits. In the loop over `data["entities"]` and `data["relationships"]`, relationship rows are copied to `relationship_candidates`. Entity rows are appended only to `excluded`, with reason `generated_summary_not_a_source_span`, and the excluded record keeps `name`, `file_path`, and `source_id`. The description text is dropped. `query_documents` returns that payload.

After the indexer exits, retain entity `description`, `entity_type`, `source_id`, and `file_path` on a separate `entity_descriptions` list. Leave those rows out of `hits`. Leave the description text out of claim `extra_texts`, or a number inside a generated summary becomes an allowed quantity. `model_f_context.normalize` already counts stripped entity rows as `entity_descriptions_not_retained` and does not invent the text. A synthetic payload test should show that diagnostic going to zero when the description is present, and staying positive when it is absent.

Do not change `index_documents`, the extraction prompt, chunk size, or the embedding function in that same edit.

## 3. Generate the four LightRAG arms from one snapshot

The inference run that starts after this snapshot is F-SOURCE, F-REL, F-DESC, and F-HOP on one frozen hybrid payload per development question. Follow `docs/implementation/MODEL_F_FOUR_ARM_EXPERIMENT.md`. `run` is both the start command and the resume command.

`docs/evaluation/MODEL_F_EXPERIMENT_SPEC.md` still describes the broader six-arm matrix, including E-DENSE and F-OFF. Those two arms are not part of this inference run. Do not combine `--model-f-context` with `--graph-relationships` or a context pad.

Use the development question ids in `host_finetune/benchmarks/dev_retrieval_benchmark.jsonl`. The corpus test split stays closed. F-DESC reads entity descriptions from the frozen raw payload. `query_documents` still drops that text until the deferred edit in section 2 is applied.

## 4. Score the saved traces

Convert each arm to `eval-bench-v1` with `profile_match` true only when the trace records that context profile. Then:

```text
host_finetune\.venv\Scripts\python.exe -X utf8 -m host_finetune.eval_bench ^
    --predictions path\to\predictions.jsonl ^
    --out experiments\EXP-YYYYMMDD-NNN-name\metrics ^
    --experiment-id EXP-YYYYMMDD-NNN ^
    --pair-left model_e ^
    --pair-right model_f_source
```

Pair on question id and medium. Ranked retrieval metrics stay null until a person fills `retrieval-annotation-v1` grades. The blank sheet from the pilot replay is a pool, not a label set. Precision, recall, MRR, and nDCG are not designated-source overlap.

Faithfulness matches `assess_factual` and does not include the separate instruction-leak flag. Report `citation_marker_leak` beside it. A leaked prompt can still have `factual_pass` true, as it did on the pilot LinkedIn drafts.

Register the run under `experiments/` with the manifest fields in `experiments/README.md`. Record that the workdir was the snapshot, not a live indexing directory.
