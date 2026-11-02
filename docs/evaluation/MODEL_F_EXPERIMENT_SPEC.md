# Model F experiment specification

Status: **PROPOSED** until a registered run executes it. The arm list is **IMPLEMENTED** in `host_finetune/model_f_context/registry.py`. No arm below has been generated against the blog expansion. Do not start this run while the indexer is writing `storage_expand_v1`.

## Arms

| Id | Retrieval | `--model-f-context` | Callback | Prompt family | What changes relative to F-SOURCE |
|---|---|---|---|---|---|
| E-DENSE | dense Chroma, k = 4 passages | not set | none | `factual-reliability-v1` | Different retriever and prompt family |
| F-OFF | LightRAG hybrid | `off` | none | `factual-reliability-v1` | LightRAG hits, no graph-context prompt |
| F-SOURCE | LightRAG hybrid | `source` | source | `model-f-graph-context-v1` | Reference Model F context arm |
| F-REL | LightRAG hybrid | `relationships` | relationships | `model-f-graph-context-v1` | Adds verified relationship lines |
| F-DESC | LightRAG hybrid | `descriptions` | descriptions | `model-f-graph-context-v1` | Adds derived descriptions |
| F-HOP | LightRAG hybrid | `multihop` | multihop | `model-f-graph-context-v1` | Adds paths and descriptions |

E-DENSE and F-OFF both have an empty graph-context callback. They are not the same arm. E-DENSE reads the dense index. F-OFF reads LightRAG and then uses the existing evidence capsule. F-SOURCE is the first arm that uses `model-f-graph-context-v1`.

## Held fixed

- Questions: `host_finetune/benchmarks/dev_retrieval_benchmark.jsonl`. The same ids on every arm.
- Generator: frozen Relabel adapter `host_finetune/output/lemkin_lora_relabel`.
- Decoding: temperature 0.7, top_p 0.9, repetition penalty 1.15, seed base 42, then the existing per-topic and per-medium offset in `compare_adapters.generate`.
- Evidence budget: 4 passages. LightRAG `top_k` counts entities or relationships. `chunk_top_k` counts chunks. Record both. Do not treat them as the passage budget.
- Context budgets when the callback is on, from `model_f_context.budget.BUDGETS`: X 480/320/80/40/40, LinkedIn 800/480/160/80/80, blog and talk 1400/800/280/160/120. The five numbers are total, primary, relationships, paths, and descriptions. Filling a budget is not a quality score.
- One frozen hybrid payload per question, reused by F-OFF, F-SOURCE, F-REL, F-DESC, and F-HOP. Do not retrieve again between those arms.
- Corpus: the finished blog snapshot after indexing exits. Validation and test documents stay out. Do not retune on the test split.
- Do not combine `--model-f-context` with `--graph-relationships` or a context pad.

## Output

Each draft is one `eval-bench-v1` record. `retrieval_config` stores the arm id, backend, mode, and context profile. `generation_config` stores the prompt family, seed, and decoding. Retrieved ids, evidence, provenance, answer, token counts, and latency are stored even when a metric is null.

Retrieval metrics and generation metrics are separate objects. There is no composite score.

Retrieval, once grades exist: precision@4, recall@4, MRR, and nDCG@4 from `retrieval_eval`, with relevance at grade 2 or 3. Designated-source overlap stays a separate field and is not recall. Provenance completeness and source diversity are diagnostics.

Generation: `assess_factual` faithfulness, numeric pass on answers that state a quantity, citation-marker leak, contradiction only when pairs are annotated, abstention only when the question is marked unanswerable. Word count is a covariate. BLEU, ROUGE, persona utility, and an LLM judge are not selection metrics.

Efficiency: retrieval seconds, generation seconds, end-to-end seconds, p50, p95, prompt tokens, output tokens. Context utilization is reported and is not a win.

## Which comparisons can be interpreted

Confirmatory generation comparisons use one frozen hybrid payload and the same prompt family. That is F-SOURCE against F-REL, F-SOURCE against F-DESC, and F-SOURCE against F-HOP, paired on question id and medium. The difference is the context profile. F-DESC is confirmatory only after entity descriptions are retained. Until then it is blocked.

Confirmatory retrieval comparison is E-DENSE against F-OFF on the graded development questions. Judge retrieval metrics only. A generation difference between those arms mixes the retriever and the evidence text, so it is not evidence that the generator changed.

Exploratory comparisons:

- E-DENSE against any graph-context arm. The prompt family and the retriever both change.
- F-OFF against F-SOURCE. The callback turns on and the prompt family changes, so a gain is not "the source channel" by itself.
- F-HOP against F-DESC. Multi-hop includes descriptions, so the pair does not isolate paths unless the description text is also recorded and compared.
- Any comparison with EXP-20261031-001 cells A–D. Those cells are a different ablation on the pilot.
- Context utilization, hit count, or answer length as a quality ranking.

Report a paired bootstrap with seed 298 for a confirmatory pair. An interval that includes zero does not become a win by dropping a medium. Blog-only and all-media results are both reported when both exist.

## Still waiting on the indexer

Querying the expansion store, retaining entity descriptions in `query_documents`, and generating these six arms. The development grades also wait on human review. The machine-readable sheet is ungraded on purpose.
