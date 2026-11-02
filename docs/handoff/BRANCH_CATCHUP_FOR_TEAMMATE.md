# Branch catch-up: Model F on `kevin/lightrag-model-f`

Audience: the teammate integrating this work into `andreah/deploy-with-react` in PersonaCloningGen, and any assistant reading the repository with them. Blog-only LightRAG indexing was still running when this note was written.

Verified locally on 2026-11-01. Companion notes, already on this branch:

- `docs/handoff/TEAMMATE_DEPLOYMENT_HANDOFF.md`
- `docs/handoff/MODEL_F_HANDOFF_MANIFEST.md`

Course scoring context for Workbook 2 and Demo 2 is in section 17. The older planning matrix `docs/rubric_matrix.md` is an October 3 baseline. Where it disagrees with the attached Workbook 2 sheet, use the sheet.

## 1. Executive summary

DATA 298 is a persona-modeling project. The system writes in a target author's style by putting a QLoRA adapter, called Relabel, on Llama 3.1 8B, then optionally grounding the draft in retrieved source text.

Model E is that generator with dense Chroma retrieval. Model F is the same generator with LightRAG. LightRAG builds a knowledge graph and can return passages, relationships, and generated descriptions. The graph-context pipeline decides which of those are safe to show the generator.

This branch makes Model F resumable to index, and it adds the graph-context pipeline, four prompt configurations of that one system, a four-arm experiment runner, and an evaluation harness. The CPU tests recorded with the handoff passed: 156 tests. No live query has been run against the expansion graph, and the four-arm generation experiment has not been run.

The branch is pushed. Tip at the time of the previous handoff: `817e159b99b904a3171a0495e7a1c9433fe98137`. The implementation commit is `dbeb21c5b0fa7e6bdd1fae6613963c5c90eb7b84`. No pull request into `andreah/deploy-with-react` was opened from that handoff. The partial index is not in Git.

Deployment integration can start without the finished index. Live Model F behavior cannot be claimed until indexing exits, the store is snapshotted and verified, and the experiment is run on that snapshot.

For the course rubrics, this branch supplies the improved model, the evaluation method, the CPU test evidence, and the source layout. It does not yet supply the Workbook 2 validation results, the response-time measurement, the visualization pack, or the Demo 2 web UI. The web UI is the largest Demo 2 item (4 of 12.5 points) and lives on the deployment branch.

## 2. Project background

The project tries to reproduce an author's writing persona: given a topic and a medium (blog, LinkedIn, X, or talk), produce a draft that sounds like that author and stays within supplied facts.

The base model is Llama 3.1 8B Instruct, loaded in 4-bit. QLoRA trains a small adapter on top of it instead of updating all of the weights. The Relabel adapter is the one selected for deployment. It is a local directory, `host_finetune/output/lemkin_lora_relabel`, and it is not part of the new handoff commit. Earlier deployment notes are in `docs/deployment/DEPLOYMENT_HANDOFF.md`.

Retrieval exists because a style adapter will invent specific numbers, employers, and attributions. Retrieved text is reduced to atomic claims. A claim is kept only when its support span occurs in the named source. The Relabel adapter then writes from those accepted claims. Numeric checks and source-copy checks run around that draft. They do not silently rewrite it into a different essay.

Model E uses a train-only Chroma collection. The active selector is `host_finetune/rag_active.json`. Dense retrieval is still the default in `compare_adapters.py` (`--retrieval-backend` defaults to `dense`).

Model F was added because passage search returns nearby text and does not represent "A is connected to B, and B is connected to C" as a structure. LightRAG extracts entities and relations while it indexes, stores them in a graph, and can query in naive, local, global, hybrid, or mix mode. Hybrid is the mode the four-arm experiment uses.

Graph context is useful only when the system can tell a supported fact from a generated summary. A relationship is shown as evidence after a source passage supports that edge. An entity description is generated graph text. It can help the model stay oriented, and it must not become an allowed fact. That is why descriptions stay in their own channel.

## 3. Architecture overview

The pieces connect in this order:

1. Base LLM: Llama 3.1 8B Instruct, 4-bit.
2. Relabel adapter: the persona weights. On for the final draft. Off during claim extraction.
3. Source corpus: cleaned author texts, assigned to train, validation, and test. The graph-ready derivative feeds indexing. The corpus test split stays closed.
4. Embedding model: Ollama `nomic-embed-text`, 768 dimensions. Query time embeds the question. Index time embeds chunks.
5. LightRAG knowledge graph: entities, relations, and a document catalog under a working directory. The expansion directory is `artifacts/lightrag/model_f/storage_expand_v1`.
6. Vector retrieval: Chroma for Model E. LightRAG also keeps its own chunk vectors inside the working directory.
7. Model F graph-context pipeline: `host_finetune/model_f_context/`. It never opens the live index. It consumes a payload already in memory.
8. Generation and evaluation: `compare_adapters.py` writes the draft. `host_finetune/eval_bench/` scores saved traces.

```mermaid
flowchart TD
    topic[Topic and medium] --> choice{Retrieval backend}
    choice -->|dense default| chroma[Chroma train-only index]
    choice -->|lightrag flag| graph[LightRAG hybrid query]
    graph --> payload[Payload in memory]
    payload --> channels[Passages, verified edges, descriptions]
    channels --> budget[Rank and token budget]
    budget --> prompt[Graph-context prompt]
    chroma --> claims[Atomic claims]
    prompt --> claims
    claims --> relabel[Relabel adapter on Llama 3.1]
    relabel --> gates[Copy, number, and leak checks]
    gates --> eval[eval_bench on saved traces]
```

`--model-f-context` defaults to `off`. With that default, LightRAG retrieval can still run, and the prompt stays the existing claim prompt. The four new profiles change only the graph-context sections.

## 4. What this branch accomplished

Verified Git state when this catch-up was drafted:

| Item | Verified value |
|---|---|
| Branch | `kevin/lightrag-model-f` |
| HEAD before this note | `817e159b99b904a3171a0495e7a1c9433fe98137` |
| Remote tracking | Matched `origin/kevin/lightrag-model-f`. Nothing was committed and left unpushed. |
| Implementation commit | `dbeb21c5b0fa7e6bdd1fae6613963c5c90eb7b84` (179 files) |
| Later doc commits | `de23109` records that hash. `817e159` records how to open the pull request. |
| Parent before the handoff commit | `95db4f54a518cb273ea9118b10b9175eeb9668be` |
| Deploy branch tip | `origin/andreah/deploy-with-react` at `0e69135fcfffd86d11779aabc093bfded1201065` |
| Merge base | `ea1d58b30cafcb4cf18a03ded39963f64ce6aee7` |
| Divergence | 34 commits on the deploy branch that this branch lacks. 38 commits on this branch that the deploy branch lacks. |

This file is a later commit. `git rev-parse HEAD` is the tip after it lands.

Committed and pushed before this note: indexing modules, graph-context package, experiment runner, evaluation harness, development benchmark, tests, implementation docs, selected experiment records, and the two earlier handoff notes.

Local and uncommitted, on purpose: style-ablation raw cells and judge cache, the extraction-cache census, the relation-recovery ledger, and the historical-replay `results.jsonl` files. Those contain generated dumps or extracted fields and were left out of the handoff.

Ignored, and not to be committed: `artifacts/` (including the live index), `deployment_handoff/`, virtualenvs, and new weights, caches, and `*.npy` files. `.gitignore` does not untrack older files. See section 12.

Not implemented on this branch: a FastAPI or React path that loads LightRAG. `backend/` has no LightRAG call. Serving Model F means wiring `--lightrag-workdir` into the service. The four-arm run `EXP-20261101-009-model-f-four-arm` has not been created.

Recorded CPU result, from the handoff on this branch, not re-run for this note: 156 passed, 2 Swig deprecation warnings, 17.21 seconds. The command is in section 14.

## 5. LightRAG indexing architecture

Entry point: `host_finetune/indexing_session.py`. Default workdir: `artifacts/lightrag/model_f/storage_expand_v1`. The running process is that module with `--resume --selection-policy blog-only --max-documents 13000 --max-runtime-minutes 480 --timeout-policy adaptive`.

Resumable sessions. `--prepare` loads the graph-ready corpus into a ledger and writes a manifest. `--resume` indexes the next eligible documents and can stop on a document cap or a time cap. A later `--resume` continues. Do not start a second session on the same directory.

SQLite ledger. `host_finetune/indexing_ledger.py` stores one row per document: status, attempts, defer reason, medium, and timing. The file is `indexing_ledger.sqlite` inside the workdir. Status values include pending, processing, completed, interrupted, failed_permanent, and failed_retryable.

Checkpointing. Progress is the ledger plus LightRAG's own files in the workdir. On resume, the session reconciles rows left in `processing` or `interrupted`. `--snapshot` on the indexing CLI can copy a workdir after a clean shutdown. The experiment snapshot command is separate and is the one to use before queries. See section 14.

Blog-only selection. `indexing_selection.py` and `indexing_schedule.py`. The store already contains a mixed-media seed. Blog-only expansion schedules eligible blogs and leaves other media in the ledger. It does not delete documents already indexed. The ledger summary calls this `mixed_media_seed_with_blog_only_expansion`.

Deferred and retry queues. A document that stalls can be deferred, often with reason `expensive_timeout`. The normal queue skips those rows. They return only with `--retry-timeouts` or `--include-deferred`. Deferred work is not part of the default remaining blog queue.

Adaptive timeout. `indexing_timeout.py`. `adaptive` sets the budget from a chunk-size tier. `fixed` uses `--max-document-seconds` for every document. The running process uses adaptive. A timeout becomes an interrupted or deferred row, not a silent success.

Failure classification. `indexing_classify.py` maps an exception or a timeout into a category and decides whether another attempt is allowed. Permanent failures stay out of the queue. The blog summary at the last checkpoint had one permanently failed blog.

Extraction quality. `indexing_quality.py` counts format warnings and weak entity names while extraction runs. `indexing_extraction_audit.py` can census cached extraction records. That census is a read-style report. It is not a second indexer.

Relationship normalization. `host_finetune/relation_recovery.py` implements `relation-normalization-v1`. It can dry-run a ledger of comma-packed or schema-labeled relation rows. `apply` refuses a directory named `storage_expand_v1`. Repairs belong on a copy, after indexing stops. The dry run did not insert edges.

Omission handling. Some source rows contain an omission marker. The scheduler counts them. Query-time filters drop omission-marker chunks. Segment-building code in `indexing_segments.py` must not be inserted into `storage_expand_v1`.

What is production-shaped, and what is a lab utility:

| Production-shaped | Lab or one-run utility |
|---|---|
| Session, ledger, schedule, timeout, failure class, quality warnings | Package-snapshot text files inside experiment directories |
| Blog-only policy and the defer queue | Scripts under `experiments/EXP-20261031-002-lightrag-resumable-index/raw/` that sealed or marked old rows |
| Normalization that refuses the live store | The dry-run recovery ledger, which was not committed |

Read next: `docs/architecture/LIGHTRAG_INDEXING_AND_CHECKPOINTING.md`.

## 6. Model F graph-context architecture

Package: `host_finetune/model_f_context/`. Schema version: `model-f-evidence-v1` in `schema.py`. Importing the package does not open graph storage.

`normalize.py` turns a payload into three channels.

- Source passages are verbatim chunks. A chunk that is not an exact substring of the catalog document is excluded. These are the only texts that can support a claim.
- Source-verified relationships copy relationship rows, then keep an edge only when a source span supports it. The predicate comes from that check, not from gluing the endpoint names together. A missing endpoint is malformed and has no predicate.
- Derived entity descriptions are generated graph text. They are classified `derived_context`. Confidence is left empty. If the text matches an instruction-injection pattern, the description is emptied and rejected.

The split exists because a generated sentence can contain a number or a title that never appeared in a source. If that sentence were treated as a passage, numeric grounding would treat the number as allowed. Keeping descriptions out of the claim channel prevents that.

Multi-hop selection is `multihop.py`, function `select_paths`. A hop must be direct, accepted, predicated, and tied to a verbatim span and a source id. Inferred and rejected edges are withheld. A path cannot repeat a node. The same edge sequence is kept once. `max_hops` is 1 or 2. A two-hop path records each hop's documents. It does not say one source stated the connection between the endpoints. `explicit_endpoint_claim` is true for one hop, and for a longer path only when a separate direct edge links the first node to the last.

Provenance sits on passages, edges, and paths: document ids, chunk ids, and whether the edge was directly stated. Source references are resolved, unresolved, missing, or not checked. An unresolved description stays derived context.

Ranking is `ranking.py`. One key orders evidence. A LightRAG weight is not added to a Chroma distance. Weight breaks ties only when the score type is `lightrag_weight`. Weak names, inferred edges, and redundant descriptions are dropped or demoted. A relationship whose predicate is already inside a verified claim can be omitted at render time as redundant with the primary source.

Token budgets are `budget.py`. Defaults differ by medium. The four-arm runner does not use those defaults unchanged. It gives every arm the same source cap and the same graph cap. See section 8.

Prompt construction is `prompt.py`. The contract is `model-f-graph-context-v1`. Section headings change with the profile. `marker_leak` checks whether prompt labels were copied into the draft.

Leakage control is `leakage.py`. It can screen documents, withhold graph context, and record a paired comparison. Evaluation configurations are listed there and again in `registry.py`. The registry keeps Model E (`E-DENSE`, dense, no graph callback) distinct from Model F with context off (`F-OFF`).

`pipeline.py` is the composition root: `prepare_context`, `render_prepared`, `context_media_jobs`, and `graph_context_callback`. The callback returns `None` for `off`.

`compare_adapters.generate` accepts `graph_context_for`. The CLI flag `--model-f-context` accepts `off`, `source`, `relationships`, `descriptions`, and `multihop`. It requires `--retrieval --retrieval-backend lightrag`. It refuses to combine with `--graph-relationships` or a context pad.

Known gap, stated in `docs/implementation/MODEL_F_POST_INDEX_CHECKLIST.md` and still true in `lightrag_backend.py`: `query_documents` keeps relationship rows and drops entity description text from hits. The normalizer counts those stripped rows as `entity_descriptions_not_retained` and does not invent the text. F-DESC can read descriptions when the raw payload still has `data.entities`. That pass-through is not finished, because the indexer imports `lightrag_backend.py`.

Read next: `docs/implementation/MODEL_F_GRAPH_CONTEXT_IMPLEMENTATION.md`.

## 7. Four inference profiles

These are four prompt configurations of one LightRAG-backed Model F. They share one Relabel adapter, one index, and one hybrid retrieval call per question. They are not four models and not four databases.

`host_finetune/model_f_experiment/arms.py` builds candidates with a richer profile, then deletes every channel the arm is not allowed to show. What the generator sees is `RENDERED_CHANNELS`.

| Profile | What the arm actually shows the generator |
|---|---|
| F-SOURCE | Source passages only |
| F-REL | Source passages and verified direct relationships |
| F-DESC | Source passages and derived descriptions |
| F-HOP | Source passages and provenance-preserving multi-hop paths |

Internal overlap, which the arm then hides:

| Arm | Profile used to build candidates | Channels removed before the prompt |
|---|---|---|
| F-SOURCE | `source` | Anything beyond passages is not built for this profile |
| F-REL | `relationships` | Paths and descriptions are not the rendered set |
| F-DESC | `descriptions` | That profile also keeps direct edges. The arm removes relationship and path sections. |
| F-HOP | `multihop` | That profile also keeps edges and descriptions. The arm removes both, and keeps paths. |

A one-hop edge is not rendered as an F-HOP path. Paths are built only from direct, accepted, sourced edges. An inferred cross-document link stays out of F-REL. A description is not added to grounding texts, so a number inside it is not an allowed quantity.

F-REL can match F-SOURCE on a question whose edges only repeat the claims. The runner records that per trace. It is not treated as a retrieval difference.

`E-DENSE` and `F-OFF` are real registry rows. They are not arms of this inference run. E-DENSE is Chroma. F-OFF is LightRAG with the graph-context callback empty and the older prompt family `factual-reliability-v1`.

## 8. Experiment runner and experimental methodology

Package: `host_finetune/model_f_experiment/`. Plan id: `EXP-20261101-009-model-f-four-arm`. Status: implemented and tested on synthetic fixtures. No snapshot has been copied. No draft has been generated from the expansion graph.

`snapshot.py` copies a finished workdir and writes `snapshot_hashes.json`. It refuses a held `indexing.lock`, refuses documents still processing, and refuses a destination whose name is `storage_expand_v1`. Copying the live expansion directory requires `--allow-expansion-source`, and only after the lock is gone. `verify` checks required graph, catalog, and vector files, requires 768-dimensional chunk vectors, and recomputes SHA-256 file hashes. It refuses a path that still contains `storage_expand_v1`.

`graph_audit.py` reads a snapshot. It classifies weak names, alias groups, duplicate edges, missing provenance, orphan vectors, and malformed extraction lines. It writes the report beside the snapshot, not inside it, unless an export file is already present. `recovery_applied` stays false.

`retrieval.py` and `runner.py` freeze one hybrid payload per question. All four arms read that file. Retrieval settings, from `arms.py`: backend `lightrag`, mode `hybrid`, passage budget 4, `top_k` 20. Query `num_ctx` is 16384 in the experiment write-up, matching the expansion index rather than the library default.

Shared evidence extraction means the claims come from that one payload. Arms do not retrieve again and do not get a private claim set.

Generation uses the Relabel adapter at `host_finetune/output/lemkin_lora_relabel`. Decoding is temperature 0.7, `top_p` 0.9, repetition penalty 1.15, seed base 42, with the existing per-topic and per-medium offset. The prompt family on every arm is `model-f-graph-context-v1`.

Resume: `run` is also the resume command. A cell whose file already has `"status": "complete"` is skipped. `plan` prints the fingerprint, question count, and pending cells without generating.

Failure handling: a sample that would overflow the window is refused instead of truncated. The runner checks the planned cells against the window before loading the model. The live store name is refused, so a path ending in `storage_expand_v1` cannot be the run input.

Context-window preflight lives in `context_window.py` and `experiment_budget` in `arms.py`. Blog and talk use 557 source tokens and 83 graph tokens so the chat prompt, a retry line, and 768 new tokens stay inside the 2048-token generation window. LinkedIn keeps 480 and 80. X keeps 320 and 40. Default `BUDGETS` in `budget.py` are larger and are not what this run uses. Every arm gets the same caps, so a longer relationship budget cannot masquerade as a better profile.

Questions are the 36 rows in `host_finetune/benchmarks/dev_retrieval_benchmark.jsonl`. Media are blog, LinkedIn, X, and talk. That is 36 × 4 × 4 = 576 drafts. The corpus test split is not read.

This isolates the graph-context wording. The retrieved payload, adapter, decoder, questions, media, and token caps stay fixed. If two arms differ, the difference is which sections were rendered.

Output goes under the `--out` directory: a manifest with the fingerprint, one frozen retrieval payload per question, and per-arm generation files. `evaluate` converts those traces for `eval_bench`. Exact commands are in section 14.

Read next: `docs/implementation/MODEL_F_FOUR_ARM_EXPERIMENT.md`.

## 9. Evaluation infrastructure

Package: `host_finetune/eval_bench/`. Schema module: `schema.py` (`eval-bench-v1`). The CLI is `python -m host_finetune.eval_bench`. It scores a predictions file. It refuses an output path that contains `storage_expand_v1`.

What the harness can score now, from `metrics.py` and `stats.py`:

| Measure | What it is | Ready without human grades |
|---|---|---|
| Factual reliability | `assess_factual` on the draft against stored evidence | Yes, after drafts exist |
| Numeric grounding | Unsupported quantities versus allowed evidence | Yes |
| Citation-marker leakage | Prompt tags copied into the draft. This is separate from factual pass. A leaked draft can still have `factual_pass` true. | Yes |
| Source copying | Long spans copied from retrieved text | Yes |
| Latency and tokens | Retrieval seconds, generation seconds, end-to-end seconds, prompt and output tokens | Yes, when the trace recorded them |
| Paired comparisons | Same question and medium, left arm minus right arm | Yes, after both arms exist |
| Bootstrap interval | `paired_mean_difference`: 1000 draws, seed 298, 95 percent percentile interval of the mean difference | Yes, on the recorded questions. It is not a single winner score. |
| Wilson interval | 95 percent interval for a rate | Yes, for rates the scorer computes |
| Precision, recall, MRR, nDCG | Need graded relevant documents | No |

`score_retrieval` stays null until a person fills `retrieval-annotation-v1` grades. The development sheet `host_finetune/benchmarks/dev_retrieval_benchmark.jsonl` has 36 questions, eight phenomena, document ids, and null grades. The withheld file holds review notes and must not be treated as labels. Grades in that sheet are empty on purpose.

Coverage of the development set is factual, entity identification, direct relationship, multi-hop, temporal comparison, conflicting evidence, attribution, and unanswerable. That is a prompt-and-retrieval worksheet. It is not a labeled relevance set.

Keep these questions separate:

- Retrieval quality: did the right sources come back. Needs human grades. Designated-source overlap in older runs is not precision or recall.
- Generation quality: is the draft faithful, numeric-safe, and free of copied prompt markers. The harness can score this from saved traces.
- Relationship accuracy: did a rendered edge match a source. Stays `not_annotated` until a person labels it.
- Multi-hop reasoning: did a path respect provenance, and did the draft avoid treating the path as one source's claim. The selector tests this on synthetic graphs. The finished index has not been audited for it.
- Persona consistency: style distance and voice scores from older Model E work. They are not a Model F score in this harness, and they are not part of the four-arm decision.

Human grading is optional for launching generation. The runner does not read grades. Human grading is required before anyone reports ranked retrieval quality.

There is no composite score. `evaluate` prints `composite` as null.

Read next: `docs/evaluation/MODEL_F_EVALUATION_INFRASTRUCTURE.md`, `docs/evaluation/MODEL_F_EXPERIMENT_SPEC.md`, and `docs/evaluation/RETRIEVAL_ANNOTATION_GUIDE.md`.

## 10. Historical experiments and existing results

These runs are real and they are not the new four profiles.

| Experiment | What it measured | What it is not |
|---|---|---|
| EXP-20261027-002 | Three-document LightRAG smoke | A ranking or a generation study |
| EXP-20261029-004 | 160-document pilot: 152 processed, 8 extraction failures | The blog expansion index |
| EXP-20261029-005 | Five-mode retrieval on eight questions | Precision or recall |
| EXP-20261029-006 and 007 | Three-topic, four-medium Relabel generation, including Chroma v2 | The five context profiles |
| EXP-20261029-008 | Read-only integrity audit of the pilot graph | An audit of `storage_expand_v1` |
| EXP-20261029-009 | Hardened four-arm generation on three topics, plus a writing judge | F-SOURCE, F-REL, F-DESC, F-HOP |
| EXP-20261030-010 | Prompt contract `factual-reliability-v1` on frozen retrieval | A graph-context result |
| EXP-20261030-011 | Span repairs of EXP-009 drafts | A replacement of EXP-009 |
| EXP-20261031-001 | Source-verified relationship conditioning on the 152-document pilot, plus a small generation ablation | The new profiles. Cells A–D used older prompt conditions. |
| EXP-20261031-002 through 005 | Resumable indexer, hardening, adaptive timeout, blog-only policy | A finished blog graph |
| EXP-20261101-006 and 007 | Extraction census and a dry-run recovery ledger | Inserted edges |
| EXP-20261101-008 | Replay of the October 31 traces through `eval_bench` | A new measurement. `profile_match` is false. |

`tests/test_eval_bench_historical.py` asserts that October 31 cells are not these profiles. `crosswalk_to_profile("D_blog", "descriptions")` is rejected. `crosswalk_to_profile("A", "source")` is not comparable. Cell letters compared a dense arm, a LightRAG arm, and a length pad. They did not hold one hybrid payload fixed and swap graph-context sections.

E-DENSE and F-OFF stay out of `EXP-20261101-009` because that run answers a narrower question: given one frozen hybrid payload, what changes when the prompt gains relationships, descriptions, or paths. Adding Chroma or the old prompt would mix retrieval changes into that comparison. They remain available for a later matrix. The broader matrix is described in `docs/evaluation/MODEL_F_EXPERIMENT_SPEC.md`.

`docs/architecture/LIGHTRAG_MODEL_F.md` records the older measured runs. Do not quote those numbers as Model F graph-context results.

## 11. Current indexing status

The indexer was still running when this note was drafted. The audit did not call `--status`, because `IndexingLedger` opens SQLite with a normal connection and a 30-second timeout. That can contend with the writer.

The latest status transcript already on this machine, from `indexing_session --status --selection-policy blog-only`, is a checkpoint. Indexing continued after that transcript. Treat every count below as last-known at draft time, not live.

| Ledger field | Last captured value |
|---|---|
| Eligible documents | 13140 |
| Completed graph documents | 2284 |
| Pending | 10751 |
| Interrupted | 102 |
| Processing | 1 |
| Permanently failed | 2 |
| Eligible blogs | 2042 |
| Completed blogs | 1461 |
| Blogs in the normal pending queue | 482 |
| Deferred blogs | 98 |
| Deferred blog retries | 91 |
| Permanently failed blogs | 1 |
| Completed non-blog seed | 823 |
| Non-blog rows excluded from the blog queue | 10261 |
| Blogs with an omission marker | 340 |
| Remaining blog chunk estimate | 2991 |

At that same moment the blog-only time scenarios were about 3.7, 6.0, and 11.7 hours for the 482 pending blogs, using median, failure-adjusted, and conservative assumptions. The transcript says those hours assume future blogs resemble completed blogs, and that a few stalls dominate the tail. They are not a guarantee. The all-media normal queue was about 10743 documents. Blog-only does not index that queue.

`last_graph_counts` in that transcript says 13401 nodes and 22027 edges with source `measured_after_shutdown_flush`. That is an older flush measurement stored in the ledger. It is not a live graph size.

An earlier report of 2210 completed graph documents and 1387 completed blogs is older than this transcript.

What remains: the pending blog queue, plus any deferred blog retries only if someone explicitly enables that queue. One blog was already a permanent failure. Non-blog documents stay in the ledger and are outside this expansion policy.

Indexing is complete for this policy when the blog normal queue is empty, no document is left `processing`, the process has exited, and `indexing.lock` is absent. Deferred rows can remain deferred without blocking that statement, as long as the report says they were not retried.

The partial directory must not be deployed or committed. It is missing most remaining blogs, it is being written, and a reader can see a torn graph. Querying it also calls Ollama while the indexer is using the same models.

The finished directory is distributed outside Git because it is large, it contains derived graph text from the corpus, and the repository ignore rules exist so that an index, a snapshot, or a response cache is not pushed. After the process exits, copy it with the experiment `snapshot` command, then `verify`. Send the verified copy by disk or object storage. Point the service at that copy with `--lightrag-workdir`. Do not merge a partial copy into the finished directory.

## 12. GitHub branch and deployment handoff

Remote `origin` is `https://github.com/andreahcruz/298P.git`. GitHub redirects that repository to `andreahcruz/PersonaCloningGen`. The remote URL in this clone was not changed.

No pull request was opened from the earlier handoff. After `gh auth login`:

https://github.com/andreahcruz/PersonaCloningGen/compare/andreah/deploy-with-react...kevin/lightrag-model-f?expand=1

`andreah/deploy-with-react` is the EC2 and React deployment line. Both sides have edited `.dockerignore`, `.gitignore`, `README.md`, and `docker-compose.yml` since the merge base, so those four files are conflict-prone. The audit did not run a merge. `kevin/direct-qlora` repeats a patch already on the deploy line. `docs/deployment/DEPLOYMENT_HANDOFF.md` describes the older Relabel plus Chroma package. Its branch, `kevin/298b-integration`, is an ancestor of this branch.

A blind merge would also bring PUK-28 and PUK-34 evaluation work, RAG v2 notes, and the Model F tree into a branch that has since grown its own demo UI. Some of that may be wanted. It should be chosen, not absorbed in one merge.

Safer sequence, for the teammate to perform later:

1. Create an integration branch from `andreah/deploy-with-react`.
2. Read `docs/handoff/MODEL_F_HANDOFF_MANIFEST.md` and this note, and list the Model F modules and their imports.
3. Bring over `host_finetune/model_f_context/`, `host_finetune/model_f_experiment/`, `host_finetune/eval_bench/`, the indexing and relation-recovery modules, the tests, and the small edits in `compare_adapters.py`, `evidence_capsule.py`, `factual_reliability.py`, `numeric_grounding.py`, and `lightrag_backend.py`. Leave React, Docker, and service files on the deploy version unless a Model F change truly requires them.
4. Keep the existing generate route and UI behavior.
5. Run the CPU tests in section 14.
6. Confirm `--retrieval-backend` still defaults to `dense` and `--model-f-context` still defaults to `off`.
7. Add a configuration slot for an external LightRAG directory. Do not point it at `storage_expand_v1` while that directory is being written. The code default for `--lightrag-workdir` is `artifacts/lightrag/model_f/storage`, which is not the expansion directory.
8. When a verified snapshot arrives, set that slot to the snapshot path.

Repository hygiene, as a separate task: `.gitignore` now ignores `artifacts/`, `storage_expand_v1/`, response caches, weights, `*.npy`, `data/cleaned/`, and `deployment_handoff/`. Git still tracks older copies of cleaned JSONL files, `chroma_db/chroma.sqlite3`, and two historical adapter weight files, because ignore rules do not drop tracked blobs. Those blobs remain reachable from history on the shared remote. A hygiene review should inventory tracked large and source-like files and decide whether history needs rewriting. This note does not list their contents. Do not do that rewrite on the deployment branch as part of the Model F integration.

## 13. Remaining work and ownership

Already implemented and tested on CPU, without the live index:

- Resumable indexing, ledger, blog-only scheduling, timeouts, and failure classes.
- Graph-context pipeline and the four rendered profiles.
- Four-arm runner, snapshot, verify, and audit.
- Evaluation harness and the ungraded development benchmark.
- Tests and the implementation docs.

Implemented, and not exercised on the expansion graph:

- A live LightRAG query through `query_documents`.
- Entity-description pass-through inside that function.
- Graph-context generation with the Relabel adapter.
- The 576-cell four-arm run.

Waiting on indexing:

- A finished blog-only store.
- Snapshot and `verify`.
- A read-only graph audit of that snapshot.
- The four-arm experiment and its `eval_bench` report.

Teammate, and can start now:

- Integration branch, file selection, conflict resolution.
- Service configuration for an external workdir, with Model E left as the default.
- Deployment tests that do not open the live store.
- Deployment notes for where the snapshot will be mounted.
- The Demo 2 web UI, which is already the subject of `andreah/deploy-with-react`.

Teammate, after the snapshot exists:

- Point the service at the verified directory.
- Smoke-test one hybrid query on a copy, not on a directory that is still being written.

Still owned with the indexing machine:

- Let the current indexer finish. Do not start a second one.
- Snapshot and verify.
- Apply the deferred `lightrag_backend.py` description pass-through only after the indexer has exited.
- Run and interpret `EXP-20261101-009-model-f-four-arm`.
- Report those results separately from the October pilot.

Independent of indexing: reading the code, merging a selective integration branch, keeping Model E as the default, and writing the service hook for a path that does not exist yet.

Depends on indexing: any command in section 14 except the CPU tests and `plan` against a fake snapshot path. `plan` only writes a manifest. `snapshot`, `verify`, `audit`, `run`, and live queries wait until the lock is gone.

## 14. Commands and practical reference

From the repository root. The venv is `host_finetune\.venv`. Do not recreate that venv while indexing is using it.

CPU tests:

```text
host_finetune\.venv\Scripts\python.exe -X utf8 -m pytest tests/test_model_f_graph_context.py tests/test_model_f_experiment.py tests/test_eval_bench.py tests/test_eval_bench_historical.py tests/test_dev_benchmark.py tests/test_graph_relationships.py tests/test_relation_normalization.py tests/test_indexing_checkpoint.py tests/test_indexing_extraction_audit.py tests/test_indexing_quality.py tests/test_indexing_selection.py tests/test_indexing_timeout.py tests/test_evidence_capsule.py tests/test_factual_reliability.py tests/test_numeric_grounding.py tests/test_lightrag_retrieval.py -q --tb=line -p no:cacheprovider
```

Indexing status, only when a concurrent read of the ledger is acceptable. Prefer this after the writer has exited:

```text
host_finetune\.venv\Scripts\python.exe -X utf8 -m host_finetune.indexing_session --status --selection-policy blog-only
```

After the process has exited and `indexing.lock` is absent:

```text
host_finetune\.venv\Scripts\python.exe -X utf8 -m host_finetune.model_f_experiment snapshot --source artifacts\lightrag\model_f\storage_expand_v1 --dest artifacts\lightrag\model_f\storage_expand_v1_snapshot --allow-expansion-source
```

```text
host_finetune\.venv\Scripts\python.exe -X utf8 -m host_finetune.model_f_experiment verify --snapshot artifacts\lightrag\model_f\storage_expand_v1_snapshot
```

```text
host_finetune\.venv\Scripts\python.exe -X utf8 -m host_finetune.model_f_experiment audit --snapshot artifacts\lightrag\model_f\storage_expand_v1_snapshot
```

`audit` writes beside the snapshot and does not apply recovery.

Optional dry plan, after a snapshot path exists:

```text
host_finetune\.venv\Scripts\python.exe -X utf8 -m host_finetune.model_f_experiment plan --questions host_finetune\benchmarks\dev_retrieval_benchmark.jsonl --out experiments\EXP-20261101-009-model-f-four-arm --snapshot artifacts\lightrag\model_f\storage_expand_v1_snapshot
```

Four-arm run. This loads the Relabel adapter and calls Ollama. Run one generation at a time. The same command resumes.

```text
host_finetune\.venv\Scripts\python.exe -X utf8 -m host_finetune.model_f_experiment run --questions host_finetune\benchmarks\dev_retrieval_benchmark.jsonl --out experiments\EXP-20261101-009-model-f-four-arm --snapshot artifacts\lightrag\model_f\storage_expand_v1_snapshot
```

```text
host_finetune\.venv\Scripts\python.exe -X utf8 -m host_finetune.model_f_experiment evaluate --out experiments\EXP-20261101-009-model-f-four-arm
```

A service-shaped generation call, with the graph prompt still off until a snapshot is accepted:

```text
host_finetune\.venv\Scripts\python.exe -X utf8 -m host_finetune.compare_adapters --adapters relabel=host_finetune/output/lemkin_lora_relabel --retrieval --retrieval-backend lightrag --lightrag-workdir PATH\TO\VERIFIED\SNAPSHOT --lightrag-mode hybrid --model-f-context off --skip-score --out-dir experiments\EXP-YYYYMMDD-NNN-name
```

Flags are defined in `compare_adapters.py` and `model_f_experiment/__main__.py`. There is no environment variable for the LightRAG directory.

## 15. Important caveats and known limitations

- The expansion index is partial and was locked by a writer when this note was drafted. Do not query it, copy it into Git, or mount it for deployment.
- `query_documents` does not yet return entity descriptions as their own list. F-DESC on live `query_documents` output will see `entity_descriptions_not_retained` greater than zero.
- October 31 cells A–D are not F-SOURCE, F-REL, F-DESC, or F-HOP.
- A two-hop path is not a single-source claim.
- Inferred, rejected, and unsourced edges stay out of the prompt.
- Relationship recovery was a dry run. Nothing was inserted.
- Ranked retrieval metrics are empty until grades exist.
- Style distance is not a score in this harness.
- The four-arm blog and talk budgets are the window-fitted 557 and 83 tokens, not the larger defaults in `budget.py`.
- `last_graph_counts` in the ledger is a flush-time measurement, not the live graph.
- Historical cleaned datasets, a Chroma database, and two adapter weight files remain in Git history. Ignore rules do not remove them.
- `deployment_handoff/` is the older local adapter and Chroma bundle. It is not this graph.
- The attached rubrics use classification language (accuracy, loss, ROC/AUC, confusion matrices). This project scores generation and retrieval. Do not invent those classification numbers to fill the sheet.

## 16. Recommended reading order

1. This note, sections 1, 3, 7, and 17.
2. `docs/handoff/TEAMMATE_DEPLOYMENT_HANDOFF.md` for setup and the workdir flag.
3. `docs/handoff/MODEL_F_HANDOFF_MANIFEST.md` for which files were shared and which stayed local.
4. `docs/implementation/MODEL_F_GRAPH_CONTEXT_IMPLEMENTATION.md` for channels, ranking, and budgets.
5. `docs/implementation/MODEL_F_FOUR_ARM_EXPERIMENT.md` for the run that is still blocked.
6. `docs/implementation/MODEL_F_POST_INDEX_CHECKLIST.md` for the description pass-through that waits on the indexer.
7. `docs/architecture/LIGHTRAG_INDEXING_AND_CHECKPOINTING.md` for the ledger and resume rules.
8. `docs/evaluation/MODEL_F_EXPERIMENT_SPEC.md` and `docs/evaluation/RETRIEVAL_ANNOTATION_GUIDE.md` before anyone cites precision or recall.
9. `docs/architecture/LIGHTRAG_MODEL_F.md` for older measured runs, with section 10 of this note beside it.
10. `host_finetune/model_f_experiment/arms.py` if the profile table and the code need to be checked against each other.
11. `docs/rubric_matrix.md` only as the October 3 planning baseline. Section 17 is the map to the attached Workbook 2 and Demo 2 sheets.

## 17. Workbook 2 and Demo 2 rubric map

The attached sheets are the scoring reference for this handoff. Point values below are copied from those sheets. Status words follow `AGENTS.md`: PROPOSED, IMPLEMENTED, TESTED, MEASURED, VALIDATED. Nothing in this section is VALIDATED against the finished blog graph.

`docs/rubric_matrix.md` still says Workbook 4.1 excellent work proposes five or more models. The attached Workbook 2 sheet says three or more, and it adds 4.5, section 6, and three appendices that the October matrix does not list. Use the attached sheet for the workbook that is due now.

### Workbook 2, section 4 — models (3.0 points on the attached sheet)

| ID | Points | What the sheet asks for | Where this branch stands |
|---|---:|---|---|
| 4.1 Model proposals | 0.5 | Three or more models, in terms of concepts, features, architectures, and algorithms. At least one is improved relative to an existing model. | Model E dense Relabel, Model F with graph context off, and the four graph-context profiles are specified. The improved piece is source-verified graph context on the same Relabel adapter. The profiles are configurations, not extra trained models. Status: IMPLEMENTED. A graded write-up still has to say this in workbook language. |
| 4.2 Model supports | 0.5 | Platform, environment, and tools for each model, plus diagrams of architecture, components, and data flow. | Section 3 of this note, `docs/architecture/LIGHTRAG_MODEL_F.md`, and `docs/implementation/MODEL_F_GRAPH_CONTEXT_IMPLEMENTATION.md`. Stack: Python 3.11, Unsloth/QLoRA, Ollama, LightRAG 1.5.7, Chroma, the host venv. Status: IMPLEMENTED as documentation. The diagrams are Markdown, not a separate figure pack. |
| 4.3 Comparison and justification | 0.5 | Compare problems, features, approaches, strengths, and limits, and justify each model. | The comparison this branch can support is Model E versus Model F, then F-SOURCE versus F-REL, F-DESC, and F-HOP under a frozen payload. Strengths and limits are written in sections 6, 7, and 15. The measured comparison is still the October pilot, which is not these profiles. Status: IMPLEMENTED as a design. The result table is not MEASURED yet. |
| 4.4 Evaluation methods | 0.5 | Methods and metrics for each model. The sheet names accuracy, loss, ROC/AUC, and MSRE as examples. | This system does not train a classifier, so those examples do not apply. The methods that do exist are in section 9: faithfulness, numeric grounding, citation-marker leak, source copying, latency, tokens, paired differences, and bootstrap intervals. Precision and recall wait on human grades. Status: IMPLEMENTED and TESTED as code. Say explicitly in the workbook that ROC/AUC is not the metric. |
| 4.5 Validation and evaluation results | 1.0 | Present and compare detailed results, with the solution shown against validated outputs. | Older registered runs in section 10 are historical evidence. `EXP-20261101-009` has not been run. Do not fill this row with October 31 cell letters. Status: PROPOSED until the four-arm run is MEASURED. |

### Workbook 2, section 5 — system (2.0 points on the attached sheet)

| ID | Points | What the sheet asks for | Where this branch stands |
|---|---:|---|---|
| 5.1 Requirements | 0.5 | System boundary, actors, use cases, and the high-level data and learning functions. | Actors implied by the code: a person who asks for a draft, the retrieval backend, the Relabel generator, and the evaluator. Boundary: generation stays in `compare_adapters`; the React UI is outside this branch. A formal use-case write-up is not in this commit. Status: PROPOSED as a workbook section. The functions themselves are IMPLEMENTED. |
| 5.2 System design | 0.5 | AI architecture, integration with platforms and cloud, data management, and UI or visualization. | Local design is section 3. Data stores are Chroma, the LightRAG workdir, and the SQLite indexing ledger. Cloud and the web UI belong to `andreah/deploy-with-react`, not to this branch. Status: IMPLEMENTED for the modeling path. The UI line of this criterion is the teammate's deployment work. |
| 5.3 Intelligent solution | 0.5 | The learning system for each targeted problem, plus inputs, outputs, context, and APIs. | Targeted problem: persona draft grounded in author text. Input: topic, medium, and either Chroma or a LightRAG directory. Output: a draft plus claim and gate records. API surface today is the `compare_adapters` CLI, not a route in `backend/`. Status: IMPLEMENTED as a CLI. The service API is not wired. |
| 5.4 Support environment | 0.5 | Technologies, platforms, and frameworks. | `host_finetune/requirements.txt`, `host_finetune/requirements-lightrag.txt`, Ollama models `llama3.1:latest` and `nomic-embed-text`, CUDA torch only for generation. Status: IMPLEMENTED. Freeze the package list from the machine that runs the four-arm experiment when that run exists. |

### Workbook 2, section 6 and appendices

The attached page lists 6.1 through 6.4 at 2 points each, then Appendices A–C at 1 point each. The page footer reads 15 points. That footer may include rows above the photographed region. The row values below are the ones printed on the criteria.

| ID | Points | What the sheet asks for | Where this branch stands |
|---|---:|---|---|
| 6.1 Analysis of model execution and evaluation | 2.0 | Methodology for accuracy, loss, precision, recall, F-score, AUC, or confusion measures, applied to tagged outputs. | Methodology for this project is section 9. There is no accuracy, loss, F-score, AUC, or confusion matrix for the generator. Precision and recall are defined and unimplemented until grades exist. Status: the method is IMPLEMENTED. The analysis of a finished Model F run is not MEASURED. |
| 6.2 Achievements and constraints | 2.0 | What was achieved on the target problems, and which constraints were encountered. | Achieved: resumable blog-only indexing, graph-context code, four profiles, runner, harness, and CPU tests. Constraints: indexer still writing the partial store, entity descriptions not passed through `query_documents`, no live four-arm result, deploy-branch divergence, and the index cannot travel in Git. Status: the narrative is this note. It is not a graded workbook chapter. |
| 6.3 System quality and response time | 2.0 | Correctness of the model and runtime against response-time targets. | Correctness checks exist (faithfulness, numeric gate, copy check). Latency fields exist in `eval_bench`. No response-time target is defined, and no end-to-end time has been measured on the finished index. The October pilot's per-draft times are historical, not this run. Status: PROPOSED as a measured claim. |
| 6.4 System visualization | 2.0 | Charts or other visualizations of data, analysis, and learning outcomes. | This branch did not add a chart pack. Older figures live under `docs/evaluation/`. The four-arm runner writes JSON and Markdown, not plots. Status: PROPOSED for Model F. Do not reuse October charts as the new profile comparison. |
| Appendix A — system testing | 1.0 | Required use-case tests with CI parameters: performance, consistency, accuracy, and loss. | CPU pytest command in section 14. Recorded result: 156 passed. There is no CI dashboard score in this handoff, and loss is not a test output. Status: TESTED for the CPU suite. |
| Appendix B — data source and management | 1.0 | Project data sources, in the required formats, placed in the designated folder. | Cleaned sources and split manifests already exist in the repository from earlier commits. The LightRAG index is not a Git artifact. The corpus test split stays closed. This note does not list document text. Status: the dataset question is separate from this branch. Confirm the designated upload folder against the course instructions before treating Git as that folder. |
| Appendix C — program library, presentation, and demo | 1.0 | Program artifacts uploaded with a clear file layout: code, notebooks, decks, and demo video. | This branch is the code drop: `host_finetune/model_f_context/`, `host_finetune/model_f_experiment/`, `host_finetune/eval_bench/`, `host_finetune/indexing_*.py`, `tests/test_model_f_*.py`, `tests/test_eval_bench*.py`, and `docs/implementation/`. A demo video and a slide deck are not in this commit. Status: IMPLEMENTED for the library. Presentation files are still PROPOSED. |

### Demo 2 (12.5 points on the attached sheet)

| Criterion | Points | What an excellent score requires | What to do with this branch |
|---|---:|---|---|
| Machine learning modeling | 3.0 | Complete modeling, including selection, improvement, innovation, and comparison. | Selection is Relabel plus a retrieval backend. The improvement is graph context with source checks. Comparison waits on `EXP-20261101-009`. Do not present the October cells as that comparison. |
| Machine learning evaluation results | 3.0 | Correct comparative results on defined metrics. | The metric definitions are in section 9. Results for the new profiles do not exist yet. Historical numbers in section 10 can be shown only with that label. |
| Web-based system design and demo UI | 4.0 | A finished web system with dynamic demo features. | Not this branch. `andreah/deploy-with-react` is the UI line. Integrating Model F means a workdir setting and a default that stays on Model E until the snapshot exists. This is the largest Demo 2 item and it is the teammate's track. |
| Presentation | 2.5 | Clear progress story, visuals, and a logical flow. | Suggested spine: persona problem, Model E, why a graph was added, four profiles of one model, what the CPU tests prove, what is blocked on the index, and what the demo UI will call. |

### What an assistant should not claim

- Do not mark 4.5, 6.1, 6.3, or Demo 2 evaluation as measured Model F graph-context results.
- Do not report precision, recall, F-score, ROC, or AUC for this generator.
- Do not describe F-SOURCE, F-REL, F-DESC, and F-HOP as four trained models or four indexes.
- Do not point the demo at `artifacts/lightrag/model_f/storage_expand_v1` while indexing is running.
- Do not merge this whole branch into the deploy branch to satisfy the web-UI criterion.
