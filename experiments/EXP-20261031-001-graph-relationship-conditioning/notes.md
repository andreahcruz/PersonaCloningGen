# Source-verified relationship conditioning

Question: on the existing 152-document LightRAG pilot, do source-supported graph relationships add useful evidence that matched dense retrieval misses, and does giving those relationships to the frozen Relabel generator improve drafts when the passages are held fixed?

Held fixed for arms A and B: user query, medium, hybrid LightRAG passages, accepted atomic claims, Relabel adapter, decoding, seed 42, output-token budget, and the existing prompt instructions. Arm B adds only the relationship block. Arm D adds a non-factual pad sized to that block and was generated for blog only. Arm C is matched pilot dense retrieval over the same 160 texts, truncated to the existing 400-character dense preview. It does not use the same passages as A and B.

The pilot graph was read and not edited. No new documents were indexed. `--fixed-style` stayed off. `factual-reliability-v1` was already the prompt contract and was not switched on for this run.

## Why relationship text was excluded

`normalize_lightrag_data` keeps verbatim source chunks as hits. Entity and relationship rows are logged as `generated_summary_not_a_source_span` and are not passed to claim extraction. LightRAG descriptions are model summaries, often merged with `<SEP>`, and are not quotations. The generator therefore saw passages the graph helped retrieve, not the graph edges themselves.

## Contract

`source-verified-relationship-v1` accepts a directly stated relationship only when one train passage contains both endpoints and at least three description content words, a multi-word keyword, or a verbatim description clause. The generator receives at most four such relationships whose source id is one of the retrieved passages. Inferred cross-document links, failed documents, omission-marker spans, weak names, unresolved conflicts, misattributed quotations, and instruction-like text are rejected or quarantined and are not prompt claims. A read-only pass over the 2,000 pilot edges accepted 520, quarantined 1, and rejected 1,479. The largest rejection reason was `generated_summary_not_supported_by_source` (1,039), then weak entity names (293).

## Retrieval

Twelve questions. Mean seconds: matched dense 2.047, naive 2.831, hybrid 3.603, mix 3.091. Overlap with a preselected id is not precision or recall.

Dense and all three LightRAG modes recovered the preselected source for Brex, corporate VC, the VP tension pair, 5 percent share, Anneka Gupta, the craft contrast, the cap table, and the B2C playbook. Dense recovered all three first-VP posts; naive, hybrid, and mix recovered two. Dense missed the 2021-boom promotion post and the late-investor-update post; naive, hybrid, and mix returned both. Every mode returned four passages for the Olympic question. None of those passages is about that ceremony.

The relationships admitted for generation mostly restated a sentence already in a retrieved passage. On the Anneka question they were off-topic edges from the retrieved set. On the Olympic question hybrid and mix still marked four in-passage relationships for generation. Those edges are about AI, Algolia, Google, and a SaaStr Annual AMA. They are not about the ceremony. That question was a retrieval control and was not generated.

## Generation

Five topics, four media, arms A, B, and C, plus five blog pad controls. Every attempted draft was scored. Qwen 2.5 14B writing means are diagnostic. Fixed-length blog voice was not scored: the Kevin reference worktree is not in this checkout. Legacy eligibility is not the decision metric.

Blog prompt tokens, B minus A: Brex 88, VP tension 121, Anneka 88, craft 214, cap table 297. The blog pad landed within 41 tokens of arm B, and within 18 tokens on four of the five topics.

| Arm | Drafts | Legacy eligible | Blog factual pass | Blog writing mean | Blog style distance |
|---|---:|---:|---:|---:|---:|
| A passages | 20 | 5 | 4/5 | 2.07 | 2.01 |
| B passages plus relationships | 20 | 3 | 2/5 | 2.27 | 1.30 |
| C matched dense | 20 | 4 | 4/5 | 2.40 | 1.91 |
| D blog token pad | 5 | 1 | 4/5 | 3.23 | 1.23 |

Lower style distance is closer to the reference profile. The pad's blog writing mean is higher than arm B, so the small writing movement from A to B is not evidence that relationship content helped. Arm B's blog factual pass rate fell from 4/5 to 2/5.

Read examples, not the means alone:

- Craft blog: A already stated the win/lose contrast and added an unsupported $250M aside. B repeated the contrast more briefly and still opened with the question. The relationship block restated the passage.
- Cap-table blog: B was shown the sentence that a messy cap table can scare off investors and slow fundraising. The draft instead said the company does not value itself highly.
- VP-tension blog: A and B are the same short refusal, "No answer found in the evidence." The admitted edges describe a mediocre VP of Sales, not the asked contrast.
- Anneka blog: A and B both name Anneka Gupta. B then drifts toward an Asana intents discussion that matches the off-topic relationship lines.
- Four of the five LinkedIn drafts in arm B copy the prompt instructions into the post, including `Prompt contract: factual-reliability-v1`. Arm A does that only in the Brex X draft.

## Decision

Model F does retrieve two relevant pilot passages that this matched dense search missed, and it misses one relevant first-VP passage that dense retrieved. That is not a general retrieval win. Source-verified relationship conditioning does not improve the frozen Relabel drafts. Broader indexing is not justified. Another indexing pass would spend time on the seven known timeouts and would not repair the generation result.

Keep Model E dense retrieval on the active Chroma index as the default. Leave `--graph-relationships` off. Do not enable `--fixed-style` for this comparison. Do not index the remaining graph-ready corpus.

## Reproduce

From the repository root, with `host_finetune\.venv`, and without a concurrent QLoRA job:

```powershell
host_finetune\.venv\Scripts\python.exe -X utf8 -m pytest tests\test_graph_relationships.py tests\test_lightrag_retrieval.py tests\test_evidence_capsule.py -q
host_finetune\.venv\Scripts\python.exe -X utf8 experiments\EXP-20261031-001-graph-relationship-conditioning\config\run_retrieval.py
ollama stop llama3.1:latest
host_finetune\.venv\Scripts\python.exe -X utf8 experiments\EXP-20261031-001-graph-relationship-conditioning\config\run_generation.py
host_finetune\.venv\Scripts\python.exe -X utf8 experiments\EXP-20261031-001-graph-relationship-conditioning\config\score_generation.py --judge
```

The generation script skips cell files that already exist. Retrieval skips a frozen file that already exists. Neither command writes the pilot graph or either Chroma directory.
