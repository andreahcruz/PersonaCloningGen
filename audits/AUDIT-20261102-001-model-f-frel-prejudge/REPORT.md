# Model F F-REL pre-judge quality audit

**Audit ID:** AUDIT-20261102-001-model-f-frel-prejudge  
**Experiment:** `experiments/EXP-20261102-001-model-f-unified-panel`  
**Profile:** F-REL  
**Panel:** `gold_001`–`gold_030` × blog, LinkedIn, X, talk (120 cells)  
**Gold SHA-256:** `d7c4f997899e28f28eb3ea9aa3417857da6f0666acbf72fd8ac31422034e520e`  
**Manifest status:** `generated_not_judged` (written 2026-11-02 16:24)  
**This audit:** read-only on experiment files. No new generations, no retrieval rerun, no GPU model load, no LLM judge.

Derived files in this directory: `summary.json`, `cells.jsonl`, `questions.json`, `sample_answers.json`, `build_audit.py`.

## Overall summary

All 120 expected cells are on disk, non-empty, and marked `complete`. There are no duplicate question-medium cells and no missing pairs. Progress records 120 completed and 0 failed.

The drafts are not clean enough to treat an LLM judge as a factuality result.

- **52/120 cells (43.3%)** fail the existing unsupported-number check. Those cells contain **185** flagged quantities. On the worst cells, the flagged figures are absent from both the accepted claims and the saved retrieved passages.
- **7/120 cells (5.8%)** copy 12 or more contiguous words from a retrieved passage. **10/120** share 8 or more words, which is the prompt’s own copying rule. The training-memorization gate, re-read from `unified_eval/rows.jsonl` after 16:43, fails **0/120**.
- **4/120 cells** are `gold_030`. Extraction status is `extraction_failed`, with **0 accepted claims**. The blog and talk still state specific Figma revenues that are not in the saved passages.
- **36/120 cells (30.0%)** fail the sentence-final completion gate. **36/120** also emitted exactly the medium’s `max_new_tokens`. Those stops concentrate on LinkedIn (17/30) and X (19/30). Blog and talk never hit their 768-token cap.
- **56/120 cells (46.7%)** contain the full topic string (`embeds_topic`). **55/120** open with it or are only the topic (`repeats_topic`). X accounts for 23 of the 30 topic echoes.
- Format-band pass is **116/120**. That band lets an X post of 3 words pass, so it does not catch title-only posts. The prompt’s own floors (LinkedIn ≥ 40 words, X ≥ 20 words) fail **12/120**.

**Judging recommendation:** the panel is complete and can be scored later. It is not ready for a judge score to stand in for factual reliability. Keep the deterministic flags beside any later judge. Do not describe F-REL as factually reliable on this panel.

`unified_eval/rows.jsonl` and `judge_cache.jsonl` were rewritten at 16:43 and now contain writing-quality fields on all 120 rows. `generation_manifest.json` and `progress.json` still say `generated_not_judged` and `judge_overlay: not_applied`. This audit did not produce those judge fields and does not use them.

## Four-medium comparison

Denominators are 30 cells per medium unless noted. Rates are count/30.

| Metric | Blog | LinkedIn | X | Talk | All (120) |
|---|---:|---:|---:|---:|---:|
| Saved, non-empty, `complete` | 30 | 30 | 30 | 30 | 120 |
| Accepted claims = 0 | 1 | 1 | 1 | 1 | 4 |
| Unsupported-number cells | 19 (63.3%) | 9 (30.0%) | 10 (33.3%) | 14 (46.7%) | 52 (43.3%) |
| `repeats_topic` | 13 | 15 | 23 | 4 | 55 |
| `embeds_topic` | 13 | 16 | 23 | 4 | 56 |
| RAG copy fail (span ≥ 12) | 3 | 1 | 1 | 2 | 7 |
| Prompt copy span ≥ 8 | 3 | 3 | 2 | 2 | 10 |
| Completion pass | 30 | 13 | 12 | 29 | 84 (70.0%) |
| Mid-sentence stop | 0 | 17 | 18 | 1 | 36 |
| Output tokens = max | 0 | 17 | 19 | 0 | 36 |
| Format-band pass | 29 | 28 | 30 | 29 | 116 |
| Prompt length floor fail | 0 | 6 | 6 | 0 | 12 |
| Words, mean / median / p95 | 246 / 261 / 318 | 186 / 238 / 271 | 90 / 117 / 130 | 290 / 298 / 341 | 203 / 244 / 323 |
| Output tokens, mean / median | 320 / 342 | 238 / 320 | 121 / 160 | 364 / 372 | 261 / 320 |
| Generation seconds, mean / median | 24.0 / 26.0 | 19.7 / 23.2 | 9.6 / 9.3 | 27.0 / 28.6 | 20.1 / 23.7 |
| Tokens/second, mean / median | 14.4 / 13.0 | 12.2 / 12.9 | 12.7 / 12.8 | 14.5 / 13.1 | — |

Word range: min 8 (`gold_010` LinkedIn), max 465 (talk). X median output tokens equal the 160 cap. LinkedIn median output tokens equal the 320 cap.

## Metric definitions and denominators

| Metric | Definition | Denominator | Status |
|---|---|---|---|
| Saved / empty / exceptions | Generation file exists; `answer` strip-empty; `status != complete` | 120 expected cells | Executed. 120 / 0 / 0 |
| Missing fields | Required keys present and non-null: ids, query, answer, prompt, status, evidence, claims, retrieved ids, tokens, latency, generation config, grounding texts, window omissions, truncation flag | 120 | Executed. 0 missing |
| Duplicate cells | More than one file per `(question_id, medium)` | 120 pairs | Executed. 0 |
| Coverage | `gold_001`–`gold_030` × 4 media | 120 | Executed. 120/120 |
| Retrieved chunks | `graph_summary.raw_chunk_count` | 30 questions | Executed. 12 on every question |
| Accepted / rejected graph edges | `accepted_relationships`, `rejected_relationships` | 30 questions | Executed. Means 33.7 accepted, 75.1 rejected |
| Accepted / rejected claims | `evidence/*.json` `verified` vs `validation_rejected` + `verifier_rejected` | 30 questions | Executed. Mean 10.7 accepted, 3.8 rejected. `gold_030` accepted = 0 |
| Extraction truncation | `extraction_input_truncated` | 30 questions and 120 cells | Executed. 0 |
| Whole-passage omission | `extraction_hits_omitted` | 30 questions | Executed. 5 questions, 20 cells |
| Graph-item omission | non-empty `window_omitted` | 120 cells | Executed. 2 cells, both `gold_011`, 9 items each |
| Unsupported numbers | `evaluate_numeric_grounding` inside `score_generation` | 120 cells, all `numeric_applicable` | Executed. 52 cells, 185 quantities |
| `repeats_topic` / `embeds_topic` | Existing functions in `evidence_capsule` and `factual_reliability` | 120 | Executed |
| Role assertions | `role_assertions` via `assess_factual` | 120 | Executed. 1 cell with an unsupported person-role-org triple (`gold_006` LinkedIn) |
| Speaker attribution | `speaker_attribution` | 8 cells where a claim has `quoted_speakers`; 112 not applicable | Executed. 8/8 applicable cells fail |
| Faithfulness pass | `assess_factual` `pass` (no accepted evidence, numeric, role, speaker, or RAG copy failure) | 120 | Executed. 61 fail |
| RAG `copy_pass` | `evaluate_rag_source_copying`, fail at 12 contiguous words | 120, retrieval on | Executed. 7 fail |
| Training copy gate | Saved `gates.copying_memorization`, fail at 12 words after dropping spans that also occur in the prompt | 120 rows on the 16:43 file | Recorded gate, re-read. 0 fail. Not rebuilt from the training corpus in this audit |
| Citation / marker leak | `instruction_leak` or `marker_leak` | 120 | Executed. 2 cells. `gold_016` X repeats a prompt sentence the marker list does not contain; found by reading |
| Lexical grounding gate | `grounding_gate`, sentence overlap ≥ 0.2 with saved passage text | 120 | Executed. 8 pass. This gate fights the prompt’s anti-copy rule, so a low pass rate is a weak factuality signal here |
| Claim presence | Audit diagnostic only: 4 contiguous claim tokens in the answer, or the whole claim if shorter | 1,284 claims on 116 cells | Not an existing scorer. 48/1,284 (3.7%). Too strict to treat as “evidence unused,” and too lexical to treat as entailment |
| Relationship endpoint mention | Audit diagnostic: both endpoint names from a `[R]` line appear in the answer | 475 lines | Not an existing scorer. 40/475 (8.4%). Not proof a relationship was used correctly |
| EOS / early EOS | `stop_reason` or `early_eos` on the generation record | 120 | **Not measured.** Neither field is stored |
| Max-token stop | `tokens.output >= max_new_tokens` | 120 | Inferred from counts, not from a stop reason. 36 cells |
| Mid-sentence | `completion_gate` / `sentence_final` | 120 | Executed. 36 fail |
| Format band | blog/talk 40–1200 words, LinkedIn 15–500, X 3–280 | 120 | Executed. 116 pass |
| Prompt floors | LinkedIn ≥ 40, X ≥ 20, from the prompt text | 60 LinkedIn+X cells; blog/talk not given that sentence | Executed for LinkedIn and X. 12 fail |
| GPU memory | `gpu_memory_bytes` | 120 | **Not measured.** Field is null |
| Verification latency | `latency.verification_seconds` | 120 | Measured and recorded as 0.0 |

Retrieval and extraction latency are reported **once per question**, from the retrieval and evidence files. The same retrieval time is also copied onto all four generation cells; summing cells would quadruple it.

Per question (n=30): retrieval seconds mean 12.9, median 12.9, p95 13.7. Extraction seconds mean 36.4, median 29.6, p95 74.3 (min 14.4, max 75.0). Prompt tokens on cells: mean 833, median 801, p95 1,159. Final graph-context tokens (`tokens.context_used`): mean 232, median 201, p95 477. Extraction prompt tokens are stored per question; the five omission records sit between 1,602 and 1,640 against a 2,048 context limit, with truncation false.

## Distribution notes

- Every retrieval file status is `complete`. Every question kept 12 raw chunks. Accepted graph edges range from 4 to 74. Rejected edges range from 16 to 139. Two-hop paths are recorded; they are not what the F-REL prompt is selected to render.
- Whole-passage omissions: `gold_003`, `gold_006`, `gold_012`, `gold_017`, `gold_030`. No passage text was character-truncated.
- Graph-item omissions: `gold_011` blog and `gold_011` talk, 9 items each. Both drafts are 22–23 word title echoes and fail the format band.
- Retries (`accepted_attempt > 1`): 20/120. Talk 10, LinkedIn 5, X 3, blog 2. A retry means the first sample failed a copy or numeric guard. The saved answer is the accepted attempt.
- Repeated full lines of 6+ words: 0/120. Four-gram repetition is low; it is in `summary.json` and is not a failure mode.
- `gold_004` X, `gold_006` talk, `gold_009` blog, and `gold_023` blog share 8–10 training words and still pass the 12-word training gate.

## Comparison with saved A–E rows

The repository does not label architectures A–D as the 120-cell panel. The comparable seed-42 `unified_eval.v1` files are Base, the PersonaRAG stand-in, Relabel with RAG off, and Relabel + RAG (Model E). Compared fields are only `gates.completion`, `gates.format`, `gates.copying_memorization`, and format-gate word count. Those definitions match Model F’s recomputed completion and format gates and the saved training-copy gate.

Writing quality, claim-support rate, style distance, and the historical “60 of 120” Model E number flag are **not compared**. The number flag is a note in `docs/evaluation/METRIC_MAP.md`. The Model E generation traces are not in this workspace, so that 60/120 figure was not recomputed. Model F’s 52/120 uses `evaluate_numeric_grounding` on these cells. Early EOS is recorded for Relabel and Relabel+RAG and is **not measured** for Model F, so those rates are not compared. RAG passage-copy (7/120) is a different check from the training-copy gate (0/120).

| Architecture | n | Completion pass | Format pass | Training-copy fails | Mid-sentence | Words, mean |
|---|---:|---:|---:|---:|---:|---:|
| Base | 120 | 45 (37.5%) | 120 | 0 | 75 | 352.1 |
| PersonaRAG stand-in | 120 | 71 (59.2%) | 120 | 0 | 49 | 296.5 |
| Relabel, RAG off | 120 | 79 (65.8%) | 115 | 1 | 41 | 192.9 |
| Relabel + RAG (Model E) | 120 | 84 (70.0%) | 117 | 2 | 36 | 183.1 |
| Model F F-REL | 120 | 84 (70.0%) | 116 | 0 | 36 | 203.1 |

Completion by medium (pass count / 30):

| Architecture | Blog | LinkedIn | X | Talk |
|---|---:|---:|---:|---:|
| Base | 20 | 5 | 3 | 17 |
| PersonaRAG stand-in | 28 | 10 | 4 | 29 |
| Relabel, RAG off | 29 | 7 | 14 | 29 |
| Relabel + RAG (Model E) | 29 | 10 | 15 | 30 |
| Model F F-REL | 30 | 13 | 12 | 29 |

Model F matches Model E on overall completion (84/120) and mid-sentence count (36/120). Model F X drafts are longer (mean 90 words vs 39.5) and still usually hit the token cap. Model F blog completion is 30/30. LinkedIn and X remain the incomplete media for every Relabel-family row file.

## Highest-priority problematic generations

Ranked by unsupported quantities, then source copy, then missing evidence, then cutoff, then length. Diagnoses marked hypothesis were not isolated by a controlled rerun.

1. **`gold_028` talk** — DigitalOcean earnings. The answer states $52 million EBITDA, $0.38–$0.42 EPS, $48 million and $28 million free cash flow, 150,000 customers, and $49/month. None of those strings are in the 27 accepted claims or the four saved passages. The claims do contain other DigitalOcean figures ($152m EBITDA, $490m ARR, 83% of revenue). Retrieval returned on-topic claims. **Generation** stated a different set of figures. Hypothesis: memorized or fabricated earnings, while the accepted claims were ignored.
2. **`gold_024` blog** — Cloudflare. Accepted claims include 221,000 customers and 65% of revenue from 3,200 large customers. The draft says 220,000, $900M Q1 revenue, 300 enterprise logos, 60% under $5k, and $180/user/month. `$900M`, `220,000`, and `36%` are absent from claims and passages. **Generation** changed a supported count and added figures. The 65% / 3,200 sentence does match accepted claims.
3. **`gold_002` LinkedIn** — Monday.com. Cuts off at 320 tokens on “$250m”. States 70/90/80/71/130/119/122 percent splits. `71%` and `180,000` are not in the claims; `$550m` is in the topic, so the checker allows it. **Generation**, plus a max-token stop. One nearby claim is “70% of features driven by customer feedback,” which is not the sentence the draft wrote.
4. **`gold_002` talk** — same topic. States a 2010 founding, $6 billion market cap, 180,000 paid customers, and 30 million free users. Those strings are not in the claims. “CAC payback < 10 months” is an accepted claim and does appear. **Generation** mixed one supported claim with unsupported figures.
5. **`gold_009` talk** — Atlassian-style figures ($600 million cash flow, $1.2 billion valuation, 25%, 32%, 90%, 60%, 85%). `$1.2 billion` is not in the saved passages. `43,000` is in both the passage and the claims. **Generation** added figures around a thinner supported core.
6. **`gold_012` blog** — 19-word copy: “the most common gap in ai sdr deployments isn't the ai it's what happens when it works.” The talk repeats the same span. **Generation** copied a retrieved passage. Speaker check is applicable and fails on all four `gold_012` media.
7. **`gold_006` blog** — 17-word copy of a Zendesk session title. LinkedIn also fails the role check and the speaker check. A whole passage was omitted before extraction. **Generation** copied source text. Hypothesis: the omitted passage is not required to explain the copied span, because the copied words are in a passage that was kept.
8. **`gold_030` blog** — 0 accepted claims after `extraction_failed` (2 attempts, EOS, one whole passage omitted, truncation false). The draft states $700 million, $253 million, and $695 million quarterly revenue. Those strings are not in the saved passages. `$304M` and `40%` are in the topic. The three revenue sentences conflict with each other. **Evidence verification failed, then generation filled numbers.** Retrieval had returned Figma passages, including a “13 million” monthly-active-user span that never became an accepted claim.
9. **`gold_030` LinkedIn** — same empty claim set. The draft leaves Figma after the title and discusses Cursor, with a `cursor.com` campaign URL. **Generation** on top of failed extraction. The URL is not a retrieved-passage overlap I verified; it reads as a citation-style leak of a tracking link.
10. **`gold_016` X** — the entire post is “A LinkedIn post must be at least 40 words. An X post must be at least 20 words.” Attempt 2. The existing leak marker list does not contain that sentence, so `instruction_leak` is false. **Generation** emitted prompt text. This is the clearest instruction leak in the panel.

Also inspect, below the top 10: `gold_011` blog and talk (title only, 9 graph items dropped), `gold_010` LinkedIn (8 words, the title), `gold_015` LinkedIn (10 words, the title) and `gold_015` blog (`[E1]` left in the prose), `gold_027` LinkedIn (the heading `FACTUALITY AND ATTRIBUTION` appears as a section title).

## Twelve samples

Factual support is unknown unless the string was checked against the saved claims or passage text.

### Representative

**`gold_027` blog.** Topic: “We Have 30 AI Agents in Production. Here Are the Top 5 Issues No One Talks About.”  
Path: `experiments/EXP-20261102-001-model-f-unified-panel/raw/generations/gold_027__F-REL__blog.json`  
289 words, 377/768 tokens, attempt 1, 11 claims, 6 relationship lines, no deterministic fail.  
Excerpt: “We’ve been running 30+ AI agents across our entire GTM stack since late summer… the issues aren’t technical anymore. They’re operational… let me tell you about the 15 things no one talks about… The First 90 Days Aren’t Enough. People think once they deploy an agent, it runs forever. It does not.”  
The prose is coherent and reads as a blog. It does not deliver five issues. It says “15 things” and develops two headings. Accepted claims include “2.5 humans + 20 AI agents,” “275,000 startup valuations,” “11 AI Agents in production,” and “AI closed a $70K deal.” None of those strings appear in the draft. The “30” matches the title, which the numeric checker treats as allowed. Relationship lines are malformed (“get most production learn often train”) and are not used as stated facts. **Evidence was available and largely unused.** Selected because it is the cleanest blog near the median length.

**`gold_021` LinkedIn.** Topic: “If Your AI Feature Didn’t Materially Boost Revenue, It Doesn’t Count. Try Again.”  
Path: `.../gold_021__F-REL__linkedin.json`  
214 words, 279/320 tokens, 8 claims. Opens by repeating the topic, so `embeds_topic` is true; other deterministic checks pass.  
Excerpt: “You’re probably using AI wrong if your AI feature hasn’t actually improved revenue yet… Don’t just slap a chatbot on top of existing workflows… AI is like electricity.”  
Coherent LinkedIn prose, on the topic, and finished. Accepted claims (“90% accuracy in AI is not good enough in sales,” “Maintaining two products is not recommended”) do not appear. Support for the electricity analogy is **unknown**. Selected as the least-warned LinkedIn draft near that medium’s median.

**`gold_029` X.** Topic: “SaaS Isn’t Dead. But the Way You Used to Win in B2B? That’s Gone.”  
Path: `.../gold_029__F-REL__x.json`  
115 words, **160/160 tokens**, ends with “And”. Completion fails. No X cell was free of a deterministic warning; this one is near the X median length.  
Excerpt: “SaaS Isn’t Dead. But the way you used to win in B2B? That’s gone: Growth rates have fallen by half since 2021… There are 5x-10x more vendors competing for every deal… And”  
The list form fits X. It echoes the topic and stops mid-sentence at the cap. Accepted claims are about 30% to 25% growth, 6.9% flat-or-negative, and the $10M–$100M path. “Fallen by half” and “5x-10x” were not in those four claims. **Generation cutoff plus claims not carried into the post.** Hypothesis: the 160-token budget, not retrieval, ends the post.

**`gold_007` talk.** Topic: “Why Now Is the Biggest Change in SaaS Sales in 15+ Years.”  
Path: `.../gold_007__F-REL__talk.json`  
298 words, 372/768 tokens, **attempt 2**, 8 claims, completion passes.  
Excerpt: “this is the biggest change in SaaS since we started selling software 15 years ago… the Age of AI. It was sort of launched by Claude in late November, early December of '23… Ninth, they expect your product to improve constantly.”  
Spoken cadence fits a talk. The list ends on a sentence, so the completion gate passes, and it does not close the argument. Claims about multiyear pricing and upsells are not the nine points in the draft. “15 years” is in the topic. Support for the Claude date is **unknown** (the numeric checker did not flag it). Selected as the cleanest talk near the talk median. The retry means attempt 1 failed a guard; the saved text is attempt 2.

### Strongest warnings

**`gold_028` talk**, **`gold_024` blog**, **`gold_002` LinkedIn**, **`gold_002` talk.** Described in the ranked list above. Paths:

- `raw/generations/gold_028__F-REL__talk.json`
- `raw/generations/gold_024__F-REL__blog.json`
- `raw/generations/gold_002__F-REL__linkedin.json`
- `raw/generations/gold_002__F-REL__talk.json`

`gold_028` is on the DigitalOcean topic and fluent. The specific quarter it narrates is not the quarter in the accepted claims. `gold_024` keeps one real claim pair (65%, 3,200) and invents the lead numbers. `gold_002` LinkedIn is cut off. `gold_002` talk is a finished monologue with a different customer count than the claims.

### `gold_030` (zero accepted claims)

Evidence: `raw/evidence/gold_030.json` (`extraction_failed`, 0 verified, 2 attempts, stop reason recorded as EOS, omitted hit `gd_66bbfcd9b1f8b6bca75fe3f6`). Retrieval: `raw/retrieval/gold_030.json`.

- **Blog** (175 words): conflicting quarterly revenues ($304M, $700M, $253M, $695M). `$700 million`, `$253 million`, and `$695 million` are not in the saved passages. Coherent grammar, internally inconsistent facts.
- **LinkedIn** (239 words, 320/320 tokens): title, then Cursor and `https://cursor.com/ai-apps?utm_source=twitter&utm_medium=social&utm_campaign=cursor_ai_apps`. Off topic. Cuts at “Cursor lets designers”.
- **X** (18 words): the topic only. Below the prompt’s 20-word floor. The format band still passes.
- **Talk** (272 words): “almost $300 million in ARR in a single quarter,” then “actual revenue,” then “hyperbolic growth” defined as milestones taking longer, then “exponential,” then “doubled their growth rate every two years.” Those history claims were not in the accepted set, because the accepted set is empty. Passage support for this timeline was **not found** in the saved hits I checked.

Graph lines for this question include endpoints such as “Four-Year CAGR of 53%” and pair Figma with Klarna and Databricks. Hypothesis: relationship construction emitted noisy edges. The drafts do not state that 53% figure. The numeric damage is from generation after extraction returned nothing.

## Paths to inspect

Generations:

- `experiments/EXP-20261102-001-model-f-unified-panel/raw/generations/gold_028__F-REL__talk.json`
- `experiments/EXP-20261102-001-model-f-unified-panel/raw/generations/gold_024__F-REL__blog.json`
- `experiments/EXP-20261102-001-model-f-unified-panel/raw/generations/gold_002__F-REL__linkedin.json`
- `experiments/EXP-20261102-001-model-f-unified-panel/raw/generations/gold_002__F-REL__talk.json`
- `experiments/EXP-20261102-001-model-f-unified-panel/raw/generations/gold_009__F-REL__talk.json`
- `experiments/EXP-20261102-001-model-f-unified-panel/raw/generations/gold_012__F-REL__blog.json`
- `experiments/EXP-20261102-001-model-f-unified-panel/raw/generations/gold_006__F-REL__blog.json`
- `experiments/EXP-20261102-001-model-f-unified-panel/raw/generations/gold_030__F-REL__blog.json`
- `experiments/EXP-20261102-001-model-f-unified-panel/raw/generations/gold_030__F-REL__linkedin.json`
- `experiments/EXP-20261102-001-model-f-unified-panel/raw/generations/gold_016__F-REL__x.json`
- `experiments/EXP-20261102-001-model-f-unified-panel/raw/generations/gold_011__F-REL__blog.json`
- `experiments/EXP-20261102-001-model-f-unified-panel/raw/generations/gold_015__F-REL__blog.json`
- `experiments/EXP-20261102-001-model-f-unified-panel/raw/generations/gold_027__F-REL__linkedin.json`

Evidence and diagnostics:

- `raw/evidence/gold_030.json`
- `raw/retrieval/gold_030.json`
- `diagnostics/descriptive_omissions.json`
- Cell-level flags: `audits/AUDIT-20261102-001-model-f-frel-prejudge/cells.jsonl`

## Ready for judging?

The generation job finished cleanly: 120/120 files, 0 failures, 0 extraction-prompt truncations. Completion and length, at the shared gate definitions, look like Model E.

Factual behavior is not in shape for a judge to be the quality decision. About 43% of cells introduce quantities the existing checker cannot find in the topic, the accepted claims, or the grounding lines. The worst of those quantities are also absent from the saved passages. X and LinkedIn often stop on the token cap, in the middle of a sentence. `gold_030` was generated with no accepted claims and still asserts revenue. A fluent judge score will not show that.

If a judge is run later, publish it next to these flags. Do not drop the numeric, copy, echo, and completion results. The writing-quality fields already present in `unified_eval/rows.jsonl` are outside this audit.
