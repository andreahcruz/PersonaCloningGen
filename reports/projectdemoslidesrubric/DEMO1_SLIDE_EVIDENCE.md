# Demo 1 slide evidence

Mapped to the five rubric rows in `reports/Project Demo slides rubric.pdf`.
Each row is 2.5 points. Excellent means the result is complete, correct, and clearly tied to the project problem.

Project: AI-driven persona cloning for B2B content, Jason Lemkin / SaaStr.
Problem the slides should keep in view: generate on-brief posts in Lemkin’s voice, grounded in his own writing, without training on held-out text or treating a cut-off fragment as a finished post.

Current candidate, as of 2026-10-05: Relabel QLoRA (`lemkin_lora_relabel`) plus Factual Dense RAG v1 (`lemkin_train_only`, 21,193 train-only documents) plus atomic evidence extraction. Deployment handoff is `docs/deployment/DEPLOYMENT_HANDOFF.md`.

Status words below follow the repo rules. MEASURED means a registered `experiments/EXP-*` run produced the number. It does not mean an independent grader has signed off.

---

## Suggested deck arc (presentation row)

1. Problem and what “good” means: voice, grounding, task completion, and copying are separate scores.
2. Where the text came from, and what cleaning removed.
3. Why the first training file taught the model to stop mid-sentence.
4. How train, validation, and test stay in separate document families.
5. What the corpus actually looks like after that filter.
6. Model path: base Llama 3.1 8B, earlier adapters, repaired adapter, relabel adapter, then train-only retrieval.
7. The numbers that decide the current candidate, and the numbers that do not.
8. What is still open: authorship of transcripts, whole-source content quality, human preference.

Say the gold set is 30 topics × 4 mediums = 120 prompts (`data/openai_ft/lemkin_gold_eval.jsonl`). It is a development comparison set, not an untouched final exam. Expected facts are scored after generation. They are not placed in the prompt.

---

## 1. Data pre-processing (2.5 pts)

**Slide claim.** Raw captures were cleaned with explicit drop reasons, then turned into supervised rows whose instruction matches the text the model is asked to finish.

### Funnel (replayed, `docs/data/PROVENANCE_PROPAGATION.md`)

| Stage | In | Out | What left |
|---|---:|---:|---|
| Raw → cleaned | 32,835 | 30,370 | 2,465 drops: event promo 450, LinkedIn chrome 623, too short 1,203, empty 150, empty transcript 39 |
| Cleaned → SFT | 30,370 | 42,771 | 5,518 under the word floor, 124 whose chunks were all removed; 3,312 lines became more than one row. Promo filter later drops 170 chunks |
| SFT → 512-token fit | 42,771 | 43,012 | Paragraphs and colon-led lists stay together. 947 extra parts come from splits. No parent row is dropped |
| Fit → balanced | 43,012 | 31,328 | SaaStr talks dropped 5,492. Extra X rows over the blog cap dropped 6,192. Kept: blog 14,315, X 14,316 |
| Balanced → relabel | 31,328 | 26,545 | 4,783 rows whose target was a word cut, a mid-sentence cut, an unfinished last slice, or a transcript fragment |

The raw files were not edited. The balanced file `dataset_fit512_keepbreaks_balanced.jsonl` (sha256 `6e99b414…f057`) was not edited. Relabel is a new file beside it (sha256 `1231835e…0897b`).

### The preprocessing problem that changed the model

Many 512-token rows were slices of a longer post, but the prompt still said “write the whole post.” The end-of-text token was therefore trained on fragments. EXP-20261003-004 fixed the labels without rewriting the targets.

Of 25,062 train rows, 21,377 remain:

| Action | Train rows | Meaning |
|---|---:|---|
| Unchanged | 11,616 | Whole posts (10,974 X, 642 blog or LinkedIn) |
| Opening | 1,711 | First slice. Prompt asks for the opening only |
| Continuation | 8,050 | Later slice that ends on a sentence, colon, or heading. Prompt quotes the previous ending |
| Dropped | 3,685 | 1,148 word cuts, 1,332 other mid-sentence cuts, 1,015 unfinished last slices, 190 transcript fragments |

Validation 2,588 and test 2,580 used the same rules. No document moved between splits. Every kept chat string fits in 512 tokens.

### Separate whole-source attempt (do not present as the training set)

EXP-20261003-001 → 002 → 003 rebuilt candidates from complete blog bodies instead of already-cut chunks.

- v3: 1,011 blogs, 601 train / 206 validation / 204 test, at a 2,048-token budget.
- Structural check passed: every row, and 288,434 cross-split pairs, with zero normalized-exact and zero token-Jaccard ≥ 0.8 crossings.
- Content-quality gate did **not** pass. The fixed 16-example sample still has split words, guest recaps, and missing-embed introductions. `training_ready` is false. No adapter was trained on this file.

### What to show

- One before/after pair: a mid-sentence slice whose old prompt said “write the whole post,” and the same text relabeled as a continuation.
- The funnel table above.
- One dropped example for each reason: promo, too short, word-cut, transcript fragment.

### Say this carefully

Cleaning removes chrome and event copy. It does not prove every remaining transcript is Lemkin speaking. Dropping SaaStr talks does not assign the other videos to him.

---

## 2. Training and test data preparation (2.5 pts)

**Slide claim.** Related chunks, exact copies, and near-duplicates stay inside one split. The test split and the 30-topic gold set are held out of training and out of the retrieval index.

### How a split is made

`host_finetune/split_groups.py` joins rows that share a source file and line, a normalized title, the same normalized output, or token-set Jaccard ≥ 0.8. The whole group is assigned together. Target mix is about 80 / 10 / 10. There is no random shuffle of individual chunks.

### Files that matter

| File | Rows | Train / val / test | Role |
|---|---:|---|---|
| First grouped file, EXP-20260928-002 | 19,321 | 15,457 / 1,932 / 1,932 | Proved the group rule. 4,379 groups. Zero group, title, exact, or Jaccard ≥ 0.8 crossings. Not the current trainer input |
| Cleaned fit512, EXP-20261001-002 | 42,877 | 34,302 / 4,287 / 4,288 | Historical default. Not the repaired or relabel run |
| Repaired balanced, EXP-20261001-008 | 31,328 | 25,062 / 3,133 / 3,133 | Paragraph-preserving fit. Stored report: zero group, exact, and near-duplicate crossings |
| Relabel, EXP-20261003-004 | 26,545 | 21,377 / 2,588 / 2,580 | What `lemkin_lora_relabel` trained on. Same families as EXP-008. Dropped rows left the file; nobody was reassigned |
| Train-only index | 21,193 | train only | 21,377 train rows minus 183 gold-overlap-family rows minus `train_26148` |

Validation loss from these files cannot be compared across rows. The prompts changed when the labels changed.

### Leakage controls that belong on a slide

- Test rows are not in the trainer and not in `lemkin_train_only`.
- 13 documents in families that already crossed splits, and 39 documents in families that overlap the gold set, were quarantined in the whole-source runs. They were not moved into train.
- The older index `lemkin_content` does not enforce this boundary. The candidate path uses `lemkin_train_only`.
- Gold tasks are a different set from the source-document test split. The 30 topics have been used while developing prompts, so call them a held-out comparison set, not a sealed final test.

### What to show

- A small diagram: one blog becomes several 512-token rows, all stamped with the same `group_id` and the same split.
- The relabel counts 21,377 / 2,588 / 2,580, plus “183 gold-overlap families excluded from retrieval.”
- One sentence on the check: zero group crossings on the repaired split; the relabel file inherits those assignments.

---

## 3. Data analytics results (2.5 pts)

**Slide claim.** After the split and the label filter, the retrieval corpus is train-only, blog-heavy in the top hits, and full of short or partial chunks. Those facts explain the RAG design.

### Corpus the model retrieves (EXP-20261004-009, collection `lemkin_train_only`, 21,193 documents)

| Finding | Count | Share |
|---|---:|---:|
| Under 20 words | 5,141 | 24.3% |
| Under 40 words | 10,427 | 49.2% |
| Continuation rows | 7,895 | 37.3% |
| Shares a group with another chunk | 11,137 | 52.6% |
| No sentence-ending punctuation | 7,036 | 33.2% |
| Headline or link, under 30 words with a URL | 4,333 | 20.4% |
| Starts as `#N` or `N.` | 1,525 | 7.2% |
| YouTube transcripts | 40 | — |

Word counts: 10th percentile 15, median 41, 90th percentile 263. Largest group: 52 chunks. X is 11,185 of 21,193 documents.

Over 30 gold topics, top 4, generator not loaded:

- Empty retrieval: 0. Duplicate document ids: 0.
- 11 of 30 queries returned two chunks from the same post.
- Mean cosine distance 0.243.
- Top-4 slots: blog 69, X 39, LinkedIn 12, YouTube 0.
- X is the majority of the index and is not the majority of the hits.

Top-20 pool for later grading (EXP-20261004-010): 600 hits, 499 judgment units after collapsing same-post chunks, 574 unique documents, 442 groups. Ranked IR metrics (precision, MRR, nDCG, recall) were not computed. No relevance grades exist yet.

### Voice analytics used later as references

Held-out style references in the unified scorer: 80 blog, 80 LinkedIn, 80 X, 0 talk. Talk has no style selector because there are no held-out talk controls.

Surface profile in `data/persona_profile.json` (generated 2026-05-12, 1,000 documents, 79,583 sentences, 626,986 content words) is older than the frozen relabel corpus. Use it as a historical picture of themes, not as the current index audit. Themes it records: sales, revenue, growth, SaaStr, ARR, customers. Framing it records: “the bottom line,” “the reality is,” “what matters is,” “dear saastr.”

Style-distance calibration (EXP-20261004-007), lower means closer to held-out Lemkin text:

- Blog is usable as a selector. Real posts sit nearer the profile (mean distance 0.853) than generic controls (2.702) or base-model drafts (1.867).
- LinkedIn and X did not pass that check. Their distances stay diagnostic.
- The scorer states the limit in the metrics file: closer surface counts are not authorship accuracy.

### What to show

- Histogram or three callouts: median 41 words, 37% continuations, 53% of documents share a post with another chunk.
- Top-4 medium mix: blog 69, X 39, LinkedIn 12, YouTube 0, against an index that is mostly X.
- One retrieval example: a topic whose top hit is a numbered fragment (`#7. Monday.com…` on `gold_002`) so the audience sees why evidence is extracted before generation.

---

## 4. Ongoing machine learning results (2.5 pts)

**Slide claim.** The selected model is Llama 3.1 8B Instruct, 4-bit, with a rank-16 QLoRA adapter trained on the relabeled file. Retrieval is added at inference from the train-only index. The label fix reduced cut-off answers. Adding retrieval made claims checkable and did not yet improve blog voice.

### Setup held for the relabel train (EXP-20261003-005)

- Base: `unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit`
- LoRA rank 16, alpha 32, dropout 0, seven attention and MLP targets
- Learning rate 1e-4, 2 epochs, batch 1, gradient accumulation 16, sequence length 512, seed 42, AdamW 8-bit
- About 3.4 hours, exit 0. Train loss 1.797. Best eval loss 1.3316 at step 2,674
- Eval loss by checkpoint: 1.4016, 1.3645, 1.3421, 1.3410, 1.3342, 1.3317, 1.3316
- Per-medium eval ran out of memory and was skipped. The repaired adapter’s eval loss (1.5311) is a different validation file, so the two losses are not a comparison
- Adapter path: `host_finetune/output/lemkin_lora_relabel`. The repaired adapter was not overwritten

### Path of models (say what each comparison actually measured)

| Step | Experiment | What was measured | Result to say out loud |
|---|---|---|---|
| Three earlier adapters, retrieval off | EXP-20261001-007 | Gold-rubric overall on 120 prompts | cleaned 0.235, balanced 0.235, rank 32 0.269. Rank 32 led this lexical rubric. X fact coverage stayed low because the facts were written for blog-length answers |
| Repaired adapter vs base and older clones | EXP-20261001-010 | Same 120 prompts, gold-rubric overall | repaired 0.244, untrained Llama 3.1 0.368, clone v5 0.258, clone v6 0.264. v5 and v6 share one Ollama weight id, so the gap is a second sample. The base model wins this rubric because it follows headings and length more often. BLEU stays near 0.004 and ROUGE-L near 0.08 for every model. This rubric does not pick the voice model |
| Label fix, retrieval off | EXP-20261004-003, seeds also 43 and 44 | Early `<\|eot_id\|>` before half the budget, and mid-sentence stops. 120 answers each | Seed 42: repaired early 0.650 / mid-sentence 0.633; relabel 0.550 / 0.342. Three-seed means: repaired 0.625 / 0.625; relabel 0.567 / 0.322. Relabel is the cutoff winner. It is not a voice winner |
| Train-only retrieval, no generator | EXP-20261004-009 | What the index returns | See analytics section. Empty rate 0 |
| LoRA scale at inference | EXP-20261004-018 | Can a weaker adapter stop invented numbers | Scales 1.00, 0.75, 0.50, 0.25. No intermediate scale is deployable. At 0.25 the drafts that pass move toward base voice, and a blog can still invent quantities |
| Full gold set, retrieval off vs frozen RAG | EXP-20261004-008 scores the off set; EXP-20261005-003 is the RAG set | Gates, then persona utility on blog drafts that pass every gate | Table below |

Generation settings for the 120-prompt comparisons: temperature 0.7, top_p 0.9, repetition penalty 1.15, budgets blog 768 / LinkedIn 320 / X 160 / talk 768. Judge for writing and stance: blinded `qwen2.5:14b`, temperature 0. Persona utility weights, a project choice: style 0.45, perspective 0.20, writing 0.25, task 0.10. BLEU, ROUGE, BERTScore, and the old gold-rubric overall are logged and do not select the model.

### Current head-to-head (120 drafts each)

Retrieval-off Relabel is the baseline inside `experiments/EXP-20261005-003-relabel-rag-final/metrics/comparison.json`. RAG is the same adapter with `lemkin_train_only`, k=4, topic-only query, then atomic evidence before generation.

| | Retrieval off | Relabel + factual RAG |
|---|---:|---:|
| Drafts | 120 | 120 |
| Eligible after gates | 77 | 72 |
| Completion failures | 41 | 36 |
| Copying failures | 1 | 2 |
| Format failures | 5 | 3 |
| Grounding failures | not applicable | 14 |
| Verifiable claims supported | — | 495 of 498 (0.994) |
| Eligible blog voice distance (lower is closer) | 0.712 (n=28) | 0.777 (n=25) |
| Eligible writing quality (about 1–5) | 3.15 | 3.06 |
| Persona utility on gate-passing blogs | 0.705 (n=23) | 0.688 (n=19) |
| Stop reasons | 101 end-of-text, 19 length | 103 end-of-text, 17 length |
| Unsupported-number diagnostic | — | 60 drafts |
| Source-copy flags | — | 2 drafts |

The selector ranks retrieval-off first because blog voice is the first ranking key. RAG is the path that can show where a claim came from: 495 of 498 verifiable claims match the retrieved evidence. Fourteen drafts still fail the grounding gate. Sixty drafts carry a numeric diagnostic. Those two facts belong on the slide next to the 99.4% support rate.

LinkedIn is the weak medium on completion: 23 of 30 retrieval-off LinkedIn drafts fail completion, and only 7 are eligible. Blog and talk are the strong mediums (28 of 30 eligible with retrieval off).

### What to show

- Loss curve for the relabel train (seven descending eval points, best 1.3316).
- Cutoff bar chart: early stop and mid-sentence, repaired vs relabel, three-seed means.
- Gate table: eligible 77 vs 72, claim support 495/498, utility 0.705 vs 0.688.
- One blog draft with the four evidence chunks and the claims marked supported or not. The generation file is `experiments/EXP-20261005-003-relabel-rag-final/raw/relabel.jsonl` (120 drafts, sha256 `266507ac…ec550`).

### Say this carefully

- Eval loss did not “beat” the repaired run. The validation prompts changed.
- Gold-rubric overall favoring the base model is a format result. The later selector refuses that score.
- Persona utility is filled only for blog drafts that pass every gate, and only after writing quality and stance passed their sanity checks. LinkedIn, X, and talk utility stay empty because those style checks are not calibrated.
- Human preference has not been used to pick the model.
- Ontology-aware retrieval is still a stated requirement and is not an implemented component. The measured retriever is dense Chroma search over train-owned chunks.

---

## 5. Presentation (2.5 pts)

Excellent on this row is clarity, visuals, and a logical flow. The evidence above is enough for that if each slide answers one question and shows one figure.

### Slide list that matches the rubric order

1. Title and problem: clone Lemkin’s B2B voice without copying held-out posts or stopping mid-sentence.
2. What success is: four separate scores (voice, grounding, task, copying), and why BLEU is logged but not used.
3. Sources and cleaning funnel (section 1 table).
4. The label bug and the relabel counts.
5. Split rule and the 21,377 / 2,588 / 2,580 file, plus the 21,193-document index.
6. Corpus analytics: length, continuations, and which medium retrieval actually returns.
7. Training card: Llama 3.1 8B, QLoRA r=16, α=32, 3.4 hours, eval loss 1.3316.
8. Cutoff result: relabel vs repaired.
9. Why the base model won the old gold rubric, and why that did not select the adapter.
10. RAG result: 99.4% claim support, utility 0.688 vs 0.705, retrieval-off still leads on blog voice.
11. Honest limits: transcript authorship, whole-source content gate, ungraded qrels, no human preference yet.
12. Live or screenshot: one topic in, four train-only passages, one draft out.

### Figures already in the repo

- Split notes and counts: `experiments/EXP-20261001-008-repaired-balanced-split/notes.md`, `experiments/EXP-20261003-004-relabel-continuations/notes.md`
- Train log: `experiments/EXP-20261003-005-relabel-qlora/raw/train.log`
- Cutoff table: `experiments/EXP-20261004-003-repaired-vs-relabel/notes.md`
- Index audit: `experiments/EXP-20261004-009-dense-rag-v1/notes.md`
- Head-to-head: `experiments/EXP-20261005-003-relabel-rag-final/metrics/comparison.json`
- Drafts to screenshot: `experiments/EXP-20261005-003-relabel-rag-final/raw/relabel.jsonl`

### Claims to leave off the slides

- Any sentence that the relabel adapter is a validated voice clone.
- Any comparison of 1.3316 vs 1.5311 as “the new model fits better.”
- Any precision, recall, or nDCG for retrieval. Those were not computed.
- The May 2026 persona profile as if it were computed on the 21,193-document index.
- Whole-source v3 as a trained model. It failed the content review and was not trained.
- `lemkin_content` as the leakage-safe index. The candidate index is `lemkin_train_only`.

---

## Journey in one paragraph

The corpus started as 32,835 raw blog, LinkedIn, X, and transcript records. Cleaning and a promo filter produced a 512-token supervised file, then a balanced 31,328-row file that dropped SaaStr talks and capped X. A grouped split kept near-duplicates together. Training on that file still paired “write the whole post” with mid-sentence slices, so the adapter learned to stop early. Relabeling kept 21,377 train rows, renamed openings and continuations, and dropped 3,685 broken targets. The rank-16 QLoRA run on Llama 3.1 8B saved `lemkin_lora_relabel` with eval loss 1.3316. On 120 prompts it cut mid-sentence stops from about 0.63 to about 0.32. A separate CPU index of 21,193 train-only rows now supplies factual RAG. On the full gold set, 495 of 498 verifiable claims are supported, and blog persona utility is 0.688 with retrieval versus 0.705 without it. Retrieval-off still leads on voice. Grounding is the gain. Voice is not yet improved by the index.

## Chats that match this story

- [Data cleaning review](863bc75e-69ab-4113-b922-db42635f14a1) — cleaning, the relabel train, and how to read the log.
- [Project architecture and pipeline](10458e8f-f3d4-4bd7-b95b-d14328cdfec2) — controlled comparison before another fine-tune.
- [RAG system design review](541000a2-cdae-44a1-a4a5-7b805f5fa812) — passages, then an evidence capsule, then the relabel adapter.
- [QLORA RAG generation test results](925bf2eb-0ab2-4857-affb-d108035a14da) — generation tests on that path.
- [Qlora relabeled model outputs](1c431112-c12a-4390-8eaa-38db7fad7a23) — the scored drafts themselves.
- [Workbook deliverable review](29840b5b-06c0-44fa-9f92-81a5347a9d72) — how this adapter sits in the model lineup for the workbook.
