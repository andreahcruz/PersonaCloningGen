# Metric map and review of evaluation methods (PUK-28)

Author: Vrushabh Bodarya. Written 2026-10-24 for the DATA 298B Team 2 group. Linear: PUK-28.

## 1. What this page is for

PUK-28 asks us to research metrics for the new architecture, review the old evaluation methods, and make sure voice, writing style, grammar, coherence and truth are all covered and can be compared across architectures.

This page does three things:

1. It maps each quality we care about to the metric that measures it today, and says what that metric really measures.
2. It reviews the older evaluation methods and says which ones to keep.
3. It lists the gaps we found, with the evidence, and what to do about each.

Sources. The current harness is `host_finetune.unified_eval` and its helpers (`voice_distance`, `writing_quality`, `claim_judge`, `numeric_grounding`) on branch `kevin/298b-integration`. The reasoning is in `docs/evaluation/EVALUATION_PLAN.md` and DEC-007, DEC-008 and DEC-009 of `docs/DECISION_LOG.md`. All numbers come from saved experiments in `experiments/EXP-*` on that branch, mainly EXP-20261005-003 (the final paired run) and EXP-20261004-007 (the unified rescore). We did not regenerate any drafts or rerun any judge for this page. That branch is not merged into `QLoRAFT` yet.

## 2. The architectures we compare

The team has not agreed one numbering for the models (the old slides use M1 to M5, a later report uses M0 to M6). This page uses plain names so nothing is silently renamed.

| Name used here | What it is | Saved results that exist today |
|---|---|---|
| Base Llama | Untrained Llama 3.1 8B with an author-style prompt, no retrieval | 120 drafts, no retrieval (EXP-20261001-010) |
| PersonaRAG | Base Llama with the persona profile and vector retrieval over the `lemkin_content` index (the `generate.py` and web demo path) | Only old traces from earlier work. No 120 prompt run |
| Repaired QLoRA | QLoRA adapter trained on the repaired 512 token data | Retrieval off (3 seeds) and plain retrieval on (one seed) |
| Relabel QLoRA | QLoRA adapter trained on the relabeled data | Retrieval off (3 seeds) and plain retrieval on (one seed) |
| Relabel QLoRA + factual RAG + atomic evidence | Kevin's frozen candidate | Both arms fully scored on the same 120 prompts (EXP-20261005-003) |

## 3. The metric map

How to read the columns. "Role" is what the harness lets the metric do: a selector may rank models, a gate decides whether a draft may be ranked, a diagnostic is shown but never used to choose.

### 3.1 Is the draft usable at all

| Quality | Current metric | What it really measures | Applies to | Role | Main weakness |
|---|---|---|---|---|---|
| Completion | Sentence-final check, mid sentence stop rate, early end of text rate | Whether the draft ends cleanly | All | Gate | Early end of text is recorded but does not fail a finished draft. A heuristic sentence-end test can pass a text that is cut at a full stop |
| Format and task | Length band for the requested medium, topic word coverage | Right size and mentions the topic | All | Gate and a 10 percent part of Persona Utility | Topic coverage is lexical, so it can miss a draft that is on topic but uses other words |
| Copying | Longest word span shared with the training targets, 12 words fails | Memorization of training text | All fine-tuned systems | Gate | Checks training targets, not retrieved passages (a separate source copy check exists for RAG) |

### 3.2 Voice and style

| Quality | Current metric | What it really measures | Applies to | Role | Main weakness |
|---|---|---|---|---|---|
| Voice (style) | Mean absolute z score of about 9 surface features and function word rates against held-out Lemkin posts of the same medium | How close the surface counts are to the average Lemkin post | All | Selector for blog only. LinkedIn and X are diagnostic only. Talk has no references | See gaps G2 and G3 |
| Perspective | Share of advice claims that a judge calls consistent with held-out Lemkin evidence (stance consistency). Framing rate distance is kept as a diagnostic | Whether the draft takes positions Lemkin takes | All | Selector | Almost every claim is judged consistent (gap G5) |
| Reference overlap | BLEU, ROUGE-L, BERTScore against a constructed reference | Word overlap with one reference text | All | Diagnostic | Near zero for every system (BLEU 0.011, ROUGE-L 0.077 on the final run), so it cannot separate anything |

### 3.3 Writing quality

| Quality | Current metric | What it really measures | Applies to | Role | Main weakness |
|---|---|---|---|---|---|
| Coherence, logical progression, relevance, specificity, usefulness | A judge (qwen2.5:14b, a different family from Llama) scores each from 1 to 5, blinded to the model | The judge's view of overall writing | All | Selector, after a calibration check passes | One judge, one pass. Gap G9 |
| Fluency | One of the six judge items above | Whether sentences read naturally | All | Part of the writing mean | Not a grammar check (gap G1) |
| Grammar | None | Nothing. Grammar is covered only by the single fluency item | All | None | Gap G1 |

### 3.4 Truth

| Quality | Current metric | What it really measures | Applies to | Role | Main weakness |
|---|---|---|---|---|---|
| Grounding in retrieved passages | Claims extracted by the judge, each labeled supported, unsupported, contradicted or not verifiable against the saved retrieved passages | Whether claims follow from what was retrieved | RAG systems only | Gate (adds a gate for RAG arms only) | Not defined for systems without retrieval (gap G7). Very high and flat (gap G5) |
| Support from held-out Lemkin posts | Same claims checked against a held-out Lemkin evidence set | Whether claims agree with what Lemkin wrote elsewhere | All | Diagnostic | Not a check against the real world |
| Numbers not in the evidence | Deterministic check of quantities against the accepted evidence | Invented figures | RAG systems with evidence extraction | Flag, draft still ships | Disagrees with the claim judge (gap G6) |
| Truth against the outside world | None | Nothing | All | None | Gap G10 |

### 3.5 Overall and cost

| Quality | Current metric | What it really measures | Role | Main weakness |
|---|---|---|---|---|
| Persona Utility | 45 percent style, 20 percent perspective, 25 percent writing quality, 10 percent task | A project specific blend | Computed only when all four parts are valid selectors | Gap G8 |
| Cost and speed | Latency, tokens, memory | Practical cost of each architecture | Not in the final comparison (latency is empty) | Needed to compare architectures fairly (gap G11) |

## 4. Review of the older methods

| Old method | Where | Problem | Verdict |
|---|---|---|---|
| Gold rubric overall (format, brief alignment, expected facts) | `eval_gold_rag.py` | It rewards format and word overlap. The untrained Llama scores 0.368 and the fine-tunes score 0.244 to 0.264 (EXP-20261001-010), which says more about formatting than about voice | Keep as a diagnostic called brief agreement. Never a selector |
| G-Eval with a Llama judge | `eval_rag.py` (judge defaults to llama3.1) | The judge is the same family as the candidates, so it can favor its own style. The newer harness already refuses Llama family judges | Replace with the qwen judge path. Do not reuse old scores |
| TF-IDF authorship classifier | `eval_rag.py` | The vectorizer is fit before cross validation (leakage), and it falls back to synthetic negatives when the dataset is not reachable | Drop as a selector |
| BERTScore, BLEU, ROUGE-L with the best of K references | `eval_rag.py` | Taking the best of 20 sampled references inflates scores and measures overlap, not voice | Diagnostic only |
| RAGAS faithfulness and answer relevancy | `eval_rag.py`, `persona_pipeline` | The persona pipeline call uses a dummy context. Needs its own judge backend | Replace with the claim level grounding already in the harness |
| Stop rates (early end of text, mid sentence) | DEC-007 | A good proxy for unfinished drafts, but not for voice | Keep as a gate, as DEC-008 decided |

## 5. Gaps, with evidence

**G1. Grammar is not measured.** The only grammar signal is the judge's "fluency" item. A judge rating is a weak grammar test, and nothing separates real errors (spelling, agreement, punctuation) from Lemkin's deliberate informal style. *Update 2026-10-24:* a first rule based checker now exists (`eval_tools/grammar.py`, section 10). On Kevin's saved Relabel drafts it finds almost nothing: 0 to 0.04 mechanical errors per 100 words, against 0.015 to 0.14 for real Lemkin posts. So mechanics is not where the drafts are weak, but the checker is coarse and cannot see tense or awkward phrasing.

**G2. Voice is only a valid selector for blog.** In the final run the style score is a selector for blog only. LinkedIn and X are diagnostic only, and talk has no held-out references (reference counts are 80 blog, 80 LinkedIn, 80 X and 0 talk). The reference counts come from EXP-20261004-007 and the selector roles from EXP-20261005-003. So we cannot compare architectures on voice for three of the four mediums.

**G3. The style score may reward average looking text.** The score is the mean absolute z score against the average Lemkin profile. In EXP-20261004-007, real held-out Lemkin blog posts sit at a mean distance of 0.8531. In EXP-20261005-003 the eligible blog drafts sit at 0.71 (retrieval off) and 0.78 (retrieval on). If both runs use the same reference pack, the drafts look closer to the profile than real posts do. We did not confirm that the packs match. A score that real posts do not win suggests it measures typical surface counts, not authorship. This is cheap to test on the saved data.

**G4. The calibration controls are very small.** The style check passes because real posts beat a few controls: 4 hand written generic sentences (mean 2.70) and 2 degraded strings (mean 6.13), plus 30 base model drafts (mean 1.87). Beating nonsense and corporate filler is a low bar. It does not show the score can tell Lemkin from another good SaaS writer.

**G5. Some scores sit at the ceiling.** Mean stance consistency is 0.9906, claim support against retrieved passages is 0.994 (495 of 498 verifiable claims), and support from held-out posts is 0.9602. When nearly everything scores near 1, these metrics cannot separate architectures.

**G6. Two truth signals disagree.** The claim judge says 99.4 percent of verifiable claims are supported. The deterministic number check flags 60 of 120 drafts for numbers that are not in the accepted evidence. The judge is probably lenient about figures. Truth should not rest on the judge alone.

**G7. The gates differ between arms, so the arms are not compared on the same drafts.** A retrieval arm gets one extra gate (grounding). In EXP-20261005-003, 72 of 120 drafts are eligible with RAG and 77 of 120 without. The voice, writing and utility numbers are computed only on eligible drafts, so each arm is measured on a different subset. Persona Utility itself uses only 19 blogs with RAG and 23 without. *Update 2026-10-24:* `eval_tools.compare` shows the bias directly. On the common gates (completion, copying, format) the RAG run passes 83 of 120 drafts and the run without RAG passes 77. The headline "72 against 77 eligible" looks worse for RAG only because the grounding gate removes 11 more drafts from the RAG run.

**G8. Persona Utility mixes things that are not on the same scale.** The style distance is turned into a score with 1 divided by (1 + distance), writing is the 1 to 5 mean divided by 5 (so the lowest possible score is 0.2), and stance is a 0 to 1 rate. The weights of 45, 20, 25 and 10 are a project decision with no sensitivity check. The plan also ranks drafts by style first, then perspective, then writing, which is a different rule from the weighted blend.

**G9. The statistics are thin.** There are 30 topics in one development panel, one seed in the final run, one judge at temperature 0, and no confidence intervals. A gap such as writing quality 3.06 against 3.15 is not shown to be real. Kevin's own audit says small single seed gaps are not a settled ranking. *Update 2026-10-24:* `eval_tools.compare` adds the missing intervals (section 10). On the Relabel RAG pair, only the blog rows are clear: blog writing quality is lower with RAG (difference -0.22, interval -0.43 to -0.03) and blog style distance is farther from the profile (+0.10, interval 0.02 to 0.21). Overall rows and the other mediums show no clear difference.

**G10. Nothing checks truth against the outside world.** Claims are checked against retrieved passages and against Lemkin's own posts. A claim that is consistent with Lemkin but factually wrong would pass.

**G11. Cost and speed are missing.** Latency is empty in the final summary. To compare architectures we also need time per draft, tokens and memory, because the RAG pipelines are much heavier than a plain adapter.

**G12. There is no human check of the judge.** DEC-007 removed human voice preference, because nobody on the team can reliably judge whether a draft sounds like Lemkin. Humans can still judge grammar, coherence and whether a claim is supported by its source, and those are the items the judge is least tested on.

## 6. What we recommend, in order

| # | Recommendation | Fixes | Runs where | Effort |
|---|---|---|---|---|
| R1 | Report every metric with a bootstrap confidence interval, paired by topic, and use at least 3 seeds for the headline comparisons | G9 | Saved drafts, CPU | Small |
| R2 | For every comparison, report two views: all drafts, and the drafts that pass the same common gates in every arm. Show the eligibility rate beside every score | G7 | Saved drafts, CPU | Small |
| R3 | Test whether the style score rewards typical text: compare real held-out posts and drafts under one reference pack. If real posts do not win, add a spread aware term or an authorship verification model trained on a proper held-out split | G3, G4 | Saved drafts and Kevin's reference pack | Medium |
| R4 | Add harder calibration controls: real posts by other SaaS writers and the existing "good persona prompt" drafts, with more than 2 or 4 controls per bucket | G4 | Needs control texts | Medium |
| R5 | Add a grammar axis: a rule based error rate per 100 words, reported relative to the same rate on real Lemkin posts of the same medium (so informal style is not punished). Keep the judge fluency item as a second opinion | G1 | CPU. Needs a grammar checker that we would install only with the team's approval | Medium |
| R6 | Make the deterministic number check a main truth metric next to the claim judge, and add atomic claim precision (the share of atomic claims that are supported) as in FActScore. Kevin's atomic evidence step already produces atomic claims | G5, G6 | Saved drafts, CPU | Medium |
| R7 | Check truth against the outside world on a small manual sample (30 drafts) as a separate, clearly labeled number | G10 | Manual | Medium |
| R8 | Report the four Persona Utility parts separately and rank by them without the blend, or run a sensitivity check of the weights and publish the range | G8 | Saved drafts, CPU | Small |
| R9 | Add a small human check (30 drafts) for grammar, coherence and claim support only. Measure how well the judge agrees with people (rank correlation). Keep voice out of it, as DEC-007 decided | G12 | Manual | Medium |
| R10 | Add time per draft, tokens and memory per architecture to every comparison table | G11 | Kevin's machine | Small |

**Status on 2026-10-24.** Built in `eval_tools/` (section 10): R1 (confidence intervals, by topic), R2 (all drafts and common gates views) and a first rule based version of R5 (grammar). R8 is partly covered, because `eval_tools.compare` prints each axis on its own and never a blended headline. R3 waits for Kevin's reference pack. R4, R6, R7, R9 and R10 are open.

## 7. How this feeds the next steps

PUK-34 (before and after RAG per architecture) will use these rules: the same 120 prompts for every architecture, the separate axes from section 3 with no blended score as the headline, paired differences with confidence intervals (R1), both the all draft view and the common gate view (R2), and a clear "pending" label for any architecture without results. Today only the Relabel pair is complete.

For the workbook, sections 3 and 4 answer "evaluation methods and metrics for each model" (item 4.4), and section 2 gives the model list for the comparison (item 4.3).

## 8. Open questions for Kevin

1. Do base Llama and PersonaRAG have saved 120 prompt results with retrieval on and off, or can you run them?
2. Is the persona prompted base the same run as the untrained Llama in EXP-20261001-010?
3. Do the style numbers in EXP-20261004-007 and EXP-20261005-003 use the same reference pack? (This decides how far we can trust G3.)
4. Is a grammar axis (R5) wanted? A first rule based version exists. Do you want a stronger checker such as LanguageTool (needs Java and about 200 MB), and may we install it?

## 9. Limits of this page

- It is a review of the design and of saved numbers. We did not rerun the harness or the judge, because the judge model (qwen2.5:14b) and the 8B adapter do not run on the laptop used for this task.
- G3 and G4 are hypotheses supported by the numbers above and should be tested before anything changes.
- The grammar tool is a small rule set. It catches mechanical slips only and finds almost none in the saved drafts, so it cannot yet separate runs.
- The intervals in section 10 come from one run per arm and 30 topics, with many comparisons (4 mediums, 10 metrics). Rows by medium are leads, not findings.
- Nothing here claims a model is better or worse. It is about whether the measurements can be trusted and compared.

## 10. Tools built for this task

All in `eval_tools/` (see `eval_tools/README.md`). They read saved results, call no model and no judge, use only the Python standard library, and have 44 tests (`python -m pytest tests/test_eval_tools.py`).

| Tool | What it does | Fixes |
|---|---|---|
| `python -m eval_tools.compare` | Compares two saved runs on the same drafts. Shows gates, an all drafts view and a common gates view, and for every metric the difference with a 95 percent interval (resampled by topic), a p value and a plain verdict | G7, G9 |
| `python -m eval_tools.grammar` | Counts mechanical errors per 100 words and scores each draft against real Lemkin posts of the same medium | G1 (first version) |

Check on real data. On Kevin's saved Relabel runs (RAG off against RAG on, 120 drafts each), the mean differences from `eval_tools.compare` equal the ones in his `metrics/comparison.json`: writing -0.104, task +0.008, stance +0.003 on 82 drafts, blog style distance +0.104 on 30, persona utility -0.026 on 15, word count -9.84. What is new is the interval and the p value, which his file marks as not computed. The full report is in `docs/evaluation/examples/relabel_rag_off_vs_on.md`.

What the example shows:

- Gates. On the common gates RAG passes 83 of 120 drafts and no RAG passes 77, so the lower "eligible" count for RAG comes from its extra grounding gate (G7).
- Blog drafts. With RAG, blog writing quality is lower (-0.22) and blog style distance is farther from the real profile (+0.10). Both intervals exclude zero. Persona Utility is also lower (-0.026, 15 blogs), which matches Kevin's result.
- Everything else. The overall rows, LinkedIn, X and talk show no clear difference, and the stance and evidence support scores stay near 1 in both runs (G5).
- Grammar. Both runs are near zero errors per 100 words, below the real posts, so mechanics does not separate them.

Rule checks on the grammar tool. Its rules were tried on 400 real posts per medium, and the false alarms found there were removed: "Series A in" read as a wrong article, the double dot that Lemkin uses on purpose, the word "hashtag" that LinkedIn exports insert, and bracket or quote counts. What is left in the real posts is mostly real slips (a doubled word, "you is", "dont", a lowercase sentence start).
