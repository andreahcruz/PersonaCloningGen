# Model F results

Two layers are easy to mix up. Both are in `experiments/EXP-20261102-001-model-f-unified-panel/unified_eval/` locally, and both are exported without draft text in [`results/model-f-handoff-v1`](../results/model-f-handoff-v1).

1. **Judge metrics** (`unified_eval.v1`, Qwen 2.5 14B). Eligible means live in `summary.json`. Row means over every scored draft live in `comparison.json`.
2. **Deterministic audit** (`audits/AUDIT-20261102-001-model-f-frel-prejudge`). Counts of unsupported numbers, source copy, topic echo, and completion. These are not judge scores.

A high writing-quality score does not mean the draft's quantities were in the evidence.

## Judge summary for F-REL

From `summary.json`. Eligible drafts: **67 / 120** (eligible rate 0.5583). Generation SHA-256 `6ed2af4f47c3f592aa833c643dc6354956b0ea7f96b7634b98a902b2a746724c`. n traces = 120.

| Field | Value | How to read it |
|---|---:|---|
| Completion pass rate | 0.700 | Gate on all 120 |
| Copying pass rate | 1.000 | Training-span gate on all 120 |
| Format pass rate | 0.9667 | Gate on all 120 |
| Grounding pass rate | 0.8167 | Gate on all 120 |
| Eligible style distance | 0.7724 | Blogs only. Lower is closer. Not authorship |
| Eligible writing quality | 3.2662 | Selector metric, eligible drafts |
| Eligible task adherence | 0.7033 | Eligible drafts |
| Eligible stance consistency | 1.000 | Perspective score. Framing distance is diagnostic |
| Eligible framing distance | 0.832 | `mean_distance_eligible` under perspective fidelity |
| Claim support vs saved hits | 1.000 | Method `claim_vs_saved_retrieval_hits` |
| Lemkin-evidence support | 0.9941 | Marked `not_rag_grounding: true` |
| Persona utility | null | Not computed. n = 0 in the comparison file |
| BLEU / ROUGE-L | 0.0112 / 0.0769 | Diagnostic. Not persona fidelity |
| Mean latency | 20.069 s | All traces |

Writing quality on this summary is the eligible mean. The comparison file's writing-quality mean on all 120 drafts is 3.165278. Both numbers are in the export. They are not interchangeable.

## Per medium

From `comparison.json`, architecture `Model F`. Means are over the `n` in that cell, not the eligible-only summary above. Persona utility is null in every medium (n = 0).

| Medium | Drafts | Eligible | Completion failures | Format failures | Grounding failures | Writing quality (n=30) | Task adherence | Words |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Blog | 30 | 22 | 0 | 1 | 7 | 3.57777 | 0.700267 | 246.4 |
| LinkedIn | 30 | 11 | 17 | 2 | 4 | 3.072223 | 0.68538 | 186.033 |
| X | 30 | 11 | 18 | 0 | 6 | 2.661113 | 0.725217 | 90.033 |
| Talk | 30 | 23 | 1 | 1 | 5 | 3.350007 | 0.670637 | 289.967 |
| All | 120 | 67 | 36 | 4 | 22 | 3.165278 (n=120) | 0.695375 | 203.108 |

Style distance is present for blog (0.819967, n=30), LinkedIn (0.794733, n=30), and X (0.602307, n=30). Talk style distance is null (n=0). The all-row style mean is 0.739002 on n=90. The eligible blog-only distance in the summary is 0.7724. Claim-support means in the comparison file are 1.0 on the rows that have a score, with 22 drafts missing a claim-support value overall.

Training-copy failures in this comparison file: 0 of 120.

## Comparison with the saved A–E rows

`comparison.json` labels the 120-draft rows Base, PersonaRAG, Relabel RAG off, Relabel + RAG, and Model F. The pre-judge audit maps Relabel + RAG to Model E and does not assign letter names to Base or PersonaRAG. This table uses the file's names. All figures are the `medium: all` rows.

| Architecture | Eligible | Completion pass | Format pass | Training-copy pass | Grounding pass | Writing quality | Style distance | Persona utility |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Base | 45/120 | 0.375 | 1.000 | 1.000 | not scored | 3.57778 (n=120) | 1.043073 (n=90) | 0.619635 (n=20) |
| PersonaRAG | 57/120 | 0.591667 | 1.000 | 1.000 | 0.833333 | 3.598603 (n=120) | 0.906062 (n=90) | 0.6306 (n=19) |
| Relabel RAG off | 77/120 | 0.658333 | 0.958333 | 0.991667 | not scored | 3.000003 (n=120) | 0.749901 (n=90) | 0.70497 (n=23) |
| Relabel + RAG | 72/120 | 0.700 | 0.975 | 0.983333 | 0.883333 | 2.895834 (n=120) | 0.805381 (n=90) | 0.687753 (n=19) |
| Model F | 67/120 | 0.700 | 0.966667 | 1.000 | 0.816667 | 3.165278 (n=120) | 0.739002 (n=90) | null (n=0) |

Eligible writing quality for Model F in `summary.json` is 3.2662, which is higher than the all-draft mean in this table. Relabel + RAG's eligible writing quality is not restated here; the README's older Model E table is a different eligible cut (3.0602) from an earlier scoring note. Use this comparison file when the question is the shared `comparison.json` sheet, and `summary.json` when the question is Model F's eligible selector metrics.

Model F matches Relabel + RAG on completion pass rate (0.700, 36 failures). Model F has fewer training-copy failures in this file (0 versus 2) and a lower grounding pass rate (0.816667 versus 0.883333). Persona utility cannot be compared: Model F did not produce one.

The audit's medium completion counts (pass/30) are Base 20/5/3/17, PersonaRAG stand-in 28/10/4/29, Relabel RAG off 29/7/14/29, Relabel + RAG 29/10/15/30, Model F 30/13/12/29 for blog, LinkedIn, X, and talk. Those are completion counts, not judge quality.

## Deterministic audit

Source: `audits/AUDIT-20261102-001-model-f-frel-prejudge/summary.json` and `REPORT.md`. The audit did not call the judge. It read the saved generations.

| Check | Count | Denominator in `summary.json` |
|---|---:|---:|
| Saved non-empty complete cells | 120 | 120 |
| Unsupported-number cells | 52 | 120 |
| Unsupported quantities | 185 | 120 cells |
| RAG copy fail, span at least 12 words | 7 | 120 |
| Prompt copy span at least 8 words | 10 | 120 |
| Completion pass | 84 | 120 |
| Mid-sentence stop | 36 | 120 |
| Output tokens equal to the medium cap | 36 | 120 |
| Format pass | 116 | 120 |
| Prompt length-floor fail | 12 | 120 |
| Topic echo (`repeats_topic`) | 55 | 120 |
| Full topic embedded | 56 | 120 |
| Faithfulness fail | 61 | 120 |
| Zero accepted claims | 4 | 120 (`gold_030`, all four media) |
| Retry, accepted attempt greater than 1 | 20 | 120 |
| Role-unsupported cells | 1 | 120 |
| Speaker-attribution failures | 8 of 8 applicable | 112 not applicable |

Training-copy failures are 0. `summary.json` stores denominator **104**. The audit report, after re-reading `unified_eval/rows.jsonl`, states **0 of 120**. `comparison.json` also shows 0 copying failures on n=120. This handoff does not recompute the gate. The 104-versus-120 denominator is unresolved in the audit summary file.

The audit also records that early EOS and GPU memory were not stored, and that a 4-token claim-overlap diagnostic found 48 of 1,284 claims. That overlap rate is not entailment and is not a judge metric.

Lexical grounding passed 8 of 120. The audit says that gate fights the prompt's anti-copy rule, so a low pass rate is a weak factuality signal.

## Hallucinations and completion failures

These are audit findings, not a new annotation pass. Draft text is not copied here.

- 52 cells contain quantities the checker could not find in the topic, the accepted claims, or the grounding lines. The report names `gold_028` talk, `gold_024` blog, `gold_002` LinkedIn, `gold_002` talk, and `gold_009` talk as the sharpest numeric cases. The diagnosis recorded there is generation, including cases where retrieval had returned on-topic claims.
- `gold_030` extraction status is `extraction_failed` with 0 accepted claims. The blog and talk still state revenues. The LinkedIn draft leaves the topic. The X draft is the topic line and fails the prompt's 20-word floor.
- 7 cells copy 12 or more contiguous words from a retrieved passage. `gold_012` blog and talk are the longest cited case (19 words). The verbatim span is left in the local generation files.
- 36 cells fail sentence-final completion. LinkedIn and X dominate. Blog had 0 completion failures in the comparison file. X's median output length sits on the 160-token cap (audit: median output tokens 160). LinkedIn's median sits on the 320-token cap.
- `gold_016` X emitted a prompt instruction as the post. `gold_015` blog left an evidence tag in the prose. `gold_011` blog and talk are short title echoes after 9 graph items were omitted.

## What the judge does not establish

Eligible writing quality 3.2662 and claim-support 1.0 can both be true while 52 cells fail the number check. Claim support compares extracted claims with saved retrieval hits. It does not check whether the draft introduced a different quantity. Publish the judge table next to the 52/120 number-failure count. Do not describe F-REL as factually reliable on this panel.
