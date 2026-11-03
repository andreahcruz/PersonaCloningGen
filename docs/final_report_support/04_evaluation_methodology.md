# Evaluation methodology

## Panel

| Item | Value | Source |
|---|---|---|
| Gold file | `data/openai_ft/lemkin_gold_eval.jsonl` | `provenance.json` |
| Gold SHA-256 | `d7c4f997899e28f28eb3ea9aa3417857da6f0666acbf72fd8ac31422034e520e` | same |
| Topics | `gold_001`–`gold_030` | `UNIFIED_120_COMPARISON.md` |
| Media | blog, LinkedIn, X, talk | same |
| Cells | 120 per architecture | `architecture_comparison.json` |
| Schema | `unified_eval.v1` | `aggregate_summary.json` |
| Judge | `qwen2.5:14b`, blinded, temperature 0, candidate family llama | `provenance.json` |
| Judge time | `2026-11-03T00:43:46.431318+00:00` | same |
| Judge git | `31b27c03eb04faf88f00a7c7930c3e9f5a32e7ed`, dirty worktree, Python 3.11.9 | same |
| Generation SHA-256 | `6ed2af4f47c3f592aa833c643dc6354956b0ea7f96b7634b98a902b2a746724c` | hash of `raw/judge_inputs.jsonl`, which is not in Git |

The 120 topics are a repeated evaluation set. The protocol note says they are not an untouched corpus test split, and the 36-question development file shares no ids with them. Overlap of one content word of length 5 or more is listed in `UNIFIED_120_COMPARISON.md` and is not treated as the same task.

Development arm selection and the official panel are separate. The panel did not select F-REL.

## Gates and scores

`DEC-008` and `DEC-009` define the harness in `host_finetune/unified_eval.py`.

Eligibility gates, not averaged into a style score:

- Completion: a non-empty sentence-final answer passes. Mid-sentence stops fail.
- Copying / memorization: longest shared training-word span, threshold 12, excluding spans that also occur in the prompt.
- Format: requested-medium length band.
- Grounding: scored when retrieval was on, against saved retrieval hits. Not applicable when retrieval was off. The comparison file stores that case as missing, not as a pass. This reporting script does the same. A row can still contain `grounding.pass = true` with `applicable = false`; that true is not counted.

Selector and quality fields:

- Writing quality: six judge dimensions, mean on a 1–5 scale, calibrated when `writing_order_holds`. Model F summary says calibrated true.
- Task adherence: brief alignment and format, role `quality_dimension`.
- Style distance: held-out surface profile, medium-matched for blog, LinkedIn, and X. Talk is null on every saved architecture (30 missing values). Lower is closer. It is not an authorship test.
- Stance consistency: diagnostic unless calibration controls exist. Model F stance calibration is false, reason `stance_controls_missing`.
- Persona utility: 0.45 style, 0.20 perspective, 0.25 writing quality, 0.10 task adherence, and only when the draft is eligible and the components are selector metrics. Model F utility is null on all 120 rows.

Diagnostics that are not persona fidelity: BLEU 0.0112 and ROUGE-L 0.0769 on the Model F summary. BERTScore is null.

`lemkin_evidence_support` retrieves a frozen held-out validation corpus after generation. Model F mean claim support on that diagnostic is 0.9941, and the field is marked `not_rag_grounding: true`. It is not the grounding gate.

## Two Model F measurements

| Measurement | What it counts | Headline |
|---|---|---|
| `unified_eval.v1` judge | Eligibility, writing, task, style, claim support versus saved hits | 67/120 eligible; grounding pass 0.8167; eligible writing 3.2662; all-draft writing 3.165278 |
| Pre-judge audit `AUDIT-20261102-001-model-f-frel-prejudge` | Numbers, source copy, caps, topic echo, extraction | 52/120 unsupported-number cells; 36/120 completion failures; 7/120 source-copy failures at 12 words |

Claim support of 1.0 on the rows that have a score means the extracted claims matched saved hits. It does not mean every quantity in the draft was supported. 43 drafts failed the number check and still passed judge grounding (`tables/model_f_judge_vs_number_crosstab.csv`).

## Deterministic details that are easy to mis-cite

- Completion pass in the audit summary is 84/120 (rate 0.7). Failures are 36. The judge comparison completion pass rate is also 0.700.
- Mid-sentence stops: 36. Outputs at the medium token cap: 36. The intersection is 31 drafts (`tables/model_f_cap_and_completion.csv`). Five mid-sentence stops are not at the cap, and five at-cap drafts are not mid-sentence stops.
- Training-copy: `comparison.json` has 0 failures on n=120. The audit summary stores count 0 with denominator 104. The public flag CSV has `training_copy_pass=true` on 104 rows and a blank on 16 rows (`gold_027` through `gold_030`, all four media), and zero `false` values. The blank cells are missing flags, not recorded failures. The handoff leaves the 104-versus-120 statement unresolved. This audit records the blank ids and does not recompute the gate.
- Zero accepted claims: 4 drafts, all media of `gold_030`.
- Source-copy failures at the saved 12-word threshold: 7. Spans of at least 8 words: 10. These are retrieved-passage overlaps. They are not the training-copy gate. The judge copying pass rate is still 1.0 because that gate is the training span.

## Classification metrics

Accuracy, precision, recall, F1, ROC/AUC, and confusion matrices need a fixed label and a prediction for the same unit. This project does not have that label for "sounds like Lemkin" or for "every sentence is true."

What exists instead:

- Gate pass/fail on completion, format, training-copy, and applicable grounding. A pass rate is the right summary. A confusion matrix against an external human label was not collected.
- Style distance and writing quality are numeric scores, not classes.
- The number audit is a deterministic flag, not a human-adjudicated hallucination label. The 2×2 table in `tables/model_f_judge_vs_number_crosstab.csv` compares two automatic checks. It is not a confusion matrix against ground truth.
- `EXP-20261029-005` and the Model F docs state that retrieval precision and recall were not scored.
- No ROC curve is stored.

Using those classifier metrics would require labels this repository does not have. The report should say that, and should show the gate rates, score means, denominators, and the judge-versus-number 2×2.

## Missing values that must stay missing

- Grounding for Base and Relabel RAG off.
- Latency for every architecture except Model F.
- Style distance for talk.
- Persona utility for Model F, and for non-blog rows of the other systems.
- Claim support where the grounding method did not emit a rate (22 Model F drafts).

The plotting script drops missing values from means. It does not impute zero. Verification against `architecture_comparison.json` matched (`tables/verification_notes.txt`).

## Paired uncertainty

`tables/paired_model_f_differences.csv` is a topic-cluster bootstrap computed in this reporting directory from the saved rows: 4,000 resamples of the 30 question ids, seed 42, percentile interval, difference equals Model F minus the named architecture. It respects the shared-topic structure across media. It is not a pre-registered test inside an `EXP-*` manifest, and it is not a multiple-comparison correction across every metric. `docs/evaluation/ARCHITECTURE_COMPARISON.md` already contains seed-level intervals for earlier RAG on/off contrasts. Those intervals are a different contrast and should not be pasted onto Model F.
