# Results and interpretation

Every overall number in this section was checked against `results/model-f-handoff-v1/architecture_comparison.json` by `scripts/build_report_figures.py`. The check matched. Eligible writing quality for Model F matched `aggregate_summary.json` (3.2662). Per-medium figures are recomputed from the same row files and from `judge_scores.csv`.

## All 120 drafts

| Architecture | Eligible | Completion | Format | Training-copy pass | Grounding | Writing (n=120) | Task (n=120) | Style (n=90) | Words |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Base | 45 | 0.375 | 1.000 | 1.000 | — | 3.578 | 0.743 | 1.043 | 352.1 |
| PersonaRAG | 57 | 0.592 | 1.000 | 1.000 | 0.833 | 3.599 | 0.725 | 0.906 | 296.5 |
| Relabel RAG off | 77 | 0.658 | 0.958 | 0.992 | — | 3.000 | 0.701 | 0.750 | 192.9 |
| Relabel + RAG | 72 | 0.700 | 0.975 | 0.983 | 0.883 | 2.896 | 0.709 | 0.805 | 183.1 |
| Model F | 67 | 0.700 | 0.967 | 1.000 | 0.817 | 3.165 | 0.695 | 0.739 | 203.1 |

A dash means the metric was not applicable. It is not a zero.

Persona utility means, blog-only samples: Base 0.620 (n=20), PersonaRAG 0.631 (n=19), Relabel RAG off 0.705 (n=23), Relabel + RAG 0.688 (n=19), Model F null (n=0).

## Eligible drafts only

Writing quality on drafts that passed the saved ranking gates:

| Architecture | Eligible n | Eligible writing quality | Eligible task adherence |
|---|---:|---:|---:|
| Base | 45 | 3.733 | 0.755 |
| PersonaRAG | 57 | 3.708 | 0.727 |
| Relabel RAG off | 77 | 3.154 | 0.716 |
| Relabel + RAG | 72 | 3.060 | 0.716 |
| Model F | 67 | 3.266 | 0.703 |

Model F's eligible writing mean is higher than its all-draft mean because ineligible drafts, concentrated in LinkedIn and X, pull the 120-draft average down. Compare eligible means only with other eligible means.

## Who leads which metric

| Metric | Leader on the saved panel | Caveat |
|---|---|---|
| Eligible drafts | Relabel RAG off, 77/120 | Eligibility is a gate product, not a prose score |
| Completion pass | Relabel + RAG and Model F, both 0.700 | Tie. Base is 0.375 |
| Judge grounding | Relabel + RAG, 0.883 | Base and Relabel off are not in this comparison |
| All-draft writing quality | PersonaRAG, 3.599 | Base is 3.578. Both have low completion |
| Eligible writing quality | Base, 3.733 | n=45, so the mean is on the drafts that already finished cleanly |
| Task adherence | Base, 0.743 | Model F is 0.695 |
| Style distance | Model F, 0.739 | Lower is closer. Interval versus Relabel RAG off includes zero. Talk excluded |
| Persona utility | Not decidable | Model F is null. Other n is about 20 blogs |
| Training-copy pass | Base, PersonaRAG, and Model F at 1.000 on the comparison file | Model F flag CSV has 16 blanks. See methodology |
| Generation latency | Model F mean 20.07 s | Other architectures did not store latency |

## Medium pattern

Completion failures concentrate in short media for every Relabel system. Model F completion failures are 0 blog, 17 LinkedIn, 18 X, and 1 talk (`per_medium.json` and `tables/architecture_metrics_by_medium.csv`). Base fails even more often on LinkedIn (25) and X (27). X writing quality is the weak cell for the Relabel family: Relabel off 1.917, Relabel + RAG 2.006, Model F 2.661, against Base 3.061 and PersonaRAG 3.122, each on n=30.

Model F blog writing quality is 3.578, matching Relabel off's blog mean and below PersonaRAG's 3.722. Model F blog grounding failures are 7 of 30, the weakest Model F grounding medium.

Mean words fall from Base (blog 515, talk 536) to the Relabel systems (blog about 247–285, X about 34–90). Model F X averages 90 words, longer than Relabel + RAG X at 39 words. Length is not quality.

Figures `07` through `12` show these splits. Talk style bars are marked unavailable.

## Model F judge versus number check

Of 120 Model F drafts (`tables/model_f_judge_vs_number_crosstab.csv`):

| Judge grounding | Unsupported number | Drafts |
|---|---|---:|
| Passed | No | 55 |
| Passed | Yes | 43 |
| Failed | No | 13 |
| Failed | Yes | 9 |

Unsupported-number cells by medium: blog 19, LinkedIn 9, X 10, talk 14. The judge can assign a high claim-support rate while the draft introduces quantities the deterministic checker does not find.

## What not to conclude

- Model F is not the best system on this panel.
- A grounding pass rate of 0.817 is not a factual-accuracy rate.
- Style distance of 0.739 is not an authorship probability.
- 27,797 nodes do not explain the grounding rate.
- The all-draft writing mean and the eligible writing mean are not interchangeable.
- Historical workbook charts are not this table. `docs/PROJECT_CONTEXT.md` forbids reusing old slide scores as current results.
