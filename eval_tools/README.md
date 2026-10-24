# eval_tools

Small helpers for PUK-28 and PUK-34. They read results that are already saved and never call a model or a judge, so they run on any laptop in seconds. Background and the reasons for each tool are in `docs/evaluation/METRIC_MAP.md`.

Needs Python 3.10 or newer and nothing else (standard library only).

## 1. Compare two runs with confidence intervals

```
python -m eval_tools.compare \
  --a rows_a.jsonl --a-experiment EXP-... --a-model relabel --label-a "RAG off" \
  --b rows_b.jsonl --label-b "RAG on" \
  --out-md report.md --out-json report.json
```

Input: `metrics/rows.jsonl` files written by the unified harness. Run A is the reference and every difference is B minus A. Drafts are paired by (topic id, medium). Use the filter options when one file holds many runs.

What you get:

- Gates: how many drafts pass each gate in each run, and the eligible rate.
- Two views. All drafts: no draft dropped. Common gates: only drafts that pass completion, copying and format in both runs. Grounding is left out of the common gates on purpose, because only retrieval runs have it, so counting it would compare different sets of drafts.
- For each metric, overall and per medium: both means, the difference, a 95 percent bootstrap interval, a sign-flip p value, and a plain verdict. The interval and the p value are computed by topic, because the four mediums for one topic are not independent.
- `p holm` corrects the overall rows for the number of metrics tested.

A difference is called clear only when the interval excludes zero and the p value is below 0.05. A row by medium is not corrected for how many were tried, so treat it as a lead. "No clear difference" does not mean "no difference".

Metrics: writing_quality, task_adherence, stance_consistency, style_distance, framing_distance, persona_utility, claim_support, evidence_support, word_count, latency_seconds, grammar_rate, grammar_deviation. Choose with `--metrics`.

## 2. Grammar and mechanics score

```
python -m eval_tools.grammar --traces raw/relabel.jsonl --data-dir data \
  --out-jsonl grammar_rows.jsonl --out-json grammar_summary.json
```

Counts mechanical errors per 100 words (doubled words, missing space after punctuation, lowercase sentence start, lowercase "i", wrong a/an, missing apostrophes, "should of", common misspellings, some agreement slips, broken characters). It scores each draft against the rate for real Lemkin posts of the same medium (400 sampled per medium from `data/*.jsonl`), so his informal style is not punished. Talk has no reference, so it is scored in absolute terms only.

To use it in a comparison, pass its `--out-jsonl` file to `compare` with `--extra-a` or `--extra-b`.

Limits, please read: it is a small rule based checker, not LanguageTool and not a human editor. It does not see tense, awkward phrasing or meaning. On the saved Relabel drafts it finds almost nothing (0 to 0.04 errors per 100 words, below the real posts), so it cannot separate runs today. Treat it as a floor check, with the judge's fluency item as the second opinion. A stronger checker can replace the rules without changing anything else, because the output is just numbers per draft.

## 3. Tests

```
python -m pytest tests/test_eval_tools.py
```

44 tests on small made up data. They include a check that an extra gate on one run does not hide that run's gain, and checks that the grammar rules do not flag "Series A in", "a Uber", "an hour" or deliberate repeats.

## Example

`docs/evaluation/examples/relabel_rag_off_vs_on.md` is the output on Kevin's saved Relabel runs. Its mean differences match his own `comparison.json`.
