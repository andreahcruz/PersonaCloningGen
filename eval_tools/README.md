# eval_tools

Small helpers for PUK-28 and PUK-34. They read results that are already saved and never call a model or a judge, so they run on any laptop in seconds. Background and the reasons for each tool are in `docs/evaluation/METRIC_MAP.md` (section 10 says how far each one can be trusted).

Needs Python 3.10 or newer and nothing else (standard library only).

## 1. Compare two runs

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
- For each metric, overall and per medium: both means, the difference, a 95 percent bootstrap interval, a sign-flip p value, `p adj`, and a plain verdict. Intervals and p values are computed by topic, because the four mediums for one topic are not independent.

How to read the verdict:

- `p adj` corrects for how many tests were run (Holm). The four primary rows are corrected together: writing quality, blog style distance, stance consistency and task adherence, which are the four parts of Persona Utility. Every other row is exploratory and is corrected as its own, larger family.
- "B better" or "B worse" (or "B higher" and "B lower" for word count, where neither direction is better): the interval excludes zero and `p adj` is below 0.05.
- "(suggestive)": it looks real on its own (interval excludes zero, p below 0.05) but does not survive the correction. A lead worth another seed, not a finding.
- "no clear difference" does not mean "no difference". "too few pairs": fewer than 5 topics.

Metrics: writing_quality, task_adherence, stance_consistency, style_distance, framing_distance, persona_utility, claim_support, evidence_support, word_count, latency_seconds, grammar_rate, grammar_deviation. Choose with `--metrics`.

## 2. Does a result repeat across seeds?

```
python -m eval_tools.seeds --rows rows.jsonl --a-model repaired --b-model relabel \
  --experiments EXP-A EXP-B EXP-C --labels 42 43 44 [--scope blog]
```

Runs the same comparison once per seed and shows them side by side, with whether every seed moved the same way. A result worth trusting keeps its sign in every seed. Three seeds show a pattern; they are not a significance test.

## 3. Grammar and mechanics floor check

```
python -m eval_tools.grammar --traces raw/relabel.jsonl --data-dir data \
  --out-jsonl grammar_rows.jsonl --out-json grammar_summary.json
```

Counts mechanical errors per 100 words (doubled words, a missing space after punctuation, a space before punctuation, a lowercase sentence start, a lowercase "i", wrong a/an, missing apostrophes, "should of", common misspellings, some agreement slips, broken characters). It scores each draft against the rate for real Lemkin posts of the same medium (400 sampled per medium from `data/*.jsonl`), so his informal style is not punished. Talk has no reference, so it is scored in absolute terms only. To use the result in a comparison, pass the `--out-jsonl` file to `compare` with `--extra-a` or `--extra-b`.

**What it is not.** It is not a grammar score. Measured with `grammar_audit`:

- About 82 percent of what it flags on a fresh sample of real posts is a real error (33 of 40). Earlier versions were at 60 and then 79 percent, before the false alarms were fixed.
- It catches 56 to 100 percent of injected errors of the kinds it has rules for.
- It catches 0 of 40 ordinary grammar mistakes (tense, wrong word, missing article, preposition, run-on, pronoun case, plurals, double negative, awkward wording).

So it is a floor check for typos and punctuation slips. On the saved Relabel drafts it finds almost nothing, which means "no mechanical errors found", not "good grammar". For real coverage use a stronger checker (LanguageTool, or a grammar prompt for the judge). It can be swapped in without changing anything else, because the output is just numbers per draft.

## 4. Audit the grammar tool

```
python -m eval_tools.grammar_audit --data-dir data --out-md audit.md --out-json audit.json
```

Injects known errors into real sentences that the tool finds clean, and counts how many it catches (recall per rule). Then runs a hand written set of 40 ordinary grammar mistakes in 10 categories, to measure coverage. A stronger checker can be passed to `coverage(check=...)` in Python to compare it on the same 40 sentences.

## 4b. Grammar with LanguageTool (Kevin's choice)

```
pip install language_tool_python        # needs Java 17 or newer; the first run downloads LanguageTool (about 260 MB)
python -m eval_tools.grammar_lt --traces raw/relabel.jsonl --data-dir data   --out-jsonl grammar_lt_rows.jsonl --out-json grammar_lt_summary.json
```

The same idea as section 3, with a much better checker. It catches 19 of the 40 ordinary mistakes in section 4 (the rule based check catches 0). To keep Lemkin's informal style from being punished, it keeps only grammar, real-typo and confused-word rules, ignores names and his own vocabulary, and compares each draft with real posts of the same medium. Pass `--out-jsonl` to `compare` with `--extra-a` or `--extra-b` and use `grammar_lt_rate` or `grammar_lt_deviation`.

**What it is not.** Most of its flags on our drafts are false alarms: 14 of 103 are clear mistakes (14 percent) in our own labeling. Use it as a screener and read the flags, not as a grade. Results, the 103 labeled flags, and the audit: `docs/evaluation/GRAMMAR_LANGUAGETOOL.md`. It runs on your computer; no text is sent anywhere. It is optional, and the other tools still need only Python.

## 5. Is a style distance gap just a length effect?

```
GIT_LFS_SKIP_SMUDGE=1 git worktree add ../kevin-ws origin/kevin/298b-integration   # skips ~480 MB of adapters
python -m eval_tools.length_control --kevin-root ../kevin-ws \
  --a-drafts rag_off_raw.jsonl --b-drafts rag_on_raw.jsonl --label-a "RAG off" --label-b "RAG on" --medium blog
```

The style distance averages rates per 100 words, so longer texts land closer to the profile (log word count and distance correlate at -0.66 over real posts and drafts). This tool cuts every draft to the same number of words and repeats the paired comparison, and compares drafts with real held-out Lemkin posts at full length and at matched length. It uses Kevin's own `host_finetune` code and reference data, which are on his branch, so that checkout is needed. On the saved Relabel runs it showed that the RAG blog voice gap (+0.104) shrinks to about +0.04 with intervals that include zero at equal length.

## 5b. Voice score on a fixed number of words (Kevin's choice)

```
GIT_LFS_SKIP_SMUDGE=1 git worktree add ../kevin-ws origin/kevin/298b-integration
python -m eval_tools.voice_fixed --kevin-root ../kevin-ws --traces raw/relabel.jsonl   --out-jsonl voice_fixed_rows.jsonl --out-json voice_fixed_summary.json
```

Scores the first 100 words of every blog and LinkedIn draft with Kevin's style distance, so length cannot drive the score (section 5 shows that it does). Drafts under 100 words are not scored, and the summary says how many. X and talk are not supported (X drafts are too short, talk has no reference). Compare with `--metrics style_distance_fixed`. Use `--blog-words 150 --linkedin-words 150` to check that the conclusion does not depend on the length. Results: `docs/evaluation/VOICE_FIXED_LENGTH.md`.

## 6. Tests

```
python -m pytest tests
```

115 tests on small made up data: 76 for the tools in sections 1 to 5, 26 for the LanguageTool tool (with a fake LanguageTool, so no Java is needed) and 13 for the fixed-length voice score (with a fake distance, so no checkout of Kevin's branch is needed). They include a check that an extra gate on one run does not hide that run's gain, the correction for many tests, the seed check, and every false alarm the grammar rules have ever produced on real posts (so they stay fixed).

## Examples and evidence

In `docs/evaluation/examples/`:

- `relabel_rag_off_vs_on.md`: Kevin's saved Relabel runs, RAG off against RAG on. The mean differences match his own `comparison.json`. Copies of the input files are in `data/`, with source paths and hashes in `data/SOURCES.md`, so it can be rerun without his branch.
- `seed_stability_relabel_vs_repaired.md`: the same comparison for seeds 42, 43 and 44.
- `style_length_check.md`: the length check on the same runs.
- `grammar_audit.md`: recall and coverage of the grammar tool.
- `grammar_precision_audit.csv`: every flagged item that was labeled by hand (real error, false alarm or text artifact), across three samples of real posts.

## Known limits

- One run per arm for the RAG comparison, and 30 gold topics that form a development panel, not a test set.
- The correction for many tests is deliberately conservative. Calibration on real data with the true difference set to zero: a primary test was wrongly called clear 2.7 percent of the time (target at most 5 percent).
- The grammar precision of 82 percent comes from 40 flags, so the true value could be roughly 68 to 91 percent.
