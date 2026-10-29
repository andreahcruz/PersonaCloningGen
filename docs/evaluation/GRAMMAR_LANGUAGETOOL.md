# Grammar with LanguageTool (PUK-28)

Kevin chose LanguageTool for the grammar axis (and said to use our own judgement if we disagreed). This page says what we built, how well it works, and what it shows about our architectures. The short version: **LanguageTool finds far more than our old checker, but most of what it flags on our drafts is not a real mistake, and grammar turns out not to be a weakness of any architecture.**

## The short answer

1. **It finds ordinary grammar mistakes the old checker could not.** On our 40 test sentences (tense, wrong word, missing article, preposition, and so on) the old rule-based checker caught **0 of 40**. LanguageTool with our settings catches **19 of 40 (48%)**. It still misses about half, including most comma splices (0 of 4) and subject-verb slips (1 of 4).
2. **Most of its flags on our drafts are false alarms.** We read all 103 flags across the eight runs: **14 (14%) are clear mistakes**, 19 are colloquial or spoken-style constructions that a strict grammarian would flag, 3 are not grammar (two drafts cut off mid-word, one stray HTML fragment), and 67 are false alarms. So the number of flags is a screen, not a grade.
3. **Clear mistakes are rare in every architecture.** Only 1 to 4 clear mistakes per 120 drafts, and every run flags *less* than real Lemkin posts do. Grammar does not separate the architectures: no pair differs clearly.
4. **Practical use:** treat grammar as "no problem found" and, when it matters, read the flagged examples. The flag count cannot rank models.

## Why not LanguageTool as it comes

On 400 real Lemkin posts per medium, default LanguageTool flags **2 to 3 things per 100 words**. Almost all of it is noise for this purpose, because Lemkin writes informally on purpose:

- spelling flags on names and jargon (Coatue, SaaStr, EMW);
- spacing and typography, comma style, repeated sentence starts;
- apostrophes that were broken in the data (the `�` replacement character).

So `eval_tools.grammar_lt` keeps only the rule groups that mean "mistake" (grammar, real typos, confused words), plus a/an and doubled words. It also ignores anything touching a broken character, and it calls a word a typo only if it is all lowercase and Lemkin's own posts use it fewer than three times. Seven rules that fired on deliberate style in real posts are switched off (for example "brand new customers", "best case"). After that, real posts flag **0.11 (blog), 0.13 (LinkedIn) and 0.43 (X) per 100 words**.

As with the old checker, the useful number is the **deviation**: a draft's rate minus the rate of real posts of the same medium. Talk has no reference (it is auto captions), so it only gets a plain rate.

Everything runs on your own computer. LanguageTool starts a local server, and no draft is sent to a web service.

## Results

### Which kinds of mistake it catches (40 test sentences)

| Kind | Caught | Kind | Caught |
|---|---|---|---|
| Pronoun case ("Me and him is") | 4 of 4 | Preposition | 2 of 4 |
| Double negative | 3 of 4 | Confused words ("Their going") | 2 of 4 |
| Plural or countability | 3 of 4 | Subject-verb agreement | 1 of 4 |
| Verb tense | 2 of 4 | Missing words | 1 of 4 |
| Awkward word choice | 1 of 4 | **Comma splice, run-on** | **0 of 4** |

### What the 103 flags on our drafts really are

Hand labels for every flag are in `examples/grammar_lt_flag_labels.csv`. Examples of **clear mistakes** it found: "a SDR", "a inflection point", "a unwritten agreement", "a $8 Billion Exit", "they don't effect your ability" (affect), "diffuse the tension" (defuse), "for for decades", and typos such as "decelaration" and "materialally". The a/an rule was right 4 times out of 4. Typical **false alarms**: "when it IPO'd" (flagged as it/its), valid words like "overhire" and "wipeout", series names like "New New School", and "learned deploying agents".

| Run | Flags | Clear mistakes | Colloquial | Not grammar | False alarms | Drafts with a flag (of 120) |
|---|---|---|---|---|---|---|
| Base Llama 3.1 | 14 | 4 | 0 | 1 | 9 | 11 |
| PersonaRAG | 6 | 2 | 0 | 0 | 4 | 6 |
| Relabel, RAG off (seeds 42, 43, 44) | 10, 16, 18 | 2, 1, 1 | 1, 1, 3 | 0, 0, 1 | 7, 14, 13 | 10, 15, 17 |
| Relabel + RAG (seeds 42, 43, 44) | 16, 8, 15 | 2, 1, 1 | 6, 4, 4 | 0, 0, 1 | 8, 3, 9 | 13, 7, 13 |

### Do the architectures differ? (paired, all drafts)

| Comparison | Rate per 100 words (A to B) | Difference (95% interval) | Verdict |
|---|---|---|---|
| Relabel RAG off to on, seed 43 | 0.053 to 0.031 | -0.022 (-0.054 to +0.008) | no clear difference |
| Relabel RAG off to on, seed 44 | 0.055 to 0.043 | -0.012 (-0.053 to +0.025) | no clear difference |
| PersonaRAG to Relabel + RAG, seed 43 | 0.021 to 0.031 | +0.010 (-0.018 to +0.043) | no clear difference |
| Base to PersonaRAG | 0.035 to 0.021 | -0.014 (-0.043 to +0.014) | no clear difference |

On the deviation from real posts, every run is below zero (cleaner than real posts), and only one comparison is even "suggestive" (RAG on, seed 43), which does not hold in seed 44. Full reports are in `examples/grammar_lt/`.

## What this does not tell you

- **The precision labels are mine, not independent.** I (Claude) labeled the flags by reading them. The CSV is ready for a teammate to re-label. The 14% figure could move by a few points.
- **The settings were tuned on real posts, then checked on our drafts.** That is the right order, but the profile can miss a kind of error that real posts do not happen to contain.
- **It misses about half of ordinary mistakes**, and the audit sentences are our own small set (40), so 48% could be roughly 33% to 63%.
- **Talk drafts are spoken-style on purpose** ("there's two types", "less people"), which accounts for most of the colloquial flags. Talk has no real-post reference.
- **The error counts are tiny** (1 to 4 clear mistakes per run), so no ranking of architectures is possible. This is a finding about grammar, not a failure to measure it.
- **It is a screener, not a grade.** A human or a stronger judge prompt would be needed for a true grammar score.

## How to run it

Needs Java 17 or newer and `pip install language_tool_python` (the first run downloads LanguageTool, about 260 MB). From the repository root:

```
python -m eval_tools.grammar_lt --traces <draft file> --data-dir data --out-jsonl grammar_lt_rows.jsonl --out-json summary.json
python -m eval_tools.compare --a rows_a.jsonl --extra-a grammar_lt_a.jsonl --b rows_b.jsonl --extra-b grammar_lt_b.jsonl \
  --metrics grammar_lt_rate grammar_lt_deviation
python docs/evaluation/examples/data/grammar_lt/audit_languagetool.py      # the 40-sentence audit
python -m pytest tests/test_grammar_lt.py                                  # 26 tests, no Java needed
```

Data, hashes of the draft files and the rebuild steps: `examples/data/grammar_lt/SOURCES.md`.
