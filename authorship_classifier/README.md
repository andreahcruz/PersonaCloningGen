# Authorship classifier: does a draft look like Jason Lemkin?

Part of PUK-28 (metrics). It adds a trained authorship check next to our existing metrics. It reads saved drafts only
and changes nothing in Kevin's evaluation or in any frozen experiment.

Report with all checks and results: `docs/evaluation/AUTHORSHIP_CLASSIFIER_RESULTS.pdf`.
Research backing for every metric: `docs/evaluation/METRICS_RESEARCH_BACKING.pdf`.

## What it does
Trains a classifier on real Lemkin posts (Kevin's TRAIN split, gold-topic families removed) against two kinds of
"not Lemkin": Paul Graham essays and plain Llama text. It scores the first 100 words (blog) or 60 words (LinkedIn)
of every saved draft. Each draft is scored by a model that never saw that draft's topic (5 topic folds).
Three views: topic-hiding "style", number-free style, and all words. Statistics reuse `eval_tools/bootstrap_ci.py`.

## Results in short (final version, `outputs_colab/`)
- Real held-out Lemkin posts against unseen plain Llama text: AUC 0.966 for blogs, 0.875 for LinkedIn (exploratory).
- Blog mean P(looks like Lemkin): Base Llama 0.35, PersonaRAG 0.38, QLoRA repaired 0.79, Model E 0.86,
  E with retrieval 0.82, Model F 0.80. Real held-out posts score 0.84.
- Fine-tuning moves the score a lot. Retrieval and the graph give no clear gain. The ranking agrees with the PUK-28
  fixed-length voice distance.
- The earlier version (`outputs/`) used our own Base Llama drafts as negatives. The final version uses 900 fresh Llama texts
  made on Colab, so no system's drafts are used for training. Blog ranking and all 8 blog verdicts are unchanged;
  PersonaRAG fell from 0.51 to 0.38, now close to Base Llama.
- A proxy for style, never proof of authorship.

## Run
    git fetch --all
    python authorship_classifier/prepare_inputs.py       # copies public inputs with git show (read only)
    cd authorship_classifier
    python -X utf8 run_pipeline.py                       # earlier version, about 3 minutes, no GPU
    python -X utf8 run_pipeline.py --llm-negatives outputs_colab/llm_negatives_clean.jsonl --out-dir outputs_colab   # final version
    python -X utf8 make_report.py outputs_colab          # figures and PDF (needs all systems, see below)
    python -X utf8 -m pytest tests                       # tests for the leak-prone steps

Needs Python 3.11, scikit-learn, numpy, pandas, scipy, matplotlib, reportlab.

## The Colab step (plain Llama negatives)
    colab/make_llm_negatives.ipynb            template: run on Colab (GPU), downloads llm_negatives_bundle.zip
    colab/make_llm_negatives_executed.ipynb   the run we used (A100, bf16, about 12 minutes, outputs kept)
    colab/llm_negatives_bundle/               the 900 generated texts and manifest.json (model, seed, settings, checksum)
    python -X utf8 colab/selftest.py          checks the notebook's own code on a tiny model, no GPU needed
    python -X utf8 validate_negatives.py colab/llm_negatives_bundle/llm_negatives.jsonl outputs_colab

The notebook writes plain Llama 3.1 8B text on 450 non-gold Lemkin blog titles, with the same prompt wording and
sampling as the Base Llama baseline (temperature 0.7, top-p 0.9, repetition penalty 1.15). The weights are the Hugging
Face copy, not the Ollama Q4_K_M build used for that baseline, so wording differs slightly.
`validate_negatives.py` has pass/fail rules fixed before the results were seen: checksum, no gold-topic overlap, enough
texts, at most 2 percent refusals, at most 5 percent copying (12-word runs, ignoring the topic title as Kevin's copy gate
does), at least 90 percent usable. The first run failed the copy rule at 5.7 percent because 48 of 51 flags only repeated
the topic title; the rule was corrected to match Kevin's gate and both numbers are reported (0.3 percent real copying).

## What is and is not in the repository
| Input | Where it comes from | In the repo? |
|---|---|---|
| Kevin's split, gold file, drafts for A, D, E (and E+retrieval seed 42) | branch `kevin/lightrag-model-f` | yes |
| Paul Graham essays and split | branch `master` | yes |
| Kevin's fixed-length voice distances | `docs/evaluation/examples/data/voice_fixed/` | yes |
| PersonaRAG and E+retrieval seeds 43 and 44 | Colab outputs on the team Drive | no (`--colab-dir`) |
| Model F drafts | Kevin's private export | no, never commit (`--f-rel`) |

Without the last two rows the pipeline still runs and skips those systems. `make_report.py` expects all systems, so the
committed PDF and results come from a run that had all of them. `inputs/` is git-ignored on purpose.

## Files
| Path | What |
|---|---|
| run_pipeline.py | the experiment: training, validity checks, system scores, paired statistics |
| make_report.py | builds the figures and the PDF from a results folder |
| prepare_inputs.py | collects the inputs (read only) |
| validate_negatives.py | checks the Colab file before it is used |
| src/textprep.py, data.py, model.py, stats.py | cleaning and windows, loaders, classifier views, intervals and tests (stats.py is a copy of eval_tools/bootstrap_ci.py) |
| outputs/ | earlier version: results.json, scores_per_draft.csv (one row per draft, no draft text), figures |
| outputs_colab/ | final version, plus negatives_validation.json and the cleaned negatives |
| colab/ | notebook, builder, topics, self-test, generated texts |
| tests/ | tests |

## Limits
A proxy for style, never proof of authorship. It reads surface style (em dashes, quotation marks, sentence shape).
The negatives come from one model and one prompt. Same 30 gold topics as the rest of our evaluation, one run each for
A, C and F. LinkedIn is weaker than blog. X posts and talks are not supported. No human has checked it.
See the report for the full list.
