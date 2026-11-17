# Authorship classifier: does a draft look like Jason Lemkin?

Part of PUK-28 (metrics). It adds a trained authorship check next to our existing metrics. It reads saved drafts only
and changes nothing in Kevin's evaluation or in any frozen experiment.

Report with all checks and results: `docs/evaluation/AUTHORSHIP_CLASSIFIER_RESULTS.pdf`.
Research backing for every metric: `docs/evaluation/METRICS_RESEARCH_BACKING.pdf`.

## What it does
Trains a classifier on real Lemkin posts (Kevin's TRAIN split, gold-topic families removed) against two kinds of
"not Lemkin": Paul Graham essays and plain base-Llama text. It scores the first 100 words (blog) or 60 words
(LinkedIn) of every saved draft. Each draft is scored by a model that never saw that draft's topic (5 topic folds).
Three views: topic-hiding "style", number-free style, and all words. Statistics reuse `eval_tools/bootstrap_ci.py`.

## Run
    git fetch --all
    python authorship_classifier/prepare_inputs.py       # copies public inputs with git show (read only)
    cd authorship_classifier
    python -X utf8 run_pipeline.py                       # about 3 minutes, no GPU
    python -X utf8 make_report.py                        # figures and PDF (needs all systems, see below)
    python -X utf8 -m pytest tests                       # 13 tests for the leak-prone steps

Needs Python 3.11, scikit-learn, numpy, pandas, scipy, matplotlib, reportlab.

## What is and is not in the repository
| Input | Where it comes from | In the repo? |
|---|---|---|
| Kevin's split, gold file, drafts for A, D, E (and E+retrieval seed 42) | branch `kevin/lightrag-model-f` | yes |
| Paul Graham essays and split | branch `master` | yes |
| Kevin's fixed-length voice distances | `docs/evaluation/examples/data/voice_fixed/` | yes |
| PersonaRAG and E+retrieval seeds 43 and 44 | Colab outputs on the team Drive | no (`--colab-dir`) |
| Model F drafts | Kevin's private export | no, never commit (`--f-rel`) |

Without the last two rows the pipeline still runs and skips those systems. `make_report.py` expects all systems, so the
committed PDF and `outputs/` come from a run that had all of them. `inputs/` is git-ignored on purpose.

## Files
| Path | What |
|---|---|
| run_pipeline.py | the experiment: training, validity checks, system scores, paired statistics |
| make_report.py | builds the figures and the PDF from outputs/results.json |
| prepare_inputs.py | collects the inputs (read only) |
| src/textprep.py, data.py, model.py, stats.py | cleaning and windows, loaders, classifier views, intervals and tests (stats.py is a copy of eval_tools/bootstrap_ci.py) |
| outputs/ | results.json, scores_per_draft.csv (one row per draft, no draft text), three figures |
| tests/ | 13 tests |

## Stronger negatives (not done yet)
The default negatives are our own base-Llama drafts on other topics, cross-fitted by topic. The cleaner version feeds
several hundred plain base-Llama texts written on non-gold topics:

    python -X utf8 run_pipeline.py --llm-negatives generated.jsonl --out-dir outputs_colab

One JSON per line: `{"text": "...", "medium": "blog"}`. The option is built and tested. The Colab generation is not done.

## Limits
A proxy for style, never proof of authorship. It reads surface style (em dashes, quotation marks, sentence shape).
Same 30 gold topics as the rest of our evaluation, one run each for A, C and F. X posts and talks are not supported.
No human has checked it. See the report for the full list.
