# EXP-20261001-007 — Mixed-medium comparison

Question: if the three existing adapters are asked with the training medium line, do blog, LinkedIn, X, and talk prompts produce different answers, and does a longer budget plus a repetition penalty reduce the hard cutoff?

Held fixed: the three adapters (`lemkin_lora_cleaned`, `lemkin_lora_balanced`, `lemkin_lora_r32`) and `data/openai_ft/lemkin_gold_eval.jsonl`. Expected facts were scored after generation and were not placed in the prompt.

## Settings

Same 30 topics. Four instructions: blog post, LinkedIn post, X post, talk. Loaded context 2048. New-token budgets 768 / 320 / 160 / 768. Temperature 0.7, top-p 0.9, repetition penalty 1.15, sampling on, 4-bit load.

## Result

120 generations per adapter. Gold-rubric overall: cleaned 0.2350, balanced 0.2346, rank 32 0.2688. Rank 32 leads on this mixed set. X answers stay short, so expected-fact coverage on X is low (0.11–0.19) because the facts were written for blog-length answers. Faithfulness is 0 because no passages were retrieved. Raw answers are in `raw/<adapter>.jsonl`.
