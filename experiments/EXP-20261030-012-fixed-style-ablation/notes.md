# Fixed-style off versus on

**Status: MEASURED.** This run does not show that `--fixed-style` improves persona fidelity. Keep the flag off.

## Question

Does adding the existing train-owned, medium-matched style exemplars to the frozen Relabel QLoRA generator change writing quality or voice, when factual retrieval and accepted claims are held fixed?

## What changed in code

Before this run, `--fixed-style` was ignored whenever factual retrieval was on. `generate()` built the Relabel prompt from accepted claims and never called `style_for`. The smallest change puts `format_voice_exemplars` in front of the factual rules, only when the flag is on. An empty exemplar list leaves the prompt text unchanged. Retrieved passages are still rejected if they appear outside that voice block. Unit checks: `tests/test_evidence_capsule.py`, `tests/test_numeric_grounding.py`, and `tests/test_compare_prompts.py` (39 passed). The full historical suite was not rerun.

## What was held fixed

- Adapter: `host_finetune/output/lemkin_lora_relabel` on `unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit`, loaded with `load_in_4bit=True`. `is_loaded_in_4bit` was true. Peak allocated memory during generation was 7025 MB.
- Chroma v2, collection `lemkin_train_only_v2_shadow`, 21,193 documents, `nomic-embed-text`, k=4, no medium filter. Retrieval was frozen once to `raw/frozen_chroma_v2.jsonl` and then reused.
- Claims were extracted once per topic with the adapter off and reused for both arms. All 90 matched cells share evidence hashes, hit ids, and seeds.
- Decoding: temperature 0.7, top_p 0.9, repetition penalty 1.15, seed base 42, the same max new tokens as the existing medium table.
- LightRAG was not queried. `rag_active.json` was not edited.

## Style exemplars

The file `host_finetune/exemplars/fixed_style_candidates.json` is unchanged. Every exemplar is a train row with `sft_role=unchanged`, the text matches the Relabel output, and the group is outside the gold-overlap block. The same two exemplars are used for every topic in a medium. Talk maps to `youtube_jason`, which is empty, so style-on talk was recorded as unsupported and not generated.

The X pair is weak source material even though it passes those checks. One post promotes SaaStrDeploy. The other is a quoted remark with Harry Stebbings and Rodrigo. They were not replaced, because swapping them would no longer be a test of the existing flag.

## Sample

30 development topics. Style-off: blog, LinkedIn, X, and talk (120 drafts). Style-on: blog, LinkedIn, and X (90 drafts) plus 30 unsupported talk records. No error cells. No cells were overwritten. Wall time 4015 seconds. The smoke topic (`gold_001`) took 126 seconds of generation, and the budget of 18000 seconds covered all 30 topics.

LanguageTool is not installed in this environment, so grammar diagnostics were not run. That tool is diagnostic only in any case.

## Voice

Fixed-length distance uses the repository's `host_finetune.voice_distance` on the first N words. Lower is closer to held-out Lemkin posts. Real blog prefixes score 1.11 at 100 words and 0.86 at 150.

| Comparison | Result |
|---|---|
| Blog, 100 words, paired | 22 topics long enough in both arms. Mean difference (on minus off) −0.000, 95% interval −0.109 to +0.099. |
| Blog, 150 words, paired | 18 topics. Mean difference +0.031, interval −0.051 to +0.127. |
| LinkedIn, 100 words, paired | 22 topics. Mean difference −0.014, interval −0.062 to +0.034. Diagnostic only. |

Style-on blogs are shorter (mean 196 words versus 241), so fewer of them reach 100 or 150 words (23 versus 27 at 100 words; 20 versus 26 at 150). The unpaired means are therefore not the comparison. X and talk are not scored: X is too short for a fixed length, and talk has no reference posts.

## Writing quality

Qwen 2.5 14B, temperature 0, six writing dimensions, blind to arm. All 210 completed drafts were scored.

On the 90 matched cells, style-on minus style-off is −0.12 (interval −0.30 to +0.06). By medium the blog gap is −0.30 (interval −0.55 to −0.07). LinkedIn is −0.01 and X is −0.07, and both intervals include zero. Means: blog 3.67 versus 3.37, LinkedIn 3.62 versus 3.61, X 2.57 versus 2.50. Talk, style-off only, averages 3.59.

## Matched counts out of 30

Order is style-off, then style-on.

| Check | Blog | LinkedIn | X |
|---|---|---|---|
| Factual pass | 15, 15 | 14, 12 | 22, 23 |
| Topic echo | 4, 5 | 16, 21 | 22, 25 |
| Unsupported number | 14, 13 | 15, 18 | 7, 6 |
| Completion | 29, 29 | 16, 14 | 16, 17 |
| Format | 30, 27 | 29, 29 | 30, 30 |
| Grounding gate | 3, 6 | 5, 5 | 11, 7 |
| Unified eligible | 3, 4 | 2, 3 | 5, 3 |

Across 90 pairs, unified eligibility is 10 and 10. Eligibility stays low because the grounding gate fails on most drafts in both arms. Ranking only the eligible drafts would compare different and tiny sets. Do not do that.

Training-corpus copy failures: 0 style-off, 1 style-on. Retrieval-source copy failures: 0 style-off, 3 style-on. The training-copy gate ignores spans that also appear in the prompt, so it under-counts copies of the exemplars. A direct span against the exemplar text finds one blog overlap of 9 words and one LinkedIn overlap of 59 words.

## Examples

`gold_001` blog, same claim and same seed. Style-off writes 295 words and argues through the enterprise social contract. Style-on writes 47 words and stops at "Let me explain." That draft is too short for the fixed-length voice score.

`gold_003` LinkedIn, style-on, pastes the prompt's style instructions and then a 59-word run from the second LinkedIn exemplar ("And are just burnt now. Fried. Some tried to mask it by hiring CROs..."). The leak markers that trigger a retry look for evidence labels such as `[E1]` and `Prompt contract:`. They do not look for `Style examples` or `[Style 1]`.

## Cost

Mean extra prompt tokens: blog +523, LinkedIn +280, X +173. Mean latency falls on blog (21.6 s to 16.8 s) because the style-on drafts are shorter, and rises slightly on LinkedIn (16.9 s to 18.1 s). No dollar cost is claimed.

## Recommendation

Leave `--fixed-style` off. The exemplars add a few hundred tokens and a second prompt block. They do not move fixed-length blog voice. They lower blog writing scores. They increase question echoes on LinkedIn and X. They can be copied verbatim. Talk cannot use the flag until real talk exemplars exist.

## Fine-tuning readiness

**Do not start another QLoRA run.**

The frozen adapter was trained on `experiments/EXP-20261003-004-relabel-continuations/raw/dataset.jsonl` (sha256 `1231835e…0897b`, 26,545 rows). The assignment file matches that hash. No group sits in more than one split.

The same file still has 451 extra exact-duplicate targets, 6,392 targets under 20 words, 1,186 promotional or boilerplate regex hits, 988 openings that are long quotations, and 15 continuations missing the prior-section marker. Train coverage is uneven: 11,190 X rows, 9,434 blog rows, 713 LinkedIn rows, and 40 transcript rows. Those are reasons the current data is imperfect. They are not a new versioned training set with a held-out evaluation plan. The graph-ready v2 corpus is a retrieval derivative, not a replacement SFT file. A retrain would risk the Relabel baseline that this ablation just measured, and there is no schedule left here to evaluate a new adapter before using it.

## Reproduce

```text
host_finetune\.venv\Scripts\python.exe -X utf8 experiments\EXP-20261030-012-fixed-style-ablation\config\run_paired.py --budget-seconds 18000
host_finetune\.venv\Scripts\python.exe -X utf8 experiments\EXP-20261030-012-fixed-style-ablation\config\score_paired.py
host_finetune\.venv\Scripts\python.exe -X utf8 experiments\EXP-20261030-012-fixed-style-ablation\config\score_paired.py --judge
host_finetune\.venv\Scripts\python.exe -X utf8 experiments\EXP-20261030-012-fixed-style-ablation\config\audit_sft.py
```

The runner skips completed cells. It does not overwrite them. Stop Ollama models before generation so Qwen is not resident during QLoRA inference. The judge run loads Qwen only after the adapter process has exited.
