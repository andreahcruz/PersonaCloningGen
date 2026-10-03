# EXP-20261001-010 — Repaired adapter, Llama 3.1, and Lemkin clones

Question: on the same 30 topics and four mediums, how does the repaired rank-16 adapter compare with an untrained Llama 3.1 and with Lemkin clone v5 and v6?

The repaired adapter was scored first. Llama 3.1, `lemkin-clone-v5`, and `lemkin-clone-v6` then received the same prompts, token budgets, temperature 0.7, top-p 0.9, and repetition penalty 1.15. Expected facts stayed out of the prompt. The Ollama models were called through `/api/chat`. The adapter was generated with Unsloth.

`ollama list` reports the same model id `06b812f9374d` for clone v5 and v6, so they are one set of weights. Their score gap is a second sample, not a different model.

## Scores (gold-rubric overall, 120 prompts)

- Repaired adapter: 0.2441
- Llama 3.1, untrained: 0.3684
- Lemkin clone v5: 0.2581
- Lemkin clone v6: 0.2639

The untrained instruct model leads this rubric. The rubric includes format compliance (headings, lists, length). Llama 3.1 follows that format more often than the fine-tunes, which were trained on Lemkin's own posts. BLEU and ROUGE-L stay near 0.004 and 0.08 for every model, so this set does not show a fine-tune pulling closer to the gold paragraph than the base model.
