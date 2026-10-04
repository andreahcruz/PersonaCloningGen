# Frozen relabel audit

CPU-only read-only input audit. See metrics/audit.json and raw/findings.jsonl. All 26,545 emitted targets preserve the prior text and split/group assignments. Input hashes are unchanged. This run loaded no model, tokenizer or GPU and did not touch EXP-004/005 or training outputs. Model, judge, retrieval and random seed fields are inapplicable.

Source flags are joined from the prior registered whole-source v3 audit. They are not a new independent duplicate/gold detector or semantic quality verdict. Flag counts overlap and refer to rows. Held-out bodies were processed mechanically, not manually reviewed. No model-performance claim follows from these data diagnostics.

The accompanying code changes add stop telemetry/seeds to future comparisons, not training. Full CPU suite: 141 passed, 4 skipped. This report does not claim those GPU comparison paths were exercised. See docs/evaluation/RELABEL_FT_NEXT_STEPS.md for interpretations and the post-training protocol.
