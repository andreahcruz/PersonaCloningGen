# Whole-source preparation v2

Preparation only, measured 1,142 complete blog candidates: 653 train, 245 validation, 244 test. Source ownership and full bodies were preserved. Independent structural verification passed for 379,097 cross-split pairs, with zero detected lexical crossings; all 34 baseline artifacts remained unchanged. See metrics/summary.json and verification.json.

The first verification attempt exposed a Transformers compatibility issue: apply_chat_template returned a mapping rather than token IDs. The verifier initially counted mapping keys. It was fixed to read input_ids, tested for both return types, and rerun successfully. This was a verifier failure, not an input mutation or model result.

All 16 fixed train/validation samples were inspected. Surviving source 021feaca... is a weekly link digest; 022a454f... and 0e27083d... contain broken words; 0242fe69..., 0289b09e... and 0aa4e49f... refer to absent visuals; 0bcfddc4... ends with a related-link title. These findings motivated the separately recorded v3 conservative quarantine policy. No test bodies were reviewed or used for policy development. Other samples retain inline markup artifacts and missing embed introductions; these require source-aware recovery, not synthesized completions.

This run remains a candidate, not approved training data. Exact raw cleaner replay is provenance evidence, not authorship verification. No model weights, persona, retrieval, embedding, judge or GPU generation were used, so those manifest fields are inapplicable. Lexical overlap checks cannot establish semantic leakage absence. Source snapshots, inputs, tokenizer hashes, command and package versions are in config/ and manifest.yaml.
