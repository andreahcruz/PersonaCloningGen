# Whole-source preparation v1

Recorded 2026-10-03. Preparation only; no model training or generation. Original raw/cleaned/SFT/split inputs were hashed before and after and remained unchanged. Configuration contains the exact executed source snapshot, tokenizer-file hashes, command, environment and package snapshot. JSON-formatted manifest.yaml is valid YAML.

Result: 1,498 complete blog candidates under 2048 full-chat tokens, retaining original source ownership (867 train / 318 validation / 313 test). These proportions intentionally differ from 80/10/10: whole-document eligibility filters an existing chunk-balanced split. No assignments were rebalanced.

All 18,677 owned full documents were checked, including otherwise ineligible social/transcript sources. Found 13 documents in lexical duplicate families spanning frozen splits; all family members were quarantined. Gold title/body/fact lexical overlap propagated to 39 family documents, also quarantined. No source was reassigned. Exact cleaner replay matched raw lineage; no raw/cleaned or prior-fragment mismatch was reported. This does not independently verify authorship.

## Review failure and follow-up

The deterministic train/validation-only review sample exposed contamination that old regexes missed. Train source 0032c0e9... is an event-session advertisement; 01a20b91... is a Quora-question link roundup; 01b4a286... is a guest-executive recap. Validation 062e9801... and 0669868a... promote Annual/expo sponsorship. Several other sampled records include embedded tweet/media fragments. These are not suitable automatically approved style targets. No held-out test bodies were reviewed or used to tune rules.

Run 001 is retained unchanged as the v1 candidate and quality-review failure evidence. Version 2 adds conservative whole-document quarantine reasons for event-titled material, executive guest recaps, embedded/external-context markers and question roundups. It does not rewrite those documents, remove them from raw data, or reassign splits. These heuristics can reject good material and do not replace authorship annotation.

Long posts are deferred, not split; untitled social posts are deferred because using their opening text as a topic creates target-derived prompts. Transcripts require speaker attribution. This is a blog-only candidate, not an apples-to-apples replacement for prior mixed-medium SFT. Training readiness remains false.

RAG, embeddings, profiles, adapters, GPU/judge fields are not applicable: the operation reads a tokenizer and performs deterministic CPU data preparation. Lexical Jaccard >=0.8 does not establish semantic duplicate absence. There is no model-performance result.
