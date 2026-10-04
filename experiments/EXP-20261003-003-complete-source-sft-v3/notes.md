# Whole-source preparation v3

Question: can complete source documents produce faithful SFT targets while preserving the repaired EXP-008 ownership and avoiding known truncation/echo/leakage mechanisms?

Measured candidate: 1,011 complete blog bodies, 601 train / 206 validation / 204 test, at a 2048-token full-chat budget. Independent structural verification passed for every row and 288,434 cross-split pairs. No normalized-exact or token-Jaccard >=0.8 crossings, and no exact duplicates within the candidate, were found. All 34 prior inventoried artifacts and all input hashes remained unchanged. All 13 documents in cross-split families and 39 documents in gold-overlap families were quarantined. No split reassignment, source rewriting or rebalancing occurred.

The exact code, parameters, tokenizer identities, command and package snapshot are preserved. No model weights were loaded; model training, GPU feasibility, retrieval, persona construction, embeddings and judging are inapplicable. Source attribution is not independent authorship verification. Blog-only eligibility changes the medium distribution relative to historical SFT and prevents an apples-to-apples model comparison.

## Content-quality gate: NOT PASSED

All 16 deterministic train/validation samples were inspected. Review annotations in raw/content_review.jsonl are an assistant textual review, not independent human author labels. The sample is diagnostic, not a random quality estimate. Test bodies were not reviewed.

V3 removes several v2 contaminants, but fresh samples reveal remaining categories: 02d70363... has a word split across lines; 02d85f3e... and 02f41f9e... are article/podcast roundups; 036c89f0... is a guest session summary with an event introduction. Other samples retain missing-embed introductions. Even apparently coherent samples have no independently verified authorship. Training readiness stays false.

Stop treating regex eligibility as clean-corpus validation. Next work should recover source page author and block/link/embed structure or annotate an explicit source cohort, retaining frozen ownership and audit trails. No training or model-improvement claim is made. Existing trainer defaults and RAG indexes were not changed.

Tests after the v3 code change: `.venv-test/Scripts/python.exe -m pytest -q` produced 129 passed, 4 skipped (0.49 s). A transcription of the observed output and test-source hashes is recorded with this run. The tokenizer verifier handles both list and mapping returns; the compatibility failure occurred during v2 verification and was resolved before v3.
