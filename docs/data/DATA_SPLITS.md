# Splits and leakage boundaries

2026-10-03. Existing assignments were not changed.

`split_groups.py` joins source filename/line, normalized title, normalized-exact output, and token-set Jaccard >=0.8 pairs. It greedily assigns whole components with medium quotas around 80/10/10. This is an SFT-row diagnostic, not complete semantic duplicate detection over raw documents.

Current dataset-hash/row-count checks passed for EXP-20261001-002 (42,877 rows: 34,302/4,287/4,288) and EXP-20261001-008 (31,328 rows: 25,062/3,133/3,133). The stored EXP-008 report reports zero group, normalized-exact and detected-near-duplicate crossings. Full near-duplicate search was not independently rerun during this audit. September's 605 crossing pairs concerned a different file and approximated 90/10 split.

Trainer default uses EXP-002; repaired commands explicitly use EXP-008. Empty `SPLIT_MANIFEST` permits the legacy in-memory 90/10 fallback for other dataset paths; do not use it for a new controlled study. Validation losses from different splits are not comparable architecture outcomes.

Before new preparation, map source-family ownership across versions; quarantine conflicts/unmatched documents. No regenerated assignments without explicit approval. Review exact/near duplicates and cross-platform families; record detector limitations.

Apply ownership to training, profiles, few-shot selection, factual/style indexes, classifier references and judge anchors. Root profile extraction/index builders lack this boundary; generation has no split filter. `rebuild_chroma.py` deletes/recreates its collection and reads an entire supplied file without a split gate. Do not run it as a read-only audit.

Gold tasks are separate from the source test partition. The 30-topic comparison set has influenced development and should not be described as an untouched final test. Audit gold overlap/support and reserve a separately approved final protocol. Tune on train/validation only.

EXP-20261003-004 filters the repaired rows without reassigning families. Dropped rows leave the file. Kept rows stay in the split they had under EXP-008. The new file has 26,545 rows: train 21,377, validation 2,588, test 2,580. Sha256 `1231835ecc5dcb608fefa0357d325affb2afbdc3aeade2a63b1dc42b2920897b`. It is the input to the running EXP-005 train, not a replacement for the EXP-008 file.

Approved whole-source preparation is now measured in EXP-20261003-001/002/003.
V3 retains 601/206/204 complete blog bodies under the original EXP-008 ownership;
13 cross-split-family documents and 39 gold-overlap-family documents are quarantined.
Independent verification found zero exact/Jaccard >=0.8 crossings across 288,434
candidate cross-split pairs. This extends the original audit with full-source checks;
it does not prove semantic leakage absence. See [preparation](COMPLETE_SOURCE_SFT.md).
