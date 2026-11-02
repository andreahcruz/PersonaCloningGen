# Relation recovery dry-run

The indexer had exited and `indexing.lock` was absent. The dry-run read `storage_expand_v1` and wrote this directory only. The cache file was unchanged after the run, and a second run reproduced the ledger hash.

Policy `relation-normalization-v1` does not replace fingerprint `771baa30dd8f79c30c4c51f296385b2ec2c652b33ff1031a099a7e16f139260b`. Insertion into a clone is not authorized.
