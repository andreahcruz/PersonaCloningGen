# LightRAG relationship repair

Policy `relation-normalization-v1` is separate from configuration fingerprint `771baa30dd8f79c30c4c51f296385b2ec2c652b33ff1031a099a7e16f139260b`. Documents completed before the next resume were extracted by the original parser. Documents parsed after that resume use the normalizer. The fingerprint still names the model, embedding, chunk size, entity guidance, and gleaning count.

The effective checkpoint is the clean stop on 2026-11-01: `stop_reason=sigint`, last document `gd_6c080774239412140ab932ec`, graph file flushed at 13,401 nodes and 22,027 edges, 12 storages finalized, no `indexing.lock`.

## Future extraction

`GraphIndexSession.open` installs `install_parser_normalizer` on the process-local `lightrag.operate._process_extraction_result`. LightRAG 1.5.7 is required. The wrapper rewrites the response string, then calls the original parser. `close` removes the wrapper.

The cache still stores the model text. The wrapper does not write the cache. A later cache hit is normalized again in memory, so a historical malformed row is repaired on a future parse of that same cached response without rewriting the cache file.

Repairs:

- Six fields `relation | source | target | relationship_keywords | keywords | description` become five fields. The label match is case-insensitive and the description must be a sentence.
- Seven fields with `relationship_keywords` in field 4 and `relationship_description` in field 6 use field 5 as keywords and field 7 as the description.
- Four fields whose last field is a short keyword list, a comma, and a sentence split into keywords and description.

Passed through unchanged, and therefore still rejected by the installed parser when the shape is wrong:

- Five-field rows, including descriptions that contain commas or the characters `<|>`.
- Keyword lists with no sentence.
- Sentence-only four-field rows that have no separate keyword segment.
- Eight-field rows such as the Interest-rates / ZIRP split.
- Self-relations. The parser drops those after normalization leaves them as five fields.

Entity lines are not rewritten, so a comma inside `Smith, John` stays in the name.

Counters on the session summary: `valid_original`, `repaired_schema_label`, `repaired_comma_packed`, `rejected_missing_description`, `rejected_ambiguous`, `rejected_unsupported`. The log records counts, not document text.

## Historical recovery

```powershell
host_finetune\.venv\Scripts\python.exe -X utf8 -m host_finetune.relation_recovery --dry-run --snapshot artifacts\lightrag\model_f\storage_expand_v1 --output experiments\EXP-20261101-007-lightrag-relation-recovery\raw
```

The command reads the snapshot and writes `recovery_ledger.jsonl` plus `recovery_summary.json` in the output directory. The output directory must sit outside the snapshot. A second run with the same inputs rewrites the same ledger bytes.

A candidate is an insert candidate only when the source-verified window rule passes: both endpoints in one window, and a multi-word keyword, a verbatim description clause, or at least three description words in that window. Both names in the chunk, with no such predicate, is `needs_review`. A missing endpoint is rejected. An existing graph edge is recorded and is not an insert. The ledger sets `changes_document_status` to false.

Dry-run at this checkpoint: 2,344 unique repaired identities, 416 insert candidates, 1,537 needs review, 215 endpoint-absent, 176 supported pairs already in the graph. Another 457 review rows and 51 endpoint-absent rows also match an existing endpoint pair. Ledger SHA-256 `3feeb06480fa557534b2227665b87a56abfa9134484ed6e3ec8b6a1957c03697`. The cache file hash was unchanged.

`--apply` refuses `storage_expand_v1`, refuses a snapshot that holds `indexing.lock`, and refuses to run without `--authorize`. With `--authorize` it still stops before any embedding call. Insertion into a clone needs a separate authorization to call `nomic-embed-text`.

## Resume

```powershell
host_finetune\.venv\Scripts\python.exe -X utf8 -m host_finetune.indexing_session --resume --selection-policy blog-only --max-documents 13000 --max-runtime-minutes 480 --timeout-policy adaptive
```

That command was not started here. It keeps the blog-only queue, the completed ledger rows, and the original fingerprint. The session summary records `extraction_normalization_policy`.

## Rollback

Delete the dry-run output directory to discard the ledger. The live store was not modified by the recovery command. If a later authorized apply writes a clone, delete the clone and keep `storage_expand_v1`. Do not copy a repaired clone over the live store unless that promotion is a separate decision.

## Still rejected

Keyword-only rows, sentence-only rows with no keyword segment, ambiguous four-field rows, and extra columns that are not the two schema-label patterns. Those rows remain in the original cache and are not in the recovery ledger.
