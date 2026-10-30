# 160-document LightRAG pilot index

Question: did the resumed 160-document index finish, and which sample documents are actually in the graph?

Held fixed: graph-ready v2, sample seed 298, project entity guidance, Llama 3.1 8B, `nomic-embed-text`, LightRAG 1.5.7, and the workdir `artifacts/lightrag/model_f/storage_pilot160`. This resume did not delete that directory. The Relabel adapter was not loaded. Chroma was not opened.

The successful command started at 2026-10-29T08:37:09Z and exited 0 at 2026-10-29T11:36:37Z (10,768 seconds). It passed `--num-ctx 16384`. An earlier attempt without `--num-ctx` used the code default of 32,768 and logged 24 inserts before the long blog timed out. A later 16,384 attempt loaded 27 full-document records, deleted the failed long blog, and exited 1 while that blog was still extracting. This resume skipped the 26 documents that were already processed.

## Reconciliation

All 160 sample ids are accounted for. None are pending.

| Status | Documents |
|---|---|
| Processed, including 26 skipped as already done | 152 |
| Failed extraction | 8 |
| Pending | 0 |

Processed medium counts: 58 blog, 30 LinkedIn, 64 X. The sample was 64 blog, 32 LinkedIn, and 64 X, so the failures are 6 blogs and 2 LinkedIn posts. Every catalog row is `split: train`.

The eight failures:

| Id | Medium | What happened |
|---|---|---|
| `gd_1d361440c168ddd12ff26855` | blog, 2,168 words | Listed in `extraction_skip.json` after earlier chunk timeouts. Not inserted. |
| `gd_5ffdc3a083492246413be6f2` | blog, 297 words | Worker timeout, 1,200 seconds, chunk 1 of 2. |
| `gd_8276b7d28a0837fa1e71181a` | LinkedIn, 413 words | Worker timeout, chunk 1 of 2. This is the review source for the worst-mis-hire question. |
| `gd_8a579b23707dd4f77e169619` | blog, 577 words | Worker timeout, chunk 2 of 3. |
| `gd_a4ad0243c9bcfec4c527a703` | LinkedIn, 131 words | Worker timeout, the only chunk. |
| `gd_aa82ba1eab089122de370f68` | blog, 767 words | Worker timeout, chunk 2 of 4. |
| `gd_f76cf95212bfc67508398f63` | blog, 1,116 words | Worker timeout, chunk 2 of 5. |
| `gd_f9a237e6ccf11fe4de9924ad` | blog, 1,045 words | Worker timeout, chunk 2 of 5. |

Timeouts were not limited to the longest blogs. A 131-word LinkedIn post also hit the 1,200-second worker limit. Those eight ids were not restarted after this exit. Another indexing pass would retry the seven timeouts because they are not marked processed and are not in the skip file.

Doc status also contains `dup-44a87e93c933a4e23929caa47da2ff13`, which is not one of the 160 ids. Its error says the file name already exists for the skipped blog. It is a leftover status row.

## Graph

After the process exited, the saved graph had 1,791 nodes and 2,000 edges. NanoVectorDB stored 348 chunk vectors, 1,791 entity vectors, and 2,000 relationship vectors, all 768-dimensional. Every processed document id appears on at least one node `file_path`. None of the eight failed ids do.

The project weak-name rule flags 244 entity names: 17 generic nouns, 5 months, 3 weekdays, and 219 numeric-name hits. The numeric rule matches any digit, so it flags names such as `B2B`. It is not a count of 219 bare numbers. Names that do match the intended rule include `Tuesday`, `Monday`, `December`, `May`, `Partner`, `Company`, `23%`, `$2m in ARR`, and `50% YoY`. `Monday` is typed as an organization, so the weekday rule may be hitting a shortened company name. Entity and relationship descriptions were not treated as citable source spans.

The `SaaStr` node description is a concatenation of conflicting descriptions, including community, conference, and venture-capital wording. That is merged extraction text, not a cleaned ontology.

No second index was started. The retrieval comparison uses this graph as it stood at exit.
