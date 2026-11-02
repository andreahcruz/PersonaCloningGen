# Replay of the October 31 pilot traces

Question: can the offline harness score the saved pilot drafts and retrieval payloads, and which of those cells are comparable to the five Model F profiles?

No generator, embedding, or LightRAG query ran. The source is EXP-20261031-001. Its workdir was the 152-document pilot, not `storage_expand_v1`.

## What was scored

65 saved drafts and 48 frozen retrieval payloads. Ranked precision, recall, MRR, and nDCG are null on every retrieval row because that experiment published no graded qrels. Designated-source overlap matched the saved hit lists on all 48 rows. That overlap is not recall.

Recomputing `assess_factual` matched the saved `factual_pass`, `numeric_pass`, and word count on all 65 drafts. The calculations were not changed.

## What is not a profile comparison

| Saved cell | What it actually was | Five-arm profile |
|---|---|---|
| A | Hybrid passages, contract `factual-reliability-v1` | Not `source` |
| B | Same passages plus the older in-passage relationship block | Not `relationships` |
| C | Matched dense preview, 400 characters | Not active Chroma Model E |
| D | Blog-only token pad | Not `descriptions` or `multihop` |

A against B is the only held-fixed ablation: same question, medium, seed, prompt contract, generator, and passage ids. Twenty pairs. The faithfulness difference across all media is 0.0, with a 95 percent bootstrap interval of -0.2 to 0.2. Blog alone is 0.4 (A minus B), interval 0.0 to 0.8, on five questions. Blog pass counts are A 4/5, B 2/5, C 4/5, D 4/5, the same counts as the October 31 notes. LinkedIn moves the other way, A 2/5 and B 4/5, so the all-media mean cancels the blog drop.

A against C is flagged because the passage ids and the retrieval backend differ. B against the blog pad is a length control, not an architecture pair. The pad does not exist for the other media.

`citation_marker_leak` matches the saved `instruction_leak` counts, including 4 of 5 arm-B LinkedIn drafts. `assess_factual` does not put that leak into `factual_pass`. The replay leaves the two fields separate, as the saved score did.

## Annotation sheet

`annotation_template.json` lists the 12 development questions and the source ids returned by the frozen pilot runs. Every grade is null. The preselected review ids are not grades.

## Reproduce

```text
host_finetune\.venv\Scripts\python.exe -X utf8 -m pytest tests/test_eval_bench.py tests/test_eval_bench_historical.py -q --tb=short
```
