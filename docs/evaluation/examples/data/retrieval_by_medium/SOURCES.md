# Provenance of the retrieval-by-medium check

- `topics.json`: the 30 gold topics exactly as the pipeline queried them (`retrieval_query` in the Colab run `EXP-COLAB-relabel-rag-seed43`, `raw/relabel.jsonl`), with the top 4 passage ids that run saved, used to check that this search reproduces it. One topic string contains the replacement character because that is how it is stored in the data.
- `results.json`: output of `retrieval_by_medium.py` (SHA-256 `01c5ec68bfaa2dfaf379b6a62c5c34b458619886b72bea3fdae48eac8777436d`).
- The index: `relabel-aws-handoff-core-20261005.zip` (SHA-256 `bda5bc9a5734f3b4bb84d7a518343601f3f94abc67a63fa99582598b5bbf7a0a`), `REQUIRED/chroma_lemkin_train_only`, collection `lemkin_train_only`, 21,193 documents (blog 9,255; X 11,185; LinkedIn 713; youtube_jason 40). All 36 files in the zip matched its `SHA256SUMS`. `chroma.sqlite3` SHA-256 `a87c12d31eda36cc81f831bb6b5114d3c545e5e98e1fb20836a7ca718b1e7bb9` equals the value Kevin's `docs/deployment/RAG_V2_PROMOTION.md` records for the untouched v1 index.
- Software: chromadb 1.5.5, Ollama 0.34.2 on Windows (Colab used 0.40.11), embedding model `nomic-embed-text`, `/api/embed` on the CPU, k = 4, cosine space.
- Kevin's own counts for the same index and topics: `experiments/EXP-20261004-009-dense-rag-v1/metrics/corpus.json` and `retrieval_diagnostics.json` on branch `kevin/298b-integration` (platforms, and top-4 slots blog 69, X 39, LinkedIn 12, YouTube 0 for the unfiltered search).
- The filter is the one defined by his helper `medium_where` in `host_finetune/train_only_index.py`, which the pipeline does not use.

## Rebuild

```
python docs/evaluation/examples/data/retrieval_by_medium/retrieval_by_medium.py --chroma-path <handoff>/REQUIRED/chroma_lemkin_train_only
```
