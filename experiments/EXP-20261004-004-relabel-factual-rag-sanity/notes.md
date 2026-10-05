# Relabel factual-retrieval sanity

Development run only. Eight drafts: the first two gold topics, four mediums, relabel adapter, factual retrieval on. Scoring was skipped. This does not replace EXP-20261004-003 (retrieval off) and it is not the full 120-draft retrieval comparison. `--fixed-style` was off.

The directory number 004 is also used by `EXP-20261004-004-repaired-vs-relabel-retrieval`, which is a separate full retrieval run. This folder is the 8-draft provenance check.

## Index probe

`lemkin_train_only` at `host_finetune/output/chroma_lemkin_train_only` was read and not rebuilt.

- Count: 21,194. Space: cosine.
- Every stored row has `split=train`.
- Mediums: blog 9,255, linkedin 713, x 11,186, youtube_jason 40.
- Eight queries (gold_001 and gold_002 × blog, linkedin, x, talk), k=4. No empty result, no duplicate ids, no wrong medium, no non-train hit.

Quality flags, not a broken index:

- gold_001 on X returns `train_26148` at cosine distance 0.0087. The stored text is the same title as the gold topic plus a short link. That is a near copy of the query inside the train index.
- gold_001 blog and LinkedIn hits are only loosely about enterprise selling. Distances are about 0.33–0.39.
- Talk hits are transcript fragments. Distances are about 0.42–0.47. Several are weakly related to the topic.
- gold_002 blog hits are Monday.com posts and are on topic (distances about 0.22–0.24).

## Generation

Exit 0. Eight rows in `raw/relabel.jsonl`. Each row has cosine distances, the chat-templated `rendered_prompt`, experiment id, adapter path, base model `unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit`, collection `lemkin_train_only`, embedding model `nomic-embed-text`, top-k 4, git SHA, and timestamp. The factual-excerpt label is in every rendered prompt. No style-example block was added.

Word counts: gold_001 blog 244, LinkedIn 245, X 17, talk 278; gold_002 blog 44, LinkedIn 76, X 57, talk 256. Six of eight stopped early on EOS. The X draft for gold_001 is 17 words. The blog draft for gold_002 is 44 words. Those previews are the reason to look at the raw file before a larger retrieval-on run.

`metrics/scoring_skipped.json` records that `unified_eval` was not called.
