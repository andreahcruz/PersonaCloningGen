# Provenance of the before/after RAG data (PUK-34)

These are compact copies of saved draft files, made with `python -m eval_tools.rag_effect slim`. A slim file keeps only the fields
`eval_tools.rag_effect` reads (token counts, stop reasons, evidence seconds, retrieval hits and checks). No draft was generated and no
model or judge was run. The full draft files are about 4 MB each for the Relabel runs and 0.4 MB for the base run and 1.8 MB for PersonaRAG, which is why only the slim copies are in the repository.

| Slim file | Made from (full file) | SHA-256 of the full file | SHA-256 of this slim file |
|---|---|---|---|
| `relabel_rag_off_seed42.slim.jsonl` | Kevin's branch `kevin/298b-integration`: `experiments/EXP-20261004-003-repaired-vs-relabel/raw/relabel.jsonl` | `5d7ebfb57e93150288602e6523f83de300cd79a600c3ee119a48ca1a97a62f67` | `8cf361c9a9878e39f96a9ebd0520156a5353a45cc27570b894d357e6988b74e5` |
| `relabel_rag_off_seed43.slim.jsonl` | same branch: `experiments/EXP-20261004-005-repaired-vs-relabel-seed43/raw/relabel.jsonl` | `8c5da8462c9a14733057d07a657ef6fca66cba79f8b8674daba618baa5b56e99` | `802d0687878c2fc587a85b1af338e037bab86166f0e8ec169edd1f4a49c86313` |
| `relabel_rag_off_seed44.slim.jsonl` | same branch: `experiments/EXP-20261004-006-repaired-vs-relabel-seed44/raw/relabel.jsonl` | `50ed43f3d9f01e4091685292de0eb0632ee2ed990b13f7f5bf56cac5b03107f8` | `0e22018ef27d048cdc9878c6028265558ac9ec00b932404eddad114e5ee0c40e` |
| `relabel_rag_on_seed42.slim.jsonl` | same branch: `experiments/EXP-20261005-003-relabel-rag-final/raw/relabel.jsonl` | `1dca8e3afd4ca3cdb21225afc58bb111a007cac7e58db2ebaca8d3da7086f2d7` | `f398193a7bcfc6358859818c8a16435c18b06211426c198db8d2099262461f15` |
| `relabel_rag_on_seed43.slim.jsonl` | Colab run `EXP-COLAB-relabel-rag-seed43`, `raw/relabel.jsonl` | `32abc777fa8a96c6275a761e6895dc2e9c1eaf5e13200ef22fd7f6fe0f9c6ac5` | `0b72fd0dcee1d20c865a076f46e9ccaf807381cc658d2460c93aecc6e17f70f1` |
| `relabel_rag_on_seed44.slim.jsonl` | Colab run `EXP-COLAB-relabel-rag-seed44`, `raw/relabel.jsonl` | `12981eb4075116e713eb0887296fff02bdde6a81a3ba20cca969c0bba3ab942e` | `2565a1924c9d23cb7966029aaaf2f6c5057af706429ef0b4050ec106f24f04ef` |
| `base_llama31.slim.jsonl` | Kevin's branch: `experiments/EXP-20261001-010-repaired-and-baselines/raw/llama3.1.jsonl` | `fa460a01e5bd92694b6fccbefe2d00d4d669fa60ecb2f1c3a1536c6dd75c57c9` | `83f09fc5ec972da2539bba2d815a1355e6b3bf30e2c9d946cd22256896fa1928` |
| `personarag_seed42.slim.jsonl` | Colab run `EXP-COLAB-personarag-newdata-seed42`, `raw/llama3.1.jsonl` | `381fa68c5d04a7576eeb887e869d2687260bb99f0474102802514a232afe013f` | `7afe5652d730b7b02fd67934fdc1a32add9c29d35deee9979b162c88c171ef38` |

The Kevin-branch files come from commit `0fac41485ec3655ea1439d6d2484115f243fdab1`. The Colab full files are on the team Google Drive (`298 B/puk28_results/`);
their hashes also appear as `generation_sha256` in each run's summary (see `../architecture/SOURCES.md`). The hashes of the two seed 42 Kevin files
(RAG off and RAG on) match the ones already recorded for PUK-28 in `../SOURCES.md`.

## Rebuild

```
python docs/evaluation/examples/data/rag_effect/rebuild_reports.py
```
