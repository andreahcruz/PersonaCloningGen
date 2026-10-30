# Provenance of the example data

These files are copies of results saved by Kevin Gao on branch `kevin/298b-integration`
(commit `0fac41485ec3655ea1439d6d2484115f243fdab1`), so the example in this folder can be rerun without that branch.
Nothing was edited. The copies only keep the rows the example needs. No draft was generated and no judge was run for them.

| File here | Where it comes from | Rows | SHA-256 of this file |
|---|---|---|---|
| `relabel_rag_off_rows.jsonl` | `experiments/EXP-20261004-008-writing-evidence/metrics/rows.jsonl`, rows with source experiment `EXP-20261004-003-repaired-vs-relabel` and model `relabel` (the file holds 1800 rows for 15 runs) | 120 | `9b7c8dd28e58eea685572cda082303bd0975b9827e6765ff0b502e15fdc7ab63` |
| `relabel_rag_on_rows.jsonl` | `experiments/EXP-20261005-003-relabel-rag-final/metrics/rows.jsonl` (all rows) | 120 | `32b14e90f4912db77b90554f32d746c5f79a622897841e916c39f0487e35e91f` |
| `grammar_rag_off.jsonl` | `python -m eval_tools.grammar` on `experiments/EXP-20261004-003-repaired-vs-relabel/raw/relabel.jsonl` | 120 | `a11b742eb7829501698cbf247903f2fcf9290cb7213b381634835df52225e58f` |
| `grammar_rag_on.jsonl` | `python -m eval_tools.grammar` on `experiments/EXP-20261005-003-relabel-rag-final/raw/relabel.jsonl` | 120 | `d7114a741ac9fba486e87addae2f532dce156b11ff49e65fb8f8c48baf09efa8` |

SHA-256 of the full source files on Kevin's branch, for anyone who wants to check the copies:

- `experiments/EXP-20261004-008-writing-evidence/metrics/rows.jsonl`: `44d2e976026910dae8e97b53d6ca4135734aef7bbfdc669376e0273586941cd8`
- `experiments/EXP-20261005-003-relabel-rag-final/metrics/rows.jsonl`: `32b14e90f4912db77b90554f32d746c5f79a622897841e916c39f0487e35e91f`
- `experiments/EXP-20261004-003-repaired-vs-relabel/raw/relabel.jsonl` (drafts used for the grammar check): `5d7ebfb57e93150288602e6523f83de300cd79a600c3ee119a48ca1a97a62f67`
- `experiments/EXP-20261005-003-relabel-rag-final/raw/relabel.jsonl` (drafts used for the grammar check): `1dca8e3afd4ca3cdb21225afc58bb111a007cac7e58db2ebaca8d3da7086f2d7`

The drafts were checked to be the same ones the judge scored: for both runs, all 120 drafts reproduce a writing quality entry in Kevin's judge cache (the cache key is a hash of the judge model and the writing prompt, which is built from the first 1800 characters of the draft).

The grammar files depend on the real Lemkin posts in `data/*.jsonl` (400 sampled per medium, seed 0) and on the rules in `eval_tools/grammar.py`, so they change if the rules change.

## Rerun

```
python -m eval_tools.compare \
  --a docs/evaluation/examples/data/relabel_rag_off_rows.jsonl --extra-a docs/evaluation/examples/data/grammar_rag_off.jsonl --label-a "RAG off" \
  --b docs/evaluation/examples/data/relabel_rag_on_rows.jsonl  --extra-b docs/evaluation/examples/data/grammar_rag_on.jsonl  --label-b "RAG on" \
  --out-md report.md
```
