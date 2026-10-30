# Provenance of the fixed-length voice data

Every `<run>.voice_fixed.jsonl` file is the per-draft result of `python -m eval_tools.voice_fixed` on the full draft file of that run, with the
first 100 words of every blog and LinkedIn draft scored (folder `n150/` uses 150 words). Each line has the draft key, its real word count, the
fixed length used, a status (`scored`, `too_short`, `not_supported`) and `style_distance_fixed`. `<run>.summary.json` says how many drafts
were scored or dropped and gives the score of real Lemkin posts on the same first N words. The distance is Kevin's own style distance
(`host_finetune.voice_distance`) with his held-out reference profile, loaded from branch `kevin/298b-integration` at commit
`0fac41485ec3655ea1439d6d2484115f243fdab1`. No model or judge was run.

The draft files that were scored, and their hashes, are the same as in `../grammar_lt/SOURCES.md`.

## Rebuild

```
GIT_LFS_SKIP_SMUDGE=1 git worktree add ../kevin-ws origin/kevin/298b-integration     # skips about 480 MB of adapters
python -m eval_tools.voice_fixed --kevin-root ../kevin-ws --traces <full draft file>   --out-jsonl <run>.voice_fixed.jsonl --out-json <run>.summary.json                  # add --blog-words 150 --linkedin-words 150 for n150/
```
