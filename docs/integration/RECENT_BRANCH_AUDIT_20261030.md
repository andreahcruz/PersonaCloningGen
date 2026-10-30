# Recent branch audit — 2026-10-30

Audit of teammate work on `andreahcruz/298P` (the same GitHub repository as `andreahcruz/PersonaCloningGen`, repository id `1172180026`). This review is read-only. No merge, cherry-pick, rebase, reset, commit, or push was performed. Frozen experiment files and `rag_active.json` were not modified.

Evidence labels used below:

- **VERIFIED** — checked in this audit against source, git objects, or a test run here.
- **REPORTED** — present in a teammate document or committed artifact, not recomputed here.
- **INFERRED** — follows from the code, but not directly measured again.
- **NOT VERIFIED** — claimed somewhere, and not checked.

## A. Executive summary

Over the last 30 days the remote repository gained two evaluation branches from Vrushabh (`vb-puk-28`, `vb-puk-34`), a closed frontend attempt (`vb-puk-5`), a one-file design history (`vb`), a documentation page already merged to the default branch (`vb-contributions`), and Andrea's EC2/React demo line (`andreah/deploy-with-react`). Kevin's RAG v2 and integration commits are already the current branch. The default branch is `QLoRAFT`, not `master`. `master` has no merge base with the current history.

The work that should be integrated before the deadline is the evaluation tooling on `vb-puk-28` and `vb-puk-34`. Those branches share eight commits and then diverge. `vb-puk-34` does not contain the LanguageTool or fixed-length voice commit. Merging either branch into `kevin/lightrag-model-f` is textually clean. Merging both requires resolving three documentation files.

The scientifically important result is the fixed-length blog voice score. On the first 100 words, Relabel's mean distance sits next to held-out Lemkin blog prefixes, and well below base Llama. The older finding that turning RAG on moved Relabel farther from Lemkin on blogs does not survive that length control. That is a qualification of the old style-distance reading. It does not replace the factual Model E / Model F experiments, and it does not show that retrieval has no effect on voice in every setup. Grammar should stay a diagnostic. A medium filter on factual retrieval is a weak idea on this index. Medium-matched style examples already exist as `--fixed-style` and have not been measured.

Local LightRAG work is uncommitted on `kevin/lightrag-model-f` and is not on any remote branch. It must be committed, or otherwise preserved, before a merge. This audit stopped before that commit and before any merge.

## Repository state at audit time

| Item | Value | Label |
|---|---|---|
| Active branch | `kevin/lightrag-model-f` | VERIFIED |
| HEAD | `0fac41485ec3655ea1439d6d2484115f243fdab1` | VERIFIED |
| Upstream | none | VERIFIED |
| Same commit already on | `origin/kevin/298b-integration` | VERIFIED |
| Default branch | `origin/QLoRAFT` at `60502845b9` (2026-10-07) | VERIFIED |
| `origin/master` | `d2dd81a`, no merge base with HEAD | VERIFIED |
| Fetch | `git fetch --all --prune` added `origin/vb-puk-28` and `origin/vb-puk-34` only. HEAD and the worktree were unchanged. | VERIFIED |
| Remotes | `origin` = `https://github.com/andreahcruz/298P.git` | VERIFIED |
| Local `QLoRAFT` | `0990c93`, 4 commits behind `origin/QLoRAFT` | VERIFIED |
| Unpushed commits on this branch | none relative to `origin/kevin/298b-integration` | VERIFIED |
| Ongoing generation jobs | none. Historical terminals for EXP-005 through EXP-011 have finished. `ollama` pid 25248 has been running since 2026-10-15. | VERIFIED |

Tracked dirty files, not committed:

- `docs/DECISION_LOG.md`
- `docs/PLAN.md`
- `docs/architecture/EXPERIMENT_MATRIX.md`
- `experiments/README.md`
- `host_finetune/compare_adapters.py` (+310 / −39 in the combined dirty diff)
- `host_finetune/evidence_capsule.py`
- `host_finetune/numeric_grounding.py`
- `tests/test_evidence_capsule.py`

Untracked and not on the remote: `deployment_handoff/`, `docs/architecture/LIGHTRAG_MODEL_F.md`, LightRAG modules under `host_finetune/`, their tests, and local experiments `EXP-20261027-001` through `EXP-20261030-011`. `EXP-20261030-010` is the factual-reliability run. `EXP-20261030-011` is a later local span-repair run and is also untracked. Ignored artifacts (virtualenv, indexes, model files) were left in place.

`git merge-tree` of HEAD with `origin/vb-puk-28`, `origin/vb-puk-34`, and `origin/QLoRAFT` reports no content conflicts. Those merges still must wait until the uncommitted Model F work has a commit to sit on. A dirty merge would be textually non-overlapping with the PUK file set, and it would still leave Model F out of the merge parent.

## B. Complete recent branch inventory

Window: committer date on or after 2026-09-30, plus open pull requests whose tips are one day older. "Ahead / behind" is versus `kevin/lightrag-model-f` at `0fac414` unless noted. No pull request exists for `vb-puk-28` or `vb-puk-34`. GitHub compare URLs are branch compares, not PRs. PR metadata came from the GitHub API because `gh` is not installed.

| Branch | Last update | Contributor | Purpose | PR status | Unique changes | Recommendation |
|---|---|---|---|---|---|---|
| `origin/vb-puk-34` | 2026-10-29 `9744482` | Vrushabh | RAG cost metrics and a medium-filtered retrieval check | No PR | 11 commits ahead of `QLoRAFT`, 0 behind. 3 commits not in `vb-puk-28` | Merge after `vb-puk-28`; resolve 3 doc conflicts |
| `origin/vb-puk-28` | 2026-10-29 `eb79011` | Vrushabh | Shared eval tools, LanguageTool grammar, fixed-length voice | No PR | 9 commits ahead of `QLoRAFT`, 0 behind. 1 commit not in `vb-puk-34` (`eb79011`) | Merge first |
| `origin/vb` | 2026-10-07 `3a55dc8` | Vrushabh | PUK-11 design history, one markdown file | PR #8 closed, not merged | 1 file, `docs/MODEL_IMPROVEMENTS_AND_DESIGN_CHANGES.md` | Defer. The doc says the repo has no evaluation results, which is no longer true |
| `origin/vb-puk-5` | 2026-10-07 `8d8909e` | Vrushabh | Next.js Lemkin Studio frontend | PR #7 closed, not merged | `web/` plus `docs/FRONTEND_API_CONTRACT.md` | Do not merge. Later demo work is on `andreah/deploy-with-react` |
| `origin/QLoRAFT` | 2026-10-07 `6050284` | Vrushabh Bodarya (merge) | Default branch: FastAPI service plus a contributions page | PR #1 and PR #6 merged | 4 commits not in HEAD: `backend/*`, `docs/VRUSHABH_CONTRIBUTIONS.md` | Do not merge by itself. Those commits are ancestors of both PUK branches |
| `origin/vb-contributions` | 2026-10-07 `3b6eb08` | Vrushabh | Contributions page | PR #6 merged | Already contained in `origin/QLoRAFT` | No separate action |
| `origin/kevin/298b-integration` | 2026-10-06 `0fac414` | Kevin Gao | Current integration commit | No open PR | Identical to HEAD | Already integrated. Local uncommitted work is not on this remote |
| `origin/andreah/deploy-with-react` | 2026-10-06 `0e69135` | andreahcruz | EC2 demo, React UI, direct QLoRA relay | No PR in the API list | 34 commits ahead of the merge base with HEAD; 4 content conflicts | Defer until after the eval merge |
| `origin/kevin/direct-qlora` | 2026-10-05 `39f58a1` | Kevin Gao | One-line Relabel prompt in the demo | No PR | Patch id matches `3e1cee4` on the deploy branch | No action. Same patch, different SHA |
| `origin/kevin/rag-fixes` | 2026-10-05 `d09cfbe` | Kevin Gao | Airflow/Spark RAG v2 selector | No PR | Ancestor of HEAD | Already integrated |
| `origin/andreah/deploy-test` | 2026-10-03 `635e07f` | andreahcruz | Single-EC2 pipeline deploy | No PR | Ancestor of `kevin/direct-qlora` and of the deploy line | No separate merge |
| `origin/pukhraj/integration-docs` | 2026-09-29 `fec0f4d` | Pukhraj Rathkanthiwar | Vanilla `frontend/` plus project docs | PR #5 open. Tip is outside the 30-day window | Stacked on the unmerged frontend branches | Do not merge |
| `origin/pukhraj/frontend-runtime` | 2026-09-29 `6953c58` | Pukhraj Rathkanthiwar | Frontend runtime tests | PR #4 open. Outside the window | Part of the same stack | Do not merge |
| `origin/pukhraj/frontend-app` | 2026-09-29 `a1d0160` | Pukhraj Rathkanthiwar | Browser UI | PR #3 open. Outside the window | Part of the same stack | Do not merge |
| `kevin/lightrag-model-f` (local only) | Working tree after `0fac414` | Kevin Gao | Model F LightRAG, EXP-001 through EXP-011 | Not pushed | Uncommitted modules, tests, and experiment directories | Keep. Do not overwrite. Commit before merging anything else |

`origin/pukhraj/backend-api` and `origin/pukhraj/backend-tests` were merged on 2026-09-29 (PR #1 and PR #2). Their FastAPI service is the `backend/` tree already on `origin/QLoRAFT`.

## C. Detailed branch reviews

### `vb-puk-28` — grammar and fixed-length voice

Tip `eb79011` (2026-10-29). Author Vrushabh. Base is `origin/QLoRAFT`. Nine commits, all dated 2026-10-24 through 2026-10-29. The first eight, through `d012078`, are also on `vb-puk-34`. The last commit is only on this branch.

What it does:

- Adds `eval_tools/`: paired bootstrap comparison, seed summaries, a small rule-based grammar checker, length control, and row loading.
- `eb79011` adds `eval_tools/grammar_lt.py`, `eval_tools/voice_fixed.py`, their tests, and the result writeups.
- Scores saved drafts. It does not retrain, reindex, or change `host_finetune` gates.

Key files: `eval_tools/grammar_lt.py`, `eval_tools/voice_fixed.py`, `eval_tools/length_control.py`, `eval_tools/grammar.py`, `eval_tools/compare.py`, `tests/test_grammar_lt.py`, `tests/test_voice_fixed.py`, `tests/test_eval_tools.py`, `docs/evaluation/GRAMMAR_LANGUAGETOOL.md`, `docs/evaluation/VOICE_FIXED_LENGTH.md`, `docs/evaluation/ARCHITECTURE_COMPARISON.md`.

Tests newly executed here, in a detached worktree at `eb79011`, using the existing project virtualenv and no network:

| Command | Result |
|---|---|
| `pytest tests/test_grammar_lt.py` | 26 collected |
| `pytest tests/test_voice_fixed.py` | 13 collected |
| `pytest tests/test_eval_tools.py` | 76 collected |
| all three together | **115 passed in 0.51s** |

The teammate doc's "26" and "13" match the per-file collection counts. Those tests use a fake LanguageTool and a fake distance. They do not prove the live grammar numbers.

#### LanguageTool grammar

Implementation status: **IMPLEMENTED** and **TESTED** (unit tests). The 19/40 and flag-label figures are **MEASURED** in committed artifacts, not re-run here.

`eval_tools.grammar` is a local regex checker. `coverage_audit.json` records **0 of 40** on a synthetic set and **4 of 4** on four rule-targeted sentences. **VERIFIED** by reading that artifact.

`eval_tools.grammar_lt` keeps LanguageTool categories `GRAMMAR`, `TYPOS`, and `CONFUSED_WORDS`, plus `EN_A_VS_AN` and `ENGLISH_WORD_REPEAT_RULE`. It drops seven rules that fired on informal Lemkin style, drops matches touching `U+FFFD`, and accepts a spelling hit only for an all-lowercase word that appears fewer than three times in the reference vocabulary. **VERIFIED** in source and in the unit tests.

Detection on the 40 synthetic sentences, from `coverage_audit.json`:

| Checker | Caught | Label |
|---|---|---|
| Rule-based `eval_tools.grammar` | 0 / 40 | VERIFIED in the artifact |
| LanguageTool default | 20 / 40 | VERIFIED in the artifact |
| LanguageTool with this profile | **19 / 40** | VERIFIED in the artifact |

The profile loses the one comma splice the default checker caught (0/4 versus 1/4). Subject-verb agreement stays 1/4. The writeup's category table matches the profile block. **VERIFIED**. This audit did not re-execute LanguageTool. `language_tool_python` is not installed. Java 17.0.16 is installed. The code calls `language_tool_python.LanguageTool("en-US")`, which is the local server class, not `LanguageToolPublicAPI`. **VERIFIED** from the call site. Draft text is not sent to the public API by this call. **INFERRED** from that class choice; no packet capture was done. The first run still downloads LanguageTool (about 260 MB, **REPORTED**). The version is not pinned in code. `SOURCES.md` says LanguageTool 6.8. **REPORTED**.

Flag audit on generated drafts, from `grammar_lt_flag_labels.csv` (103 rows):

| Label | Count | Share |
|---|---|---|
| false alarm | 67 | 65% |
| colloquial or prescriptive | 19 | 18% |
| real mistake | 14 | 14% |
| not grammar (truncation or markup) | 3 | 3% |

Per-run counts in the writeup match the CSV. **VERIFIED** against the CSV. The labels are one reader's labels. The writeup says they were made by Claude and are not an independent annotation. **REPORTED**, and the file itself does not record a second rater. Clear mistakes are 1 to 4 per 120-draft run. Paired rate intervals in the writeup all include zero. **REPORTED**. Grammar does not separate these models.

Other limits:

- The reference rate and the spelling vocabulary come from up to 400 raw `data/*.jsonl` posts per medium, seed 0, with no train/validation filter (`grammar.reference_texts`). That is a baseline of Lemkin's raw writing, not a held-out split. **VERIFIED** in source.
- Talk has no reference. **VERIFIED**.
- Precision on real flags is about 14% if only "real mistake" counts. That is too low for an eligibility gate.
- The tool does not modify `host_finetune.unified_eval`. Wiring it in later should add `grammar_lt_rate` as a diagnostic column only.

Recommendation: merge the module with `vb-puk-28`. Do not fail drafts on grammar flags. Do not rank adapters by flag counts.

#### Fixed-length voice

Implementation status: **IMPLEMENTED**, **TESTED** (unit tests), and **MEASURED** on saved drafts. The blog means for drafts stored in this repository were recomputed in this audit.

How text is selected. `truncate_words` keeps the first N whitespace-separated tokens and rejoins them with single spaces. Newlines are collapsed. The profile is still fit on full held-out posts. Every truncated text then has one paragraph, so `paragraphs_per_100w` is a shared offset and does not rank models. Type-token ratio and sentence length still move because a prefix is not a full post. **VERIFIED** in `length_control.py` and `voice_distance.py`.

Length matching. Blog and LinkedIn use N = 100 by default, and the published comparison also uses 150. A draft shorter than N is `too_short` and is dropped. `compare` pairs only drafts scored on both sides. X and talk are `not_supported`. **VERIFIED**.

Holdout. `load_kevin_distance` calls `held_out_reference_texts` (validation, `sft_role=unchanged`, gold-overlap groups blocked), then `partition_references`. This audit recomputed the blog split: 80 validation posts, first 40 fit the profile, last 40 are the control. The control mean on the first 100 words is **1.1096**, with 38 of 40 posts long enough. That matches `relabel_rag_off_seed42.summary.json`. **VERIFIED**. The 40/40 cut follows dataset order, not a random shuffle. **INFERRED** limitation: the two halves can differ systematically if file order is structured. Train rows are not in the profile. **VERIFIED** from the split filter. Talk references: **0**. **VERIFIED**.

Leakage. Voice references are validation text. Grammar references are raw files and can include training posts. Those are different choices. Near-duplicate posts inside the 80 blog validation texts were **NOT VERIFIED** in this audit. Gold topics and the real-post controls are not the same topics. The score is surface stylometry (sentence length, punctuation, type-token ratio, function-word rates). `voice_distance.py` already says closer surface counts are not authorship. Fixed length does not change that. **VERIFIED**.

Recomputed blog means at N = 100, from git blobs (CRLF in the working tree changes the file hash; the git bytes match the SHA-256 values in `SOURCES.md`):

| Run | This audit | Their summary | Drafts scored |
|---|---|---|---|
| Real control prefixes | 1.1096 | 1.1096 | 38 of 40 |
| Relabel RAG off, seed 42 | 1.0843 | 1.0843 | 30 |
| Relabel RAG off, seed 43 | 1.0486 | 1.0486 | 30 |
| Relabel RAG off, seed 44 | 1.1273 | 1.1273 | 29 |
| Relabel RAG on, seed 42 | 1.1312 | 1.1312 | 29 |
| Base Llama 3.1 | 2.0276 | 2.0276 | 30 |

PersonaRAG blog **1.7086** and Relabel RAG-on seeds 43 and 44 are in the committed summary files. Those drafts are Colab outputs and are not in this repository, so those two means were **NOT VERIFIED** by recomputation. The writeup's rounded table (real 1.11, Relabel about 1.05–1.13, PersonaRAG 1.71, base 2.03) matches the summary files. **VERIFIED** for in-repo runs. **REPORTED** for Colab runs.

LinkedIn at N = 100, from the same summary files, does not separate models the way blog does: real control 0.792 (24 of 40 long enough), base 0.710, PersonaRAG 0.680, Relabel RAG-off seed 42 0.668. **REPORTED** from the summary JSON, except the real-post loader path which is the same code. Do not describe LinkedIn as "Relabel matches the author and the others do not."

Four claims, kept separate:

1. **Relabel resembles real author writing.** Supported only as a blog-prefix mean comparison: Relabel means 1.05–1.13 sit next to the control mean 1.11. There is no interval on the real-post mean. It is not an authorship test. It is not a LinkedIn or X result. X cannot use this N because Relabel's median X draft is 16 words (**VERIFIED** in the seed-42 summary).
2. **Relabel outperforms the other models on this voice score.** Supported on blogs versus base (recomputed gap about 0.9) and, from their paired intervals, versus PersonaRAG (about 0.6, interval excluding zero at N = 100 and N = 150, three seeds). **REPORTED** for the PersonaRAG pairing. One PersonaRAG seed is reused against all three Relabel seeds.
3. **Retrieval has no independent voice effect.** Not established. Relabel RAG-on versus RAG-off at N = 100 is about +0.05, +0.06, and −0.07, and the reported intervals include zero or touch it. Base to PersonaRAG is about −0.32 with an interval that excludes zero, on one seed. Retrieval can move the base model and still show no clear extra change once the generator is Relabel.
4. **Length confounded the older blog RAG penalty.** Supported for that specific comparison. Full-length blog gap about +0.10 with an interval excluding zero; equal-prefix gaps shrink and the intervals include zero (`style_length_check.md` and `VOICE_FIXED_LENGTH.md`). **REPORTED** for the paired intervals. The seed-42 means recomputed here are consistent with the +0.05 description (1.1312 − 1.0843 = 0.047).

The "drafts look more typical than real posts" reading is also a length effect in their length-matched check: probability a RAG-off blog draft is closer than a real post falls from 0.70 at full length to 0.53 when lengths are matched. **REPORTED**.

Sample size is 30 gold topics, three Relabel seeds, one base seed, one PersonaRAG seed. Short drafts are dropped, and RAG-on blogs drop more of them (26–29 of 30 at N = 100). The score ignores everything after word N, so a real difference in endings is invisible. **VERIFIED** as a property of the method.

Recommendation: merge `voice_fixed` into the evaluation framework as the length-controlled blog voice report. Keep the old full-length distances in the frozen experiments. Do not overwrite them. State the blog result, the LinkedIn non-separation, and the unsupported X/talk cases separately.

#### Architecture comparison (shared history, also on `vb-puk-34`)

`docs/evaluation/ARCHITECTURE_COMPARISON.md` compares base Llama, a new Colab "PersonaRAG" (base Llama plus dense train-only retrieval), Relabel RAG-off, and Relabel RAG-on, on 30 topics × 4 mediums. It reports a large blog voice advantage for Relabel over PersonaRAG that survives equal length, a large writing-quality disadvantage for Relabel, and shorter Relabel drafts. Multiple-testing notes are in the page. PersonaRAG and base have one seed. Seeds 43 and 44 retrieved the same passages as Kevin's seed 42 on 29 of 30 topics, and the extracted evidence differed on 15 of 30 topics. **REPORTED**. That means the three seeds are not pure generation-seed replicates.

This is documentation of saved and Colab runs, plus the `eval_tools` harness. It is worth keeping. It is not a new Model E versus Model F result. The index they describe is the v1 train-only collection (21,193 documents), not the local LightRAG pilot.

### `vb-puk-34` — retrieval cost and medium filter

Tip `9744482` (2026-10-29). Contains `d012078` and then three commits that `vb-puk-28` does not have: `bfa0aa4`, `8208f34`, `9744482`. It does not contain `grammar_lt.py` or `voice_fixed.py`.

Tests newly executed in a detached worktree at `9744482`:

| Command | Result |
|---|---|
| `pytest tests/test_rag_effect.py` | 20 collected |
| `pytest tests/test_retrieval_by_medium.py` | 6 collected |
| `pytest tests/test_eval_tools.py` | 76 collected |
| all three together | **102 passed in 0.47s** |

#### Strategy A versus strategy B

The retrieval script implements **strategy A**: the same topic embedding, Chroma `where={"medium": ...}`, k = 4, on collection `lemkin_train_only`. Talk maps to `youtube_jason`. **VERIFIED** in `retrieval_by_medium.py`. It does not retrieve facts from the full corpus and then add style examples.

**Strategy B is already implemented** on this branch's ancestor and in the current worktree:

- `build_runtime_retriever` embeds the topic and queries with `where=None`. The comment says the output medium selects the generation instruction only. **VERIFIED** in `host_finetune/compare_adapters.py`.
- `--fixed-style` loads `host_finetune/exemplars/fixed_style_candidates.json` only when the flag is set. `format_voice_exemplars` labels those posts as voice, not evidence. Factual excerpts stay a separate block. **VERIFIED**.
- The candidate file status is `review_candidate`, `loaded_by_default: false`. `empty_platforms` is `youtube_jason` only. Blog, LinkedIn, and X have train-owned unchanged exemplars with gold-overlap families excluded. **VERIFIED** by reading the file.

Nothing in `vb-puk-34` adds a new style-RAG subsystem. The unmeasured piece is a generation comparison with `--fixed-style` on, factual retrieval left unfiltered, and the copying gate left on. That run should be a new `experiments/EXP-*` directory. It should not be implemented as part of this audit.

#### Medium-filter measurements (strategy A)

Committed `results.json` on the branch:

| Filter | Pool | Mean distance, unfiltered → filtered | Worst topic | Words, unfiltered → filtered | Share under 20 words | IDs kept from the unfiltered top 4 |
|---|---|---|---|---|---|---|
| Blog | 9,255 | 0.243 → 0.255 | +0.046 | 123 → 169 | 14% → 2% | 58% |
| LinkedIn | 713 | 0.243 → 0.305 | +0.165 | 123 → 169 | 14% → 1% | 10% |
| X | 11,185 | 0.243 → 0.270 | +0.087 | 123 → 44 | 14% → 35% | 32% |
| Talk | 40 | 0.243 → 0.429 | +0.334 | 123 → 264 | 14% → 0% | 0% |

**VERIFIED** by parsing `results.json`. The writeup rounds these figures and does not overclaim. Reproduction check inside the artifact: unfiltered top 4 matched the Colab run on **28 of 30** topics. **VERIFIED** as a field in the artifact. The search was **NOT VERIFIED** by re-querying Chroma. Opening the frozen index with a persistent client can write to it, so this audit only hashed `chroma.sqlite3` in the local handoff. The hash is `a87c12d31eda36cc81f831bb6b5114d3c545e5e98e1fb20836a7ca718b1e7bb9`, which matches `SOURCES.md` and the v1 hash recorded for the untouched index. **VERIFIED**.

Distance is cosine distance, not a relevance judgment. The script and the writeup say so. There is no precision or recall. Talk's 40 chunks and LinkedIn's 713 documents make a medium filter a coverage problem. Blog is the only medium where the filter is nearly a no-op. This supports leaving factual retrieval unfiltered. It does not measure whether `--fixed-style` would change voice.

The script targets v1 `lemkin_train_only`. Model E generation in the local experiments uses Chroma v2 (`lemkin_train_only_v2_shadow`, 21,193 documents in the run logs). Do not treat the v1 filter table as a v2 or LightRAG result.

#### `eval_tools.rag_effect`

Reads fields already stored on drafts: prompt tokens, generated tokens, words, stop reason, `early_eos`, evidence seconds counted once per topic, `numeric_grounding.pass`, `rag_source_copying.flagged`, claim counts, hit mediums, and hit distances. **VERIFIED** in source. Those field names match the trace writer in `compare_adapters.py`, including LightRAG hits that carry `medium`. **VERIFIED** for the schema. It was not pointed at EXP-009 or EXP-010 traces in this audit.

What the writeup claims, all **REPORTED** from slim copies:

- Prompt tokens rise from about 70 to about 360. Generated tokens and draft length do not show a clear paired change.
- Evidence extraction is about 28 seconds per topic on the local GPU run and about 33 seconds on the Colab seeds. Generation wall-clock time is not in the drafts.
- About 25% of retrieved passages match the requested medium (talk 0%).
- About 47–55% of RAG drafts have an unsupported number. About 80–82% of proposed claims are counted as verified. The page itself says these two checks disagree and that there is no RAG-off baseline for the number check.
- Copy flags are 2–3% of drafts.

The tool does not compute lexical overlap and does not call it entailment. "Claims that survive verification" is the saved verifier count, not a new factual gold set. Passage distance is not precision or recall. Infrastructure cost in dollars is not calculated. Failed requests are not a separate analyzed class beyond missing fields, which the paired code skips.

Use it as an additional reporter beside `unified_eval`, not as a replacement. Point it at saved JSONL, including Model E and Model F traces that use this schema. Do not rescore frozen metrics in place.

### Other branches

#### `andreah/deploy-with-react` (`0e69135`, 2026-10-06)

Demo stack: FastAPI relay, Next.js `web/`, Caddy, GPU and AWS compose files, direct Relabel QLoRA prompt alignment, password gate, and EC2 pipeline sample tooling. It contains a recommitted copy of the PUK-5 frontend (`620729f`) and of the FastAPI service (`3a5cad0`), plus later demo fixes. `git merge-tree` against HEAD conflicts in `.dockerignore`, `.gitignore`, `README.md`, and `docker-compose.yml`. **VERIFIED**.

This is the branch to use if the course demo is integrated. It is not required to interpret Model E, Model F, or voice. Merge it after the evaluation branches, in its own commit, with those four conflicts resolved by hand. Do not also merge `vb-puk-5`.

#### `kevin/direct-qlora` (`39f58a1`)

`git patch-id --stable` of `39f58a1` equals `3e1cee4` on `andreah/deploy-with-react` (`a0d78432e00e17ea3987408378a33c2ac7779666`). **VERIFIED**. The histories diverged, so Git does not see one as an ancestor of the other. Merging both would replay one patch. No separate merge.

#### `vb` / PUK-11 (`3a55dc8`)

One new file, `docs/MODEL_IMPROVEMENTS_AND_DESIGN_CHANGES.md`. PR #8 was closed unmerged on 2026-10-07. The opening says the repo has the evaluation harness and no evaluation results. That was already stale relative to the October experiments, and it does not know about LightRAG. `merge-tree` with HEAD is clean. Defer. If it is kept later, add a header that it is a 2026-10-07 historical note.

#### `vb-puk-5` (`8d8909e`)

Next.js app under `web/`. PR #7 closed unmerged. `andreah/deploy-with-react` has that frontend plus a generate route, Docker, and later page changes (about 448 insertions and 148 deletions under `web/` and `backend/` relative to this tip). Merging this branch and the deploy branch would duplicate the UI. Do not merge.

#### `origin/QLoRAFT` and `vb-contributions`

`origin/QLoRAFT` is 4 commits ahead of the merge base with HEAD (`0990c93`):

- `7b2caeb` FastAPI generation service (`backend/Dockerfile`, `backend/main.py`, `backend/requirements.txt`)
- `efde946` merge of PR #1
- `3b6eb08` `docs/VRUSHABH_CONTRIBUTIONS.md`
- `6050284` merge of PR #6

`merge-tree` with HEAD is clean. Both PUK branches already contain these commits. A PUK merge brings them in. Do not merge `QLoRAFT` as an extra step. The contributions page is a status note, not a result. The FastAPI service is a small generation API and is not the host QLoRA evaluation path. It was not executed in this audit.

#### `kevin/rag-fixes` and `kevin/298b-integration`

`d09cfbe` is an ancestor of HEAD. `0fac414` is HEAD. No action, except that the uncommitted LightRAG work still needs a commit of its own before it exists on the branch history.

#### `andreah/deploy-test`

Ancestor of the direct-QLoRA and deploy-with-react line. No separate merge.

#### Open Pukhraj pull requests (tips 2026-09-29, outside the commit window)

PR #3, #4, and #5 are still open and stacked: `frontend-app` into `backend-tests`, `frontend-runtime` into `frontend-app`, `integration-docs` into `frontend-runtime`. They add `frontend/` (static HTML/JS), not `web/`. PR #1 already put the FastAPI service on `QLoRAFT`. Merging this stack would add a second UI and reopen an older backend line. Leave the PRs open or close them as superseded by `andreah/deploy-with-react`. Do not merge them into `kevin/lightrag-model-f`.

## D. Merge readiness matrix

| Contribution | Engineering quality | Scientific validity | Conflict risk | Merge strategy | Priority |
|---|---|---|---|---|---|
| `eval_tools` compare, bootstrap, seeds, length control | Strong. 76 unit tests passed here | Appropriate paired intervals. Multiple-test notes are documented | Low versus HEAD. Shared with both PUK tips | Merge via `vb-puk-28` | High |
| Fixed-length voice | Strong. 13 tests passed. Blog means recomputed exactly for in-repo drafts | Good length control for blog prefixes. Not authorship. LinkedIn does not separate models. X and talk unsupported | Low versus HEAD | Merge via `vb-puk-28`. Use as a report, not a rewrite of old metrics | Highest |
| LanguageTool grammar | Good filter and tests. Live tool not re-run. Version unpinned. Extra Java dependency | Detection 19/40. Draft precision about 14% clear mistakes. Does not rank models | Low versus HEAD | Merge the module. Diagnostic only. Not a gate | High for the code, low as a selector |
| Architecture comparison writeup | Reproducible from saved rows and the notebook path they document | Careful, with seed and hardware limits. Evidence text differs on 15/30 topics between machines | Low | Merge as documentation with `vb-puk-28` | Medium |
| `rag_effect` | Strong. 20 tests passed. Matches current trace field names | Valid cost and behavior report. Not entailment, not precision/recall, no generation latency | Low versus HEAD. Doc conflict with `vb-puk-28` | Merge via `vb-puk-34` after `vb-puk-28` | Medium |
| Medium-filter retrieval check | Script is clear. Tests cover the summary math only. Live search not re-run | Honest negative result for strategy A on the v1 index. Distance is not graded relevance | Low | Merge the doc and script. Do not change the retriever | Medium |
| `--fixed-style` generation run | Already implemented. Not part of either PUK branch | Not measured | None until a new experiment | Do not build a new style-RAG path. Schedule one registered run later | Later |
| FastAPI `backend/` and contributions page | Small, not executed here | Not a model result | Clean versus HEAD | Arrives with the PUK merge. Accept it. Do not treat it as the evaluator | Low |
| `andreah/deploy-with-react` | Substantial demo code with later fixes | Not a scientific result | Four content conflicts | Defer until after the eval merge | Medium for the demo only |
| `vb-puk-5`, `kevin/direct-qlora`, `andreah/deploy-test`, `vb-contributions`, `kevin/rag-fixes`, `kevin/298b-integration` | — | — | Duplicate or already contained | No separate merge | None |
| `vb` PUK-11 markdown | One historical doc, now stale | Describes an older repo | Clean, but misleading if unlabeled | Defer | Low |
| Open Pukhraj frontend PRs | Older static UI | — | Second frontend | Do not merge | None |

## E. Recommended merge order

Preconditions:

1. Stay on `kevin/lightrag-model-f`.
2. Commit the local LightRAG and experiment work first, in a commit the user explicitly requests. That commit is not part of this audit. Until it exists, a merge parent will not contain Model F.
3. Do not `stash`, `reset`, `clean`, or check out another branch in this dirty worktree.
4. Confirm `rag_active.json` and the `experiments/EXP-20261027-*` through `EXP-20261030-*` directories are untouched by the merge. The PUK diffs do not modify them. **VERIFIED** by the added-file list from `origin/QLoRAFT...origin/vb-puk-28` and `...vb-puk-34`.

Order:

1. Merge `origin/vb-puk-28` with `--no-ff`. Expected additions: `eval_tools/` except `rag_effect.py`, grammar and voice docs and artifacts, `tests/test_grammar_lt.py`, `tests/test_voice_fixed.py`, `tests/test_eval_tools.py`, plus `backend/` and `docs/VRUSHABH_CONTRIBUTIONS.md` from the `QLoRAFT` ancestors. `merge-tree` reports no conflicts with `0fac414`.
2. Run `python -m pytest tests/test_grammar_lt.py tests/test_voice_fixed.py tests/test_eval_tools.py`. Expect the same 115 passes if the tree matches `eb79011` for those files.
3. Merge `origin/vb-puk-34` with `--no-ff`. Git will conflict in:
   - `docs/evaluation/README.md`
   - `docs/evaluation/figures/make_figures.py`
   - `eval_tools/README.md`
   Keep both sides: the grammar and fixed-voice sections from PUK-28, and the RAG-cost and medium-filter sections from PUK-34. `ARCHITECTURE_COMPARISON.md` is not in the conflict list.
4. Run the union: the PUK-28 three files plus `tests/test_rag_effect.py` and `tests/test_retrieval_by_medium.py`. Then run the existing CPU tests that do not load the LLM (`tests/test_evidence_capsule.py` and the other non-GPU host tests already used on this branch).
5. Stop. Do not merge the demo branch in the same commit.

Later, only if the demo is in scope:

6. Merge `origin/andreah/deploy-with-react` and resolve `.dockerignore`, `.gitignore`, `README.md`, and `docker-compose.yml`. Do not merge `vb-puk-5` or `kevin/direct-qlora`.

Rollback:

- If the merge has not been committed, `git merge --abort`.
- If a merge commit was created and not pushed, `git revert -m 1 <merge-commit>` rather than `git reset --hard`. A hard reset would also threaten any uncommitted Model F files still in the worktree.
- Do not revert or edit files inside an existing `experiments/EXP-*` directory as part of rollback.

Blockers:

- Uncommitted Model F work has no commit yet.
- `language_tool_python` is absent, so a live grammar smoke test is a separate install and a 260 MB download. It is not required to merge the unit-tested module.
- The three documentation conflicts are small and manual. They are not code conflicts.

## F. Updated project understanding

### Model E (Chroma dense)

No teammate branch replaces Model E or the v2 index. PUK-34's medium filter is a v1 counterfactual and argues against filtering facts by output medium. `rag_effect` can describe token cost and saved grounding flags on Model E traces. It does not change the EXP-005 or EXP-009 retrieval comparison.

### Model F (LightRAG)

No remote branch in this window implements or evaluates LightRAG. Local experiments remain the only Model F evidence. The teammate voice and grammar results use dense retrieval or no retrieval. They do not show that graph retrieval improved factuality or voice. The current conclusion stands: LightRAG has not demonstrated superior generation quality on the small paired evaluation.

### Relabel QLoRA

Still the frozen generator. The fixed-length blog score is the best current evidence that its surface style, on blog openings of 100 words, lands in the same range as held-out Lemkin prefixes and clearly closer than base Llama. That is style distance, not identity, and it was measured on the October 4–5 draft files plus Colab repeats. It is not a reason to retrain.

### Voice evaluation

Old observation, kept as a historical measurement: on the full-length style distance, Relabel RAG-on blogs looked about 0.10 farther from the profile than RAG-off, and intervals excluded zero before length control.

New interpretation, for the report and the presentation: output length explains that blog gap. After scoring the first 100 or 150 words, the RAG-on versus RAG-off difference is not clear. Relabel's blog-prefix distance is about the same as real held-out prefixes. Base and PersonaRAG remain farther on blogs. LinkedIn does not show the same separation. X and talk are out of range for this control. EXP-009's note that style distances were not ranked across arms, because eligible medium sets differed, still stands. EXP-010's writing-quality drop under the stricter factual prompt is a different metric and is unchanged.

### Factual evaluation

Unchanged. Lexical overlap is not entailment. Unsupported-number rates in `rag_effect` are a diagnostic the teammate writeup already refuses to treat as a verdict, because they disagree with the claim judge. Stronger factual prompts in EXP-010 remain the local evidence that unsupported numbers fell and writing quality fell with them. Do not replace those files.

### Grammar evaluation

Add it as an optional diagnostic after Java and `language_tool_python` are available. Do not add it to the eligibility gates. The measured outcome is that ordinary grammar is not where these systems differ, and that raw flag counts are mostly false alarms.

### Retrieval design

Keep unfiltered factual retrieval from the train-only corpus. Strategy A (medium-filtered facts) is a poor default on this index, especially for talk (40 chunks) and LinkedIn (713 documents), and it shortens X passages. Strategy B (`--fixed-style` exemplars plus unfiltered facts) is the implemented option and has no generation measurement yet. Talk has no style exemplar in the candidate file.

### Final demonstration

Relabel remains the generator to show. Dense retrieval remains the factual context path that has actually been paired with it. LightRAG stays an implemented retrieval option with a negative or inconclusive generation comparison, not a winning system. Voice slides should show full-length and fixed-length blog numbers side by side and say which conclusion each one supports. Grammar should be one limitation slide, not a leaderboard.

## G. Work remaining

Ranked for the time left. Academic importance first.

1. **Write the interpretation update, without editing frozen metrics.** State the length confound, the blog-only fixed-length result, and the non-results for LinkedIn, X, talk, and Model F. Effort: low. Risk: low. This is the change the voice result actually justifies.
2. **Commit the local Model F work, then merge PUK-28 and PUK-34** in the order in section E. Effort: low once the commit exists. Risk: low if the three doc conflicts are unioned and the experiment directories are not touched.
3. **One registered `--fixed-style` generation comparison.** Same frozen Relabel adapter, unfiltered factual retrieval, style exemplars on versus off, copying gate on, fixed-length blog voice as a reported metric. This is the missing test of strategy B. Effort: one GPU run. Risk: medium, because exemplars are training text and the copying gate has to hold. Do not start it inside this audit.
4. **Do not build medium-filtered factual RAG.** The v1 check is enough to deprioritize it. A v2 or LightRAG repeat would be confirmatory only.
5. **Optional live LanguageTool pin.** Install `language_tool_python`, record the jar version, and re-run the 40-sentence audit once. Until that exists, quote `coverage_audit.json` as the teammate measurement. Effort: low. Not a selector.
6. **Demo merge of `andreah/deploy-with-react`.** Useful for a live demo, irrelevant to the voice and factual claims. Do it after the eval merge, with the four file conflicts.
7. **Leave closed.** PUK-11 as a current status doc, the open vanilla-frontend PR stack, a second grammar gate, and any rewrite of EXP-009 or EXP-010 numbers.

## H. Final recommendation

Integrate now, after the local Model F commit exists:

- `origin/vb-puk-28` at `eb79011`, including the shared `eval_tools` harness, fixed-length voice, and LanguageTool as a diagnostic.
- `origin/vb-puk-34` at `9744482` immediately after, keeping both sides of the three documentation conflicts.
- The FastAPI files and contributions page that those merges will carry in from `QLoRAFT`.

Needs a human pass during the second merge, not a redesign:

- The three documentation conflicts.
- A one-paragraph note in the workbook that grammar flags are not gates and that fixed-length voice does not replace frozen scores.

Keep experimental:

- Medium-filtered retrieval. Record the negative v1 check. Do not switch the retriever.
- `--fixed-style`. The code path exists. The experiment does not.
- Live LanguageTool until the package version is pinned.

Defer explicitly:

- `andreah/deploy-with-react` until the evaluation merge is done.
- `vb`, `vb-puk-5`, `kevin/direct-qlora`, `andreah/deploy-test`, and the open Pukhraj frontend PRs.
- Any edit to frozen experiment outputs, including EXP-009, EXP-010, and EXP-011, or to `rag_active.json`.

## Commands for the next integration stage

Not executed. Run them only after the uncommitted LightRAG work has been committed on `kevin/lightrag-model-f` and `git status` shows a clean tree.

```powershell
git switch kevin/lightrag-model-f
git rev-parse HEAD
git merge --no-ff origin/vb-puk-28 -m "Merge PUK-28 eval tools, fixed-length voice, and LanguageTool as a diagnostic."
python -m pytest tests/test_grammar_lt.py tests/test_voice_fixed.py tests/test_eval_tools.py
git merge --no-ff origin/vb-puk-34 -m "Merge PUK-34 RAG cost report and the medium-filter retrieval check."
# Resolve by keeping both sides:
#   docs/evaluation/README.md
#   docs/evaluation/figures/make_figures.py
#   eval_tools/README.md
git add docs/evaluation/README.md docs/evaluation/figures/make_figures.py eval_tools/README.md
git commit -m "Merge PUK-34 RAG cost report and the medium-filter retrieval check."
python -m pytest tests/test_grammar_lt.py tests/test_voice_fixed.py tests/test_eval_tools.py tests/test_rag_effect.py tests/test_retrieval_by_medium.py
```

If the second merge is still in progress and should be abandoned: `git merge --abort`.

Do not run `git merge origin/vb-puk-5`, `git merge origin/kevin/direct-qlora`, `git merge origin/QLoRAFT` as an extra step, or `git push`.
