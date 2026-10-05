# DATA 298B project state

Deployment status as of 2026-10-05: the candidate is Relabel QLoRA plus Factual Dense RAG v1 (21,193 train-only documents) plus atomic evidence extraction. Start with the [deployment handoff](deployment/DEPLOYMENT_HANDOFF.md) and the [README](../README.md). The audit below is the 2026-10-04 record. It is not the deployment description.

## 2026-10-04 update: relabel adapter saved

EXP-20261003-004 relabeled the repaired 512-token file. EXP-20261003-005 finished and saved `host_finetune/output/lemkin_lora_relabel`. Train loss 1.797. Best eval loss 1.3316 at step 2,674. That loss is not comparable to the repaired adapter’s 1.5311, because the validation prompts changed. Per-medium training eval ran out of memory and was skipped. A separate CPU embed finished collection `lemkin_train_only` from the train split only. It does not replace `lemkin_content`, and it was not an input to the trainer. EXP-20261004-003 scored both adapters with retrieval off. Relabel is the cutoff baseline: early stop 0.550 versus 0.650, mid-sentence 0.342 versus 0.633. That is not a voice promotion. EXP-20261004-004 repeated the pair with `lemkin_train_only` excerpts. Seeds 43 and 44 are EXP-20261004-005 and EXP-20261004-006. No new fine-tune was started. No Ollama export. Follow [the relabeled FT protocol](evaluation/RELABEL_FT_NEXT_STEPS.md). Whole-source v3 remains a separate candidate.

Last updated 2026-10-04, America/Los_Angeles, after the train exited 0.

### Training result

The relaunch ran about 3.4 hours and exited 0. The log line `Saving LoRA adapter ->` … `lemkin_lora_relabel` is present. The repaired adapter was not overwritten. Eval loss by step: 1.4016, 1.3645, 1.3421, 1.3410, 1.3342, 1.3317, 1.3316. Generation stop rates are in EXP-20261004-003.

### Train-only index snapshot

`python -m host_finetune.rebuild_chroma --train-only --batch-size 16` finished at 03:55:45 with `train-only index count=21194`. Selection from the EXP-004 file: 21,377 train rows, 5,168 validation/test rows excluded, 183 gold-overlap-family train rows excluded, 0 empty outputs. Embedder was Ollama `nomic-embed-text` with `num_gpu: 0`. A 64-row batch stalled and was stopped; batches of 16 completed. Persist path `host_finetune/output/chroma_lemkin_train_only`, collection `lemkin_train_only`, metadata `group_id`, `medium`, `row_id`, `split`. This build has no `experiments/EXP-*` manifest. The count is the Chroma collection count logged at exit, not a retrieval-quality score.

`compare_adapters.py` can append factual excerpts from that collection only when `--retrieval` is set. The default, including `repaired-vs-relabel`, stays retrieval-off and keeps the existing one-line prompts. Streamlit and `generate.py` still default to collection `lemkin_content` and an Ollama generator. A finished adapter plus a finished index is not a deployed vertical: scoring, GGUF/Ollama registration, and pointing the UI at `lemkin_train_only` are still open.

## Current training run

The early-stop audit is recorded in the conversation and in EXP-20261003-004. Supervised `<|eot_id|>` sat on fragments whose prompt still said “write the whole post.” EXP-004 built a new JSONL beside the repaired file. Raw files and `host_finetune/data/dataset_fit512_keepbreaks_balanced.jsonl` (sha256 `6e99b414733eafdfa27a5e94dd26beff1498bbfd58af132251e24044fef4f057`) were not modified. Targets were copied. The fitter was not rerun.

Of 25,062 train rows, 21,377 remain: 11,616 whole posts unchanged, 1,711 openings, 8,050 continuations, and 3,685 drops (word cuts, other mid-sentence cuts, unfinished last slices, and transcript fragments). Validation is 2,588 and test is 2,580. Surviving rows kept their EXP-008 split. New dataset sha256 `1231835ecc5dcb608fefa0357d325affb2afbdc3aeade2a63b1dc42b2920897b`. Group 3 is 8,430 rather than the summary table’s 8,315 because 115 sentence-regex endings, including ellipses, were relabeled. `tests/test_relabel_continuations.py` passed (8 tests). That is not a full-suite rerun.

EXP-20261003-005 finished under the same Llama 3.1 8B 4-bit setup as the repaired adapter: rank 16, alpha 32, learning rate 1e-4, 2 epochs, batch 1, gradient accumulation 16, sequence length 512, seed 42. A first launch exited before step 1 on a Windows cp1252 print error; the relaunch set `PYTHONUTF8=1` and exited 0. The log is `experiments/EXP-20261003-005-relabel-qlora/raw/train.log`. EXP-20261004-003 later measured stopping. The lower stop rates are a cutoff result, not a voice result.

## Current preparation result

`host_finetune/prepare_complete_sft.py` and its independent verifier are IMPLEMENTED and TESTED (129 passed, 4 skipped). EXP-20261003-003 records 1,011 whole-source blog candidates: 601 train, 206 validation, 204 test. Structural verification independently checked every target and 288,434 cross-split pairs, finding no normalized-exact or token-Jaccard >=0.8 crossings. All 34 inventoried artifacts and preparation inputs remain unchanged. Thirteen cross-split family documents and 39 gold-overlap family documents were quarantined without reassignment.

Content quality is NOT validated: the train/validation sample still contains missing embeds, guest summaries and scrape artifacts. `training_ready` remains false. That preparation run produced no adapter and no model-performance result. The later `lemkin_train_only` build is a different index, described at the top, and is not a whole-source retrieval result. See [complete-source preparation](data/COMPLETE_SOURCE_SFT.md). The audit sections below describe the starting state unless explicitly updated. The worktree also currently reports `data/cleaned.zip` deleted and `data/cleaned (2).zip` untracked; preparation did not operate on either archive.

## Repository and objective

- Branch: `kevin/298b-integration`; verified HEAD: `ea1d58b30cafcb4cf18a03ded39963f64ce6aee7`.
- Initial status: no tracked edits; untracked `data/cleaned (2).zip` and `docs/research/`. This audit adds documentation and is not committed.
- Recent history: `fd16806` records cleaned-corpus training/comparisons; `ea1d58b` adds rank-32 and repaired adapters via LFS. Tag `298b-baseline-pre-rerun` at `6a01d45` is older, not HEAD.
- Local master contains Paul Graham files. Local remote-tracking HEAD points to origin/master; no remote fetch was performed.
- Identity: Jason Lemkin/SaaStr B2B generation, DATA 298B. Objective: reproducible QLoRA-centered generation with evidence-based retrieval improvements. Phase: relabel train is saved and the cutoff comparison is registered; the train-only index has a local collection of 21,194 documents and is not a deployed result. The whole-source quality gate remains open.

## Implemented architecture

1. Root Compose application: `dags/lemkin_pipeline_dag.py` loads five JSONL sources into MinIO; Spark cleans/chunks/embeds; Chroma `lemkin_content` serves retrieval. A parallel raw-source task builds persona statistics. `generate.py`/Streamlit combine topic-based retrieval, persona, format and brief with a selectable Ollama generator. Registered adapter exports can use this path, but recent comparisons did not.
2. Host QLoRA: cleaned-source builder, promo filter, token fitter, grouped split loader, Unsloth/TRL assistant-only training, export and registration. Defaults: `unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit`, 512 tokens, rank 16/alpha 32, dropout 0, one epoch, LR 1e-4, seven attention/MLP targets. Runs have overrides.
3. Separate `persona_pipeline/`: collectors, Spark, DuckDB, profile, persistent Chroma `lemkin_persona`, query rewrite/vocabulary rerank, HF/PEFT training/inference. Executable source exists; it is not the Compose-mounted DAG and was not runtime-tested here. Its name does not establish fidelity to the PersonaRAG paper.
4. `compare_adapters.py` still defaults to one-line medium instructions and no retrieval. EXP-007/010 record that path. A `--retrieval` switch and a `repaired-vs-relabel` adapter preset now exist; retrieval stays off unless requested. The preset has not been executed. The unadapted comparator is still author-prompted, not an unconditioned base baseline.
5. `host_finetune/train_only_index.py` selects EXP-004 train rows, drops gold-overlap families recorded in the EXP-003 dispositions, and can upsert `lemkin_train_only`. The live build is described at the top. `generate.py` does not read that collection unless `CHROMA_COLLECTION_NAME` and the index path are pointed at it.

No ontology implementation was found in inspected Python source. A controlled retrieval comparison, validated voice evaluation, and export of `lemkin_lora_relabel` remain unrun. See [architecture](architecture/ARCHITECTURE.md).

## Verified data and splits

[artifact_hashes.csv](data/artifact_hashes.csv) records current source/cleaned/SFT/split/profile/adapter identities. Hashes do not recover missing cleaning lineage.

`assert_manifest_matches` passed for:

- Default `dataset_from_cleaned_sources_fit512.jsonl` and EXP-20261001-002: 42,877 rows, 34,302 train / 4,287 validation / 4,288 test.
- Repaired `dataset_fit512_keepbreaks_balanced.jsonl` and EXP-20261001-008: 31,328 rows, 25,062 / 3,133 / 3,133. SHA256: `6e99b414733eafdfa27a5e94dd26beff1498bbfd58af132251e24044fef4f057`.

EXP-008's stored report reports zero group, normalized-exact and Jaccard>=0.8 near-duplicate crossings. These diagnostics were inspected, not independently recomputed. They do not establish semantic duplicate absence or train-only profiles/indexes. Splits were not changed. Repaired data is not the trainer default. Source keys identify lines in cleaned files, not stable raw document IDs. Gold tasks are separate from source test assignments.

## Experiments and results

[experiment_inventory.csv](experiment_inventory.csv): 13 EXP directories; five have manifest.yaml, eight do not. Two directories reuse October sequence 004 with different full names.

- EXP-20260927-001 records offline rescoring of copied historical traces plus leakage diagnostics. Its manifest explicitly says fresh generation was not invoked and model outputs were not validated. MEASURED describes that offline operation, not a new model benchmark.
- EXP-20260928-002 and EXP-20261001-001/002/008 contain split records. EXP-008 leaves source/split hashes and environment as n/a; recorded is not complete provenance.
- EXP-20261001-003/004/005/009 contain training logs; EXP-006/007/010 contain comparison traces/metrics. They support recorded work, but lack required top-level manifests: **incomplete registration**, not independently VALIDATED results.
- Rank-32 and repaired adapter configs/safetensors exist and are LFS-tracked. Configs confirm ranks 32/16, alpha 64/32 and the bnb-4bit base path; revision is null. Current weight hashes do not recover training-time environment.
- Cursor's reported losses and gold-score ordering have corresponding notes/metric files, but were not regenerated here. Different validation splits invalidate direct loss comparison. Gold-overlap ordering is not voice ordering.
- Clone v5/v6 identical-weight claim is recorded in EXP-010 notes; live Ollama identity was not checked.
- Current independently validated model-quality results: **none established by this audit**.

## Verification performed

On 2026-10-03, `.venv-test/Scripts/python.exe -m pytest -q` passed: **103 passed, 4 skipped**. Optional PySpark, Streamlit and NLTK gates explain skips; fixtures stub unavailable infrastructure. `docker compose config -q` passed with obsolete version-key warning. These are lightweight/configuration checks, not GPU, service, retrieval or model-quality validation.

Inspected: branch/history/tree, instructions, README, compose/dependencies, relevant source/call sites/tests, splits and run records, adapter configs, notebook, rubrics, historical report/slides, derived guidance and selected primary papers. No local `.cursor/` or `.codex/` repository directory was present. All 126 research PDF/PPTX files were inventoried/hashed/text-extracted; claims review is selective. Extraction warned about some PDF object/font metadata but returned text; no visual check of every paper was attempted. Rubric first pages were rendered and inspected.

## Conflicts, debt, and unknowns

- **Stale documentation:** prior state named eaae794 as HEAD/67 tests; experiment README said no runs. Prior versions remain in Git. README epochs 3 conflicts with config 1; root Chroma 0.4.22 differs from Compose/UI 1.5.5.
- **Watcher input mismatch:** watcher downloads/prepares dataset.jsonl then invokes finetune without overriding input. Trainer defaults to cleaned fit512 with EXP-002. A new sentinel need not train new downloaded data.
- **Incomplete targets:** builder chunks near 1600 characters; fitter can fall back to words. The repaired train file still pairs many of those slices with a full-post instruction. EXP-004 relabels or drops those train rows without restoring text the chunker removed. EXP-20261004-003 compared the adapters. Early stops are less common on the relabel adapter and are still common. Whole long articles remain sections at 512 tokens.
- **Target-derived prompts:** untitled sources use first 120 body characters as topic, creating prompt/target overlap.
- **Evaluation mismatch:** gold overall measures format/lexical overlap. X prompts ask for a post, scorer expects x_thread; one-line prompts omit scored structure requirements. Backends/quantization and missing sampling seeds confound comparisons; empty-context zeros are not RAG metrics.
- **Style scorer validity:** non-split-aware references/anchors; vectorizer fit before classifier CV; synthetic-negative fallback; topic/medium confounds. Best-of-K BERTScore against sampled corpus text is not proof of voice.
- **Leakage boundaries:** `lemkin_content`, persona extraction, and `generate.py` still do not enforce train-only ownership. The in-progress `lemkin_train_only` build does filter to train rows and excludes gold-overlap families; that filter does not apply until a caller selects that collection. Grouped SFT splits alone do not protect the default RAG path or evaluator references. EXP-20261004-002 also flags 183 train rows in gold-overlap families and 13 in full-source cross-split families inside the relabeled file.
- **Attribution/cleaning:** regex cleanup exists; reliable speaker/author/template labels and raw-to-cleaned lineage do not. Dropping SaaStr talks does not establish authorship of remaining transcripts.
- **Historical branch mixture:** report/presentation discuss etl_jasonl, FTv2, QLoRAFT and metricsft. DuckDB belongs to the separate path. Notebook section 3.5 describes retrieval-only preparation despite current SFT implementation. Historical M1-M5 and derived M0-M6 labels conflict.
- **Other existing risks:** Chroma loader splits JSON text with splitlines(); watcher retries the entire chain on registration failure. Documented, not fixed.
- **Unknown runtime:** live index/profile provenance, services, export parity and longer-context GPU capacity. Local rubric deadline is not live LMS verification; exact ontology grading language remains user-reported.

## Active and next task

Active: DEC-009 scores saved drafts with blinded `qwen2.5:14b`. Writing quality and stance both passed their sanity checks, so Persona Utility can be filled on gate-passing blog drafts. RAG grounding stays empty unless that run retrieved passages. Framing distance is diagnostic. EXP-20261004-008 is the rescore; it does not retrain or regenerate. Human voice preference is still not a selection step. See [decisions](DECISION_LOG.md).
