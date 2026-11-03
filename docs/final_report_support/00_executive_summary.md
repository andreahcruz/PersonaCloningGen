# Executive summary

AI-Driven Persona Cloning for B2B Content Generation (SJSU DATA 298B) generates Jason Lemkin / SaaStr text in four media: blog, LinkedIn, X, and talk. The measured comparison uses one 30-topic gold file crossed with those four media, 120 drafts per architecture, schema `unified_eval.v1`, judge `qwen2.5:14b` at temperature 0. Gold SHA-256 is `d7c4f997899e28f28eb3ea9aa3417857da6f0666acbf72fd8ac31422034e520e` (`data/openai_ft/lemkin_gold_eval.jsonl`).

This checkout does not assign letters A–D to the saved architectures. The six-arm registry calls dense Relabel plus RAG **Model E**. LightRAG plus the same Relabel generator is **Model F**. The scored comparison names are Base, PersonaRAG, Relabel RAG off, Relabel + RAG, and Model F. See `docs/evaluation/UNIFIED_120_COMPARISON.md`.

## What was built

The generator that the later experiments share is a rank-16 QLoRA adapter, `lemkin_lora_relabel`, on `unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit`. Training is registered as `EXP-20261003-005` (best eval loss 1.3316). The adapter exists because truncated training rows had been labeled as finished posts (`DEC-006`). Retrieval, when used, does not paste raw passages into the final prompt. An evidence step with the adapter off keeps claims whose spans occur in a retrieved source.

Two retrieval systems were measured on the 120-draft panel:

- Dense Chroma, collection `lemkin_train_only`, 21,193 train documents, `nomic-embed-text`, top 4. This is Relabel + RAG.
- LightRAG 1.5.7 hybrid search on a frozen partial graph. Official profile **F-REL** adds relationship lines only when the endpoints and predicate words occur in a retrieved passage. The snapshot has 27,797 nodes and 47,897 edges from 2,713 completed documents out of 13,140 eligible (`results/model-f-handoff-v1/provenance.json`).

An 8-question frozen pilot selected **F-SOURCE**, not F-REL (`EXP-20261102-002-model-f-quick-pilot`). A 26-question development comparison later recommended F-REL (`EXP-20261102-003-model-f-desc-vs-rel`). The 120-prompt panel did not choose the arm.

## What the 120-draft panel shows

All-draft means below are the `medium=all` rows in `results/model-f-handoff-v1/architecture_comparison.json`. This reporting pass recomputed them from the saved row files and matched every published overall figure (`tables/verification_notes.txt`).

| Architecture | Eligible | Completion | Grounding | Writing quality (n=120) | Style distance (n=90) | Task adherence |
|---|---:|---:|---:|---:|---:|---:|
| Base | 45 | 0.375 | not scored | 3.578 | 1.043 | 0.743 |
| PersonaRAG | 57 | 0.592 | 0.833 | 3.599 | 0.906 | 0.725 |
| Relabel RAG off | 77 | 0.658 | not scored | 3.000 | 0.750 | 0.701 |
| Relabel + RAG | 72 | 0.700 | 0.883 | 2.896 | 0.805 | 0.709 |
| Model F (F-REL) | 67 | 0.700 | 0.817 | 3.165 | 0.739 | 0.695 |

No architecture leads every column. Relabel RAG off has the most eligible drafts. Relabel + RAG has the highest grounding pass rate among retrieval systems. PersonaRAG has the highest all-draft writing quality. Model F has the lowest style distance; a topic-cluster interval against Relabel RAG off includes zero, so that style edge is not a clear separation (`tables/paired_model_f_differences.csv`). Model F persona utility is null.

Model F eligible writing quality is **3.2662** on 67 drafts (`aggregate_summary.json`). The all-draft mean is **3.1653**. Those are different denominators.

The deterministic audit is a different measurement. On Model F, 52 of 120 drafts contain a quantity the checker could not find in the topic, accepted claims, or grounding lines. Of those 52, 43 still passed the judge grounding gate (`tables/model_f_judge_vs_number_crosstab.csv`). Judge grounding is not factual reliability.

## Current system boundary

Model F generation is a command-line experiment runner. `backend/main.py` in this checkout calls the older `generate.py` path. The React demo is on `andreah/deploy-with-react` and is not in this tree. The handoff says that service has no LightRAG route (`docs/teammate_handoff.md`). The graph snapshot and Relabel adapter are not in Git.

`docs/evaluation/UNIFIED_120_COMPARISON.md` still says the Model F panel was not measured. Later files, `docs/model_f_results.md` and `results/model-f-handoff-v1/`, record the measured panel. Use the handoff JSON for scores.

## What this package adds

`docs/final_report_support/` holds the retrospective, comparison, graph notes, rubric matrix, outlines, and figures regenerated from the saved rows. It does not rescore the official experiment.
