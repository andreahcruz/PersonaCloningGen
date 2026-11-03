# F-REL 120-cell row export (public package)

This public directory does not contain `generations.jsonl`. The 120 responses are in the private archive `model-f-frel-120-evaluation.zip`, unchanged. Public publication was withheld because source-copy flags, a URL inside a draft, and the lack of a publication grant make the response text unsafe for this public repository.

The official experiment directory was not modified. Scores and flags are copied from saved files. This export does not rescore them.

## Files

| File | Rows | Contents |
|---|---:|---|
| `generations.jsonl` | 120 | Private only. Question, medium, response text, model identifiers, recorded output token count |
| `judge_scores.csv` | 120 | Qwen 2.5 14B `unified_eval.v1` row from `unified_eval/rows.jsonl` |
| `deterministic_flags.csv` | 120 | Pre-judge audit cell from `audits/AUDIT-20261102-001-model-f-frel-prejudge/cells.jsonl` |
| `manifest.json` | — | Identities and SHA-256 hashes |

## Join

The key is `question_id` + `medium`. Question IDs are `gold_001` through `gold_030`. Media are `blog`, `linkedin`, `x`, and `talk`. Each file has one row per pair and no duplicate pairs.

## Field notes

Empty CSV cells are missing values. They are not zeros.

`output_tokens` is `tokens.output` on the generation record, with `tokens.source` recorded there as `caller_tokenizer`.

`writing_quality` is the judge mean. Dimension columns are the stored subscores. `style_distance` is empty when the saved row has no distance, which is the case for talk. `persona_utility` is empty on every Model F row.

`grounding_pass` and the grounding rates are the judge's claim-versus-saved-retrieval-hit score. `unsupported_numeric_count` is the deterministic quantity check. Those two columns do not measure the same thing.

`source_copy_pass` fails when the audit recorded a retrieved-passage overlap of at least 12 words. `training_copy_pass` is the separate training-span gate. A true training-copy pass can sit beside a false source-copy pass.

`mid_sentence_stop` comes from the audit completion object. `at_max_tokens` is the audit flag that output tokens equal the medium cap. The generation record does not store a stop reason.

## Evaluation limits

The judge is blinded `qwen2.5:14b` at temperature 0. It is not the Llama generator. Eligible writing quality and claim-support can look strong while the deterministic number check fails. On this panel the audit recorded 52 cells with an unsupported number. The judge grounding gate is a different test: it scores extracted claims against saved retrieval hits, and a draft can introduce quantities that never become those claims.

Stance is diagnostic. Persona utility was not computed. Style distance is not an authorship test.

The generation manifest still says `generated_not_judged`. The judge file is `unified_eval/rows.jsonl`.
