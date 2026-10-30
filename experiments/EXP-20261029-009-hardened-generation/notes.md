# Hardened four-arm generation

Question: after the evidence-path fixes, do full-corpus Chroma v2, matched pilot dense retrieval, LightRAG naive, and LightRAG mix differ on the same three topics and four media?

All 12 topic-medium cells were generated in every arm. Paired comparisons below use that common set. Persona-style means use only eligible rows whose medium is a style selector. Those eligible sets are not the same, so the style distances are not a ranking.

## Settings

Adapter `host_finetune/output/lemkin_lora_relabel`. Seed 42. Evidence budget 4, at most one chunk per source. LightRAG `num_ctx` 16384. Retrieval was frozen before the adapter loaded. The graph was not rewritten and no new documents were indexed.

Arms:

| Arm | Frozen retrieval | Wall seconds |
|---|---|---|
| chroma_v2 | full train-only Chroma v2, 21,193 documents | 281.469 |
| pilot_dense | dense search over the 160-document pilot | 206.309 |
| naive | LightRAG naive on the finished pilot graph | 215.934 |
| mix | LightRAG mix on the same graph | 234.598 |

Wall time includes adapter load, claim extraction, and one retry when a draft echoed the question, ran short, stopped mid-sentence, or stated an unsupported number. Naive and mix wrote identical answers on the Brex and VP topics (8 of 12 drafts). Mix changed only the speaker topic.

## Unified gates

The copying gate is the training-corpus 12-word span check. It passed on every stored draft. A separate retrieval-source copy check flagged the naive and mix VP X drafts at 16 contiguous words. Numeric grounding is diagnostic and does not change eligibility. Lexical grounding passes at overlap 0.2 against retrieved passages.

| Arm | Eligible | Completion | Format | Grounding | Copying gate |
|---|---|---|---|---|---|
| chroma_v2 | 2/12 (0.1667) | 0.9167 | 0.9167 | 0.1667 | 1.0 |
| pilot_dense | 4/12 (0.3333) | 0.75 | 0.8333 | 0.5 | 1.0 |
| naive | 1/12 (0.0833) | 0.8333 | 0.9167 | 0.25 | 1.0 |
| mix | 1/12 (0.0833) | 0.8333 | 0.9167 | 0.3333 | 1.0 |

Eligible style distance: Chroma 0.9778 on LinkedIn only; pilot dense 0.9775 on blog and LinkedIn; naive and mix have no selector-eligible medium, so their style means are null. Do not compare those distances.

Stance consistency was null. Framing distance is diagnostic and was averaged beyond the selector-eligible rows, so it is not reported here as a perspective score.

## Writing quality

Judge `qwen2.5:14b`, temperature 0. Mean of the writing dimensions over all 12 drafts:

| Arm | Mean, all 12 | Mean, eligible only |
|---|---|---|
| chroma_v2 | 3.2778 | 3.25 (n=2) |
| pilot_dense | 2.9444 | 2.8334 (n=4) |
| naive | 2.8194 | 3.3333 (n=1) |
| mix | 2.8333 | 2.6667 (n=1) |

The all-12 means are paired. The eligible-only means are not.

## What the fixes changed, and what remained

Chroma Brex evidence JSON failed twice. The run continued with `evidence_status=extraction_failed` instead of dropping the topic. That status means the extractor failed. It does not mean the retrieved pages were irrelevant. The generator still wrote unsupported amounts, including `$5 billion`, `$1 million`, and `6,000` on LinkedIn. That LinkedIn draft passed lexical grounding at 0.5. The evidence block itself was not filled with invented claims.

Question echoes remain on X in every arm, and on additional blog drafts in the pilot, naive, and mix arms. The X length band starts at 3 words, so a 13-word echo can pass format even though the prompt asks for at least 20 words. Several blogs remained under the 40-word floor after the retry.

Unsupported numbers remain common on the Brex answers. Naive and mix blogs both state `$60 billion`. The supported `$5.15 billion` figure appears in some pilot, naive, and mix answers and in none of the Chroma answers, because Chroma extraction failed.

Anneka Gupta is named in the pilot-dense blog, the naive blog and talk, and the mix X post and talk. Chroma does not name her. The titles attached to her are not reliable: the drafts call her CEO of Gorgias, Chief Product Officer at Owner.com, and CEO of Asana. The mix X draft also pasted the speaker instruction into the post. After this run, the prompt was changed so the evidence block keeps only `Speaker:` and tells the model not to copy that label. That wording is TESTED and was not used for these measured drafts.

## Workbook and presentation

LightRAG did not improve voice, eligibility, or numeric reliability on this fixed set. The graph-enhanced mix arm matched naive on two of the three topics. The useful change is that a broken evidence parse no longer deletes the topic, and failed or omission-marker chunks are dropped at query time without editing the stored graph. Limitations: three topics, one seed, a 152-document graph, lexical grounding that can pass when extraction failed, numeric checks that do not block, and style distances that cannot be ranked across unequal eligible subsets. No precision or recall was computed. QLoRA was not retrained.
