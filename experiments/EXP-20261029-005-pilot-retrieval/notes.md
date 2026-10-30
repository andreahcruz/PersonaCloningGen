# Pilot retrieval comparison

Question: on the same eight review questions, what does each retriever return, and do the graph modes differ from vector search?

Held fixed: the eight questions in `host_finetune/pilot_questions.json`, evidence budget 4, one chunk per LightRAG source, the finished pilot index, and frozen Chroma v2 (`lemkin_train_only_v2_shadow`, 21,193 documents). Review source ids were not sent to any retriever. No precision or recall is claimed. A designated-source hit only means a preselected id appeared in the top hits.

The process exited 0 in 177.1 seconds. Per-query times below do not include the one-time embedding of the 160 pilot texts.

## Designated-source checks

Seven questions have a preselected source. The Olympic question expects an empty list.

| Question | dense pilot | Chroma v2 | naive | local | global | hybrid | mix |
|---|---|---|---|---|---|---|---|
| Brex price | yes | yes | yes | yes | yes | yes | yes |
| Corporate VC second check | yes | yes | yes | yes | yes | yes | yes |
| Worst mis-hire | yes | no | yes | no | no | no | yes |
| First VP of Sales | all 3 | none of 3 | 2 of 3 | 2 of 3 | 2 of 3 | 2 of 3 | 2 of 3 |
| 5 percent share | yes | yes | yes | no | yes | yes | yes |
| VP Sales tension | both | X post only | both | X post only | both | both | both |
| Anneka Gupta speaker | yes | yes | yes | yes | yes | yes | yes |
| Olympic ceremony | 4 hits | 4 hits | 4 hits | 4 hits | 4 hits | 4 hits | 4 hits |

Mean recorded seconds: dense 2.07, Chroma 2.73, naive 2.85, local 2.96, global 2.95, hybrid 3.55, mix 3.05. Every hit list that returned text was marked train-attributed. One Chroma result, the corporate-VC query, carried `duplicate_source: true`. Two of those hits are near-copies of the same "second and third checks" passage under different train-row ids.

## What the differences mean

Matched dense search and LightRAG `naive` both read source text. They found the Brex price, the corporate-VC post, the 5 percent post, both sides of the VP of Sales tension, and the Anneka Gupta quotation.

`local` missed the 5 percent post and kept only one side of the tension. `global` and `hybrid` found that post and both tension sources. `mix` tracked `naive` more closely than `local`, including the mis-hire post.

The mis-hire LinkedIn post is an extraction failure. Its doc status is `failed`, and its id is not on a graph node. It still has 2 stored chunks. Matched dense searches the pilot file directly, so it can return that post without a graph. `naive` and `mix` returned those leftover chunks. `local`, `global`, and `hybrid` did not. A failed extraction is therefore not the same as absence from the chunk store.

Every Anneka hit is labeled `third_party_quotation` with speaker Anneka Gupta. LightRAG's stored chunk includes the attribution note. The dense and Chroma previews start at the quoted sentence because those arms embed the source text; the attribution field is still set.

No mode abstained on the Olympic ceremony question. Each returned four train-split passages. That is a shared failure to return an empty list, not a precision score.

Chroma's full 21,193-row index did not place the three preselected first-VP posts in its top four. The matched 160-document dense search did. Those Chroma rows were not graded, so this is a corpus-coverage observation, not a claim that Chroma retrieved worse evidence.

## Generation choice

`naive` is the same-corpus vector control. `hybrid` is the graph mode that recovered the conditional post and both tension sources, which `local` did not. Generation uses those two LightRAG modes plus full-corpus Chroma. The pilot is not being expanded: the review questions already separate the modes, and another indexing pass would spend the remaining time on timeouts instead of generation.
