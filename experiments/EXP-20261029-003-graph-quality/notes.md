# Graph-ready v2 and extraction probe

Question: do the known quote, weak-entity, and stacked-section problems need a new derivative before a 160-document index?

v1 was not rewritten. v2 is `artifacts/lightrag/model_f/graph_ready_v2/`. Its document hash is `d7bd01a01f288dda3582eddcfd2fc1539574f97b6e168b1124a0b43b76d9a118`.

## Stacked blog

`jasonlemkin_blog.jsonl` line 1462 is the Rubrik post that starts at section 2 in the derived text. The five eligible chunks are adjacent: every interior gap is 0 words. The cleaned page has 174 words before those chunks. That lead is the author's introduction plus an embedded post from Bipul Sinha. v2 does not copy it. The section stack is the article's own numbered list, not a join of unrelated posts.

Across all 948 ordered joins, 355 have at least one interior gap. The largest is 1,038 words, on blog line 345. v2 inserts a marker that states the omitted word count and does not copy the omitted text. 312 joins also drop a lead. Those leads stay out for the same reason as the Rubrik lead: the residual can mix author prose with someone else's embedded post.

## Quotations

The Anneka Gupta LinkedIn post is now `third_party_quotation` with speaker Anneka Gupta. The original sentences remain in the text, under an attribution note. Nine documents in the full derivative change attribution for this reason. 137 documents have at least one detected third-party quotation. A heading such as "Day Plan" or "Deep Dives" is rejected. Names such as Harry Stebbings and Tomasz Tunguz remain. A few odd labels remain outside the pilot, including "Cat Rik" and "Descript Andrew". The 160-document sample contains two quoted documents: Alex Rosenblatt, kept as a publication page, and Anneka Gupta, marked as a third-party quotation.

## Extraction

The same three documents were indexed twice with Llama 3.1 8B at temperature 0. The library prompt produced 30 nodes and 11 names flagged by the weak-entity rules, including `Tuesday` and `December`. The project prompt produced 31 nodes and 10 flagged names. `Tuesday` remained. Salesforce Ventures and GV remained. The library description said Anneka Gupta is associated with Webflow as a CPO. The project description says she is CPO of Rubrik and mentions Webflow. That is the better sentence. It is not a large change in entity count.

Qwen 2.5 7B, at a 4,096 context, did not follow the LightRAG output format. The graph contains `<source_entity>` and `<target_entity>`. It is not a usable extractor. The pilot stays on Llama 3.1 with the project guidance. One 2,168-word blog, `gd_1d361440c168ddd12ff26855`, timed out twice on chunk 5 of 11, first at a 32,768 context and then at 16,384. It is listed in `extraction_skip.json` and is not in the pilot graph. The other documents continue. The first 26 processed documents used a 32,768 context. The resume uses 16,384 because the larger context held the GPU for more than eight minutes on one chunk.

## Separate test fix

`write_parameters` now fills `lemkin_train_only` when retrieval is on, the collection argument is empty, and the Chroma path is the v1 directory. An explicit collection still wins, including `lemkin_train_only_v2_shadow`. `tests/test_evidence_capsule.py` passes. `rag_active.json` was not edited.
