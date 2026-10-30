# Pilot retrieval and a small paired generation

Question: after the 160-document index stopped, do the five LightRAG modes retrieve the review sources, and does a small Relabel generation keep quoted speech attributed to the speaker?

Held fixed: graph-ready v2, the frozen 160-id sample, the finished pilot workdir, the Relabel adapter files, `rag_active.json`, and both Chroma indexes. This run did not start another index. Gold expected facts were not sent to a retriever. Precision and recall were not computed.

The index reconciliation is EXP-20261029-004. The same retrieval sheet, plus a read-only frozen Chroma v2 arm, is also in EXP-20261029-005. Hit ids for pilot dense and the five LightRAG modes match between this directory and EXP-20261029-005.

## Index counts

The catalog contains all 160 sample parents. Every catalog row is train. Doc status after exit:

| State | Count | Meaning |
|---|---|---|
| Processed | 152 | Entity and relationship extraction finished |
| Not extracted | 8 | 7 worker timeouts plus the skipped 2,168-word blog |
| Pending | 0 | None |

The skipped blog is graph id `gd_1d361440c168ddd12ff26855`, parent `gd_de0b05c2882793b64fd76389`. A leftover `dup-` status row points at that id and is not an extra sample document. The seven timeouts are named in EXP-20261029-004. Failed document ids are not on graph node or edge `source_id` values. Their text chunks can still be in the vector store: naive and mix retrieved `gd_8276b7d28a0837fa1e71181a` even though its entity extraction failed. Local, global, and hybrid did not.

Processed is not the same thing as a clean graph. The saved graph has 1,791 nodes and 2,000 edges. The three-document smoke graph is still 29 nodes and 29 edges.

## Extraction profile

This resume ran 10,768 seconds at `--num-ctx 16384`, chunk size 400, overlap 100. The log has 287 successful chunk lines, summing to 2,504 entity mentions and 2,019 relation mentions, with 38 chunks that recorded zero relations. Seven documents each hit one 1,200-second worker timeout. Those seven waits are 8,400 seconds of the wall clock. The log has no per-line timestamps, no input or output token counts, and no retry counter. Documents already marked processed came from earlier attempts, including a 32,768-context run. This log does not store a context size on each document.

The weak-name audit flags 244 names: 17 generic nouns, 5 months, 3 weekdays (`Monday`, `Tuesday`, `Friday`), and 219 numeric-name hits. The numeric rule matches any digit, so it includes names such as `B2B`. It is not a count of 219 bare numbers. No entity was named `<source_entity>`.

## Omission markers

Thirteen extracted pilot documents contain an omission marker. Nineteen stored chunks contain that sentence, and 238 edges cite at least one of those chunks. A chunk that contains the marker includes text from both sides of the gap. That can create a relationship the source never stated. This run did not score how many of those 238 edges are false. The safe next segmentation is to split a document into separate LightRAG documents at each marker, keep the parent id on every piece, and leave the marker out of the chunk that a model extracts. Do not join across the gap inside one 400-token chunk.

## Retrieval

Eight questions, evidence budget 4. Overlap with the preselected review ids is not precision or recall.

Pilot dense, which embeds the 160 source texts and does not open Chroma, placed a designated source first on every supported question and returned all three first-VP-of-Sales sources. Naive, global, hybrid, and mix missed one of those three. Local also missed the 5 percent source and one of the two VP-of-Sales tension sources. Every mode, including dense, returned four hits for the Olympic-ceremony question. None abstained.

The Anneka Gupta post was rank 1 in every mode. LightRAG chunk text includes the attribution note and `quoted_speakers: ["Anneka Gupta"]` with status `third_party_quotation`. The dense preview is the original post text, which names Anneka Gupta in the span, and the same metadata is on the hit. Review ids were not part of the query.

Frozen Chroma v2, recorded in EXP-20261029-005, searches the full train index. Its top 4 often includes `train_*` rows outside the pilot. It is not a matched comparison. The matched dense arm is the 160 source texts.

## Generation

Three topics were generated with the frozen Relabel adapter, seed 42, in blog, LinkedIn, X, and talk: the Brex price, the VP-of-Sales tension, and the Anneka Gupta attribution question. Mix was the graph mode because, on the retrieval sheet, it matched the best designated-source overlap among the graph modes and also returned the mis-hire post through its vector chunks. That choice was made before reading the generated text.

The first LightRAG generation passed chunk ids such as `gd_22550cc95a46ded2d9a37c99:gd_22550cc95a46ded2d9a37c99-chunk-000`. The evidence stage rejected every LightRAG claim as `unknown_source_id`. Those answers were written with no accepted evidence. The naive Brex blog in that run said Capital One was buying Brex for $5.3 billion.

A second run aliased each hit id to the parent document id, without changing the generator. Claims were then accepted. Results:

- Pilot dense Brex blog states $5.15 billion and also adds an unsupported `$2 Billion` figure. Numeric grounding flagged it and did not block the output.
- Naive and mix Brex blogs also state $5.15 billion, then add other dollar amounts. Numeric grounding flagged those amounts, including a 57 percent premium, and did not block the output.
- The accepted attribution claim is "AI means everything in SaaS needs to be a personalized experience." It does not name Anneka Gupta. The dense blog attributes the sentence to Marc Benioff. The naive blog says "I did." The mix blog attributes it to a Harry Stebbings post. None of the generated answers name Anneka Gupta.

Retrieval stored the speaker. The accepted claim dropped the speaker, and the writing prompt uses the claims. The generated text is not a Lemkin endorsement of that sentence; it is a failure to keep the quoted speaker.

## Recommendation

Do not expand indexing to the full corpus. Re-index a controlled sample only after two changes: split documents at omission markers, and stop the 1,200-second timeouts from discarding an otherwise short post. Keep the current 152-document graph as the measured pilot. A larger paired generation should wait until an accepted claim still names the quoted speaker. The three-topic run already shows what the current evidence stage does.
