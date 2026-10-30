# Span repair of the EXP-009 drafts

Question: can a post-generation edit remove unsupported facts from the EXP-009 drafts without the writing collapse measured in EXP-010?

EXP-009 and EXP-010 raw files were not modified. No graph indexing and no QLoRA training. The repair reads a finished draft and deletes or narrows a span. It does not generate a new draft.

## Why EXP-010 writing fell

Accepted-claim counts are identical on all 48 paired cells. Extraction failure is the four Chroma Brex drafts in both experiments, so it does not explain the other arms.

Of the 20 cells whose EXP-010 draft fell below 60 percent of the EXP-009 length:

- 14 contain the question or an evidence-instruction line
- 13 kept a short first draft under the stricter prompt
- 5 kept a retry that was much shorter than that arm's own first attempt
- 2 are the pre-existing Chroma extraction failure

The stricter prompt made the model repeat the question or paste the instructions. The retry rule then treated a short echo as an improvement when it contained fewer unsupported numbers.

## Repair rule

Contract `span-repair-v1`. An unsupported quantity is deleted, including a glued `+` and a preceding preposition when the deletion would otherwise leave "for and" or "over annually". An unsupported title is replaced by the speaker's name when that name is in the accepted claims. A quoted 6-word span with no speaker name gets that name inserted. The edit is rejected, and the original draft is kept, when it adds a copying or factual failure, drops more than 15 percent of the words, breaks the ending, or leaves a dangling preposition. The warning is stored beside the draft and is not inserted into the prose.

`substantive-quality-v1` requires completion, the length band, both copying checks, no question echo, strict factual pass, at least 85 percent of the original words, and overlap with an accepted claim when claims exist. Lexical overlap is not an input.

## Measured comparison

Twenty-one drafts were edited: 3 Chroma, 6 pilot dense, 7 naive, and 5 mix. Unchanged drafts reuse the EXP-009 writing score. Edited drafts were rescored by blinded `qwen2.5:14b`.

| Arm | Unsupported numbers, 009 / repair / 010 | Mean words, 009 / repair / 010 | Echoes, 009 / repair / 010 | Writing, 009 / repair / 010 |
|---|---|---|---|---|
| Chroma v2 | 0.4167 / 0.1667 / 0.0833 | 173.25 / 173.08 / 102.0 | 0.25 / 0.25 / 0.4167 | 3.2778 / 3.1944 / 2.3472 |
| Pilot dense | 0.4167 / 0.0 / 0.1667 | 142.33 / 141.08 / 150.5 | 0.3333 / 0.3333 / 0.5 | 2.9444 / 2.7083 / 2.0556 |
| Naive | 0.5 / 0.0 / 0.0 | 150.33 / 147.75 / 115.25 | 0.3333 / 0.3333 / 0.5833 | 2.8194 / 2.6389 / 2.1111 |
| Mix | 0.4167 / 0.0 / 0.0 | 159.33 / 157.08 / 161.17 | 0.3333 / 0.3333 / 0.5 | 2.8333 / 2.6944 / 2.3333 |

Strict pass moved from 0.4167, 0.4167, 0.25, and 0.3333 to 0.6667, 0.6667, 0.75, and 0.75. Claim-overlap means did not change. Training-copy stayed 1.0. Retrieval-copy stayed at the EXP-009 rates. Completion and format stayed at the EXP-009 rates. Unified eligibility stayed 0.1667 for Chroma and 0.0833 for naive and mix, and moved from 0.3333 to 0.25 for pilot dense. Substantive eligibility stays near zero because the EXP-009 echoes and missing quote overlap already fail that rule.

Short drafts that are mostly an unsupported number were not edited, because the deletion would remove more than 15 percent of the words. They keep the original text and a warning. Chroma Brex LinkedIn is one of those: deleting `$5 billion`, `6,000+`, and `$1 million` would leave "acquire Brex for and".

## Recommendation

Use span repair of the EXP-009 drafts, not the EXP-010 prompt, when the goal is fewer unsupported numbers without collapsing the draft. The repair is the better balance of grounding, length, echoes, and originality. It is not writing-neutral: all-12 writing means still fall by about 0.08 to 0.24, far less than the EXP-010 drops. Drafts with no defensible short edit remain the EXP-009 text plus a visible warning. This result is MEASURED. It is not VALIDATED as authorship improvement, and it does not show that LightRAG writes better than dense retrieval.

Human-reviewable before-and-after text is in `metrics/examples.json`.
