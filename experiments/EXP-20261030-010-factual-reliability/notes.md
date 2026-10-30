# Factual reliability contract and paired regeneration

Question: do deterministic factual checks, and one versioned prompt change, change Model E versus Model F on the same 12 topic-medium cells?

EXP-20261029-009 raw files were not modified. The LightRAG graph was not modified. Retrieval was reread from that experiment's frozen files. The Relabel adapter, seed 42, and token budgets were unchanged. The prompt contract is `factual-reliability-v1`.

## Evaluation contract

The unified eligibility rule is completion, training-corpus copy under 12 words, the medium length band, and lexical sentence overlap of at least 0.2. That overlap is token overlap with retrieved passages. It is not claim entailment. An `extraction_failed` draft can pass it and be eligible. In EXP-009 the Chroma Brex LinkedIn draft had no accepted claims, stated `$5 billion`, `$1 million`, and `6,000`, passed lexical grounding at 0.5, and was unified-eligible.

The checks are different objects:

| Check | What it measures | Effect on unified eligibility |
|---|---|---|
| Lexical overlap | Share of answer sentences with token overlap at least 0.2 | Can pass or fail eligibility |
| Verifiable claim support | Accepted claim text contains the detected fact | Not an eligibility input |
| Numeric grounding | Currency, percent, date, year, and large counts absent from the topic and accepted claims | Diagnostic only |
| Source attribution | A 6-word quoted-claim span appears with the quoted speaker's name | Not an eligibility input |
| Training-corpus copying | 12-word span from training targets, excluding the prompt | Can fail eligibility |
| Retrieval-source copying | 12-word span from the retrieved passage | Can leave a flagged draft eligible |

Operational generation before this run could retry a copy, an echo, a short draft, an unfinished sentence, or an unsupported number. The generation loop also rebuilt the prompt without the `extraction_failed` status, so that status note was not in the EXP-009 prompts. This run uses the instruction already built for the job, including that status. A retry is kept only when the targeted failure improves and the retry does not add a retrieval-source copy, a new unsupported number, a new unsupported or unverifiable role, or a new speaker failure. The scientific length band for X still starts at 3 words. The operational retry asks for 20.

Strict grounding in this experiment passes only when accepted claims exist and none of these fire: unsupported number, unsupported or unverifiable role, speaker omission or unverifiable speaker use, or retrieval-source copying. A role without an organization is not verifiable. No detected role assertion is not treated as a hidden true role.

## Measured comparison

Same 12 cells. Writing judge is `qwen2.5:14b` on the new drafts. EXP-009 writing means are the previously recorded all-12 means.

| Arm | Strict pass 009 | Strict pass 010 | Unsupported numbers 009 | Unsupported numbers 010 | Echoes 009 | Echoes 010 | Unified eligible 009 | Unified eligible 010 | Writing 009 | Writing 010 | Wall seconds 010 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Chroma v2 | 0.4167 | 0.6667 | 0.4167 | 0.0833 | 0.25 | 0.4167 | 0.1667 | 0.0833 | 3.2778 | 2.3472 | 250.971 |
| Pilot dense | 0.4167 | 0.5833 | 0.4167 | 0.1667 | 0.3333 | 0.5 | 0.3333 | 0.1667 | 2.9444 | 2.0556 | 214.109 |
| Naive | 0.25 | 0.8333 | 0.5 | 0.0 | 0.3333 | 0.5833 | 0.0833 | 0.0833 | 2.8194 | 2.1111 | 228.313 |
| Mix | 0.3333 | 0.6667 | 0.4167 | 0.0 | 0.3333 | 0.5 | 0.0833 | 0.0833 | 2.8333 | 2.3333 | 241.903 |

Training-corpus copying passed on every stored draft in both experiments. Retrieval-source copying failed one naive draft and one mix draft in EXP-009, and one mix draft in this run. Chroma Brex remained `extraction_failed` after two extraction attempts. Its new drafts no longer contain the earlier invented dollar amounts. Several of those drafts are 8 or 9 words, so the numeric improvement is omission, not a supported price.

Unsupported role rates fell from 0.0833 on pilot, naive, and mix to 0. Speaker omissions were 0 in EXP-009 and are 0.0833, 0.1667, and 0.0833 for pilot, naive, and mix here. Strict eligibility is 0.3333, 0.1667, 0.25, and 0.25. Selector-valid style means are not ranked: Chroma and naive and mix have no selector-eligible medium in this run, and pilot dense has blog only.

## Findings for the workbook

1. Model F adds a LightRAG index and five query modes beside Model E's Chroma retrieval. Both paths still extract atomic claims, check spans, and generate with the same Relabel adapter.
2. Graph retrieval did not improve retrieval quality on the measured eight-question sheet in EXP-20261029-005. Matched dense found all three first-VP sources; naive and mix found two. Mix also returned chunks from a failed document. Precision and recall were not scored. Mean query time was 2.07 seconds for dense, 2.85 for naive, and 3.05 for mix.
3. Graph retrieval did not improve factual correctness over pilot dense or Chroma. Naive and mix removed unsupported numbers in this run, and they also removed most of the draft on several cells. Mix still matched naive on the Brex and VP drafts often enough that the graph mode is not a separate factual result.
4. Writing style did not improve. All-12 writing means fell for every arm. Persona-style distances are not comparable because the eligible medium sets differ.
5. Indexing the 160-document pilot took 10,768 seconds. One four-arm generation in this contract took 251, 214, 228, and 242 seconds, including adapter load and retries. Mean retrieval time on the eight-question sheet was 2.07 seconds for dense, 2.85 for naive, and 3.05 for mix.
6. Limits: three topics, one seed, a 152-document graph, lexical eligibility that is not entailment, strict checks that only see quantities, parsed roles, and 6-word quote spans, and a prompt change that trades invented numbers for short echoes.
7. The strict checker and prompt contract are IMPLEMENTED and TESTED. This paired run is MEASURED. It is not VALIDATED as a quality improvement. The frozen graph and EXP-009 artifacts remain in place.
