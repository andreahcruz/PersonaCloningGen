# Data provenance, dictionary, and cleaning

2026-10-03. [artifact_hashes.csv](artifact_hashes.csv) inventories current local bytes; hashes do not establish authorship or reconstruct cleaning history.

## Layers and fields

- Raw blog: `content`, `title`, URL/date; LinkedIn: `content`, publication metadata; X: `text`, URL/created-at and reply/repost fields; YouTube: `transcript_text`, `video_title`, `video_url`. See root DAG normalizers for field variants. Channel is not speaker identity.
- `data/cleaned/*.jsonl`: source-shaped derived inputs. Raw-to-cleaned transformation/version and authorship assurance are incomplete. Pre-existing untracked `data/cleaned (2).zip` was not extracted or made canonical.
- Host builder SFT: `instruction`, empty `input`, target `output`, medium `source`, `source_file`, `source_line`. Line number belongs to the cleaned source and is not stable across cleaning versions. Raw document IDs/URLs are not retained in this schema.
- Split JSONL: metadata contains dataset hash/count/Jaccard threshold; assignments contain row index, group, split, title and provenance. Valid only for exact dataset bytes.
- `data/openai_ft/lemkin_gold_eval.jsonl`: task/brief, expected facts, reference and rubric fields. These are not authentic author-style gold labels merely because they are named gold. Factual support and construction provenance remain to audit.
- Root `data/persona_profile.json`: vocabulary/structure/framing/theme statistics and meta. Separate persona pipeline uses a different profile schema/path.

## Implemented cleaning

Root normalizers skip unusable rows and some replies/reposts. Spark strips HTML, chunks and embeds. Host builder uses 50-word minimum except X's 12-word minimum, default 1600-character chunks, and a first-120-character topic fallback for untitled content. Promo filter uses event-title/trailing-CTA regexes. Token fitter preserves paragraphing and colon/list associations where possible but may recurse to words. Tests do not guarantee semantic completeness of every answer.

`queue_followup_trains.select_balanced_rows` drops SaaStr talks and keeps whole X documents in sorted source-key order until X rows reach blog rows. This is neither randomized medium sampling nor token-loss balancing. It does not establish that remaining transcripts belong only to Lemkin.

## Proposed contract

Preserve raw bytes; create versioned derived artifacts with lineage and reason codes. Treat authorship, promotion/template status, boilerplate, duplication, completeness and medium separately. Quarantine uncertain examples; human-audit a fixed sample. Organic/promotional corpus views are proposals, not production labels. Do not silently normalize away style, synthesize endings, or reassign speakers.

Complete-post SFT must begin from source documents; increasing the fit limit cannot restore text lost in upstream chunking. Explicitly name section/continuation tasks or omit them from full-post supervision. Punctuation is only a completeness heuristic. Every row needs inspectable source ownership.

EXP-20261003-004 is a derived view of the repaired balanced file, not a new cleaning pass. Added fields on kept rows are `sft_role` (`unchanged`, `opening`, or `continuation`) and `prior_group`. Dispositions for every input row, including drops, are in that experiment’s `raw/dispositions.jsonl`. The prior balanced file’s bytes are unchanged.

Record source/collection terms where known; public availability is not proof of sole authorship, license or consent. Evaluate copying against source spans. Rights and attribution coverage remain unknown.
