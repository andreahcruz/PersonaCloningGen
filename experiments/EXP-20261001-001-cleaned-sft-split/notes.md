# EXP-20261001-001 — Cleaned-source SFT split

Question: after the cleaned corpus keeps essay text past tweet embeds and rejoins newlines inside URL paths, can that corpus become a 512-token QLoRA file with a medium in each instruction, and a grouped 80/10/10 split that keeps every chunk of one source line in one split?

Held fixed: the five raw files in `data/`. `host_finetune/data/dataset.jsonl` was not overwritten. `data/openai_ft/lemkin_gold_eval.jsonl` was not used. EXP-20260928-002 was not edited. No fine-tune was started.

## Method

`spark_jobs/corpus_footer_scrub.py` removes a Jason `(@jasonlk)` attribution card and leaves the essay after it. It rejoins a newline inside a URL path when the next piece continues the path, and it leaves a following host such as `pic.twitter.com` alone.

`python -m host_finetune.clean_scraped` rewrote `data/cleaned/`. The SFT builder then chunked those files, named the medium in the instruction (`blog post`, `LinkedIn post`, `X post`, or `talk`), and kept X posts of at least 12 words. Blog, LinkedIn, and YouTube kept a 50-word floor. The builder reads physical lines, because `str.splitlines()` also splits on U+2028 and that character appears inside three X posts.

The promo filter drops a title that is a direct ticket, speaker, or sponsor announcement. `Dear SaaStr` titles stay. A trailing event CTA is removed only from the last 45% of a chunk, and the chunk is dropped for being under 20 words only when that CTA was actually removed.

`fit_sft_to_context.py` re-chunked with the Llama 3.1 chat template until every example was at most 512 tokens. `token_length_audit.py` counted the same template and exited with zero rows over the limit.

`split_groups.py` unions rows that share `source_file` and `source_line`, then unions titles, normalized-exact copies, and Jaccard ≥ 0.8 near-duplicates. Whole groups go to the split with the most remaining quota. There is no random seed.

## Counts

Cleaned documents: blog 3,502 → 3,466; LinkedIn 2,675 → 2,006; X 25,997 → 24,277; Jason YouTube 462 → 428; SaaStr YouTube 199 → 193.

SFT build: 39,742 rows (blog 11,467, LinkedIn 980, X 20,105, Jason talks 1,698, SaaStr talks 5,492). Documents kept: blog 3,466, LinkedIn 809, X 19,960, Jason talks 424, SaaStr talks 193. Dropped for length: LinkedIn 1,197, X 4,317, Jason talks 4. Bad JSON: 0.

Promo filter: 39,742 → 39,575. Direct announcement chunks dropped: 166, from 123 source documents. Trailing CTAs removed: 9. Chunks dropped because fewer than 20 words remained after a CTA: 1.

Token fit: 39,575 → 39,688. Titles shortened: 3,829. Rows dropped for still exceeding the limit: 0. Audit: 39,688 examples, max 512, p50 159, p95 445, over-limit 0.

Split of the fit file: 31,750 train / 3,969 validation / 3,969 test. 22,288 groups. Zero groups, exact copies, or near-duplicate pairs cross a split.

## Spot checks

The cleaned “48 Types of VP Sales” essay is 11,584 characters, contains “one key criterion,” and does not contain `(@jasonlk)`. Fit row 10950 is that essay, instructed as a blog post, and the attribution card is gone.

Fit row 13234 is an X post whose URL is `https://saastr.ai/valuation-calculator`. The path is not split across a newline.

Fit row 12503 is an X post of 38 words, under the old 50-word floor, instructed as an X post.

Fit row 187 is “Dear SaaStr: When Should I Vibe Code An App vs. Buy One?” The body still mentions SaaStr Annual and 10,000 attendees. The title was not treated as a ticket announcement.

## What a later training run must do

`SPLIT_MANIFEST` is still optional. Unset, `finetune.py` loads `host_finetune/data/dataset.jsonl` and uses the in-memory 90/10 shuffle. This assignment hashes `dataset_from_cleaned_sources_fit512.jsonl`. Point `DATASET_LOCAL` at that file and set `SPLIT_MANIFEST` to `experiments/EXP-20261001-001-cleaned-sft-split/raw/split_assignments.jsonl` before training. The hash check will refuse the new manifest against the old file.

Validation and test are single-row groups. Most of those rows are X posts (3,748 and 3,713). Multi-chunk blog posts and talks were assigned to train while that quota was still the largest. A later comparison should not treat validation loss as a blog-essay score.

YouTube auto-captions and multi-speaker interviews are still in the file, labeled as talks.
