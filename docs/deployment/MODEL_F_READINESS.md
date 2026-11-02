# Model F LightRAG readiness

This document records the pre-index integration boundary for Model F. Model F
is not enabled in the deployed generator yet.

## Current state

- Integration branch: `codex/model-f-integration`.
- Upstream handoff: `kevin/lightrag-model-f` at `817e159b99b904a3171a0495e7a1c9433fe98137`.
- Dense PersonaRAG remains the default retrieval method.
- `MODEL_F_ENABLED=false` is the production default.
- The API's `model_f` method returns a controlled 503 until a verified index
  and query adapter are available.
- The reserved request contract is `retrieval_backend="lightrag"` plus one of
  `model_f_context=off|source|relationships|descriptions|multihop`. Existing
  methods retain `retrieval_backend="dense"` and `model_f_context="off"`.
- No LightRAG package is installed in the web/backend image yet, and no index
  is mounted or queried during this phase.

## Snapshot acceptance gate

Kevin must provide a completed snapshot outside Git. Before a candidate points
at it, retain the following with the deployment record:

1. Output from the Model F `snapshot` and `verify` commands.
2. SHA-256 checksum and catalog hash.
3. Size of the snapshot and enough dedicated candidate storage for it.
4. LightRAG version, embedding model, corpus/split identity, and configuration.
5. Confirmation that `indexing.lock` is absent and no documents are still
   `processing`.
6. One successful hybrid-query smoke result against the immutable copy.

Never mount `storage_expand_v1`, any directory containing `indexing.lock`, or
the writer's live index directory. Mount the accepted snapshot read-only and
outside the Git checkout.

## Post-index work

After acceptance, add the private LightRAG query adapter and package pin from
`host_finetune/requirements-lightrag.txt`. Run F-SOURCE first; F-REL, F-DESC,
and F-HOP remain experimental until the four-arm experiment and human
retrieval grades are recorded. Do not replace the Chroma corpus or existing
PersonaRAG route.
