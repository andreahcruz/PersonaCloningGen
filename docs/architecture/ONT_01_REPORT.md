# ONT-01 — schema contract and five-document admission pilot

Status: **IMPLEMENTED and TESTED contracts; actual source admission PENDING human review**. No real claims were extracted or marked retrievable. DEC-013 records the approved initial direction; DEC-012 and its benchmark are unchanged.

## Delivered contracts

`host_finetune/ontology_contract.py` is a standard-library-only module with no model, retrieval, DuckDB, Chroma or filesystem operations. Frozen dataclasses and strict typed JSON decoding cover document/source identities, frozen dataset/split/gold lineage, immutable admission decisions, exact passages/spans, canonical entities and aliases, typed statements/relationships, conditions, polarity/modality, temporal scope, quantities, source-slot bindings, human attestations and conflicts.

The initial predicate set is deliberately small: metric observations, person-role-organization relationships, and author recommendations/discouragement. Unknown predicates fail closed. Conditions support explicit all/any groups with typed comparisons; unknown scope is distinct from no condition stated. Dates are not inferred from publication time. Quantities support exact USD, percentages, counts and ratios, with explicit subject, metric, period and basis. Unsupported numeric formats are excluded; the old diagnostic parser is not changed. Decimal arithmetic preserves cents and large values without silent rounding.

`validate_snapshot` computes claim eligibility; there is no user-supplied eligible flag to trust. `retrievable_assertions` revalidates the full supplied snapshot before returning immutable assertions. States are `pending_review`, `eligible`, `quarantined`, `rejected`, `duplicate`, and `conflict`. Candidate/reviewed lifecycle alone never grants eligibility.

Required admission proof includes body identity, eligible train lineage, allowed source class, nonconflicting author signals, an actual byline-proof capture matching its hash and URL, and a named/timestamped content-bound human approval. A byline capture is not automatically interpreted as endorsement: each assertion's reviewed source binding and wording must also be approved. Author identity can be supported by the admission proof instead of requiring the author's name literally in every first-person prose span.

Each assertion references an independently annotated `SupportBinding` containing the same structured statement and source spans for its required slots. Subject/quantity/time inconsistencies are rejected independently of numeric token presence. Exact offsets and hashes are checked against supplied immutable source bodies. The reviewer digest covers assertion wording/lifecycle, source bindings, supporting documents/admission decisions, passages/spans, and relevant entity/alias records. Changing either the claim or its source annotation invalidates the prior attestation. Adding an unrelated source does not invalidate an existing review.

Duplicate IDs fail closed; duplicate structured propositions are deduplicated for retrieval without removing underlying records. Opposing polarity, same-scope differing values, and opposing recommendation/discouragement are treated conservatively as conflicts. Explicit conflict records are checked across the supplied snapshot before any future top-k or prompt fitting. ONT-01 does not implement conflict resolution.

## Five-document manifest

Registered artifact: `experiments/EXP-20261108-004-ont-01-schema-admission/raw/pilot_admission.json`.

Five candidates were chosen from the frozen 24-question source locators for topic diversity:

- `dev_candidate_001`: cleaned blog line 108, Relabel row 718; customer account handoff.
- `dev_candidate_013`: line 623, row 3977; pricing versus growth.
- `dev_candidate_014`: line 1943, row 8068; conditional custom-feature decisions.
- `dev_candidate_020`: line 2871, row 10250; cash reserves.
- `dev_candidate_024`: line 3178, row 10931; expansion revenue.

Every constituent source row was checked through the existing `graph_ready.eligible_train_ids` / gold-family exclusion functions. Source hashes, record hashes, source URL/date, cleaned-file line, row/group IDs, body hash, full-body passage offsets, decision hashes and development-question IDs are recorded. File/line locators refer to the cleaned source; the raw file hash identifies its upstream corpus, not an asserted one-to-one raw-line mapping.

All **five** decisions remain **pending_review**, with empty actual reviewer/byline-proof fields. There are **zero admitted documents, zero real assertions, and zero retrievable real assertions**. Previous relevance review of a development question is not authorship approval. No blog body or verbatim supporting span was copied into the artifact; passage identity is represented by offsets/hashes and an immutable source locator.

Human review must check the source version, byline and conflicting signals, provenance, guest/interview/third-party content, and quotation/endorsement boundaries. Keep proof captures local. After review, append a new `AdmissionDecision` referencing the previous decision ID and bind its approval to `admission_subject`; do not edit the recorded pending decision or invent historical approvals. The pure `append_decision` helper returns a new validated history rather than mutating the old one.

## Tests, reproduction, and limitations

Tests in `tests/test_ontology_contract.py` use explicitly synthetic source bodies and reviewer attestations, plus one controlled read of the real pending manifest. They cover the user's requested missing/conflicting attribution, source-span/offset failures, missing bindings, wrong company-number association, wrong period, unverified/quarantined claims, unsupported relationships, duplicates/conflicts, immutable decisions, stale attestations, alias collisions, exact quantities, and conditional advice. These are contract tests, not measured extraction accuracy or independently verified real authorship.

Verification: **53 ontology tests passed; 169 relevant tests passed in total**, including existing evidence, numeric, source-selection, split, audit and project-management tests. Manifest validation passed without warnings. `python -m project_pm check` passed with zero errors and two known historical warnings for unregistered F-SOURCE/F-DESC runs. Exact commands and captured outputs are in the new run's `raw/verification.json`; post-handoff checks are in `raw/closure_verification.json`.

The recorded builder command is the virtualenv Python executing `experiments/EXP-20261108-004-ont-01-schema-admission/config/build_admission.py`. It reuses existing source-selection functions, constructs hashes/passages without rewriting source strings, round-trips the schema, and refuses overwriting run outputs. Reproduction uses the recorded inputs/code in a clean location where the run output files do not exist. Existing outputs remain immutable. `raw/provenance.json` records the builder-stage hashes; final verification additionally records final code hashes and revalidates the same manifest after numeric/date edge-case hardening. Source snapshots accompany final verification because these changes are not committed.

Semantic entailment and reviewer identity cannot be authenticated by these structural contracts. The approved source binding is a trusted, content-bound human attestation, not an NLP theorem or cryptographic identity signature. Numeric/company/time checks compare the proposed assertion with that approved binding; they do not discover a correct association from arbitrary prose. Complete source-registry/manifest integrity must be verified by a later store adapter. Conflict detection covers supplied structured assertions and declared conflict links, not every unannotated contradiction in a source corpus. Full typed claim extraction, graph reasoning and automated source attribution are not implemented.

Generation output, output verification and benchmark eligibility remain separate future interfaces. This module computes **evidence retrieval eligibility only**; its state must not be reused as a generated-output or benchmark-pass label.

## Proposed integration changes — not implemented

1. **Evidence capsule:** add a future immutable validated-evidence pack with assertion ID, source-version/span identities, review identities, and structured bindings. Revalidate the pack at the rendering boundary and keep a trace sidecar keyed by assertion ID and emitted evidence label. Do not route rich ontology records through the existing field-dropping `partition_evidence` path without preserving that map. Reuse source-span/copying checks; a legacy accepted flag or `attribution_status` string is insufficient. Keep the existing capsule and historical traces unchanged in ONT-01.
2. **Existing Chroma to DuckDB expansion:** later validate a mapping from admitted document/row IDs to the selected frozen train-only index version, then expand a common retrieved candidate pool to approved claims. Verify the source/index snapshot identities and do not modify the existing index or selector. ONT-01 creates neither DuckDB nor a new claim-vector collection.
3. **Numeric verification:** later consume canonical subject/metric/quantity/period/basis tuples and their approved source bindings. Keep topic numbers and style-example numbers out of source authority. Preserve raw generated text and produce a separate verification record and separate benchmark-eligibility record under the same rules for matched architectures. Do not add ontology-specific repair/revision or silently replace the current historical numeric diagnostics.
4. **Conflicts and coverage:** the later store must supply complete relevant conflict groups before truncating top-k evidence. Preserve qualifier/conflict bundles or record an explicit insufficient-evidence outcome under the common comparison policy. Retain every benchmark cell and do not count abstention as factual perfection.

## Stop and review

Review the contract's trusted-human-attestation boundary, four-predicate initial scope, five candidate sources, admission-proof requirements, and future capsule/numeric integration. ONT-01 completion does not mean the sources are verified and does not authorize ONT-02. Obtain user review before starting extraction. The 30-topic gold benchmark remains unchanged and primary; all 24 questions remain development material. No index build, graph database, claim-vector index, inference, training, output revision, split changes, commits or deployment occurred.
