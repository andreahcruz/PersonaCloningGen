"""Synthetic contracts, not author verification or model inference."""
from dataclasses import FrozenInstanceError, asdict, replace
import json

import pytest

from host_finetune.ontology_contract import (
    AUTHOR, AdmissionDecision, AdmissionState, Assertion, Comparator, Condition,
    ConditionKind, Conditions, Conflict, ContractError, Document, Entity, Kind,
    Lifecycle, Modality, Passage, Polarity, Predicate, Provenance, Quantity, Reason,
    Review, ReviewState, SlotEvidence, Snapshot, Span, State, Statement,
    SupportBinding, TimeKind, TimeScope, admission_subject, append_decision,
    assertion_subject, fingerprint, retrievable_assertions, sha256,
    snapshot_from_dict, validate_admission, validate_snapshot,
    quantity_canonical,
)

DATE = "2026-11-08T00:00:00+00:00"
FIRST = "Jason Lemkin reports: Acme retention was 50% in 2024."
SECOND = "Acme retention was 60% in 2024."
BODY = FIRST + " " + SECOND + " Beta retention was 80% in 2023. I recommend hiring carefully if Acme growth is below 10%."


def approved(subject):
    return Review(subject, ReviewState.APPROVED, "synthetic-reviewer", DATE, "test-fixture://approval")


def sign_claims(snapshot):
    snapshot = replace(snapshot, assertions=tuple(replace(c, lifecycle=Lifecycle.REVIEWED, review=None) for c in snapshot.assertions))
    return replace(snapshot, assertions=tuple(replace(c, review=approved(assertion_subject(c, snapshot))) for c in snapshot.assertions))


@pytest.fixture
def fixture():
    author = Entity(Kind.PERSON, AUTHOR)
    acme = Entity(Kind.ORGANIZATION, "Acme")
    beta = Entity(Kind.ORGANIZATION, "Beta")
    action = Entity(Kind.ACTION, "hiring carefully")
    provenance = Provenance("synthetic_blog.jsonl", 1, *(sha256(x) for x in
                            ("raw", "clean", "record", "dataset", "split", "gold", "policy")),
                            row_ids=(0,), group_ids=(sha256("family"),), selection_eligible=True)
    doc = Document(provenance, "https://example.test/blog", sha256(BODY), DATE, (AUTHOR,))
    proof = "Synthetic byline proof: Jason Lemkin for https://example.test/blog; test-only."
    decision = AdmissionDecision(doc.id, fingerprint(doc), AdmissionState.ADMITTED,
                                 AUTHOR, sha256(proof), doc.source_url)
    decision = replace(decision, review=approved(admission_subject(doc, decision)))
    passage = Passage(doc.id, 0, len(BODY), sha256(BODY))
    span = Span(passage.id, 0, len(FIRST), sha256(FIRST))
    period = TimeScope(TimeKind.EXPLICIT, "2024")
    quantity = Quantity(acme.id, "retention", "50%", "pct:50", period, "retention")
    statement = Statement(acme.id, Predicate.METRIC, author.id, quantity=quantity, time=period)
    binding = SupportBinding(statement, tuple(SlotEvidence(slot, span.id) for slot in
                                             ("subject", "predicate", "quantity", "metric", "time")))
    claim = Assertion(statement, "Acme retention was 50% in 2024.", binding.id)
    snapshot = Snapshot((doc,), (decision,), (passage,), (span,), (author, acme, beta, action), (binding,), (claim,))
    return sign_claims(snapshot), {doc.id: BODY}, {sha256(proof): proof}


def result(fixture, snapshot=None):
    s, bodies, proofs = fixture
    return validate_snapshot(snapshot or s, bodies, proofs).assertions[0]


def with_binding(snapshot, binding, statement=None):
    claim = replace(snapshot.assertions[0], support_id=binding.id, statement=statement or binding.statement)
    return sign_claims(replace(snapshot, supports=(binding,), assertions=(claim,)))


def test_reviewed_supported_synthetic_claim_can_be_eligible(fixture):
    s, bodies, proofs = fixture
    assert result(fixture).retrievable
    assert retrievable_assertions(s, bodies, proofs) == s.assertions
    assert validate_admission(s.documents[0], s.decisions[0], BODY, proofs).retrievable


def test_json_roundtrip_is_strict_and_immutable(fixture):
    s = fixture[0]
    assert snapshot_from_dict(json.loads(json.dumps(asdict(s)))) == s
    with pytest.raises(FrozenInstanceError):
        s.documents[0].source_url = "changed"
    payload = json.loads(json.dumps(asdict(s)))
    payload["assertions"][0]["eligibility"] = "eligible"
    with pytest.raises(ContractError, match="unknown fields"):
        snapshot_from_dict(payload)
    payload["assertions"][0].pop("eligibility")
    payload["documents"][0]["provenance"]["source_line"] = True
    with pytest.raises(ContractError):
        snapshot_from_dict(payload)


def test_direct_constructor_cannot_bypass_enum_and_tuple_types(fixture):
    s, bodies, proofs = fixture
    with pytest.raises(ContractError):
        validate_snapshot(replace(s, assertions=list(s.assertions)), bodies, proofs)
    claim = replace(s.assertions[0], lifecycle="reviewed")
    with pytest.raises(ContractError):
        validate_snapshot(replace(s, assertions=(claim,)), bodies, proofs)


def test_pending_admission_does_not_invent_attribution(fixture):
    s, bodies, proofs = fixture
    decision = AdmissionDecision(s.documents[0].id, fingerprint(s.documents[0]))
    s = replace(s, decisions=(decision,))
    assert validate_snapshot(s, bodies, proofs).documents[0].state is State.PENDING
    assert not retrievable_assertions(s, bodies, proofs)


def test_missing_attribution_and_unavailable_proof_fail_closed(fixture):
    s, bodies, _ = fixture
    assert validate_snapshot(s, bodies).documents[0].reasons == (Reason.MISSING_ATTRIBUTION,)
    assert not retrievable_assertions(s, bodies)


@pytest.mark.parametrize("signals", [(AUTHOR, "Guest Author"), ("Unknown",), ("Another Author",)])
def test_conflicting_author_signals_are_excluded(fixture, signals):
    s, bodies, proofs = fixture
    doc = replace(s.documents[0], author_signals=signals)
    decision = replace(s.decisions[0], document_sha256=fingerprint(doc))
    check = validate_admission(doc, decision, BODY, proofs)
    assert Reason.CONFLICTING_AUTHORS in check.reasons
    assert not check.retrievable


@pytest.mark.parametrize("kind", ["guest", "interview", "transcript", "repost", "third_party", "ambiguous"])
def test_excluded_source_classes(fixture, kind):
    s, _, proofs = fixture
    doc = replace(s.documents[0], content_kind=kind)
    decision = replace(s.decisions[0], document_sha256=fingerprint(doc))
    assert Reason.EXCLUDED_SOURCE in validate_admission(doc, decision, BODY, proofs).reasons


def test_source_mutation_or_stale_admission_cannot_be_used(fixture):
    s, bodies, proofs = fixture
    assert Reason.BODY_HASH in validate_admission(s.documents[0], s.decisions[0], BODY + "!", proofs).reasons
    decision = replace(s.decisions[0], review=approved(sha256("different contents")))
    assert validate_admission(s.documents[0], decision, BODY, proofs).state is State.PENDING


@pytest.mark.parametrize("start,end", [(-1, 10), (1, 0), (0, 99999), (True, 5)])
def test_invalid_passage_offsets(fixture, start, end):
    s, bodies, proofs = fixture
    s = replace(s, passages=(replace(s.passages[0], start=start, end=end),))
    with pytest.raises(ContractError):
        validate_snapshot(s, bodies, proofs)


def test_unsupported_source_span_hash_or_offsets(fixture):
    s, bodies, proofs = fixture
    with pytest.raises(ContractError, match="unsupported_source_span"):
        validate_snapshot(replace(s, spans=(replace(s.spans[0], text_sha256=sha256("invented")),)), bodies, proofs)
    with pytest.raises(ContractError, match="offsets"):
        validate_snapshot(replace(s, spans=(replace(s.spans[0], end=99999),)), bodies, proofs)


def test_missing_assertion_slot_or_support_is_not_eligible(fixture):
    s = fixture[0]
    binding = replace(s.supports[0], evidence=(s.supports[0].evidence[0],))
    assert Reason.MISSING_BINDING in result(fixture, with_binding(s, binding)).reasons
    claim = replace(s.assertions[0], support_id="missing")
    assert result(fixture, replace(s, assertions=(claim,))).state is State.REJECTED


def test_wrong_company_number_association_rejected_despite_all_words_present(fixture):
    s = fixture[0]
    beta = next(e for e in s.entities if e.canonical_name == "Beta")
    claim = s.assertions[0]
    wrong = replace(claim.statement, subject_id=beta.id,
                    quantity=replace(claim.statement.quantity, subject_id=beta.id))
    s = sign_claims(replace(s, assertions=(replace(claim, statement=wrong),)))
    assert Reason.WRONG_SUBJECT in result(fixture, s).reasons
    assert not result(fixture, s).retrievable


def test_internal_quantity_subject_must_match_assertion(fixture):
    s = fixture[0]
    beta = next(e for e in s.entities if e.canonical_name == "Beta")
    statement = replace(s.assertions[0].statement, quantity=replace(s.assertions[0].statement.quantity, subject_id=beta.id))
    binding = replace(s.supports[0], statement=statement)
    assert Reason.WRONG_SUBJECT in result(fixture, with_binding(s, binding)).reasons


def test_wrong_temporal_scope_and_missing_period(fixture):
    s = fixture[0]
    claim = s.assertions[0]
    year = TimeScope(TimeKind.EXPLICIT, "2023")
    wrong = replace(claim.statement, time=year, quantity=replace(claim.statement.quantity, time=year))
    changed = sign_claims(replace(s, assertions=(replace(claim, statement=wrong),)))
    assert Reason.WRONG_TIME in result(fixture, changed).reasons
    undated = replace(claim.statement, time=TimeScope())
    assert Reason.WRONG_TIME in result(fixture, with_binding(s, replace(s.supports[0], statement=undated))).reasons


@pytest.mark.parametrize("period", ["2024-13", "2024-02-30", "2024-Q5", "0000-Q1", "last year"])
def test_invalid_explicit_periods(fixture, period):
    s = fixture[0]
    time = TimeScope(TimeKind.EXPLICIT, period)
    statement = replace(s.assertions[0].statement, time=time, quantity=replace(s.assertions[0].statement.quantity, time=time))
    assert Reason.WRONG_TIME in result(fixture, with_binding(s, replace(s.supports[0], statement=statement))).reasons


def test_uncertain_scope_is_not_silently_unconditional(fixture):
    s = fixture[0]
    statement = replace(s.assertions[0].statement, conditions=Conditions(ConditionKind.UNKNOWN))
    assert Reason.UNKNOWN_SCOPE in result(fixture, with_binding(s, replace(s.supports[0], statement=statement))).reasons


@pytest.mark.parametrize("lifecycle,state", [(Lifecycle.CANDIDATE, State.PENDING),
                                             (Lifecycle.QUARANTINED, State.QUARANTINED),
                                             (Lifecycle.REJECTED, State.REJECTED)])
def test_unverified_quarantined_and_rejected_claims_cannot_promote(fixture, lifecycle, state):
    s = fixture[0]
    s = replace(s, assertions=(replace(s.assertions[0], lifecycle=lifecycle),))
    assert result(fixture, s).state is state
    assert not retrievable_assertions(s, fixture[1], fixture[2])


def test_review_hash_covers_claim_wording_and_bindings(fixture):
    s = fixture[0]
    changed = replace(s, assertions=(replace(s.assertions[0], text="Invented paraphrase"),))
    assert result(fixture, changed).state is State.PENDING
    time = replace(s.assertions[0].statement.time, period="2025")
    statement = replace(s.assertions[0].statement, time=time, quantity=replace(s.assertions[0].statement.quantity, time=time))
    binding = replace(s.supports[0], statement=statement)
    # Changing both the model tuple and the source annotation still invalidates
    # the prior manual attestation; it cannot manufacture semantic verification.
    claim = replace(s.assertions[0], statement=statement, support_id=binding.id)
    changed = replace(s, supports=(binding,), assertions=(claim,))
    assert not result(fixture, changed).retrievable


def test_unsupported_relationship_or_unknown_predicate(fixture):
    s = fixture[0]
    statement = replace(s.assertions[0].statement, predicate=Predicate.ROLE,
                        object_id=next(e.id for e in s.entities if e.kind is Kind.ACTION), quantity=None)
    assert Reason.UNSUPPORTED_RELATION in result(fixture, with_binding(s, replace(s.supports[0], statement=statement))).reasons
    payload = json.loads(json.dumps(asdict(s)))
    payload["assertions"][0]["statement"]["predicate"] = "invented_relation"
    with pytest.raises(ContractError):
        snapshot_from_dict(payload)


def test_duplicate_ids_do_not_duplicate_retrievable_evidence(fixture):
    s, bodies, proofs = fixture
    duplicated = replace(s, assertions=s.assertions * 2)
    checks = validate_snapshot(duplicated, bodies, proofs)
    assert all(c.state is State.DUPLICATE for c in checks.assertions)
    assert not retrievable_assertions(duplicated, bodies, proofs)


def conflicting_snapshot(s):
    start = BODY.index(SECOND)
    span = Span(s.passages[0].id, start, start + len(SECOND), sha256(SECOND))
    statement = replace(s.assertions[0].statement,
                        quantity=replace(s.assertions[0].statement.quantity, surface="60%", canonical="pct:60"))
    binding = SupportBinding(statement, tuple(SlotEvidence(slot, span.id) for slot in
                                             ("subject", "predicate", "quantity", "metric", "time")))
    claim = Assertion(statement, SECOND, binding.id)
    return sign_claims(replace(s, spans=s.spans + (span,), supports=s.supports + (binding,), assertions=s.assertions + (claim,)))


def test_conflicting_values_are_found_across_the_snapshot(fixture):
    s, bodies, proofs = fixture
    s = conflicting_snapshot(s)
    assert all(c.state is State.CONFLICT for c in validate_snapshot(s, bodies, proofs).assertions)
    assert not retrievable_assertions(s, bodies, proofs)


def test_explicit_conflict_ledger_and_dangling_conflict_fail_closed(fixture):
    s, bodies, proofs = fixture
    s = conflicting_snapshot(s)
    s = replace(s, conflicts=(Conflict(tuple(c.id for c in s.assertions), "reviewer flagged inconsistency"),))
    assert all(c.state is State.CONFLICT for c in validate_snapshot(s, bodies, proofs).assertions)
    with pytest.raises(ContractError):
        validate_snapshot(replace(s, conflicts=(Conflict((s.assertions[0].id, "missing"), "unknown"),)), bodies, proofs)


def test_admission_decisions_are_content_bound_append_only_records(fixture):
    decision = fixture[0].decisions[0]
    history = append_decision((), decision)
    revoked = replace(decision, status=AdmissionState.QUARANTINED, previous_decision_id=decision.id,
                      reasons=(Reason.CONFLICTING_AUTHORS,), review=None)
    new_history = append_decision(history, revoked)
    assert history == (decision,)
    assert new_history == (decision, revoked)
    with pytest.raises(ContractError):
        append_decision(history, decision)
    with pytest.raises(FrozenInstanceError):
        decision.status = AdmissionState.ADMITTED


def test_review_records_require_reviewer_time_and_reference(fixture):
    s = fixture[0]
    for changes in ({"reviewer": None}, {"reviewed_at": "2026-11-08"}, {"reference": None}):
        claim = replace(s.assertions[0], review=replace(s.assertions[0].review, **changes))
        assert result(fixture, replace(s, assertions=(claim,))).state is State.PENDING


def test_ambiguous_aliases_fail_closed(fixture):
    s, bodies, proofs = fixture
    beta = next(e for e in s.entities if e.canonical_name == "Beta")
    entities = tuple(replace(e, aliases=("Acme",)) if e.id == beta.id else e for e in s.entities)
    with pytest.raises(ContractError, match="ambiguous_alias"):
        validate_snapshot(replace(s, entities=entities), bodies, proofs)


def test_conditional_author_advice_preserves_conditions_without_name_in_prose(fixture):
    s, bodies, proofs = fixture
    author = next(e for e in s.entities if e.canonical_name == AUTHOR)
    action = next(e for e in s.entities if e.kind is Kind.ACTION)
    acme = next(e for e in s.entities if e.canonical_name == "Acme")
    text = "I recommend hiring carefully if Acme growth is below 10%."
    start = BODY.index(text)
    span = Span(s.passages[0].id, start, len(BODY), sha256(text))
    conditions = Conditions(ConditionKind.EXPLICIT, "all", (Condition(acme.id, "growth", Comparator.LT, "10", "percent"),))
    statement = Statement(author.id, Predicate.RECOMMENDS, author.id, object_id=action.id,
                          conditions=conditions, modality=Modality.QUALIFIED)
    binding = SupportBinding(statement, tuple(SlotEvidence(slot, span.id) for slot in
                                             ("predicate", "object", "conditions", "modality")))
    claim = Assertion(statement, "I recommend hiring carefully if Acme growth is below 10%.", binding.id)
    s = sign_claims(replace(s, spans=(span,), supports=(binding,), assertions=(claim,)))
    assert validate_snapshot(s, bodies, proofs).assertions[0].retrievable
    # Removing a condition while retaining the reviewed binding cannot promote.
    claim = replace(s.assertions[0], statement=replace(statement, conditions=Conditions()))
    assert not retrievable_assertions(replace(s, assertions=(claim,)), bodies, proofs)


@pytest.mark.parametrize("surface,unit,expected", [
    ("$1.50", "usd", "usd:1.5"), ("$5m", "usd", "usd:5000000"),
    ("$123456789012345678901234567890.12", "usd", "usd:123456789012345678901234567890.12"),
    ("10", "n", "n:10"), ("0.5", "ratio", "ratio:0.5"),
    ("50%", "pct", "pct:50"), ("2024", "year", None),
    ("1.5", "n", None), ("a lot", "usd", None), ("$1,23", "usd", None),
])
def test_exact_supported_quantity_units(surface, unit, expected):
    assert quantity_canonical(surface, unit) == expected


def test_registered_pilot_manifest_is_pending_not_verified():
    from pathlib import Path
    from host_finetune.train_only_index import load_jsonl
    root = Path(__file__).resolve().parents[1]
    path = root / "experiments/EXP-20261108-004-ont-01-schema-admission/raw/pilot_admission.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    snapshot = snapshot_from_dict(manifest["snapshot"])
    records = load_jsonl(root / "data/cleaned/jasonlemkin_blog.jsonl")
    bodies = {d.id: records[d.provenance.source_line - 1]["content"] for d in snapshot.documents}
    validation = validate_snapshot(snapshot, bodies)
    assert len(snapshot.documents) == 5
    assert all(r.state is State.PENDING for r in validation.documents)
    assert all(d.review.status is ReviewState.PENDING and d.review.reviewer is None for d in snapshot.decisions)
    assert all(not d.author and not d.authorship_proof_sha256 for d in snapshot.decisions)
    assert not snapshot.assertions and not retrievable_assertions(snapshot, bodies)
    assert all(not link["manual_approval_recorded"] for link in manifest["source_links"])
