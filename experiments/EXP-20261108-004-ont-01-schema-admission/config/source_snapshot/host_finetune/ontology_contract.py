"""ONT-01: immutable, CPU-only evidence contracts. No retrieval/model/storage IO.

Exact spans establish identity, not entailment. Human attestations are trusted
inputs bound to content hashes; this module cannot authenticate a reviewer or
infer semantic company/quantity associations from prose. Real pilot approvals
must remain pending until performed. Only validate_snapshot computes eligibility.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, dataclass, fields, is_dataclass
from datetime import datetime
from decimal import Decimal, localcontext
from enum import Enum
import hashlib
import json
import re
import types
from typing import Mapping, Union, get_args, get_origin, get_type_hints

VERSION = "author_claim.v1"
AUTHOR = "Jason Lemkin"


class ContractError(ValueError):
    pass


class State(str, Enum):
    PENDING = "pending_review"
    ELIGIBLE = "eligible"
    QUARANTINED = "quarantined"
    REJECTED = "rejected"
    DUPLICATE = "duplicate"
    CONFLICT = "conflict"


class AdmissionState(str, Enum):
    PENDING = "pending_review"
    ADMITTED = "admitted"
    QUARANTINED = "quarantined"
    REJECTED = "rejected"


class ReviewState(str, Enum):
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"


class Lifecycle(str, Enum):
    CANDIDATE = "candidate"
    REVIEWED = "reviewed"
    QUARANTINED = "quarantined"
    REJECTED = "rejected"


class Kind(str, Enum):
    PERSON = "person"
    ORGANIZATION = "organization"
    PRODUCT = "product"
    ROLE = "role"
    ACTION = "action"
    CONCEPT = "concept"


class Predicate(str, Enum):
    METRIC = "has_metric"
    ROLE = "has_role"
    RECOMMENDS = "recommends"
    DISCOURAGES = "discourages"


class Polarity(str, Enum):
    POSITIVE = "positive"
    NEGATIVE = "negative"


class Modality(str, Enum):
    ASSERTED = "asserted"
    QUALIFIED = "qualified"
    POSSIBLE = "possible"


class TimeKind(str, Enum):
    EXPLICIT = "explicit"
    UNDATED = "not_stated"
    UNKNOWN = "unknown"


class ConditionKind(str, Enum):
    EXPLICIT = "explicit"
    NONE = "not_stated"
    UNKNOWN = "unknown"


class Comparator(str, Enum):
    EQ = "eq"
    LT = "lt"
    LE = "le"
    GT = "gt"
    GE = "ge"


class Reason(str, Enum):
    INVALID_RECORD = "invalid_record"
    WRONG_VERSION = "wrong_version"
    INVALID_IDENTITY = "invalid_identity"
    EXCLUDED_SOURCE = "excluded_source"
    INELIGIBLE_LINEAGE = "ineligible_lineage"
    BODY_HASH = "body_hash_mismatch"
    MISSING_ATTRIBUTION = "missing_attribution"
    CONFLICTING_AUTHORS = "conflicting_author_signals"
    STALE_REVIEW = "missing_or_stale_review"
    PENDING_REVIEW = "human_review_pending"
    NOT_ADMITTED = "document_not_admitted"
    INVALID_OFFSETS = "invalid_passage_or_span_offsets"
    SPAN_HASH = "unsupported_source_span"
    MISSING_ENTITY = "missing_entity"
    AMBIGUOUS_ALIAS = "ambiguous_alias"
    MISSING_BINDING = "missing_assertion_binding"
    WRONG_SUBJECT = "wrong_company_or_subject_binding"
    WRONG_TIME = "wrong_temporal_scope"
    UNSUPPORTED_RELATION = "unsupported_relationship"
    UNSUPPORTED_QUANTITY = "unsupported_quantity"
    UNSUPPORTED_BINDING = "unsupported_binding"
    UNKNOWN_SCOPE = "uncertain_condition_or_time"
    QUARANTINED = "quarantined_claim"
    REJECTED = "rejected_claim"
    DUPLICATE = "duplicate_assertion"
    CONFLICT = "unresolved_conflicting_assertions"
    INVALID_LEDGER = "invalid_admission_history"


def sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def fingerprint(value) -> str:
    payload = asdict(value) if is_dataclass(value) else value
    return sha256(json.dumps(payload, sort_keys=True, ensure_ascii=True, allow_nan=False, separators=(",", ":")))


def identity(namespace: str, value) -> str:
    return namespace + ":" + fingerprint({"schema": VERSION, "value": asdict(value) if is_dataclass(value) else value})


@dataclass(frozen=True)
class Provenance:
    source_file: str
    source_line: int
    raw_file_sha256: str
    cleaned_file_sha256: str
    cleaned_record_sha256: str
    dataset_sha256: str
    split_sha256: str
    gold_sha256: str
    selection_code_sha256: str
    row_ids: tuple[int, ...]
    group_ids: tuple[str, ...]
    split: str = "train"
    selection_eligible: bool = False


@dataclass(frozen=True)
class Document:
    provenance: Provenance
    source_url: str
    body_sha256: str
    publication_date: str | None = None
    author_signals: tuple[str, ...] = ()
    content_kind: str = "blog"
    body_locator: str = "cleaned.content.unmodified"

    @property
    def id(self) -> str:
        return identity("document", (self.provenance.source_file, self.provenance.cleaned_file_sha256,
                                     self.provenance.source_line, self.body_sha256, self.body_locator))


@dataclass(frozen=True)
class Review:
    subject_sha256: str
    status: ReviewState = ReviewState.PENDING
    reviewer: str | None = None
    reviewed_at: str | None = None
    reference: str | None = None


@dataclass(frozen=True)
class AdmissionDecision:
    document_id: str
    document_sha256: str
    status: AdmissionState = AdmissionState.PENDING
    author: str | None = None
    authorship_proof_sha256: str | None = None
    proof_source_url: str | None = None
    review: Review | None = None
    reasons: tuple[Reason, ...] = ()
    previous_decision_id: str | None = None

    @property
    def id(self) -> str:
        return identity("admission", self)


def admission_subject(document: Document, decision: AdmissionDecision) -> str:
    payload = asdict(decision)
    payload.pop("review")
    return fingerprint({"document": asdict(document), "decision": payload})


def append_decision(history: tuple[AdmissionDecision, ...], decision: AdmissionDecision) -> tuple[AdmissionDecision, ...]:
    """Return a new chain. Existing frozen decisions are never overwritten."""
    prior = [d for d in history if d.document_id == decision.document_id]
    expected = prior[-1].id if prior else None
    if decision.previous_decision_id != expected or any(d.id == decision.id for d in history):
        raise ContractError(Reason.INVALID_LEDGER.value)
    return history + (decision,)


@dataclass(frozen=True)
class Passage:
    document_id: str
    start: int
    end: int
    text_sha256: str

    @property
    def id(self) -> str:
        return identity("passage", self)


@dataclass(frozen=True)
class Span:
    passage_id: str
    start: int
    end: int
    text_sha256: str

    @property
    def id(self) -> str:
        return identity("span", self)


@dataclass(frozen=True)
class Entity:
    kind: Kind
    canonical_name: str
    aliases: tuple[str, ...] = ()

    @property
    def id(self) -> str:
        return identity("entity", (self.kind, self.canonical_name.casefold()))


@dataclass(frozen=True)
class TimeScope:
    kind: TimeKind = TimeKind.UNDATED
    period: str | None = None


@dataclass(frozen=True)
class Condition:
    subject_id: str
    attribute: str
    comparator: Comparator
    value: str
    unit: str | None = None


@dataclass(frozen=True)
class Conditions:
    kind: ConditionKind = ConditionKind.NONE
    operator: str = "all"
    clauses: tuple[Condition, ...] = ()


@dataclass(frozen=True)
class Quantity:
    subject_id: str
    metric: str
    surface: str
    canonical: str
    time: TimeScope
    basis: str


@dataclass(frozen=True)
class Statement:
    subject_id: str
    predicate: Predicate
    speaker_id: str
    object_id: str | None = None
    organization_id: str | None = None
    quantity: Quantity | None = None
    time: TimeScope = TimeScope()
    conditions: Conditions = Conditions()
    polarity: Polarity = Polarity.POSITIVE
    modality: Modality = Modality.ASSERTED
    scope: tuple[tuple[str, str], ...] = ()


@dataclass(frozen=True)
class SlotEvidence:
    slot: str
    span_id: str


@dataclass(frozen=True)
class SupportBinding:
    """Independently annotated source binding, not a model truth assertion."""
    statement: Statement
    evidence: tuple[SlotEvidence, ...]

    @property
    def id(self) -> str:
        return identity("support", self)


@dataclass(frozen=True)
class Assertion:
    statement: Statement
    text: str
    support_id: str
    lifecycle: Lifecycle = Lifecycle.CANDIDATE
    review: Review | None = None

    @property
    def id(self) -> str:
        # State/review changes do not change the semantic assertion identity.
        return identity("assertion", (asdict(self.statement), self.text, self.support_id))


@dataclass(frozen=True)
class Conflict:
    assertion_ids: tuple[str, ...]
    reason: str
    # ONT-01 deliberately does not implement conflict adjudication/promotion.


@dataclass(frozen=True)
class Snapshot:
    documents: tuple[Document, ...] = ()
    decisions: tuple[AdmissionDecision, ...] = ()
    passages: tuple[Passage, ...] = ()
    spans: tuple[Span, ...] = ()
    entities: tuple[Entity, ...] = ()
    supports: tuple[SupportBinding, ...] = ()
    assertions: tuple[Assertion, ...] = ()
    conflicts: tuple[Conflict, ...] = ()
    schema_version: str = VERSION


@dataclass(frozen=True)
class Eligibility:
    record_id: str
    state: State
    reasons: tuple[Reason, ...] = ()

    @property
    def retrievable(self) -> bool:
        return self.state is State.ELIGIBLE


@dataclass(frozen=True)
class Validation:
    documents: tuple[Eligibility, ...]
    assertions: tuple[Eligibility, ...]


def _decode(annotation, value, strict=False):
    """Strict typed JSON reader; no coercion, unknown keys, or mutable lists."""
    origin, args = get_origin(annotation), get_args(annotation)
    if origin in (Union, types.UnionType):
        for option in args:
            try:
                return _decode(option, value, strict)
            except (ContractError, ValueError, TypeError):
                pass
        raise ContractError("invalid optional/union value")
    if annotation is type(None):
        if value is not None:
            raise ContractError("expected null")
        return None
    if origin is tuple:
        if not isinstance(value, (tuple, list)):
            raise ContractError("expected array")
        if strict and type(value) is not tuple:
            raise ContractError("mutable array in immutable contract")
        if len(args) == 2 and args[1] is Ellipsis:
            return tuple(_decode(args[0], v, strict) for v in value)
        if len(value) != len(args):
            raise ContractError("wrong tuple length")
        return tuple(_decode(t, v, strict) for t, v in zip(args, value))
    if isinstance(annotation, type) and issubclass(annotation, Enum):
        if strict and type(value) is not annotation:
            raise ContractError("untyped enum in immutable contract")
        if not isinstance(value, str):
            raise ContractError("enum must be string")
        return annotation(value)
    if is_dataclass(annotation):
        if not isinstance(value, dict):
            raise ContractError("expected record")
        names = {f.name for f in fields(annotation)}
        if set(value) - names:
            raise ContractError("unknown fields: " + str(sorted(set(value) - names)))
        hints = get_type_hints(annotation)
        return annotation(**{k: _decode(hints[k], v, strict) for k, v in value.items()})
    if type(value) is not annotation:
        raise ContractError(f"expected {annotation}, got {type(value)}")
    return value


def snapshot_from_dict(payload: dict) -> Snapshot:
    try:
        return _decode(Snapshot, payload)
    except (TypeError, ValueError) as exc:
        raise ContractError(str(exc)) from exc


def _hash(value: str | None) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[a-f0-9]{64}", value) is not None


def quantity_canonical(surface: str, unit: str) -> str | None:
    """Exact pilot units; unsupported formats fail closed, without rounding.

    The historical diagnostic parser rounds money to integral USD and ignores
    small counts, so it is not an exact-value contract. It remains unchanged.
    Unit/basis and semantic associations still require reviewed source bindings.
    """
    number = r"([+-]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?)"
    scale = Decimal(1)
    if unit == "usd":
        match = re.fullmatch(r"\$\s*" + number + r"\s*(k|m|b|thousand|million|billion)?", surface, re.I)
        if match:
            scale = Decimal({None: 1, "k": 1000, "thousand": 1000, "m": 1000000,
                             "million": 1000000, "b": 1000000000, "billion": 1000000000}[(match[2] or "").lower() or None])
    elif unit == "pct":
        match = re.fullmatch(number + r"\s*(?:%|percent)", surface, re.I)
    elif unit in ("n", "ratio"):
        match = re.fullmatch(number, surface)
    else:
        return None
    if not match:
        return None
    digits = match[1].replace(",", "")
    with localcontext() as context:
        context.prec = max(28, len(digits) + 12)
        value = Decimal(digits) * scale
    if unit == "n" and value != value.to_integral_value():
        return None
    rendered = format(value, "f")
    if "." in rendered:
        rendered = rendered.rstrip("0").rstrip(".")
    return unit + ":" + rendered


def _approved(review: Review | None, subject: str) -> bool:
    if not review or review.status is not ReviewState.APPROVED or review.subject_sha256 != subject:
        return False
    try:
        date = datetime.fromisoformat(review.reviewed_at or "")
    except ValueError:
        return False
    return bool(review.reviewer and review.reviewer.strip() and review.reference and review.reference.strip()
                and date.tzinfo is not None and _hash(subject))


def validate_admission(document: Document, decision: AdmissionDecision, body: str,
                       proofs: Mapping[str, str] | None = None) -> Eligibility:
    _decode(Document, asdict(document), strict=True)
    _decode(AdmissionDecision, asdict(decision), strict=True)
    p = document.provenance
    errors = []
    if decision.document_id != document.id or decision.document_sha256 != fingerprint(document):
        errors.append(Reason.INVALID_IDENTITY)
    hashes = (p.raw_file_sha256, p.cleaned_file_sha256, p.cleaned_record_sha256,
              p.dataset_sha256, p.split_sha256, p.gold_sha256, p.selection_code_sha256, document.body_sha256)
    if not all(_hash(h) for h in hashes) or p.source_line < 1 or not p.source_file or not document.source_url:
        errors.append(Reason.INVALID_IDENTITY)
    if not body.strip() or sha256(body) != document.body_sha256:
        errors.append(Reason.BODY_HASH)
    if (p.split != "train" or not p.selection_eligible or not p.row_ids or not p.group_ids
            or any(i < 0 for i in p.row_ids) or len(set(p.row_ids)) != len(p.row_ids)
            or len(set(p.group_ids)) != len(p.group_ids) or not all(_hash(g) for g in p.group_ids)):
        errors.append(Reason.INELIGIBLE_LINEAGE)
    if document.content_kind != "blog":
        errors.append(Reason.EXCLUDED_SOURCE)
    if any(a.strip().casefold() != AUTHOR.casefold() for a in document.author_signals):
        errors.append(Reason.CONFLICTING_AUTHORS)
    if errors:
        return Eligibility(document.id, State.REJECTED, tuple(dict.fromkeys(errors)))
    if decision.status is AdmissionState.REJECTED:
        return Eligibility(document.id, State.REJECTED, (Reason.REJECTED,))
    if decision.status is AdmissionState.QUARANTINED:
        return Eligibility(document.id, State.QUARANTINED, (Reason.QUARANTINED,))
    if decision.status is AdmissionState.PENDING:
        return Eligibility(document.id, State.PENDING, (Reason.PENDING_REVIEW,))
    proof = (proofs or {}).get(decision.authorship_proof_sha256 or "")
    if (decision.author != AUTHOR or not proof or sha256(proof) != decision.authorship_proof_sha256
            or AUTHOR.casefold() not in proof.casefold() or decision.proof_source_url != document.source_url):
        return Eligibility(document.id, State.QUARANTINED, (Reason.MISSING_ATTRIBUTION,))
    if not _approved(decision.review, admission_subject(document, decision)):
        return Eligibility(document.id, State.PENDING, (Reason.STALE_REVIEW,))
    return Eligibility(document.id, State.ELIGIBLE)


def _indexed(records) -> dict:
    result = {}
    for record in records:
        if record.id in result:
            raise ContractError("duplicate record ID: " + record.id)
        result[record.id] = record
    return result


def _span_text(span: Span, passages: dict, documents: dict, bodies: Mapping[str, str]) -> str:
    passage = passages.get(span.passage_id)
    if not passage or passage.document_id not in documents:
        raise ContractError(Reason.INVALID_OFFSETS.value)
    body = bodies.get(passage.document_id)
    if body is None or sha256(body) != documents[passage.document_id].body_sha256:
        raise ContractError(Reason.BODY_HASH.value)
    if not (type(passage.start) is int and type(passage.end) is int
            and 0 <= passage.start < passage.end <= len(body)):
        raise ContractError(Reason.INVALID_OFFSETS.value)
    if sha256(body[passage.start:passage.end]) != passage.text_sha256:
        raise ContractError(Reason.SPAN_HASH.value)
    if not (type(span.start) is int and type(span.end) is int
            and passage.start <= span.start < span.end <= passage.end):
        raise ContractError(Reason.INVALID_OFFSETS.value)
    text = body[span.start:span.end]
    if sha256(text) != span.text_sha256:
        raise ContractError(Reason.SPAN_HASH.value)
    return text


def assertion_subject(assertion: Assertion, snapshot: Snapshot) -> str:
    """Bind manual entailment approval to assertion text AND source-slot context.

    Includes the transitive support closure, not unrelated source/claim records.
    Appending a different source therefore does not invalidate existing reviews.
    """
    supports = [b for b in snapshot.supports if b.id == assertion.support_id]
    span_ids = {e.span_id for b in supports for e in b.evidence}
    spans = [s for s in snapshot.spans if s.id in span_ids]
    passage_ids = {s.passage_id for s in spans}
    passages = [p for p in snapshot.passages if p.id in passage_ids]
    doc_ids = {p.document_id for p in passages}
    decisions = {d.document_id: d for d in snapshot.decisions if d.document_id in doc_ids}
    statements = [assertion.statement] + [s.statement for s in supports]
    entity_ids = {k for s in statements for k in (s.subject_id, s.speaker_id, s.object_id, s.organization_id)}
    entity_ids |= {c.subject_id for s in statements for c in s.conditions.clauses}
    payload = {"schema": snapshot.schema_version,
               "documents": [asdict(d) for d in snapshot.documents if d.id in doc_ids],
               "decisions": [asdict(decisions[k]) for k in sorted(decisions)],
               "passages": [asdict(p) for p in passages], "spans": [asdict(s) for s in spans],
               "supports": [asdict(s) for s in supports],
               "entities": [asdict(e) for e in snapshot.entities if e.id in entity_ids]}
    claim = asdict(assertion)
    claim.pop("review")
    return fingerprint({"schema": VERSION, "claim": claim, "evidence_registry": payload})


def _statement_errors(statement: Statement, entities: dict) -> list[Reason]:
    errors = []
    subject, speaker = entities.get(statement.subject_id), entities.get(statement.speaker_id)
    obj = entities.get(statement.object_id)
    if not subject or not speaker or speaker.kind is not Kind.PERSON or speaker.canonical_name != AUTHOR:
        errors.append(Reason.MISSING_ENTITY)
    if (statement.object_id and not obj) or (statement.organization_id and statement.organization_id not in entities):
        errors.append(Reason.MISSING_ENTITY)
    if statement.time.kind is TimeKind.UNKNOWN or statement.conditions.kind is ConditionKind.UNKNOWN:
        errors.append(Reason.UNKNOWN_SCOPE)
    if statement.time.kind is TimeKind.EXPLICIT:
        period = statement.time.period or ""
        valid = bool(re.fullmatch(r"\d{4}(?:-Q[1-4]|-\d{2}(?:-\d{2})?)?", period)) and period[:4] != "0000"
        if valid and "-Q" not in period:
            try:
                datetime.fromisoformat(period + ("-01-01" if len(period) == 4 else "-01" if len(period) == 7 else ""))
            except ValueError:
                valid = False
        if not valid:
            errors.append(Reason.WRONG_TIME)
    elif statement.time.period is not None:
        errors.append(Reason.WRONG_TIME)
    c = statement.conditions
    if c.operator not in ("all", "any") or ((c.kind is ConditionKind.EXPLICIT) != bool(c.clauses)):
        errors.append(Reason.MISSING_BINDING)
    for clause in c.clauses:
        if (clause.subject_id not in entities or not clause.attribute or not clause.value
                or not isinstance(clause.comparator, Comparator)):
            errors.append(Reason.MISSING_BINDING)
    if len(dict(statement.scope)) != len(statement.scope) or any(not k or not v for k, v in statement.scope):
        errors.append(Reason.MISSING_BINDING)
    if statement.predicate is Predicate.METRIC:
        q = statement.quantity
        if not subject or subject.kind not in (Kind.ORGANIZATION, Kind.PRODUCT) or q is None or statement.object_id is not None:
            errors.append(Reason.MISSING_BINDING)
        if q:
            if q.subject_id != statement.subject_id:
                errors.append(Reason.WRONG_SUBJECT)
            if q.time != statement.time or statement.time.kind is not TimeKind.EXPLICIT:
                errors.append(Reason.WRONG_TIME)
            if (quantity_canonical(q.surface, q.canonical.split(":", 1)[0]) != q.canonical
                    or not q.metric.strip() or not q.basis.strip()):
                errors.append(Reason.UNSUPPORTED_QUANTITY)
    elif statement.predicate is Predicate.ROLE:
        org = entities.get(statement.organization_id)
        if (not subject or subject.kind is not Kind.PERSON or not obj or obj.kind is not Kind.ROLE
                or not org or org.kind is not Kind.ORGANIZATION or statement.quantity is not None):
            errors.append(Reason.UNSUPPORTED_RELATION)
    elif statement.predicate in (Predicate.RECOMMENDS, Predicate.DISCOURAGES):
        if (not subject or subject.id != statement.speaker_id or not obj or obj.kind is not Kind.ACTION
                or statement.quantity is not None):
            errors.append(Reason.UNSUPPORTED_RELATION)
    else:
        errors.append(Reason.UNSUPPORTED_RELATION)
    return errors


def validate_snapshot(snapshot: Snapshot, bodies: Mapping[str, str],
                      proofs: Mapping[str, str] | None = None) -> Validation:
    # Also rejects mutable lists, unknown enums and type-coercion in direct
    # dataclass construction. JSON callers use the same strict contract.
    if _decode(Snapshot, asdict(snapshot), strict=True) != snapshot:
        raise ContractError(Reason.INVALID_RECORD.value)
    if snapshot.schema_version != VERSION:
        raise ContractError(Reason.WRONG_VERSION.value)
    docs, passages, spans = map(_indexed, (snapshot.documents, snapshot.passages, snapshot.spans))
    entities, supports = map(_indexed, (snapshot.entities, snapshot.supports))
    history = ()
    decisions = {}
    for decision in snapshot.decisions:
        if decision.document_id not in docs:
            raise ContractError(Reason.INVALID_LEDGER.value)
        history = append_decision(history, decision)
        decisions[decision.document_id] = decision
    doc_results = {}
    for doc in snapshot.documents:
        decision = decisions.get(doc.id)
        decision = decision or AdmissionDecision(doc.id, fingerprint(doc))
        doc_results[doc.id] = validate_admission(doc, decision, bodies.get(doc.id, ""), proofs)
    # Orphan/invalid passage contracts fail even with no extracted claims.
    for passage in snapshot.passages:
        _span_text(Span(passage.id, passage.start, passage.end, passage.text_sha256), passages, docs, bodies)
    for span in snapshot.spans:
        _span_text(span, passages, docs, bodies)
    aliases = {}
    for entity in snapshot.entities:
        if not entity.canonical_name.strip():
            raise ContractError(Reason.MISSING_ENTITY.value)
        for name in (entity.canonical_name,) + entity.aliases:
            key = (entity.kind, name.strip().casefold())
            if not key[1] or (key in aliases and aliases[key] != entity.id):
                raise ContractError(Reason.AMBIGUOUS_ALIAS.value)
            aliases[key] = entity.id
    results = []
    allowed_slots = {"subject", "predicate", "object", "speaker", "time", "quantity", "metric",
                     "conditions", "scope", "polarity", "modality", "organization"}
    for claim in snapshot.assertions:
        reasons = []
        if claim.lifecycle is Lifecycle.REJECTED:
            results.append(Eligibility(claim.id, State.REJECTED, (Reason.REJECTED,)))
            continue
        if claim.lifecycle is Lifecycle.QUARANTINED:
            results.append(Eligibility(claim.id, State.QUARANTINED, (Reason.QUARANTINED,)))
            continue
        binding = supports.get(claim.support_id)
        if not binding or not claim.text.strip():
            results.append(Eligibility(claim.id, State.REJECTED, (Reason.MISSING_BINDING,)))
            continue
        reasons += _statement_errors(claim.statement, entities)
        if claim.statement.subject_id != binding.statement.subject_id:
            reasons.append(Reason.WRONG_SUBJECT)
        if claim.statement.time != binding.statement.time:
            reasons.append(Reason.WRONG_TIME)
        if claim.statement != binding.statement:
            reasons.append(Reason.UNSUPPORTED_BINDING)
        slot_texts = defaultdict(list)
        source_ids = set()
        for anchor in binding.evidence:
            span = spans.get(anchor.span_id)
            if anchor.slot not in allowed_slots or not span:
                reasons.append(Reason.SPAN_HASH)
                continue
            text = _span_text(span, passages, docs, bodies)
            slot_texts[anchor.slot].append(text)
            source_ids.add(passages[span.passage_id].document_id)
        # MVP assertion evidence comes from one source; no stitched inference.
        if len(source_ids) != 1:
            reasons.append(Reason.UNSUPPORTED_BINDING)
        if any(not doc_results[source].retrievable for source in source_ids):
            reasons.append(Reason.NOT_ADMITTED)
        # The byline capture/admission review supports the document-author
        # identity; requiring the author's name in every prose span would reject
        # ordinary first-person advice. Claim review still attests endorsement.
        required = {"predicate"}
        s = claim.statement
        if s.predicate not in (Predicate.RECOMMENDS, Predicate.DISCOURAGES):
            required.add("subject")
        if s.object_id:
            required.add("object")
        if s.organization_id:
            required.add("organization")
        if s.quantity:
            required |= {"quantity", "metric", "time"}
            if s.quantity.surface not in " ".join(slot_texts["quantity"]):
                reasons.append(Reason.UNSUPPORTED_QUANTITY)
        if s.time.kind is TimeKind.EXPLICIT:
            required.add("time")
        if s.conditions.kind is ConditionKind.EXPLICIT:
            required.add("conditions")
            condition_text = " ".join(slot_texts["conditions"]).casefold()
            for clause in s.conditions.clauses:
                entity = entities.get(clause.subject_id)
                if (clause.attribute.casefold() not in condition_text or clause.value.casefold() not in condition_text
                        or not entity or not any(n.casefold() in condition_text for n in (entity.canonical_name,) + entity.aliases)):
                    reasons.append(Reason.UNSUPPORTED_BINDING)
        if s.scope:
            required.add("scope")
        if s.polarity is Polarity.NEGATIVE:
            required.add("polarity")
        if s.modality is not Modality.ASSERTED:
            required.add("modality")
        if not required <= slot_texts.keys():
            reasons.append(Reason.MISSING_BINDING)
        for slot, entity_id in (("subject", s.subject_id), ("speaker", s.speaker_id),
                                ("object", s.object_id), ("organization", s.organization_id)):
            entity = entities.get(entity_id)
            if entity:
                if (slot == "speaker" and slot not in slot_texts) or (
                    slot == "subject" and s.predicate in (Predicate.RECOMMENDS, Predicate.DISCOURAGES)
                    and entity_id == s.speaker_id and slot not in slot_texts
                ):
                    continue
                text = " ".join(slot_texts[slot]).casefold()
                if not any(name.casefold() in text for name in (entity.canonical_name,) + entity.aliases):
                    reasons.append(Reason.UNSUPPORTED_BINDING)
        if reasons:
            results.append(Eligibility(claim.id, State.QUARANTINED, tuple(dict.fromkeys(reasons))))
        elif claim.lifecycle is not Lifecycle.REVIEWED or not _approved(claim.review, assertion_subject(claim, snapshot)):
            results.append(Eligibility(claim.id, State.PENDING, (Reason.STALE_REVIEW,)))
        else:
            results.append(Eligibility(claim.id, State.ELIGIBLE))
    # Global conflict inspection precedes any future top-k or token truncation.
    by_id = defaultdict(list)
    semantic = defaultdict(list)
    conflict_groups = defaultdict(list)
    for i, claim in enumerate(snapshot.assertions):
        by_id[claim.id].append(i)
        if not results[i].retrievable:
            continue
        statement = asdict(claim.statement)
        semantic[fingerprint(statement)].append(i)
        if statement["quantity"]:
            statement["quantity"] = {k: v for k, v in statement["quantity"].items() if k not in ("surface", "canonical")}
        statement.pop("polarity")
        if claim.statement.predicate in (Predicate.RECOMMENDS, Predicate.DISCOURAGES):
            statement["predicate"] = "author_position"
        conflict_groups[fingerprint(statement)].append(i)
    for indices in by_id.values():
        if len(indices) > 1:
            for i in indices:
                results[i] = Eligibility(snapshot.assertions[i].id, State.DUPLICATE, (Reason.DUPLICATE,))
    for indices in semantic.values():
        for i in sorted(indices, key=lambda j: snapshot.assertions[j].id)[1:]:
            results[i] = Eligibility(snapshot.assertions[i].id, State.DUPLICATE, (Reason.DUPLICATE,))
    for indices in conflict_groups.values():
        if len({fingerprint(snapshot.assertions[i].statement) for i in indices}) > 1:
            for i in indices:
                results[i] = Eligibility(snapshot.assertions[i].id, State.CONFLICT, (Reason.CONFLICT,))
    for conflict in snapshot.conflicts:
        if len(set(conflict.assertion_ids)) < 2 or not conflict.reason.strip() or any(k not in by_id for k in conflict.assertion_ids):
            raise ContractError(Reason.CONFLICT.value)
        for key in conflict.assertion_ids:
            for i in by_id[key]:
                if results[i].retrievable or results[i].state is State.DUPLICATE:
                    results[i] = Eligibility(key, State.CONFLICT, (Reason.CONFLICT,))
    return Validation(tuple(doc_results.values()), tuple(results))


def retrievable_assertions(snapshot: Snapshot, bodies: Mapping[str, str],
                          proofs: Mapping[str, str] | None = None) -> tuple[Assertion, ...]:
    """Only computed eligibility, never a caller-provided verified flag."""
    validation = validate_snapshot(snapshot, bodies, proofs)
    return tuple(c for c, result in zip(snapshot.assertions, validation.assertions) if result.retrievable)
