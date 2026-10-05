"""Claim labels against saved retrieval passages and held-out Lemkin evidence.

Retrieval-off drafts do not receive RAG grounding. Held-out evidence support is
a separate field and is not called RAG grounding.
"""
from __future__ import annotations

import math
import re
from collections import Counter

from host_finetune.voice_distance import (
    HELD_OUT_PLATFORM_TO_MEDIUM,
    held_out_reference_texts,
    partition_references,
)
from host_finetune.train_only_index import assignment_rows, blocked_group_ids

TOKEN_RE = re.compile(r"[a-z0-9']+")
QUERY_STOP = {
    "a", "an", "the", "and", "or", "to", "of", "in", "on", "for", "with", "that", "this",
    "it", "is", "are", "was", "were", "be", "as", "at", "by", "from", "we", "i", "you",
    "they", "he", "she", "my", "our", "your", "not", "but", "if", "so", "about", "just",
    "than", "then", "there", "their", "what", "when", "who", "how", "do", "did", "have",
    "has", "had", "will", "can", "could", "would", "should", "its", "im", "were", "dont",
    "its", "also", "into", "over", "after", "before", "because", "which", "them", "his",
    "her", "been", "being", "very", "more", "most", "some", "any", "no", "yes", "one",
}
RAG_LABELS = ("supported", "unsupported", "contradicted", "not_verifiable", "not_applicable")
EVIDENCE_LABELS = ("supported", "contradicted", "not_established")
STANCE_LABELS = ("consistent", "inconsistent", "not_established")
CLAIM_SUPPORT_PASS = 0.5


def claim_prompt(draft: str, topic: str, retrieval_passages: list[str] | None, evidence_passages: list[str]) -> str:
    """Blind claim prompt. It does not name a model or adapter."""
    retrieval = "\n\n".join(retrieval_passages) if retrieval_passages else "NONE"
    evidence = "\n\n".join(evidence_passages) if evidence_passages else "NONE"
    return (
        "Extract up to 5 substantive claims from the DRAFT. Return JSON only.\n"
        "Skip greetings and pure wording. Each claim is fact or advice.\n"
        "rag compares the claim only to RETRIEVAL PASSAGES. "
        "If those passages are NONE, set every rag value to not_applicable.\n"
        "rag values: supported, unsupported, contradicted, not_verifiable, not_applicable.\n"
        "evidence compares the claim only to LEMKIN EVIDENCE. "
        "values: supported, contradicted, not_established.\n"
        "stance is for advice claims against LEMKIN EVIDENCE: "
        "consistent, inconsistent, or not_established. Fact claims use not_established "
        "unless the evidence also shows that position.\n"
        "Topic overlap is not support. Contradicted means a passage states the opposite.\n"
        "Set confidence to your own certainty between 0 and 1.\n"
        '{"claims":[{"text":"...","kind":"advice","rag":"not_applicable",'
        '"evidence":"supported","stance":"consistent","confidence":0.8}]}\n\n'
        f"Topic: {topic}\n\n"
        f"RETRIEVAL PASSAGES:\n{retrieval}\n\n"
        f"LEMKIN EVIDENCE:\n{evidence}\n\n"
        f"DRAFT:\n{draft}\n"
    )


def _label(value, allowed: tuple[str, ...], default: str) -> str:
    if isinstance(value, str) and value.strip().lower() in allowed:
        return value.strip().lower()
    return default


def parse_claims(payload: dict | None, retrieval_on: bool) -> list[dict]:
    """Normalize judge claims. Retrieval-off cannot keep a RAG support label."""
    if not isinstance(payload, dict):
        return []
    raw = payload.get("claims")
    if not isinstance(raw, list):
        return []
    claims = []
    for item in raw[:5]:
        if not isinstance(item, dict):
            continue
        text = str(item.get("text") or "").strip()
        if not text:
            continue
        confidence = item.get("confidence")
        if isinstance(confidence, bool) or not isinstance(confidence, (int, float)):
            confidence = None
        else:
            confidence = max(0.0, min(1.0, float(confidence)))
        rag = _label(item.get("rag"), RAG_LABELS, "not_verifiable")
        if not retrieval_on:
            rag = "not_applicable"
        claims.append({
            "text": text[:400],
            "kind": "advice" if str(item.get("kind") or "").lower() == "advice" else "fact",
            "rag": rag,
            "evidence": _label(item.get("evidence"), EVIDENCE_LABELS, "not_established"),
            "stance": _label(item.get("stance"), STANCE_LABELS, "not_established"),
            "confidence": confidence,
        })
    return claims


def _rate(numerator: int, denominator: int) -> float | None:
    if denominator <= 0:
        return None
    return round(numerator / denominator, 4)


def _confidence(claims: list[dict]) -> float | None:
    values = [claim["confidence"] for claim in claims if claim.get("confidence") is not None]
    if not values:
        return None
    return round(sum(values) / len(values), 4)


def rag_grounding_report(claims: list[dict], retrieval_on: bool) -> dict:
    """Claim-level RAG grounding. Retrieval-off stays empty rather than a zero."""
    if not retrieval_on:
        return {
            "applicable": False,
            "rag_grounding": None,
            "reason": "retrieval_off",
            "claim_support_rate": None,
            "contradiction_rate": None,
            "counts": None,
            "confidence": None,
            "n_claims": 0,
            "pass": True,
        }
    counts = {label: 0 for label in RAG_LABELS}
    for claim in claims:
        counts[claim["rag"]] = counts.get(claim["rag"], 0) + 1
    verifiable = counts["supported"] + counts["unsupported"] + counts["contradicted"]
    support = _rate(counts["supported"], verifiable)
    contradiction = _rate(counts["contradicted"], counts["supported"] + counts["unsupported"] + counts["contradicted"] + counts["not_verifiable"])
    passed = (
        verifiable > 0
        and counts["contradicted"] == 0
        and support is not None
        and support >= CLAIM_SUPPORT_PASS
    )
    return {
        "applicable": True,
        "method": "claim_vs_saved_retrieval_hits",
        "claim_support_rate": support,
        "contradiction_rate": contradiction,
        "counts": counts,
        "confidence": _confidence(claims),
        "n_claims": len(claims),
        "pass": passed,
        "threshold": CLAIM_SUPPORT_PASS,
    }


def evidence_report(claims: list[dict], n_passages: int) -> dict:
    counts = {label: 0 for label in EVIDENCE_LABELS}
    for claim in claims:
        counts[claim["evidence"]] = counts.get(claim["evidence"], 0) + 1
    judged = counts["supported"] + counts["contradicted"]
    return {
        "name": "lemkin_evidence_support",
        "not_rag_grounding": True,
        "claim_support_rate": _rate(counts["supported"], judged),
        "contradiction_rate": _rate(counts["contradicted"], judged + counts["not_established"]),
        "counts": counts,
        "confidence": _confidence(claims),
        "n_claims": len(claims),
        "n_passages": n_passages,
        "corpus": "held-out validation posts; gold-overlap families and style-profile texts excluded",
    }


def stance_report(claims: list[dict]) -> dict:
    advice = [claim for claim in claims if claim["kind"] == "advice"]
    counts = {label: 0 for label in STANCE_LABELS}
    for claim in advice:
        counts[claim["stance"]] = counts.get(claim["stance"], 0) + 1
    judged = counts["consistent"] + counts["inconsistent"]
    return {
        "stance_consistency_rate": _rate(counts["consistent"], judged),
        "counts": counts,
        "n_advice_claims": len(advice),
        "method": "stance_consistency_vs_held_out_evidence",
        "limits": "Framing-rate distance remains a secondary stylometric diagnostic.",
    }


def stance_calibration_gate(real_rate: float | None, contradicted_rate: float | None) -> dict:
    """Real Lemkin prose must be more stance-consistent than an opposed draft."""
    if real_rate is None or contradicted_rate is None:
        return {"calibrated": False, "role": "diagnostic_only", "reason": "stance_controls_missing"}
    passed = real_rate >= contradicted_rate + 0.2
    return {
        "calibrated": passed,
        "role": "selector" if passed else "diagnostic_only",
        "reason": "real_more_consistent_than_opposed_draft" if passed else "stance_judge_failed_sanity_check",
        "real_rate": real_rate,
        "opposed_rate": contradicted_rate,
    }


def _tokens(text: str) -> list[str]:
    return TOKEN_RE.findall((text or "").lower())


class EvidenceIndex:
    """Frozen held-out corpus. Train posts are not inserted."""

    def __init__(self, documents: list[dict]):
        self.documents = documents
        self._postings: dict[str, list[tuple[int, float]]] = {}
        self._norms: list[float] = []
        counts = [Counter(_tokens(doc.get("text") or "")) for doc in documents]
        document_frequency: Counter[str] = Counter()
        for count in counts:
            document_frequency.update(count.keys())
        total = max(len(documents), 1)
        self._idf = {
            term: math.log((total + 1) / (frequency + 1)) + 1
            for term, frequency in document_frequency.items()
        }
        for doc_id, count in enumerate(counts):
            weight_sum = 0.0
            for term, frequency in count.items():
                idf = math.log((total + 1) / (document_frequency[term] + 1)) + 1
                weight = (1 + math.log(frequency)) * idf
                self._postings.setdefault(term, []).append((doc_id, weight))
                weight_sum += weight * weight
            self._norms.append(math.sqrt(weight_sum) or 1.0)

    def search(
        self,
        query: str,
        medium: str | None = None,
        k: int = 4,
        exclude_ids: set[str] | None = None,
    ) -> list[dict]:
        """Rare content words outrank function words. A document can be excluded."""
        scores: dict[int, float] = {}
        excluded = exclude_ids or set()
        for term in set(_tokens(query)):
            if term in QUERY_STOP or len(term) < 3:
                continue
            idf = self._idf.get(term, 1.0)
            for doc_id, weight in self._postings.get(term, ()):
                scores[doc_id] = scores.get(doc_id, 0.0) + weight * idf
        ranked = []
        for doc_id, score in scores.items():
            document = self.documents[doc_id]
            if document.get("id") in excluded:
                continue
            if medium and document.get("medium") not in {medium, None}:
                continue
            ranked.append((score / self._norms[doc_id], document))
        if medium and len(ranked) < k:
            extra = []
            for doc_id, score in scores.items():
                document = self.documents[doc_id]
                if document.get("id") in excluded or document.get("medium") == medium:
                    continue
                extra.append((score / self._norms[doc_id], document))
            ranked.extend(extra)
        ranked.sort(key=lambda item: item[0], reverse=True)
        return [document for _score, document in ranked[:k]]


def evidence_documents(dataset_rows: list[dict], assignments: list[dict], dispositions: list[dict]) -> list[dict]:
    """Validation posts only, excluding the style-profile slice and gold-overlap families."""
    blocked = blocked_group_ids(assignments, dispositions)
    held = held_out_reference_texts(dataset_rows, assignments, blocked)
    profile_texts = set()
    for texts in held.values():
        profile, _real = partition_references(texts)
        profile_texts.update(profile)
    documents = []
    records = assignment_rows(assignments)
    for index, (row, record) in enumerate(zip(dataset_rows, records)):
        if record.get("split") != "validation" or record.get("sft_role") != "unchanged":
            continue
        if record.get("group_id") in blocked:
            continue
        text = (row.get("output") or "").strip()
        if not text or text in profile_texts:
            continue
        documents.append({
            "id": f"val_{index}",
            "medium": HELD_OUT_PLATFORM_TO_MEDIUM.get(record.get("source_platform")),
            "text": text[:1200],
        })
    return documents
