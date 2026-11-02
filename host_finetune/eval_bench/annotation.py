"""Blank retrieval-grade sheets. This module does not assign grades."""

from __future__ import annotations

ANNOTATION_SCHEMA = "retrieval-annotation-v1"
GRADE_SCALE = (
    {"grade": 0, "meaning": "The document does not help answer the question."},
    {"grade": 1, "meaning": "The document is on the topic and does not contain the answer."},
    {"grade": 2, "meaning": "The document contains information that answers the question."},
)


def blank_template(questions: list[dict], pools: dict[str, list[str]], *, source: str) -> dict:
    """One ungraded row per saved candidate. Preselected ids are not grades."""
    rows = []
    for question in questions:
        question_id = str(question["id"])
        rows.append({
            "question_id": question_id,
            "kind": question.get("kind"),
            "query": question.get("query"),
            "source": source,
            "split_policy": "Development review questions only. Do not annotate the corpus test split.",
            "preselected_source_ids": [str(item) for item in (question.get("review_source_ids") or [])],
            "preselected_ids_are_grades": False,
            "documents": [
                {"doc_id": doc_id, "grade": None, "annotator": None, "notes": ""}
                for doc_id in pools.get(question_id, [])
            ],
        })
    return {
        "schema": ANNOTATION_SCHEMA,
        "grade_scale": [dict(item) for item in GRADE_SCALE],
        "grades_are_human_assigned": False,
        "min_grade_for_ranked_metrics": 2,
        "relevance_scope_after_grades_exist": "corpus",
        "questions": rows,
    }


def pools_from_retrieval(records: list[dict]) -> dict[str, list[str]]:
    """Candidate ids in first-seen order. Order is not a relevance ranking."""
    pools: dict[str, list[str]] = {}
    for record in records:
        question_id = str(record["question_id"])
        seen = pools.setdefault(question_id, [])
        for doc_id in record.get("retrieved_source_ids") or []:
            if doc_id and doc_id not in seen:
                seen.append(str(doc_id))
    return pools
