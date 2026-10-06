"""Qrel pool construction must not invent relevance grades."""
from host_finetune.qrel_pool import (
    accepted_qrels,
    annotate_query,
    inherit_group_grades,
    judgment_view,
    pool_diagnostics,
)


def _hit(doc_id, group, text, distance=0.2, row_id=1):
    return {
        "id": doc_id,
        "group_id": group,
        "medium": "blog",
        "row_id": row_id,
        "split": "train",
        "text": text,
        "distance": distance,
    }


def test_raw_pool_keeps_same_group_duplicates_and_judgment_view_collapses_them():
    hits = [
        _hit("train_a", "g1", "Monday.com grew past four hundred million in ARR this year.", row_id=10),
        _hit("train_b", "g1", "The next section of the same Monday.com post continues here.", row_id=11),
        _hit("train_c", "g2", "A different company filed its S-1 and talked about net retention.", row_id=12),
    ]
    raw = annotate_query("gold_002", "5 Interesting Learnings from Monday.com", hits, {
        10: {"sft_role": "opening", "split": "train"},
        11: {"sft_role": "continuation", "split": "train"},
        12: {"sft_role": "unchanged", "split": "train"},
    })
    assert len(raw) == 3
    assert raw[1]["duplicate_of"] == "train_a"
    assert raw[1]["source_family_redundant"] is True
    assert "relevance_grade" not in raw[0]
    units = judgment_view(raw)
    assert len(units) == 2
    assert units[0]["document_id"] == "train_a"
    assert units[0]["sibling_document_ids"] == ["train_b"]


def test_unaccepted_judgments_do_not_become_qrels():
    judgments = [
        {"query_id": "gold_001", "document_id": "train_a", "relevance_grade": 3, "review_status": "proposed"},
        {"query_id": "gold_001", "document_id": "train_b", "relevance_grade": None, "review_status": "unlabeled"},
        {"query_id": "gold_001", "document_id": "train_c", "relevance_grade": 0, "review_status": "accepted"},
    ]
    qrels = accepted_qrels(judgments)
    assert qrels == [{"query_id": "gold_001", "doc_id": "train_c", "grade": 0}]


def test_explicit_sibling_grade_wins_over_group_inheritance():
    raw = [
        {"query_id": "gold_002", "document_id": "train_a", "group_id": "g1"},
        {"query_id": "gold_002", "document_id": "train_b", "group_id": "g1"},
        {"query_id": "gold_002", "document_id": "train_c", "group_id": "g2"},
    ]
    judgments = [
        {"query_id": "gold_002", "document_id": "train_a", "relevance_grade": 3, "review_status": "accepted"},
        {"query_id": "gold_002", "document_id": "train_b", "relevance_grade": 1, "review_status": "accepted"},
    ]
    expanded = inherit_group_grades(judgments, raw)
    grades = {row["doc_id"]: row["grade"] for row in expanded}
    assert grades["train_a"] == 3
    assert grades["train_b"] == 1
    assert "train_c" not in grades
    v2_only = inherit_group_grades(
        judgments,
        raw + [{
            "query_id": "gold_002",
            "document_id": "v2_chunk",
            "group_id": "g1",
            "pool_origin": "dense_rag_v2",
        }],
    )
    assert "v2_chunk" not in {row["doc_id"] for row in v2_only}


def test_diagnostics_do_not_report_precision_or_recall():
    raw = annotate_query(
        "gold_001",
        "Can an 8-person startup sell to a CIO?",
        [_hit("train_z", "gz", "#7. A short fragment http://example.com/a", row_id=3)],
        {3: {"sft_role": "continuation", "split": "train"}},
    )
    summary = pool_diagnostics([{"query_id": "gold_001", "hits": raw}], ks=(1,))
    assert raw[0]["fragment"] is True
    assert "relevance_grade" not in raw[0]
    assert "precision_at_1" not in summary
    assert "recall_at_1" not in summary
    assert summary["ranked_metrics_omitted"] is True
    assert summary["headline_only_rate"] == 1
