"""Ranked retrieval metrics require an explicit qrels scope."""
from host_finetune.retrieval_eval import chunk_flags, summarize_retrieval


def _hit(doc_id, group, medium, text, distance=0.2):
    return {
        "id": doc_id,
        "group_id": group,
        "medium": medium,
        "text": text,
        "distance": distance,
    }


def test_no_qrels_omits_ranked_metrics_and_keeps_diagnostics():
    result = summarize_retrieval(
        [{
            "query_id": "gold_001",
            "hits": [
                _hit("train_1", "g1", "x", "A short headline http://example.com/a", 0.30),
                _hit("train_2", "g1", "x", "Another post about enterprise sales today.", 0.40),
            ],
        }],
        relevance_scope="none",
    )
    summary = result["summary"]
    assert summary["corpus_recall_reported"] is False
    assert "precision_at_4" not in summary
    assert "recall_at_4" not in summary
    assert summary["duplicate_group_rate"] == 1
    assert summary["empty_retrieval_rate"] == 0
    assert summary["mean_cosine_distance"] == 0.35
    assert summary["reason_ranked_metrics_omitted"]


def test_judged_pool_reports_precision_but_not_corpus_recall():
    qrels = [
        {"query_id": "gold_002", "doc_id": "train_a", "grade": 3},
        {"query_id": "gold_002", "doc_id": "train_b", "grade": 0},
        {"query_id": "gold_002", "doc_id": "train_c", "grade": 2},
    ]
    result = summarize_retrieval(
        [{
            "query_id": "gold_002",
            "hits": [
                _hit("train_a", "ga", "blog", "Monday.com grew at $400m ARR."),
                _hit("train_b", "gb", "blog", "A different company filed to go public."),
            ],
        }],
        qrels,
        k_values=(1, 2),
        relevance_scope="judged_pool",
    )
    summary = result["summary"]
    assert summary["precision_at_1"] == 1
    assert summary["precision_at_2"] == 0.5
    assert summary["mrr"] == 1
    assert "recall_at_2" not in summary
    assert summary["pool_recall_at_2"] == 0.5
    assert summary["corpus_recall_reported"] is False


def test_incomplete_pool_judgment_does_not_invent_precision():
    qrels = [{"query_id": "gold_001", "doc_id": "train_a", "grade": 3}]
    result = summarize_retrieval(
        [{
            "query_id": "gold_001",
            "hits": [
                _hit("train_a", "ga", "blog", "Useful evidence about the buyer."),
                _hit("train_z", "gz", "x", "This hit was never judged."),
            ],
        }],
        qrels,
        k_values=(2,),
        relevance_scope="judged_pool",
    )
    assert result["queries"][0]["precision_at_2"] is None
    assert result["queries"][0]["evidence_precision_among_judged"] == 1


def test_corpus_scope_counts_unlisted_documents_as_irrelevant():
    qrels = [
        {"query_id": "gold_001", "doc_id": "train_a", "grade": 3},
        {"query_id": "gold_001", "doc_id": "train_held", "grade": 2},
    ]
    result = summarize_retrieval(
        [{
            "query_id": "gold_001",
            "hits": [
                _hit("train_a", "ga", "blog", "Direct evidence."),
                _hit("train_miss", "gm", "x", "Not in the qrels."),
            ],
        }],
        qrels,
        k_values=(2,),
        relevance_scope="corpus",
    )
    row = result["queries"][0]
    assert row["precision_at_2"] == 0.5
    assert row["recall_at_2"] == 0.5
    assert "pool_recall_at_2" not in row


def test_mid_list_chunk_is_flagged_without_becoming_a_relevance_label():
    flags = chunk_flags(
        "#7. Monday.com is growing 68% at $550m ARR and still adding customers "
        "every quarter this year across the whole product line."
    )
    assert flags["mid_list"] is True
    assert flags["very_short"] is False
