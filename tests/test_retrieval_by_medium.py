"""Tests for the pure part of docs/evaluation/examples/data/retrieval_by_medium/retrieval_by_medium.py.

The script itself needs the Chroma index and Ollama, so only its summary maths is tested, on made up hits.

    python -m pytest tests/test_retrieval_by_medium.py
"""
import importlib.util
from collections import Counter
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "docs/evaluation/examples/data/retrieval_by_medium/retrieval_by_medium.py"
spec = importlib.util.spec_from_file_location("retrieval_by_medium", SCRIPT)
rbm = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rbm)  # imports nothing heavy: chromadb and requests are only used inside main and embed


def hit(doc_id, medium, distance, words=100, group="g"):
    return {"id": doc_id, "medium": medium, "group_id": group, "distance": distance, "words": words, "text": "t"}


TOPICS = [{"id": "t1"}, {"id": "t2"}]


def build(unfiltered_hits, filtered_hits_by_medium):
    unfiltered = {"t1": unfiltered_hits, "t2": unfiltered_hits}
    filtered = {m: {"t1": hits, "t2": hits} for m, hits in filtered_hits_by_medium.items()}
    return unfiltered, filtered


def four(medium, distance, words=100, prefix="a"):
    return [hit(f"{prefix}{i}", medium, distance, words, group=f"{prefix}{i}") for i in range(4)]


def test_index_names_follow_the_pipeline_mapping():
    assert rbm.INDEX_MEDIUM == {"blog": "blog", "linkedin": "linkedin", "x": "x", "talk": "youtube_jason"}
    assert rbm.K == 4


def test_share_of_requested_medium_before_and_after_filtering():
    unfiltered_hits = [hit("a", "blog", 0.2), hit("b", "blog", 0.2), hit("c", "x", 0.2), hit("d", "linkedin", 0.2)]
    unfiltered, filtered = build(unfiltered_hits, {m: four(rbm.INDEX_MEDIUM[m], 0.3, prefix=m) for m in rbm.INDEX_MEDIUM})
    out = rbm.summarize(TOPICS, unfiltered, filtered, Counter({"blog": 9, "linkedin": 7, "x": 5, "youtube_jason": 3}))
    assert out["blog"]["unfiltered_share_of_medium"] == 0.5 and out["blog"]["filtered_share_of_medium"] == 1.0
    assert out["x"]["unfiltered_share_of_medium"] == 0.25
    assert out["talk"]["unfiltered_share_of_medium"] == 0.0 and out["talk"]["filtered_share_of_medium"] == 1.0
    assert out["talk"]["pool_documents"] == 3 and out["blog"]["pool_documents"] == 9


def test_distance_gap_is_filtered_minus_unfiltered_and_the_worst_topic_is_reported():
    unfiltered = {"t1": four("blog", 0.2, prefix="u"), "t2": four("blog", 0.2, prefix="u")}
    filtered = {m: {"t1": four(rbm.INDEX_MEDIUM[m], 0.3, prefix=m), "t2": four(rbm.INDEX_MEDIUM[m], 0.5, prefix=m)} for m in rbm.INDEX_MEDIUM}
    out = rbm.summarize(TOPICS, unfiltered, filtered, Counter())
    assert out["linkedin"]["distance_gap_mean"] == pytest.approx(0.2, abs=1e-4)   # mean of +0.1 and +0.3
    assert out["linkedin"]["distance_gap_max"] == pytest.approx(0.3, abs=1e-4)
    assert out["linkedin"]["unfiltered_mean_distance"] == 0.2 and out["linkedin"]["filtered_mean_distance"] == 0.4


def test_overlap_counts_passages_kept_from_the_unfiltered_search():
    unfiltered_hits = [hit("a", "blog", 0.2), hit("b", "blog", 0.2), hit("c", "x", 0.2), hit("d", "x", 0.2)]
    filtered_blog = [hit("a", "blog", 0.2), hit("b", "blog", 0.2), hit("e", "blog", 0.25), hit("f", "blog", 0.26)]
    unfiltered, filtered = build(unfiltered_hits, {m: four(rbm.INDEX_MEDIUM[m], 0.3, prefix=m) for m in rbm.INDEX_MEDIUM})
    filtered["blog"] = {"t1": filtered_blog, "t2": filtered_blog}
    out = rbm.summarize(TOPICS, unfiltered, filtered, Counter())
    assert out["blog"]["overlap_with_unfiltered_share"] == 0.5 and out["x"]["overlap_with_unfiltered_share"] == 0.0


def test_very_short_share_uses_the_twenty_word_flag():
    short, long_ = [hit("s%d" % i, "x", 0.2, words=10) for i in range(2)], [hit("l%d" % i, "x", 0.2, words=50) for i in range(2)]
    unfiltered, filtered = build(short + long_, {m: four(rbm.INDEX_MEDIUM[m], 0.3, words=19, prefix=m) for m in rbm.INDEX_MEDIUM})
    out = rbm.summarize(TOPICS, unfiltered, filtered, Counter())
    assert out["x"]["unfiltered_very_short_share"] == 0.5 and out["x"]["filtered_very_short_share"] == 1.0
    assert out["blog"]["unfiltered_mean_words"] == 30 and out["blog"]["filtered_mean_words"] == 19


def test_unique_groups_counts_distinct_group_ids_per_topic():
    same_group = [hit("a%d" % i, "blog", 0.2, group="g1") for i in range(4)]
    unfiltered, filtered = build(four("blog", 0.2, prefix="u"), {m: four(rbm.INDEX_MEDIUM[m], 0.3, prefix=m) for m in rbm.INDEX_MEDIUM})
    filtered["blog"] = {"t1": same_group, "t2": same_group}
    assert rbm.summarize(TOPICS, unfiltered, filtered, Counter())["blog"]["filtered_unique_groups_mean"] == 1.0
