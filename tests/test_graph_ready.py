"""Graph-ready reconstruction stays inside the train-only boundary."""

from host_finetune.graph_ready import (
    GraphReadyPolicy,
    assemble_source_documents,
    join_ordered_chunks,
    select_index_sample,
)


def _row(index, text, split, source_line, **extra):
    row = {
        "row_id": f"train_{index}",
        "row_index": index,
        "text": text,
        "split": split,
        "group_id": extra.pop("group_id", "g1"),
        "medium": extra.pop("medium", "blog"),
        "source_file": extra.pop("source_file", "blog.jsonl"),
        "source_line": source_line,
        "prior_row_index": extra.pop("prior_row_index", index),
        "part_index": extra.pop("part_index", None),
        "base_title": extra.pop("base_title", "Pricing"),
        "sft_role": extra.pop("sft_role", "unchanged"),
        "eligible": extra.pop("eligible", split == "train"),
    }
    row.update(extra)
    return row


def test_cross_split_source_is_not_stitched():
    train = _row(0, "Train claim about retention.", "train", 1)
    held = _row(1, "Validation-only revenue figure 77.", "validation", 1, eligible=False)
    cleaned = {( "blog.jsonl", 1): {"content": train["text"] + " " + held["text"]}}
    result = assemble_source_documents([train, held], cleaned)
    assert len(result["included"]) == 1
    document = result["included"][0]
    assert document["reconstruction_method"] == "single_chunk"
    assert document["cleaned_document_used"] is False
    assert "Validation-only" not in document["text"]
    assert "cross_split_sibling" in document["flags"]
    assert result["pre_counts"]["cross_split_source"] == 1


def test_unknown_split_is_excluded():
    train = _row(0, "A train sentence about hiring.", "train", 2)
    unknown = _row(1, "Mystery split sentence.", "holdout", 2, eligible=True)
    result = assemble_source_documents([train, unknown], {})
    assert result["included"] == []
    assert result["quarantined"][0]["exclusion_reasons"] == ["unknown_split"]
    assert "Mystery split" not in result["quarantined"][0]["text"]


def test_cleaned_document_used_only_when_residual_is_small():
    first = "Alpha grew from five reps to six reps."
    second = "The same team stayed flat after that."
    cleaned = {"content": f"{first} {second} Thanks."}
    rows = [
        _row(0, first, "train", 3),
        _row(1, second, "train", 3),
    ]
    result = assemble_source_documents(rows, {("blog.jsonl", 3): cleaned})
    document = result["included"][0]
    assert document["reconstruction_method"] == "cleaned_document"
    assert document["text"] == cleaned["content"]
    assert document["experiment_row_ids"] == ["train_0", "train_1"]


def test_large_cleaned_residual_does_not_enter_the_graph_text():
    first = "Alpha grew from five reps to six reps."
    second = "The same team stayed flat after that."
    extra = "This held-aside paragraph was never an eligible training chunk and should stay out."
    rows = [_row(0, first, "train", 4), _row(1, second, "train", 4)]
    result = assemble_source_documents(
        rows,
        {("blog.jsonl", 4): {"content": f"{first}\n{second}\n{extra}"}},
    )
    document = result["included"][0]
    assert document["reconstruction_method"] == "ordered_eligible_chunks"
    assert "held-aside" not in document["text"]
    assert "Alpha grew" in document["text"]
    assert "same team" in document["text"]


def test_order_conflict_does_not_join_chunks():
    rows = [
        _row(0, "First slice of the post.", "train", 5, prior_row_index=2),
        _row(1, "Second slice of the post.", "train", 5, prior_row_index=1),
    ]
    result = assemble_source_documents(rows, {})
    assert len(result["included"]) == 2
    assert all(doc["reconstruction_method"] == "single_chunk" for doc in result["included"])
    assert result["pre_counts"]["order_conflict"] == 1


def test_repost_is_quarantined_and_short_x_post_is_kept():
    repost = _row(
        0,
        "Someone else's thread about fundraising.",
        "train",
        6,
        medium="x",
        source_file="x.jsonl",
    )
    short = _row(
        1,
        "Hire slower.",
        "train",
        7,
        medium="x",
        source_file="x.jsonl",
        base_title="",
    )
    cleaned = {
        ("x.jsonl", 6): {"text": repost["text"], "username": "jasonlk", "is_repost_or_quote": True},
        ("x.jsonl", 7): {"text": short["text"], "username": "jasonlk", "is_repost_or_quote": False},
    }
    result = assemble_source_documents([repost, short], cleaned)
    assert [doc["source_line"] for doc in result["included"]] == [7]
    assert result["quarantined"][0]["exclusion_reasons"] == ["repost_or_quote"]


def test_headline_stub_is_excluded_without_dropping_a_numeric_fragment():
    stub = _row(0, "Monday board meeting", "train", 8)
    number = _row(1, "NRR hit 120.", "train", 9, medium="x", source_file="x.jsonl")
    result = assemble_source_documents([stub, number], {})
    assert [doc["source_line"] for doc in result["included"]] == [9]
    assert "headline_or_stub" in result["quarantined"][0]["exclusion_reasons"]
    assert result["included"][0]["numeric_surface_count"] >= 1


def test_exact_duplicate_keeps_one_source_line():
    text = "The same claim appears on two captured lines."
    rows = [
        _row(0, text, "train", 11),
        _row(1, text, "train", 12),
    ]
    result = assemble_source_documents(rows, {})
    assert len(result["included"]) == 1
    assert result["included"][0]["source_line"] == 11
    assert result["quarantined"][0]["exclusion_reasons"] == ["exact_duplicate_source"]
    assert result["quarantined"][0]["duplicate_of"] == result["included"][0]["graph_document_id"]


def test_join_drops_a_repeated_word_suffix():
    left = "Retention stayed above the line for the whole year and the team noticed"
    right = "for the whole year and the team noticed a hiring freeze afterward"
    joined = join_ordered_chunks([left, right])
    assert joined.count("for the whole year and the team noticed") == 1
    assert "hiring freeze" in joined


def test_sample_is_deterministic_and_covers_media():
    docs = []
    for index, medium in enumerate(["blog", "x", "linkedin", "blog", "x"]):
        docs.append(
            {
                "graph_document_id": f"gd_{index:02d}",
                "medium": medium,
                "word_count": 10 * (index + 1),
                "split": "train",
            }
        )
    first = select_index_sample(docs, 4, seed=298)
    second = select_index_sample(docs, 4, seed=298)
    assert [doc["graph_document_id"] for doc in first] == [doc["graph_document_id"] for doc in second]
    assert len({doc["medium"] for doc in first}) >= 2


def test_sample_cap_limits_a_dominant_medium():
    docs = []
    for index in range(30):
        medium = "x" if index < 24 else ("blog" if index < 28 else "linkedin")
        docs.append(
            {
                "graph_document_id": f"gd_{index:02d}",
                "medium": medium,
                "word_count": 20 + index,
                "split": "train",
            }
        )
    sample = select_index_sample(docs, 10, seed=298, max_medium_fraction=0.4)
    counts = {}
    for document in sample:
        counts[document["medium"]] = counts.get(document["medium"], 0) + 1
    assert max(counts.values()) <= 4
    assert set(counts) == {"x", "blog", "linkedin"}


def test_validation_text_is_absent_when_it_is_marked_eligible_by_mistake():
    leaked = _row(3, "Held out validation body.", "validation", 13, eligible=True)
    result = assemble_source_documents([leaked], {})
    assert result["included"] == []
    assert result["quarantined"] == []
    policy = GraphReadyPolicy()
    assert policy.exclude_unattributed_transcripts is True
