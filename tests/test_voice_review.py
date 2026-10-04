"""Surface style-distance diagnostic. Human voice preference is not scored."""
from host_finetune.voice_distance import (
    closer_side,
    reference_texts,
    score_sheet,
    style_distance,
    style_features,
)


def test_style_distance_prefers_the_matching_surface():
    references = [
        "So I've seen this. We closed the deal. The NRR was 120%.",
        "So I'm going to be direct. Our reps missed quota, and the ACV was $50k.",
    ]
    from host_finetune.voice_distance import reference_profile

    profile = reference_profile(references)
    close = style_distance(references[0], profile)
    far = style_distance("The committee convened regarding procedural compliance.", profile)
    assert close is not None and far is not None
    assert close < far
    assert closer_side(0.2, 1.4) == "a"
    assert closer_side(1.4, 0.2) == "b"
    assert closer_side(1.0, 1.02) == "tie"
    assert closer_side(None, 1.0) == "unavailable"


def test_references_skip_fragments_and_talk_has_none():
    dataset = [
        {"output": "Whole blog."},
        {"output": "Fragment."},
        {"output": "Whole post."},
    ]
    assignments = [
        {"row_index": 0, "split": "train", "group_id": "b", "source_platform": "blog",
         "sft_role": "unchanged"},
        {"row_index": 1, "split": "train", "group_id": "b", "source_platform": "blog",
         "sft_role": "continuation"},
        {"row_index": 2, "split": "train", "group_id": "x", "source_platform": "x",
         "sft_role": "unchanged"},
    ]
    chosen = reference_texts(dataset, assignments, set())
    assert chosen["blog"] == ["Whole blog."]
    assert chosen["x"] == ["Whole post."]
    assert chosen["talk"] == []
    scored = score_sheet(
        [{"pair_id": "pair_0001", "medium": "talk", "answer_a": "Hello.", "answer_b": "Hello there."}],
        chosen,
    )
    assert scored["pairs"][0]["closer"] == "unavailable"
    assert "not authorship" in scored["limits"]
    features = style_features("So I've seen 120% NRR.")
    assert features["first_person_per_100w"] > 0
    assert features["numbers_per_100w"] > 0
