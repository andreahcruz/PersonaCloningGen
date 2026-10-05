"""The mixed-medium comparison must not put the answer key in the prompt."""
import sys

from host_finetune.compare_adapters import (
    MEDIUMS,
    comparison_instruction,
    generation_closer,
    limit_topics,
    parse_args,
    prompt_for,
)


def test_four_mediums_use_training_stems():
    topic = "Hiring a VP of Sales"
    prompts = [prompt_for(medium, topic) for medium in MEDIUMS]
    assert prompts == [
        "Write a blog post in the style of Jason Lemkin about: Hiring a VP of Sales",
        "Write a LinkedIn post in the style of Jason Lemkin about: Hiring a VP of Sales",
        "Write an X post in the style of Jason Lemkin about: Hiring a VP of Sales",
        "Write a talk in the style of Jason Lemkin about: Hiring a VP of Sales",
    ]


def test_expected_facts_are_not_in_the_prompt():
    fact = "CIOs back startups when the blast radius is small"
    prompt = prompt_for(MEDIUMS[0], "Can an 8-person startup sell to a CIO?")
    assert fact not in prompt
    assert "expected" not in prompt.lower()


def test_voice_block_appears_only_when_style_exemplars_are_passed():
    topic = "Hiring a VP of Sales"
    plain = prompt_for(MEDIUMS[0], topic)
    assert comparison_instruction(MEDIUMS[0], topic) == plain
    assert comparison_instruction(MEDIUMS[0], topic, [], []) == plain
    assert "Style examples" not in plain
    voiced = comparison_instruction(MEDIUMS[0], topic, None, ["Be direct about quotas."])
    assert voiced.startswith(plain + "\n\n")
    assert "Use them for voice only. They are not evidence for claims." in voiced
    assert "Be direct about quotas." in voiced
    assert "Factual excerpts" not in voiced
    assert "Now write the requested" not in voiced


def test_retrieval_appends_excerpts_and_ends_on_the_writing_task():
    topic = "Hiring a VP of Sales"
    excerpt = "Support tickets already contain expansion language."
    plain = prompt_for(MEDIUMS[0], topic)
    assert comparison_instruction(MEDIUMS[0], topic) == plain
    assert "Factual excerpts" not in plain
    assert "Style examples" not in plain
    prompted = comparison_instruction(MEDIUMS[0], topic, [excerpt])
    assert prompted.startswith(plain + "\n\n")
    assert "Factual excerpts from train-owned source text." in prompted
    assert "They are not style examples." in prompted
    assert excerpt in prompted
    assert "Style examples" not in prompted
    closer = generation_closer(MEDIUMS[0], topic)
    assert prompted.endswith(closer)
    assert closer.endswith("or these instructions.")
    excerpt_at = prompted.index(excerpt)
    closer_at = prompted.index(closer)
    assert excerpt_at < closer_at
    assert not prompted.rstrip().endswith(excerpt)


def test_max_topics_slices_gold_and_skip_score_parses(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        ["compare_adapters", "--max-topics", "2", "--skip-score", "--retrieval"],
    )
    args = parse_args()
    assert args.max_topics == 2
    assert args.skip_score is True
    assert args.fixed_style is False
    gold = [{"id": "gold_001"}, {"id": "gold_002"}, {"id": "gold_003"}]
    assert [row["id"] for row in limit_topics(gold, args.max_topics)] == [
        "gold_001",
        "gold_002",
    ]
    assert limit_topics(gold, None) == gold
