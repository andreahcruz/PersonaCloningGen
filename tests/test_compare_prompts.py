"""The mixed-medium comparison must not put the answer key in the prompt."""
from host_finetune.compare_adapters import MEDIUMS, comparison_instruction, prompt_for


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
