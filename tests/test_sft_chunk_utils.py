"""Long documents split on section headings only when they exceed the chunk cap."""
from host_finetune.sft_chunk_utils import split_long_document


def test_short_document_stays_whole():
    body = "One short post about pricing."
    assert split_long_document(body, max_chars=1600) == [(None, body)]


def test_long_document_without_headings_uses_sentences():
    sentence = "Pricing should follow value. "
    body = sentence * 80
    pieces = split_long_document(body, max_chars=200)
    assert len(pieces) > 1
    assert all(heading is None for heading, _chunk in pieces)
    assert all(len(chunk) <= 200 for _heading, chunk in pieces)


def test_numbered_sections_stay_together_until_they_exceed_the_cap():
    body = "\n".join([
        "Intro paragraph about hiring.",
        "1. The Evangelist",
        "This person sells the vision before the process exists. " * 3,
        "2. Mr. Make-it-Repeatable",
        "This person turns the first wins into a process the next rep can run.",
    ])
    pieces = split_long_document(body, max_chars=len(body) - 1)
    headings = [heading for heading, _chunk in pieces]
    assert headings[0] is None
    assert headings[1].startswith("1. The Evangelist")
    assert headings[2].startswith("2. Mr. Make-it-Repeatable")
    assert "Mr. Make-it-Repeatable" not in pieces[1][1]
    assert "The Evangelist" in pieces[1][1]


def test_oversized_section_is_split_without_mixing_the_next_heading():
    body = "\n".join([
        "1. First section",
        ("Alpha sentence about quotas. " * 30),
        "2. Second section",
        "Beta stays in its own section.",
    ])
    pieces = split_long_document(body, max_chars=180)
    first_parts = [chunk for heading, chunk in pieces if heading and heading.startswith("1.")]
    second_parts = [chunk for heading, chunk in pieces if heading and heading.startswith("2.")]
    assert len(first_parts) > 1
    assert second_parts == ["2. Second section\nBeta stays in its own section."]
    assert all("Beta" not in chunk for chunk in first_parts)
    assert all(len(chunk) <= 180 for chunk in first_parts)
