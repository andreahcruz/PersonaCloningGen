"""The follow-up file drops other-speaker talks and caps X by whole document."""
from host_finetune.queue_followup_trains import select_balanced_rows


def _row(source: str, line: int, n: int = 1) -> list[dict]:
    return [
        {"source": source, "source_file": f"{source}.jsonl", "source_line": line, "output": f"{source}-{line}-{i}"}
        for i in range(n)
    ]


def test_drops_saastr_and_caps_x_without_splitting_a_document():
    rows = []
    rows += _row("blog", 1, 3)
    rows += _row("youtube_saastr", 1, 4)
    rows += _row("youtube_jason", 1, 2)
    rows += _row("x", 1, 2)
    rows += _row("x", 2, 2)
    rows += _row("linkedin", 1, 1)
    kept, stats = select_balanced_rows(rows)
    sources = [row["source"] for row in kept]
    assert "youtube_saastr" not in sources
    assert sources.count("blog") == 3
    assert sources.count("x") == 4  # both X docs, because the second is needed to reach 3 blog rows
    assert sources.count("youtube_jason") == 2
    assert stats["dropped_youtube_saastr"] == 4
    assert stats["x_documents_kept"] == 2
