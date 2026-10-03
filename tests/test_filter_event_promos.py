"""Promo filter drops a short remnant only after it actually removed a CTA."""
import json

from host_finetune.filter_event_promos import main


def test_short_chunk_without_a_cta_is_kept(tmp_path, capsys):
    dataset = tmp_path / "in.jsonl"
    out = tmp_path / "out.jsonl"
    short = {
        "instruction": "Write an X post in the style of Jason Lemkin about: Pricing",
        "input": "",
        "output": "Price on value. Raise prices.",
        "source": "x",
        "source_file": "jasonlk_originals.jsonl",
        "source_line": 3,
    }
    cta = {
        "instruction": "Write a blog post in the style of Jason Lemkin about: Dear SaaStr: How do I price?",
        "input": "",
        "output": (
            "Charge for the outcome, not the seat. " * 30
            + "Join us at SaaStr Annual and buy tickets."
        ),
        "source": "blog",
        "source_file": "jasonlemkin_blog.jsonl",
        "source_line": 9,
    }
    dataset.write_text(
        json.dumps(short) + "\n" + json.dumps(cta) + "\n",
        encoding="utf-8",
    )
    import sys
    argv = sys.argv
    sys.argv = ["filter", "--dataset", str(dataset), "--out", str(out)]
    try:
        main()
    finally:
        sys.argv = argv
    rows = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 2
    assert rows[0]["output"] == short["output"]
    assert "buy tickets" not in rows[1]["output"].lower()
    assert "Charge for the outcome" in rows[1]["output"]
