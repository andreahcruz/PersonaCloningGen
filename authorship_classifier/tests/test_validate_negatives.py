import json, pathlib, sys
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT))
import validate_negatives as vn

HAVE_INPUTS = (ROOT / "inputs" / "split" / "dataset.jsonl").exists() and (ROOT / "inputs" / "gold_eval.jsonl").exists()


def test_refusal_regex_flags_refusals_not_normal_text():
    assert vn.REFUSAL.search("I can't help with that request.")
    assert vn.REFUSAL.search("As an AI language model, I do not")
    assert not vn.REFUSAL.search("Hire your first sales rep before you hire a VP of Sales.")


def test_jaccard_identity_and_disjoint():
    assert vn.jac("Hire a VP of Sales", "hire a vp of sales!") == 1.0
    assert vn.jac("alpha beta gamma", "delta epsilon zeta") == 0.0


def test_load_returns_rows_and_stable_checksum(tmp_path):
    path = tmp_path / "x.jsonl"
    path.write_text(json.dumps({"text": "a b c", "medium": "blog"}) + "\n", encoding="utf-8")
    rows, manifest, sha1 = vn.load(path)
    assert len(rows) == 1 and manifest is None
    assert vn.load(path)[2] == sha1


def _rows(topic_for, n, text_for):
    return [{"id": f"blog-{i:04d}", "medium": "blog", "topic": topic_for(i), "prompt": f"Write a blog post in the style of Jason Lemkin about: {topic_for(i)}",
             "text": text_for(i), "stop": "eos", "seed": 1, "model": "t", "precision": "t", "new_tokens": 200} for i in range(n)]


@pytest.mark.skipif(not HAVE_INPUTS, reason="needs inputs/ from prepare_inputs.py")
def test_gold_topic_makes_validation_fail(tmp_path, monkeypatch):
    gold = [json.loads(l)["topic"] for l in open(ROOT / "inputs" / "gold_eval.jsonl", encoding="utf-8")]
    words = " ".join(f"word{i}" for i in range(200))
    rows = _rows(lambda i: gold[0], 3, lambda i: f"Draft {i}. {words}")
    path = tmp_path / "gold.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    monkeypatch.setattr(vn, "ROOT", tmp_path)
    out = vn.main(path, "res")
    assert out["gold_topic_overlaps"] >= 1 and out["PASS"] is False


@pytest.mark.skipif(not HAVE_INPUTS, reason="needs inputs/ from prepare_inputs.py")
def test_title_repeat_is_not_counted_as_copying(tmp_path, monkeypatch):
    """A text that only repeats its own topic title (a real Lemkin title) must not be flagged, as in Kevin's copy gate."""
    sys.path.insert(0, str(ROOT / "src"))
    from data import load_lemkin_rows
    from textprep import ngram_set
    ds, asg, _ = load_lemkin_rows()
    title = None
    for row, a in zip(ds, asg):                       # find a long real training sentence to use as a fake 'title'
        words = (row.get("output") or "").split()
        if a["split"] == "train" and a["source_platform"] == "blog" and len(words) > 40:
            title = " ".join(words[:14]); break
    assert title
    filler = " ".join(f"zz{i}" for i in range(200))
    rows = _rows(lambda i: title, 1, lambda i: f"{title}. {filler}")
    path = tmp_path / "t.jsonl"
    path.write_text(json.dumps(rows[0]) + "\n", encoding="utf-8")
    monkeypatch.setattr(vn, "ROOT", tmp_path)
    out = vn.main(path, "res")
    assert out["flags"].get("copies_lemkin_train_12gram", 0) == 0
    assert out["copy_share_raw_incl_topic_title_repeats"] > 0       # the raw number still shows it
