import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
from textprep import normalize, clean_pg_essay, windows, first_window, mask_to_style, ngram_set

def test_normalize_unifies_quotes_and_space():
    assert normalize("It’s  a “good” day\n\nhttps://x.com/a") == "It's a \"good\" day"

def test_pg_notes_and_thanks_removed():
    t = "Body text here.\n\n" + "x " * 100 + "\n\nNotes\n\nfootnote text\n\nThanks to Sam for reading drafts."
    assert "footnote" not in clean_pg_essay(t) and "Thanks to" not in clean_pg_essay(t)

def test_windows_exact_size_and_tail_dropped():
    w = windows(" ".join(str(i) for i in range(250)), 100)
    assert len(w) == 2 and all(len(x.split()) == 100 for x in w)

def test_first_window_none_when_short():
    assert first_window("one two three", 100) is None
    assert len(first_window(" ".join(["w"] * 150), 100).split()) == 100

def test_mask_hides_content_keeps_function_words_and_punct():
    m = mask_to_style("You need 3 great reps, not ARR.")
    assert m.split() == ["you", "need", "#", "*", "*", ",", "not", "*", "."] 

def test_ngram_set():
    assert len(ngram_set("a b c d e f g h i", 8)) == 2
