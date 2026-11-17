import sys, pathlib
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT))
import numpy as np
import run_pipeline as rp
from stats import holm_adjust, summarize_paired

def test_sample_weights_balance_classes_and_split_negatives():
    src = np.array(["lemkin"] * 100 + ["pg"] * 30 + ["llm"] * 10)
    w = rp.sample_weights(src)
    assert abs(w[src == "lemkin"].sum() - w[src != "lemkin"].sum()) < 1e-9      # positives = negatives
    assert abs(w[src == "pg"].sum() - w[src == "llm"].sum()) < 1e-9              # PG and Llama share negatives

def test_llm_negatives_never_contain_held_out_topic():
    drafts = [{"run": "A_base_llama", "medium": "blog", "topic": f"gold_{i:03d}", "text": " ".join(["word"] * 250)} for i in range(1, 7)]
    held = {"gold_002", "gold_005"}
    got = rp.llm_negatives(drafts, "blog", 100, held)
    assert got and all(not any(t in w["doc"] for t in held) for w in got)

def test_llm_negatives_use_only_base_llama():
    drafts = [{"run": "E_alone_seed42", "medium": "blog", "topic": "gold_001", "text": " ".join(["w"] * 300)}]
    assert rp.llm_negatives(drafts, "blog", 100, set()) == []

def test_paired_detects_clear_gap_and_keeps_noise_unclear():
    tm = {"A": {f"t{i}": 0.4 for i in range(20)}, "B": {f"t{i}": 0.4 + (0.2 if i % 2 else 0.19) for i in range(20)}}
    s = summarize_paired([tm["B"][t] - tm["A"][t] for t in tm["A"]], list(tm["A"]), n_boot=2000, n_perm=5000, seed=1)
    assert s["ci_low"] > 0 and s["p_value"] < 0.01
    s2 = summarize_paired([(-1) ** i * 0.05 for i in range(20)], [f"t{i}" for i in range(20)], n_boot=2000, n_perm=5000, seed=1)
    assert s2["ci_low"] < 0 < s2["ci_high"]

def test_holm_never_lowers_a_p_value():
    adj = holm_adjust({"a": 0.01, "b": 0.04, "c": 0.5})
    assert adj["a"] >= 0.01 and adj["b"] >= 0.04 and adj["c"] >= 0.5

def test_numberfree_view_ignores_numbers():
    from textprep import mask_to_style
    assert "#" not in mask_to_style("We hit $20M ARR, up 100%.", drop_numbers=True).split()
    assert "#" in mask_to_style("We hit $20M ARR, up 100%.").split()


def test_external_negatives_replace_cross_fitting():
    rp.EXTERNAL_NEG = [{"text": " ".join(["x"] * 250), "medium": "blog"}]
    try:
        got = rp.llm_negatives([], "blog", 100, {"gold_001"})
        assert len(got) == 2 and all(w["doc"].startswith("ext:") for w in got)
    finally:
        rp.EXTERNAL_NEG = None
