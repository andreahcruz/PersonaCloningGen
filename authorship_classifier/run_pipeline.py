"""Authorship classifier pipeline: does a model's draft look like Jason Lemkin?

Run:  python -X utf8 run_pipeline.py
Reads  inputs/   (local copies of Kevin's split, saved drafts, Paul Graham essays)
Writes outputs/  only. Nothing in any git repository is touched.

The design in four sentences
1. Positives are real Lemkin posts from Kevin's TRAIN split, gold-topic families removed.
2. Negatives are (a) another human writer (Paul Graham) and (b) plain base-Llama text, because
   a PG-only classifier learns "modern business writing", not "Lemkin" (see validity check V0).
3. A draft is scored on its first N words, so length cannot help one system.
4. Every draft is scored by a model that never saw that draft's topic (5 topic folds).
"""
from __future__ import annotations

import collections
import hashlib
import json
import pathlib
import sys
import time

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent / "src"))

from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from data import SYSTEMS, load_drafts, lemkin_windows, pg_windows, load_lemkin_rows
from model import auc_with_ci, make_vectorizer
from stats import bootstrap_ci, holm_adjust, summarize_paired
from textprep import first_window, ngram_set, normalize, windows

ROOT = pathlib.Path(__file__).resolve().parent
OUT = ROOT / "outputs"
OUT.mkdir(exist_ok=True)

MEDIA = {  # medium -> (window size in words, which base-Llama draft media supply negatives)
    "blog": (100, ("blog", "talk")),
    "linkedin": (60, ("linkedin",)),
}
VIEWS = ("style", "style_nonum", "full")
C_MAIN = 1.0
C_SENSITIVITY = (0.3, 3.0)
N_FOLDS = 5
SEED = 7


PUK28 = {  # our run label -> PUK-28 file stem (Kevin's own voice_distance, first 100 words, held-out reference profile)
    "A_base_llama": "base_llama31", "C_personarag_seed42": "personarag_seed42",
    "E_alone_seed42": "relabel_rag_off_seed42", "E_alone_seed43": "relabel_rag_off_seed43", "E_alone_seed44": "relabel_rag_off_seed44",
    "E_rag_seed42": "relabel_rag_on_seed42", "E_rag_seed43": "relabel_rag_on_seed43", "E_rag_seed44": "relabel_rag_on_seed44"}
# SHA-256 of the draft files as recorded in PUK-28 (docs/evaluation/examples/data/grammar_lt/SOURCES.md)
EXPECTED_SHA = {
    "E_alone_seed42": "5d7ebfb57e93150288602e6523f83de300cd79a600c3ee119a48ca1a97a62f67",
    "E_alone_seed43": "8c5da8462c9a14733057d07a657ef6fca66cba79f8b8674daba618baa5b56e99",
    "E_alone_seed44": "50ed43f3d9f01e4091685292de0eb0632ee2ed990b13f7f5bf56cac5b03107f8",
    "E_rag_seed42": "1dca8e3afd4ca3cdb21225afc58bb111a007cac7e58db2ebaca8d3da7086f2d7",
    "E_rag_seed43": "32abc777fa8a96c6275a761e6895dc2e9c1eaf5e13200ef22fd7f6fe0f9c6ac5",
    "E_rag_seed44": "12981eb4075116e713eb0887296fff02bdde6a81a3ba20cca969c0bba3ab942e",
    "C_personarag_seed42": "381fa68c5d04a7576eeb887e869d2687260bb99f0474102802514a232afe013f",
    "A_base_llama": "fa460a01e5bd92694b6fccbefe2d00d4d669fa60ecb2f1c3a1536c6dd75c57c9"}


def puk28_distances():
    out = {}
    for run, stem in PUK28.items():
        for line in open(ROOT / "inputs" / "puk28_voice_fixed" / f"{stem}.voice_fixed.jsonl", encoding="utf-8"):
            r = json.loads(line)
            if r.get("style_distance_fixed") is not None:
                out[(run, r["id"], r["medium"])] = float(r["style_distance_fixed"])
    return out


def check_provenance():
    from data import SYSTEMS
    files = {run: fname for run, _, _, fname in SYSTEMS}
    rep = {}
    for run, want in EXPECTED_SHA.items():
        path = ROOT / "inputs" / "kevin_drafts" / files[run]
        if not path.exists():
            rep[run] = {"sha256": None, "matches_PUK28_record": None, "note": "file not available in this checkout"}
            continue
        got = hashlib.sha256(path.read_bytes()).hexdigest()
        rep[run] = {"sha256": got, "matches_PUK28_record": got == want}
    return rep


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


# --------------------------------------------------------------------------------------
def sample_weights(src: np.ndarray) -> np.ndarray:
    """Positive total weight equals negative total weight; Paul Graham and base-Llama share the
    negative weight equally, otherwise the few Llama windows would be ignored."""
    sw = np.ones(len(src))
    n_pos = (src == "lemkin").sum()
    for name in ("pg", "llm"):
        n = (src == name).sum()
        if n:
            sw[src == name] = (n_pos / 2) / n
    return sw


def fit_model(view, train, C, shuffle=False, shuffle_seed=SEED):
    y = np.array([w["label"] for w in train])
    src = np.array([w["src"] for w in train])
    if shuffle:
        y = np.random.RandomState(shuffle_seed).permutation(y)
    vec = make_vectorizer(view)
    X = vec.fit_transform([w["text"] for w in train])
    clf = LogisticRegression(C=C, max_iter=3000, solver="liblinear").fit(X, y, sample_weight=sample_weights(src))
    return lambda texts: clf.predict_proba(vec.transform(texts))[:, 1]


EXTERNAL_NEG: list[dict] | None = None   # set by --llm-negatives: generated base-Llama text on non-gold topics


def llm_negatives(drafts, medium, size, excl_topics):
    """Plain base-Llama text used as the second kind of 'not Lemkin'.
    Default: our own base-Llama drafts on OTHER topics (cross-fitted). With --llm-negatives:
    generated text on non-gold topics, which removes the cross-fitting caveat."""
    if EXTERNAL_NEG is not None:
        out = []
        for i, r in enumerate(EXTERNAL_NEG):
            if r.get("medium", medium) in MEDIA[medium][1]:
                out += [{"text": w, "doc": f"ext:{i}", "label": 0, "src": "llm"} for w in windows(r["text"], size)]
        return out
    out = []
    for d in drafts:
        if d["run"] != "A_base_llama" or d["medium"] not in MEDIA[medium][1] or d["topic"] in excl_topics:
            continue
        out += [{"text": w, "doc": f'A:{d["topic"]}:{d["medium"]}', "label": 0, "src": "llm"} for w in windows(d["text"], size)]
    return out


# --------------------------------------------------------------------------------------
def run_medium(medium, drafts, topics, train_gram_set):
    size = MEDIA[medium][0]
    log(f"[{medium}] loading windows (size {size})")
    lem_tr = lemkin_windows(medium, "train", size, first_only=False)
    pg_tr = pg_windows("train", size)
    lem_te = lemkin_windows(medium, "test", size, first_only=True, roles=("opening", "unchanged"))
    pg_te = pg_windows("test", size)
    lem_va = lemkin_windows(medium, "validation", size, first_only=True, roles=("opening", "unchanged"))
    log(f"[{medium}] train lemkin {len(lem_tr)} pg {len(pg_tr)} | test lemkin {len(lem_te)} pg {len(pg_te)}")
    fold_of = {t: i % N_FOLDS for i, t in enumerate(topics)}

    configs = [(v, C_MAIN) for v in VIEWS] + [("style", c) for c in C_SENSITIVITY]
    med_drafts = [d for d in drafts if d["medium"] == medium]
    scores = {cfg: {} for cfg in configs}            # cfg -> (run, topic) -> p
    held_llm = {cfg: [] for cfg in configs}          # held-out base-Llama window scores
    held_llm_docs = []
    real_scores = {cfg: [] for cfg in configs}
    pg_scores = {cfg: [] for cfg in configs}
    shuffled = {}

    for k in range(N_FOLDS):
        held = {t for t in topics if fold_of[t] == k}
        train = lem_tr + pg_tr + llm_negatives(drafts, medium, size, held)
        held_windows = []
        for d in drafts:
            if d["run"] == "A_base_llama" and d["medium"] in MEDIA[medium][1] and d["topic"] in held:
                for w in windows(d["text"], size):
                    held_windows.append((w, f'A:{d["topic"]}:{d["medium"]}'))
        held_llm_docs += [doc for _, doc in held_windows]
        for cfg in configs:
            view, C = cfg
            f = fit_model(view, train, C)
            for d in med_drafts:
                if d["topic"] in held:
                    w = first_window(d["text"], size)
                    if w:
                        scores[cfg][(d["run"], d["topic"])] = float(f([w])[0])
            held_llm[cfg] += list(f([w for w, _ in held_windows]))
            real_scores[cfg].append(f([w["text"] for w in lem_te]))
            pg_scores[cfg].append(f([w["text"] for w in pg_te]))
            if cfg == ("style", C_MAIN) and k == 0:       # label-shuffle control: 20 different shuffles, fold 0
                for rep in range(20):
                    fs = fit_model("style", train, C, shuffle=True, shuffle_seed=SEED + rep)
                    shuffled[rep] = {"real_lemkin": fs([w["text"] for w in lem_te]), "pg": fs([w["text"] for w in pg_te]),
                                     "base_llama": fs([w for w, _ in held_windows])}
        log(f"[{medium}] fold {k + 1}/{N_FOLDS} done")

    # ---------------- validity ----------------
    validity = {}
    y_lem = np.ones(len(lem_te)); lem_docs = [w["doc"] for w in lem_te]
    y_pg = np.zeros(len(pg_te)); pg_docs = [w["doc"] for w in pg_te]
    for cfg in configs:
        real = np.mean(real_scores[cfg], axis=0)       # average over the 5 fold models
        pgp = np.mean(pg_scores[cfg], axis=0)
        h = np.array(held_llm[cfg])
        a1 = auc_with_ci(np.r_[y_lem, y_pg], np.r_[real, pgp], lem_docs + pg_docs, n_boot=300)
        a2 = auc_with_ci(np.r_[y_lem, np.zeros(len(h))], np.r_[real, h], lem_docs + held_llm_docs, n_boot=300)
        validity[f"{cfg[0]}_C{cfg[1]}"] = {
            "V1_real_lemkin_vs_paul_graham_auc": a1, "V2_real_lemkin_vs_heldout_topic_base_llama_auc": a2,
            "mean_P_real_lemkin": float(real.mean()), "mean_P_paul_graham": float(pgp.mean()),
            "mean_P_heldout_base_llama": float(h.mean()), "n_real": int(len(real)), "n_pg": int(len(pgp)), "n_llm": int(len(h))}
    a_pg = [roc_auc_score(np.r_[y_lem, y_pg], np.r_[v["real_lemkin"], v["pg"]]) for v in shuffled.values()]
    a_ll = [roc_auc_score(np.r_[y_lem, np.zeros(len(v["base_llama"]))], np.r_[v["real_lemkin"], v["base_llama"]]) for v in shuffled.values()]
    validity["V3_label_shuffle_control"] = {
        "n_shuffles": len(a_pg), "auc_real_vs_pg_mean": float(np.mean(a_pg)), "auc_real_vs_pg_min_max": [float(min(a_pg)), float(max(a_pg))],
        "auc_real_vs_base_llama_mean": float(np.mean(a_ll)), "auc_real_vs_base_llama_min_max": [float(min(a_ll)), float(max(a_ll))],
        "expected": "scatter around 0.5 (no consistent signal) if the real result is not a leak or a bug"}
    return {"scores": scores, "validity": validity, "real": {c: np.mean(real_scores[c], axis=0) for c in configs},
            "pg": {c: np.mean(pg_scores[c], axis=0) for c in configs},
            "heldllm": {c: np.array(held_llm[c]) for c in configs}, "lem_va": lem_va, "lem_te": lem_te, "size": size}


def v0_pg_only(medium, drafts):
    """The classifier we should NOT use: Lemkin against Paul Graham only. It separates them
    almost perfectly, yet calls plain base-Llama drafts 'Lemkin', so it is not a valid test."""
    size = MEDIA[medium][0]
    train = lemkin_windows(medium, "train", size, first_only=False) + pg_windows("train", size)
    lem_te = lemkin_windows(medium, "test", size, first_only=True, roles=("opening", "unchanged"))
    pg_te = pg_windows("test", size)
    base = [first_window(d["text"], size) for d in drafts if d["run"] == "A_base_llama" and d["medium"] == medium]
    base = [w for w in base if w]
    out = {}
    for view in ("style", "full"):
        f = fit_model(view, train, C_MAIN)
        pr, pp, pb = f([w["text"] for w in lem_te]), f([w["text"] for w in pg_te]), f(base)
        out[view] = {"auc_real_vs_pg": float(roc_auc_score(np.r_[np.ones(len(pr)), np.zeros(len(pp))], np.r_[pr, pp])),
                     "mean_P_real_lemkin": float(pr.mean()), "mean_P_paul_graham": float(pp.mean()),
                     "mean_P_base_llama_drafts": float(pb.mean()), "n_base_llama_drafts": len(pb),
                     "verdict": "INVALID: base Llama drafts score as Lemkin" if pb.mean() > 0.5 else "ok"}
    return out


# --------------------------------------------------------------------------------------
def system_tables(medium, res, drafts, train_gram_set):
    size = res["size"]
    sysname = {run: system for run, system, *_ in SYSTEMS}
    by_run = {(d["run"], d["topic"]): d for d in drafts if d["medium"] == medium}
    dist = puk28_distances()
    per_draft = []
    for cfg, sc in res["scores"].items():
        for (run, topic), p in sc.items():
            d = by_run[(run, topic)]
            w = first_window(d["text"], size)
            grams = ngram_set(w, 8)
            per_draft.append({"medium": medium, "view": cfg[0], "C": cfg[1], "run": run, "system": sysname[run],
                              "seed": d["seed"], "topic": topic, "p_lemkin": p,
                              "copies_train_8gram": bool(grams & train_gram_set),
                              "draft_words": len(normalize(d["text"]).split()),
                              "voice_distance_fixed": dist.get((d["run"], topic, medium))})
    return per_draft


def summarize_systems(rows, medium, view, C, drop_copy=False):
    sel = [r for r in rows if r["medium"] == medium and r["view"] == view and r["C"] == C and not (drop_copy and r["copies_train_8gram"])]
    # topic-level mean over the seeds a system has
    topic_means = collections.defaultdict(lambda: collections.defaultdict(list))
    for r in sel:
        topic_means[r["system"]][r["topic"]].append(r["p_lemkin"])
    tm = {s: {t: float(np.mean(v)) for t, v in d.items()} for s, d in topic_means.items()}
    out = {}
    for s, d in tm.items():
        vals = list(d.values())
        lo, hi = bootstrap_ci(vals, n_boot=5000, seed=SEED)
        out[s] = {"mean_P": float(np.mean(vals)), "ci_low": lo, "ci_high": hi, "n_topics": len(vals),
                  "share_above_0.5": float(np.mean(np.array(vals) >= 0.5))}
    return out, tm


COMPARISONS = [
    ("E Relabel, no retrieval", "A Base Llama"), ("E Relabel, no retrieval", "C PersonaRAG stand-in"),
    ("E Relabel, no retrieval", "D QLoRA repaired"), ("E Relabel + retrieval", "E Relabel, no retrieval"),
    ("E Relabel + retrieval", "C PersonaRAG stand-in"), ("F Model F (F-REL)", "E Relabel + retrieval"),
    ("F Model F (F-REL)", "E Relabel, no retrieval"), ("D QLoRA repaired", "A Base Llama"),
]


def paired(tm):
    out, praw = [], {}
    for b, a in COMPARISONS:
        if a not in tm or b not in tm:
            continue
        common = sorted(set(tm[a]) & set(tm[b]))
        diffs = [tm[b][t] - tm[a][t] for t in common]
        s = summarize_paired(diffs, common, n_boot=5000, n_perm=20000, seed=SEED)
        s["B_minus_A"] = f"{b} minus {a}"
        out.append(s); praw[s["B_minus_A"]] = s["p_value"]
    adj = holm_adjust(praw)
    for s in out:
        s["p_holm"] = adj[s["B_minus_A"]]
        lo, hi, p = s["ci_low"], s["ci_high"], s["p_holm"]
        s["verdict"] = ("B higher" if lo is not None and lo > 0 and p is not None and p < 0.05 else
                        "B lower" if hi is not None and hi < 0 and p is not None and p < 0.05 else "no clear difference")
    return out


def rank_corr(a: dict, b: dict) -> float:
    keys = sorted(set(a) & set(b))
    return float(spearmanr([a[k] for k in keys], [b[k] for k in keys]).statistic)


def explain_features(drafts):
    """What the style views look at. Explanation only: trained on all data, never used for scoring."""
    out = {}
    for medium in ("blog",):
        size = MEDIA[medium][0]
        train = lemkin_windows(medium, "train", size, first_only=False) + pg_windows("train", size) + llm_negatives(drafts, medium, size, set())
        y = np.array([w["label"] for w in train]); src = np.array([w["src"] for w in train])
        for view in ("style", "style_nonum"):
            vec = make_vectorizer(view)
            X = vec.fit_transform([w["text"] for w in train])
            clf = LogisticRegression(C=C_MAIN, max_iter=3000, solver="liblinear").fit(X, y, sample_weight=sample_weights(src))
            names = np.array(vec.get_feature_names_out()); co = clf.coef_[0]; o = np.argsort(co)
            out[f"{medium}_{view}"] = {"toward_lemkin": [str(names[i]) for i in o[::-1][:20]], "toward_other": [str(names[i]) for i in o[:20]]}
    return out


# --------------------------------------------------------------------------------------
def main():
    drafts = load_drafts()
    topics = sorted({d["topic"] for d in drafts})
    log(f"{len(drafts)} drafts, {len(topics)} topics")
    rows_all, asg_all, _ = load_lemkin_rows()
    train_gram_set = set()
    for row, a in zip(rows_all, asg_all):
        if a["split"] == "train":
            train_gram_set |= ngram_set(row.get("output") or "", 8)
    log(f"training 8-gram set: {len(train_gram_set):,}")

    report = {"provenance": check_provenance(), "settings": {"views": VIEWS, "C_main": C_MAIN, "C_sensitivity": C_SENSITIVITY, "folds": N_FOLDS,
                           "windows": {m: MEDIA[m][0] for m in MEDIA}}, "media": {}}
    per_draft_all = []
    for medium in MEDIA:
        res = run_medium(medium, drafts, topics, train_gram_set)
        rows = system_tables(medium, res, drafts, train_gram_set)
        per_draft_all += rows
        res["validity"]["V0_paul_graham_only_classifier"] = v0_pg_only(medium, drafts)
        m = {"validity": res["validity"], "systems": {}, "paired": {}, "robustness": {}}
        for view in VIEWS:
            for drop in (False, True):
                key = f"{view}{'_no_copy' if drop else ''}"
                summ, tm = summarize_systems(rows, medium, view, C_MAIN, drop_copy=drop)
                m["systems"][key] = summ
                m["paired"][key] = paired(tm)
                if not drop:
                    m.setdefault("_tm", {})[view] = tm
        # robustness: do rankings agree across views and C?
        base = {s: v["mean_P"] for s, v in m["systems"]["style"].items()}
        rob = {"style_vs_full": rank_corr(base, {s: v["mean_P"] for s, v in m["systems"]["full"].items()}),
               "style_vs_style_nonum": rank_corr(base, {s: v["mean_P"] for s, v in m["systems"]["style_nonum"].items()}),
               "style_vs_style_no_copy": rank_corr(base, {s: v["mean_P"] for s, v in m["systems"]["style_no_copy"].items()})}
        for c in C_SENSITIVITY:
            summ, _ = summarize_systems(rows, medium, "style", c)
            rob[f"style_C{c}_vs_style_C{C_MAIN}"] = rank_corr(base, {s: v["mean_P"] for s, v in summ.items()})
        # agreement with Kevin's fixed-length voice distance (lower distance = closer)
        st = [r for r in rows if r["view"] == "style" and r["C"] == C_MAIN and r["voice_distance_fixed"] is not None]
        rho = spearmanr([r["p_lemkin"] for r in st], [-r["voice_distance_fixed"] for r in st])
        sysd = collections.defaultdict(list)
        for r in st:
            sysd[r["system"]].append(-r["voice_distance_fixed"])
        sysmean = {s: float(np.mean(v)) for s, v in sysd.items()}
        rob["draft_level_spearman_P_vs_negative_PUK28_voice_distance"] = {"rho": float(rho.statistic), "p": float(rho.pvalue), "n": len(st)}
        rob["system_level_spearman_P_vs_negative_PUK28_voice_distance"] = rank_corr(base, sysmean)
        rl = spearmanr([r["p_lemkin"] for r in st], [r["draft_words"] for r in st])
        rob["draft_level_spearman_P_vs_total_draft_words"] = {"rho": float(rl.statistic), "p": float(rl.pvalue)}
        within = {}
        for sysn in sorted({r["system"] for r in st}):
            pairs = [(r["p_lemkin"], r["draft_words"]) for r in rows if r["system"] == sysn and r["view"] == "style" and r["C"] == C_MAIN]
            rr = spearmanr([a for a, _ in pairs], [b for _, b in pairs])
            within[sysn] = {"rho": float(rr.statistic), "p": float(rr.pvalue), "n": len(pairs)}
        rob["within_system_spearman_P_vs_draft_words"] = within
        m["robustness"] = rob
        m["n_drafts_scored"] = collections.Counter(r["system"] for r in rows if r["view"] == "style" and r["C"] == C_MAIN)
        m["n_drafts_dropped_for_copy"] = collections.Counter(r["system"] for r in rows if r["view"] == "style" and r["C"] == C_MAIN and r["copies_train_8gram"])
        m.pop("_tm", None)
        # keep real / PG / base score vectors for the validity picture
        np.savez(OUT / f"validity_scores_{medium}.npz", real=res["real"][("style", C_MAIN)], pg=res["pg"][("style", C_MAIN)],
                 base_llama=res["heldllm"][("style", C_MAIN)], real_full=res["real"][("full", C_MAIN)],
                 pg_full=res["pg"][("full", C_MAIN)], base_llama_full=res["heldllm"][("full", C_MAIN)])
        report["media"][medium] = m
    report["top_features"] = explain_features(drafts)
    import csv
    with open(OUT / "scores_per_draft.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(per_draft_all[0].keys())); w.writeheader(); w.writerows(per_draft_all)
    with open(OUT / "results.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=lambda o: dict(o) if isinstance(o, collections.Counter) else str(o))
    log("done. outputs written to", OUT)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Does a model draft look like Jason Lemkin? Authorship classifier pipeline")
    ap.add_argument("--llm-negatives", help="jsonl with one base-Llama text per line: {\"text\": ..., \"medium\": \"blog\"|\"talk\"|\"linkedin\" (optional)}")
    ap.add_argument("--out-dir", default="outputs", help="folder name under this directory for results")
    a = ap.parse_args()
    OUT = ROOT / a.out_dir; OUT.mkdir(exist_ok=True)
    if a.llm_negatives:
        EXTERNAL_NEG = [json.loads(l) for l in open(a.llm_negatives, encoding="utf-8") if l.strip()]
        log(f"using {len(EXTERNAL_NEG)} external base-Llama texts as negatives")
    main()
