"""Authorship classifier: real Lemkin text against another writer, two views.

view "full"  : all words (word 1-2 grams). Can use topic words, so it is the weaker proof.
view "style_nonum": same as style but numbers and $ % + are ignored (guards against rewarding invented figures).
view "style" : content words masked, only function words and punctuation remain.
               This is the one to trust, because topic cannot carry the decision.
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from functools import partial

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, balanced_accuracy_score

from textprep import mask_to_style

C_GRID = (0.1, 0.3, 1.0, 3.0, 10.0)


def make_vectorizer(view: str):
    if view == "full":
        return TfidfVectorizer(lowercase=True, ngram_range=(1, 2), min_df=3, max_features=60000, sublinear_tf=True)
    if view == "style":
        return TfidfVectorizer(lowercase=False, token_pattern=r"\S+", ngram_range=(1, 3), min_df=3,
                               max_features=60000, sublinear_tf=True, preprocessor=mask_to_style)
    if view == "style_nonum":
        return TfidfVectorizer(lowercase=False, token_pattern=r"\S+", ngram_range=(1, 3), min_df=3,
                               max_features=60000, sublinear_tf=True, preprocessor=partial(mask_to_style, drop_numbers=True))
    raise ValueError(view)


@dataclass
class Fitted:
    view: str
    vec: TfidfVectorizer
    clf: LogisticRegression
    C: float
    val_auc: float

    def score(self, texts: list[str]) -> np.ndarray:
        """P(text looks like Lemkin). 0.5 means the model cannot tell."""
        return self.clf.predict_proba(self.vec.transform(texts))[:, 1]


def fit(view: str, train: list[dict], val: list[dict]) -> Fitted:
    """Fit on train windows only. Choose C on validation AUC. Test is never touched here."""
    vec = make_vectorizer(view)
    Xtr = vec.fit_transform([w["text"] for w in train])
    ytr = np.array([w["label"] for w in train])
    Xva = vec.transform([w["text"] for w in val])
    yva = np.array([w["label"] for w in val])
    best = None
    for C in C_GRID:
        clf = LogisticRegression(C=C, class_weight="balanced", max_iter=3000, solver="liblinear")
        clf.fit(Xtr, ytr)
        auc = roc_auc_score(yva, clf.predict_proba(Xva)[:, 1])
        if best is None or auc > best[0]:
            best = (auc, C, clf)
    return Fitted(view, vec, best[2], best[1], float(best[0]))


def auc_with_ci(y: np.ndarray, p: np.ndarray, docs: list[str], n_boot: int = 500, seed: int = 0):
    """AUC with a 95% interval that resamples whole documents (windows of one post are not independent)."""
    y = np.asarray(y); p = np.asarray(p)
    base = float(roc_auc_score(y, p))
    by_doc: dict[tuple[int, str], list[int]] = {}
    for i, (lab, d) in enumerate(zip(y, docs)):
        by_doc.setdefault((int(lab), d), []).append(i)
    pos = [v for k, v in by_doc.items() if k[0] == 1]
    neg = [v for k, v in by_doc.items() if k[0] == 0]
    rng = random.Random(seed)
    stats = []
    for _ in range(n_boot):
        idx = [i for g in (rng.choice(pos) for _ in pos) for i in g] + [i for g in (rng.choice(neg) for _ in neg) for i in g]
        yy = y[idx]
        if yy.min() == yy.max():
            continue
        stats.append(roc_auc_score(yy, p[idx]))
    lo, hi = np.percentile(stats, [2.5, 97.5])
    return base, float(lo), float(hi)


def summarize(y, p) -> dict:
    y = np.asarray(y); p = np.asarray(p)
    return {"auc": float(roc_auc_score(y, p)),
            "balanced_accuracy_at_0.5": float(balanced_accuracy_score(y, (p >= 0.5).astype(int)))}
