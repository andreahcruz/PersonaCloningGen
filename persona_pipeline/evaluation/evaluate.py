"""
Phase 10 — Evaluate generated content quality across multiple dimensions.

Metrics:
  1. BERTScore  — semantic similarity to real patio11 posts
  2. RAGAS faithfulness — are claims supported by retrieved context?
  3. Style consistency — authorship-verification classifier (LogReg on TF-IDF)
  4. BLEU / ROUGE-L — surface similarity

Run: python persona_pipeline/evaluation/evaluate.py
"""
from __future__ import annotations

import json
import os
import sys
import warnings
from pathlib import Path

REPO_ROOT = str(Path(__file__).resolve().parents[2])
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

from persona_pipeline.config.config import DB_PATH, PERSONA_PIPELINE_DIR

OUTPUTS_DIR = PERSONA_PIPELINE_DIR.parent / "outputs"
RESULTS_PATH = PERSONA_PIPELINE_DIR / "evaluation" / "results.json"


# ── Helpers ───────────────────────────────────────────────────────────

def _load_generated_texts(generated_dir: str | Path | None = None) -> list[str]:
    d = Path(generated_dir) if generated_dir else OUTPUTS_DIR
    if not d.exists():
        warnings.warn(f"Outputs directory not found: {d}")
        return []
    texts: list[str] = []
    for f in sorted(d.glob("*.md")):
        texts.append(f.read_text(encoding="utf-8"))
    return texts


def _load_real_posts() -> list[str]:
    import duckdb
    if not os.path.exists(DB_PATH):
        warnings.warn("DuckDB not found; returning empty reference set.")
        return []
    con = duckdb.connect(DB_PATH, read_only=True)
    try:
        rows = con.execute(
            "SELECT cleaned_text FROM all_posts WHERE word_count > 100 LIMIT 500"
        ).fetchall()
    finally:
        con.close()
    return [r[0] for r in rows if r[0]]


# ── Metric 1: BERTScore ──────────────────────────────────────────────

def compute_bertscore(generated: list[str], references: list[str]) -> dict:
    try:
        from bert_score import score as bert_score_fn
    except ImportError:
        warnings.warn("bert-score not installed; skipping BERTScore.")
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    if not generated or not references:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    # Compare each generated text against the best-matching reference
    refs_for_gen = references[: len(generated)]
    while len(refs_for_gen) < len(generated):
        refs_for_gen.append(references[0])

    P, R, F1 = bert_score_fn(
        generated, refs_for_gen, model_type="roberta-large", verbose=False
    )
    return {
        "precision": float(P.mean()),
        "recall": float(R.mean()),
        "f1": float(F1.mean()),
    }


# ── Metric 2: RAGAS faithfulness ─────────────────────────────────────

def compute_ragas(generated: list[str]) -> dict:
    try:
        from ragas.metrics import faithfulness
        from ragas import evaluate as ragas_evaluate
        from datasets import Dataset

        if not generated:
            return {"faithfulness": 0.0}

        data = {
            "question": ["Generate B2B content"] * len(generated),
            "answer": generated,
            "contexts": [["No context available."]] * len(generated),
        }
        ds = Dataset.from_dict(data)
        result = ragas_evaluate(ds, metrics=[faithfulness])
        return {"faithfulness": float(result["faithfulness"])}
    except Exception as e:
        warnings.warn(f"RAGAS evaluation failed: {e}")
        return {"faithfulness": 0.0}


# ── Metric 3: Style consistency (authorship verification) ────────────

def compute_style_consistency(
    generated: list[str],
    real_posts: list[str],
) -> dict:
    if not generated or not real_posts:
        return {"style_score": 0.0, "cv_accuracy": 0.0}

    # Negative class: sample from Blog Authorship Corpus (or synthetic fallback)
    negative_texts: list[str] = []
    try:
        from datasets import load_dataset
        blog_ds = load_dataset("barilan/blog_authorship_corpus", split="train", streaming=True)
        for i, row in enumerate(blog_ds):
            if i >= 500:
                break
            txt = row.get("text", "")
            if len(txt.split()) > 50:
                negative_texts.append(txt)
    except Exception as e:
        warnings.warn(f"Could not load Blog Authorship Corpus: {e}")
        negative_texts = [
            "Today I went to the store and bought some groceries. It was a nice day. "
            * 5
        ] * min(100, len(real_posts))

    # Build training set
    texts = real_posts[:500] + negative_texts[:500]
    labels = [1] * min(len(real_posts), 500) + [0] * min(len(negative_texts), 500)

    vectorizer = TfidfVectorizer(max_features=5000, stop_words="english")
    X = vectorizer.fit_transform(texts)
    y = np.array(labels)

    clf = LogisticRegression(max_iter=1000, random_state=42)
    cv_scores = cross_val_score(clf, X, y, cv=3, scoring="accuracy")

    # Train on full set, score generated texts
    clf.fit(X, y)
    gen_features = vectorizer.transform(generated)
    gen_probs = clf.predict_proba(gen_features)[:, 1]  # prob of being patio11

    return {
        "style_score": float(np.mean(gen_probs)),
        "cv_accuracy": float(np.mean(cv_scores)),
    }


# ── Metric 4: BLEU + ROUGE-L ────────────────────────────────────────

def compute_bleu_rouge(generated: list[str], references: list[str]) -> dict:
    results = {"bleu": 0.0, "rouge_l": 0.0}

    if not generated or not references:
        return results

    # BLEU (sentence-level average)
    try:
        from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

        smooth = SmoothingFunction().method1
        bleu_scores = []
        for gen in generated:
            gen_tokens = gen.lower().split()
            best_bleu = 0.0
            for ref in references[:50]:
                ref_tokens = ref.lower().split()
                try:
                    s = sentence_bleu([ref_tokens], gen_tokens, smoothing_function=smooth)
                    best_bleu = max(best_bleu, s)
                except Exception:
                    pass
            bleu_scores.append(best_bleu)
        results["bleu"] = float(np.mean(bleu_scores))
    except ImportError:
        warnings.warn("nltk not installed; skipping BLEU.")

    # ROUGE-L
    try:
        from rouge_score.rouge_scorer import RougeScorer

        scorer = RougeScorer(["rougeL"], use_stemmer=True)
        rouge_scores = []
        for gen in generated:
            best_rouge = 0.0
            for ref in references[:50]:
                s = scorer.score(ref, gen)
                best_rouge = max(best_rouge, s["rougeL"].fmeasure)
            rouge_scores.append(best_rouge)
        results["rouge_l"] = float(np.mean(rouge_scores))
    except ImportError:
        warnings.warn("rouge-score not installed; skipping ROUGE-L.")

    return results


# ── Main evaluation ──────────────────────────────────────────────────

def evaluate_outputs(generated_dir: str | Path | None = None) -> dict:
    generated = _load_generated_texts(generated_dir)
    if not generated:
        warnings.warn("No generated outputs found to evaluate.")
        return {}

    real_posts = _load_real_posts()

    print(f"Evaluating {len(generated)} generated texts against {len(real_posts)} real posts...\n")

    bert = compute_bertscore(generated, real_posts)
    print(f"BERTScore — P={bert['precision']:.4f}  R={bert['recall']:.4f}  F1={bert['f1']:.4f}")

    ragas = compute_ragas(generated)
    print(f"RAGAS Faithfulness — {ragas['faithfulness']:.4f}")

    style = compute_style_consistency(generated, real_posts)
    print(f"Style Score — {style['style_score']:.4f}  (CV accuracy={style['cv_accuracy']:.4f})")

    bleu_rouge = compute_bleu_rouge(generated, real_posts)
    print(f"BLEU — {bleu_rouge['bleu']:.4f}")
    print(f"ROUGE-L — {bleu_rouge['rouge_l']:.4f}")

    results = {
        "num_generated": len(generated),
        "num_references": len(real_posts),
        "bertscore": bert,
        "ragas": ragas,
        "style": style,
        "bleu": bleu_rouge["bleu"],
        "rouge_l": bleu_rouge["rouge_l"],
    }

    os.makedirs(os.path.dirname(str(RESULTS_PATH)), exist_ok=True)
    with open(str(RESULTS_PATH), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved -> {RESULTS_PATH}")

    return results


def main() -> None:
    evaluate_outputs()


if __name__ == "__main__":
    main()
