#!/usr/bin/env python3
"""End-to-end RAG eval harness for the Lemkin fine-tune.

For each row in an eval-set JSONL:
  1. Build the same RAG prompt as ``generate.py`` / Streamlit (same persona,
     same Chroma retrieval, same format spec) for every model under test.
  2. Call Ollama ``/api/generate`` and store a trace row
     ``{id, model, query, contexts, answer, latency_seconds, ...}``.
  3. Score:
       • G-Eval rubric via an Ollama judge: corpus-backed author profile
         (``persona_profile.json``), optional real-post excerpt anchors from
         ``data/jasonlemkinlinkedin.jsonl`` + ``data/jasonlemkin_blog.jsonl`` (and
         ``--dataset-jsonl`` when present), plus per-dimension Lemkin-style scores.
       • BERTScore vs a sample of real Lemkin outputs from ``dataset.jsonl``.
       • BLEU + ROUGE-L vs the same reference sample.
       • Style consistency — TF-IDF + LogisticRegression authorship classifier
         (positives: Lemkin outputs; negatives: Blog Authorship Corpus if
         reachable, else a synthetic fallback).
  4. Optionally compute RAGAS (faithfulness / answer_relevancy) if ``ragas``
     and an LLM backend are configured (best-effort — failures are reported,
     not fatal).

Example (compare base vs fine-tune, against Docker Chroma):
    set CHROMA_USE_HTTP=true
    set CHROMA_HOST=localhost
    set CHROMA_PORT=8000
    python -m host_finetune.eval_rag --models llama3.1,lemkin-clone

Reuse cached traces (skip generation; re-score only):
    python -m host_finetune.eval_rag --skip-generate

Outputs (under ``host_finetune/output/eval/`` by default):
    traces.jsonl   — one trace row per (model, eval row)
    geval.jsonl    — judge scores per trace
    summary.json   — per-model aggregates (G-Eval averages, latency, optional RAGAS)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Reuse the same RAG functions Streamlit / generate.py use so the eval reflects
# production behavior (persona prompt, Chroma collection, context window).
from generate import (  # noqa: E402
    CONTEXT_MAX_CHARS,
    DEFAULT_EMBED_MODEL,
    OLLAMA_BASE_DEFAULT,
    build_prompt,
    format_context,
    load_format_spec,
    load_persona,
    ollama_embed,
    ollama_generate,
    retrieve,
    summarize_persona,
)

DEFAULT_EVAL_SET = REPO_ROOT / "host_finetune" / "eval_set.example.jsonl"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "host_finetune" / "output" / "eval"
DEFAULT_PERSONA = REPO_ROOT / "data" / "persona_profile.json"
DEFAULT_FORMAT_SPECS = REPO_ROOT / "formats" / "format_specs.yaml"
DEFAULT_INDEX = REPO_ROOT / "data" / "index"
DEFAULT_JUDGE_MODEL = "llama3.1"
DEFAULT_DATASET_JSONL = REPO_ROOT / "host_finetune" / "data" / "dataset.jsonl"
DEFAULT_BERTSCORE_MODEL = "roberta-large"
# Compare each generated text against this many reference samples (max-F1 per generated).
DEFAULT_REF_PER_CANDIDATE = 20
# Reference corpus sizes (load this many positives, train style classifier on this many).
REF_SAMPLE_SIZE = 500
STYLE_NEG_SAMPLE_SIZE = 500
# Raw Lemkin sources used only for G-Eval style anchors (short excerpts in judge prompt).
DEFAULT_STYLE_ANCHOR_JSONLS: tuple[Path, ...] = (
    REPO_ROOT / "data" / "jasonlemkinlinkedin.jsonl",
    REPO_ROOT / "data" / "jasonlemkin_blog.jsonl",
)


# ── Eval set ───────────────────────────────────────────────────────────

def load_eval_set(path: Path) -> list[dict]:
    rows: list[dict] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("//"):
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"[eval_rag] skipping malformed eval row: {e}")
    if not rows:
        raise RuntimeError(f"No usable rows in {path}.")
    return rows


# ── RAG generation pass ────────────────────────────────────────────────

def generate_trace(
    row: dict,
    *,
    gen_model: str,
    ollama_base: str,
    embed_model: str,
    format_specs_path: Path,
    persona_path: Path,
    index_path: Path,
) -> dict:
    fmt = row.get("format", "linkedin_post")
    topic = (row.get("topic") or "").strip()
    if not topic:
        raise ValueError(f"eval row {row.get('id')!r} missing topic")
    audience = (row.get("audience") or "").strip()
    goal = (row.get("goal") or "").strip()
    cta = (row.get("cta") or "none").strip() or "none"
    k = int(row.get("k", 8))

    format_spec = load_format_spec(format_specs_path, fmt)
    persona = load_persona(persona_path)
    persona_summary = summarize_persona(persona)

    # Same query construction as user_interface.py / generate.py.
    query_text = f"{topic}. {goal}. Audience: {audience}."
    query_embed = ollama_embed(query_text, ollama_base, embed_model)
    chunks = retrieve(index_path, query_embed, k, embed_model)
    context_block = format_context(chunks, max_chars=CONTEXT_MAX_CHARS)
    prompt = build_prompt(
        format_spec, persona_summary, topic, audience, goal, cta, context_block,
    )

    t0 = time.time()
    answer = ollama_generate(prompt, ollama_base, gen_model)
    latency = time.time() - t0

    return {
        "id": row.get("id") or topic[:48],
        "model": gen_model,
        "format": fmt,
        "topic": topic,
        "audience": audience,
        "goal": goal,
        "cta": cta,
        "k": k,
        "query": query_text,
        "contexts": [str(c.get("chunk_text", "")) for c in chunks],
        "context_meta": [
            {"doc_id": c.get("doc_id"), "source": c.get("source")} for c in chunks
        ],
        "answer": answer,
        "gold_reference": row.get("gold_reference"),
        "latency_seconds": round(latency, 2),
        "prompt_chars": len(prompt),
        "answer_chars": len(answer),
    }


# ── G-Eval rubric (Ollama judge) ──────────────────────────────────────
# Style dimensions are grounded in raw Lemkin posts (LinkedIn + SaaStr blog JSONL)
# and in ``persona_profile.json`` (corpus statistics). The judge sees a short real
# excerpt as a texture anchor, not as a fact source to agree with.

RUBRIC_DIMENSIONS: list[tuple[str, str]] = [
    (
        "lemkin_concreteness",
        "Uses specific numbers, $, %, time horizons, headcounts, or other quantified "
        "facts when the topic allows; penalize empty superlatives ('massive traction') "
        "with no anchors.",
    ),
    (
        "lemkin_operator_voice",
        "Direct founder-to-operator advice: blunt where useful, practical, confident. "
        "Penalize generic LinkedIn inspo, corporate polish, or hypey listicle voice.",
    ),
    (
        "lemkin_structure_rhythm",
        "High-signal pacing: short paragraphs, clear pivots ('Here's what...', "
        "contrasts, numbered actions when fitting). Penalize long essay walls that "
        "read unlike Lemkin's usual punch.",
    ),
    (
        "lemkin_domain_register",
        "Natural SaaS / GTM vocabulary when relevant (ARR, NRR/MRR, churn, pipeline, "
        "ACV, CS, AE, enterprise vs SMB, etc.) used correctly — not jargon salad.",
    ),
    (
        "coherence",
        "Logical flow; no internal contradictions; claims hang together.",
    ),
    (
        "format_adherence",
        "Matches the requested format (length band, structural cues from the prompt).",
    ),
    (
        "context_use",
        "Uses retrieved context for substance without copying long chunks verbatim; "
        "no invented 'facts' unsupported by context or obvious common knowledge.",
    ),
]


def _row_text_for_style_anchor(row: dict) -> str:
    for key in ("output", "content", "text", "body"):
        val = row.get(key)
        if isinstance(val, str) and val.strip():
            return val.strip()
    return ""


def _strip_trailing_blog_noise(text: str) -> str:
    """Drop trailing boilerplate so anchors focus on Lemkin's voice."""
    lower = text.lower()
    cut = lower.rfind("\nrelated posts")
    if cut > 120:
        text = text[:cut].strip()
    return text


def collect_style_anchor_source_paths(primary_dataset: Path) -> list[Path]:
    """De-duplicated list of JSONL files that may supply real-post anchors."""
    paths: list[Path] = [primary_dataset, *DEFAULT_STYLE_ANCHOR_JSONLS]
    seen: set[Path] = set()
    out: list[Path] = []
    for p in paths:
        rp = p.resolve()
        if rp in seen or not rp.is_file():
            continue
        seen.add(rp)
        out.append(rp)
    return out


def load_style_anchor_texts(
    paths: list[Path],
    *,
    max_samples: int = 400,
    min_words: int = 32,
) -> list[str]:
    """Load short-post candidates from JSONL (``output`` / ``content`` / …)."""
    texts: list[str] = []
    seen_hashes: set[str] = set()
    for path in paths:
        if len(texts) >= max_samples:
            break
        try:
            with open(path, encoding="utf-8") as f:
                for line in f:
                    if len(texts) >= max_samples:
                        break
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        row = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    raw = _row_text_for_style_anchor(row)
                    if not raw:
                        continue
                    raw = _strip_trailing_blog_noise(raw)
                    if len(raw.split()) < min_words:
                        continue
                    h = hashlib.sha256(raw[:240].encode("utf-8", errors="ignore")).hexdigest()[:16]
                    if h in seen_hashes:
                        continue
                    seen_hashes.add(h)
                    texts.append(raw)
        except OSError:
            continue
    return texts


def _truncate_anchor(text: str, max_chars: int = 680) -> str:
    if len(text) <= max_chars:
        return text
    chunk = text[:max_chars]
    last_period = chunk.rfind(". ")
    if last_period > max_chars // 2:
        return chunk[: last_period + 1].strip()
    return chunk.rsplit(maxsplit=1)[0].strip() + "…"


def _pick_style_anchor(anchors: list[str], seed_key: str) -> str | None:
    if not anchors:
        return None
    h = int(hashlib.md5(seed_key.encode("utf-8")).hexdigest()[:8], 16)
    return _truncate_anchor(anchors[h % len(anchors)])


def _geval_author_profile_for_judge(persona: dict) -> str:
    """Factual checklist from ``persona_profile.json`` (same stats used for generation)."""
    vocab = persona.get("vocabulary_tendencies", {})
    structure = persona.get("structure_patterns", {})
    framing = persona.get("favorite_framing", {})
    themes = persona.get("favorite_themes", {})
    meta = persona.get("meta", {})
    words = ", ".join(vocab.get("top_50_content_words", [])[:22])
    starters = ", ".join(structure.get("common_sentence_starters_bigrams", [])[:10])
    phrases = ", ".join(framing.get("favorite_framing_phrases", [])[:12])
    theme_bits = themes.get("top_content_words_and_bigrams", [])[:14]
    theme_line = ", ".join(str(t) for t in theme_bits) if theme_bits else "B2B SaaS, GTM, fundraising"
    src = meta.get("source", "Jason Lemkin — blog, LinkedIn, SaaStr")
    return (
        f"Target author (corpus-derived): {src}\n"
        f"- Frequent operator vocabulary (examples): {words}.\n"
        f"- Recurrent openings / pivots (natural use, not forced every sentence): {starters}.\n"
        f"- Common framing moves: {phrases}.\n"
        f"- Recurring themes / collocations: {theme_line}."
    )


def _geval_json_shape() -> str:
    lines = [f'  "{k}": <int>,' for k, _ in RUBRIC_DIMENSIONS]
    lines.append('  "comment": "<one short sentence>"')
    return "{\n" + "\n".join(lines) + "\n}"


def _geval_prompt(
    trace: dict,
    *,
    author_profile: str,
    style_anchor: str | None,
) -> str:
    contexts = "\n\n---\n".join(trace.get("contexts", [])[:6])
    rubric_lines = "\n".join(f"- {k}: {desc}" for k, desc in RUBRIC_DIMENSIONS)
    gold = (trace.get("gold_reference") or "").strip()
    anchor_block = ""
    if gold:
        anchor_block = (
            "OPTIONAL HUMAN REFERENCE (same assignment if provided — compare tone/texture, "
            "not factual overlap):\n"
            f"{_truncate_anchor(gold, 720)}\n\n"
        )
    elif style_anchor:
        anchor_block = (
            "REAL POST EXCERPT (same author, unrelated topic — judge voice, pacing, and "
            "texture only; do not penalize the DRAFT for disagreeing on facts):\n"
            f"{style_anchor}\n\n"
        )
    return (
        "You are an impartial reviewer. Score the DRAFT on each rubric dimension "
        "1–5 (integer), where 5 = strong match to that dimension.\n"
        "For the four lemkin_* dimensions, prioritize similarity to Jason Lemkin's "
        "observed writing (see AUTHOR PROFILE and optional excerpt below), not generic "
        "'good writing'.\n"
        "Return STRICT JSON only, with this exact shape and nothing else:\n"
        f"{_geval_json_shape()}\n\n"
        "AUTHOR PROFILE (corpus statistics — style target):\n"
        f"{author_profile}\n\n"
        f"{anchor_block}"
        f"Rubric:\n{rubric_lines}\n\n"
        f"Topic: {trace.get('topic')}\n"
        f"Audience: {trace.get('audience')}\n"
        f"Goal: {trace.get('goal')}\n"
        f"Format: {trace.get('format')}\n\n"
        "Retrieved context (excerpts; use for context_use and factual grounding):\n"
        f"{contexts[:3500]}\n\n"
        "DRAFT:\n"
        f"{(trace.get('answer') or '')[:6000]}\n\n"
        "JSON ONLY:\n"
    )


def _extract_json_object(text: str) -> dict | None:
    if not text:
        return None
    s = text.strip()
    # Strip markdown fences if the judge wrapped them.
    if s.startswith("```"):
        s = s.strip("`")
        if s.lower().startswith("json"):
            s = s[4:].lstrip()
    start = s.find("{")
    end = s.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        return json.loads(s[start : end + 1])
    except json.JSONDecodeError:
        return None


def geval_score(
    trace: dict,
    judge_model: str,
    ollama_base: str,
    *,
    author_profile: str,
    style_anchors: list[str],
) -> dict | None:
    key = f"{trace.get('id', '')}|{trace.get('model', '')}|{trace.get('topic', '')}"
    anchor = _pick_style_anchor(style_anchors, key) if style_anchors else None
    raw = ollama_generate(
        _geval_prompt(trace, author_profile=author_profile, style_anchor=anchor),
        ollama_base,
        judge_model,
    )
    parsed = _extract_json_object(raw)
    if not parsed:
        return None
    out: dict = {}
    for dim, _ in RUBRIC_DIMENSIONS:
        v = parsed.get(dim)
        if isinstance(v, bool):
            continue
        if isinstance(v, (int, float)) and 1 <= v <= 5:
            out[dim] = float(v)
    if not out:
        return None
    if isinstance(parsed.get("comment"), str):
        out["comment"] = parsed["comment"][:240]
    return out


# ── Reference corpus from dataset.jsonl ───────────────────────────────

def load_reference_corpus(
    dataset_jsonl: Path,
    *,
    max_samples: int = REF_SAMPLE_SIZE,
    min_words: int = 80,
) -> list[str]:
    """Sample ``output`` fields from ``dataset.jsonl`` as the real-Lemkin corpus.

    Skips short / empty rows so BERTScore + BLEU references are informative.
    """
    out: list[str] = []
    if not dataset_jsonl.is_file():
        return out
    with open(dataset_jsonl, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            text = (row.get("output") or "").strip()
            if not text or len(text.split()) < min_words:
                continue
            out.append(text)
            if len(out) >= max_samples:
                break
    return out


def _negative_corpus(n: int = STYLE_NEG_SAMPLE_SIZE) -> list[str]:
    """Non-Lemkin text for the authorship classifier. Falls back to synthetic
    English when the Blog Authorship Corpus isn't reachable (e.g. offline)."""
    try:
        from datasets import load_dataset

        blog = load_dataset(
            "barilan/blog_authorship_corpus", split="train", streaming=True,
        )
        out: list[str] = []
        for i, row in enumerate(blog):
            if i >= n:
                break
            t = (row.get("text") or "").strip()
            if len(t.split()) > 50:
                out.append(t)
        if out:
            return out
    except Exception as e:
        print(f"[eval_rag] negative corpus fallback (synthetic): {e!r}")
    # Weak fallback: keeps the classifier trainable when offline.
    return [
        "Today I worked on a feature and refactored a few files. We discussed the "
        "roadmap and aligned on priorities. The week was productive overall."
    ] * n


# ── BERTScore ─────────────────────────────────────────────────────────

def compute_bertscore(
    generated: list[str],
    references: list[str],
    *,
    model_type: str = DEFAULT_BERTSCORE_MODEL,
    refs_per_candidate: int = DEFAULT_REF_PER_CANDIDATE,
) -> dict | None:
    """Best-of-K BERTScore per candidate. Returns ``None`` if bert-score missing."""
    if not generated or not references:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0, "n_candidates": 0}
    try:
        from bert_score import score as bert_score_fn  # type: ignore
    except ImportError:
        return None

    import random
    rng = random.Random(42)
    refs_pool = list(references)

    cand_repeats: list[str] = []
    ref_repeats: list[str] = []
    per_cand_idx: list[int] = []
    k = min(refs_per_candidate, len(refs_pool))
    for ci, cand in enumerate(generated):
        sample = rng.sample(refs_pool, k) if k < len(refs_pool) else refs_pool
        cand_repeats.extend([cand] * len(sample))
        ref_repeats.extend(sample)
        per_cand_idx.extend([ci] * len(sample))

    try:
        P, R, F1 = bert_score_fn(
            cand_repeats, ref_repeats, model_type=model_type, verbose=False,
        )
    except Exception as e:
        return {"error": f"bertscore failed: {e!r}"}
    p_list = P.tolist()
    r_list = R.tolist()
    f_list = F1.tolist()

    # Reduce to one (p,r,f) per candidate by taking the best F1 against any reference.
    n = len(generated)
    best: list[tuple[float, float, float]] = [(-1.0, -1.0, -1.0)] * n
    for ci, p, r, f in zip(per_cand_idx, p_list, r_list, f_list):
        if f > best[ci][2]:
            best[ci] = (p, r, f)

    p_arr = [b[0] for b in best if b[2] >= 0]
    r_arr = [b[1] for b in best if b[2] >= 0]
    f_arr = [b[2] for b in best if b[2] >= 0]
    if not f_arr:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0, "n_candidates": 0}
    return {
        "precision": round(statistics.mean(p_arr), 4),
        "recall": round(statistics.mean(r_arr), 4),
        "f1": round(statistics.mean(f_arr), 4),
        "n_candidates": len(f_arr),
        "model_type": model_type,
    }


# ── BLEU + ROUGE-L ────────────────────────────────────────────────────

def compute_bleu_rouge(
    generated: list[str],
    references: list[str],
    *,
    refs_per_candidate: int = DEFAULT_REF_PER_CANDIDATE,
) -> dict:
    """Best-of-K BLEU and ROUGE-L per candidate, averaged across candidates."""
    out: dict = {"bleu": None, "rouge_l": None}
    if not generated or not references:
        out["bleu"] = 0.0
        out["rouge_l"] = 0.0
        return out

    import random
    rng = random.Random(42)
    pool = list(references)
    k = min(refs_per_candidate, len(pool))

    # BLEU (sentence-level, smoothed)
    try:
        from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu  # type: ignore

        smooth = SmoothingFunction().method1
        bleu_scores: list[float] = []
        for cand in generated:
            sample = rng.sample(pool, k) if k < len(pool) else pool
            cand_toks = cand.lower().split()
            best = 0.0
            for ref in sample:
                ref_toks = ref.lower().split()
                try:
                    s = sentence_bleu([ref_toks], cand_toks, smoothing_function=smooth)
                except Exception:
                    continue
                if s > best:
                    best = s
            bleu_scores.append(best)
        out["bleu"] = round(statistics.mean(bleu_scores), 4) if bleu_scores else 0.0
    except ImportError:
        out["bleu_error"] = "nltk not installed"

    # ROUGE-L
    try:
        from rouge_score.rouge_scorer import RougeScorer  # type: ignore

        scorer = RougeScorer(["rougeL"], use_stemmer=True)
        rouge_scores: list[float] = []
        rng2 = random.Random(43)  # independent sample to avoid double bias
        for cand in generated:
            sample = rng2.sample(pool, k) if k < len(pool) else pool
            best = 0.0
            for ref in sample:
                try:
                    s = scorer.score(ref, cand)["rougeL"].fmeasure
                except Exception:
                    continue
                if s > best:
                    best = s
            rouge_scores.append(best)
        out["rouge_l"] = round(statistics.mean(rouge_scores), 4) if rouge_scores else 0.0
    except ImportError:
        out["rouge_error"] = "rouge-score not installed"

    return out


# ── Style consistency (TF-IDF + LR authorship classifier) ─────────────

def fit_style_classifier(positives: list[str], negatives: list[str]):
    """Train a TF-IDF + LogisticRegression author-vs-other classifier.

    Returns ``(vectorizer, classifier, cv_accuracy)`` or ``None`` if scikit-learn
    isn't available or the input corpora are degenerate.
    """
    if not positives or not negatives:
        return None
    try:
        import numpy as np
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import cross_val_score
    except ImportError:
        return None

    p = positives[:REF_SAMPLE_SIZE]
    n = negatives[:STYLE_NEG_SAMPLE_SIZE]
    texts = p + n
    labels = np.array([1] * len(p) + [0] * len(n))
    vec = TfidfVectorizer(max_features=5000, stop_words="english")
    X = vec.fit_transform(texts)
    clf = LogisticRegression(max_iter=1000, random_state=42)
    try:
        cv = cross_val_score(clf, X, labels, cv=3, scoring="accuracy")
    except Exception:
        cv = np.array([0.0])
    clf.fit(X, labels)
    return vec, clf, float(cv.mean())


def score_with_style_classifier(
    generated: list[str],
    fitted,
) -> dict:
    """Per-candidate P(Lemkin) under the fitted classifier."""
    if not generated or fitted is None:
        return {"style_score": None, "cv_accuracy": None, "per_candidate": []}
    vec, clf, cv_acc = fitted
    X = vec.transform(generated)
    probs = clf.predict_proba(X)[:, 1].tolist()
    return {
        "style_score": round(statistics.mean(probs), 4),
        "cv_accuracy": round(cv_acc, 4),
        "per_candidate": [round(p, 4) for p in probs],
    }


# ── Optional RAGAS ────────────────────────────────────────────────────

def ragas_scores(traces: list[dict]) -> dict | None:
    """Best-effort RAGAS. Returns None if the package is absent, or an error
    payload if ragas couldn't reach its configured LLM/embeddings. To enable:
        pip install -U "ragas>=0.2" "datasets"
    and configure ragas's LLM backend (typically OPENAI_API_KEY or a
    LangChain LLM). Without a backend ragas raises at evaluate time."""
    try:
        from datasets import Dataset
        from ragas import evaluate as ragas_evaluate
        from ragas.metrics import answer_relevancy, faithfulness
    except ImportError:
        return None
    try:
        ds = Dataset.from_dict({
            "question": [t["query"] for t in traces],
            "answer": [t["answer"] for t in traces],
            "contexts": [t["contexts"] for t in traces],
        })
        result = ragas_evaluate(ds, metrics=[faithfulness, answer_relevancy])
        try:
            return {k: float(result[k]) for k in result.keys()}
        except Exception:
            return {"raw_result": str(result)}
    except Exception as e:
        return {"error": repr(e)}


# ── Main ──────────────────────────────────────────────────────────────

def _aggregate(
    traces: list[dict],
    judge_rows: list[dict],
    models: list[str],
    *,
    references: list[str] | None = None,
    style_classifier=None,
    do_bertscore: bool = True,
    do_bleu_rouge: bool = True,
    do_style: bool = True,
    bertscore_model: str = DEFAULT_BERTSCORE_MODEL,
    refs_per_candidate: int = DEFAULT_REF_PER_CANDIDATE,
) -> dict:
    summary: dict = {"models": {}}
    for model in models:
        m_traces = [t for t in traces if t.get("model") == model]
        m_scores = [
            r["scores"] for r in judge_rows
            if r.get("model") == model and isinstance(r.get("scores"), dict)
        ]
        per_dim: dict = {}
        for dim, _ in RUBRIC_DIMENSIONS:
            vals = [s[dim] for s in m_scores if dim in s]
            if vals:
                per_dim[dim] = round(statistics.mean(vals), 3)
        if per_dim:
            per_dim["overall"] = round(statistics.mean(per_dim.values()), 3)
        latencies = [t.get("latency_seconds", 0.0) for t in m_traces]
        entry: dict = {
            "n_traces": len(m_traces),
            "avg_answer_chars": round(
                statistics.mean([t.get("answer_chars", 0) for t in m_traces]), 1
            ) if m_traces else 0,
            "avg_latency_seconds": round(statistics.mean(latencies), 2) if latencies else 0.0,
            "geval": per_dim,
            "n_judge_parsed": len(m_scores),
        }

        generated = [t.get("answer", "") for t in m_traces if t.get("answer")]

        if do_bertscore and generated and references:
            print(f"  [bertscore] model={model} ({len(generated)} candidates, {bertscore_model})")
            bs = compute_bertscore(
                generated, references,
                model_type=bertscore_model,
                refs_per_candidate=refs_per_candidate,
            )
            if bs is None:
                entry["bertscore"] = {"error": "bert-score not installed"}
            else:
                entry["bertscore"] = bs

        if do_bleu_rouge and generated and references:
            print(f"  [bleu/rouge] model={model}")
            br = compute_bleu_rouge(
                generated, references, refs_per_candidate=refs_per_candidate,
            )
            entry.update({
                "bleu": br.get("bleu"),
                "rouge_l": br.get("rouge_l"),
            })

        if do_style and generated and style_classifier is not None:
            print(f"  [style] model={model}")
            entry["style"] = score_with_style_classifier(generated, style_classifier)

        summary["models"][model] = entry
    return summary


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--eval-set", default=str(DEFAULT_EVAL_SET),
                   help="Path to eval set JSONL.")
    p.add_argument("--models", default="llama3.1,lemkin-clone",
                   help="Comma-separated Ollama model names to compare.")
    p.add_argument("--judge-model", default=DEFAULT_JUDGE_MODEL,
                   help="Ollama model used for the G-Eval rubric.")
    p.add_argument("--ollama-base",
                   default=os.environ.get("OLLAMA_BASE", OLLAMA_BASE_DEFAULT),
                   help="Ollama base URL.")
    p.add_argument("--embed-model", default=DEFAULT_EMBED_MODEL)
    p.add_argument("--index", default=str(DEFAULT_INDEX),
                   help="Local Chroma path (ignored when CHROMA_USE_HTTP=true).")
    p.add_argument("--persona", default=str(DEFAULT_PERSONA))
    p.add_argument("--format-specs", default=str(DEFAULT_FORMAT_SPECS))
    p.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    p.add_argument("--skip-generate", action="store_true",
                   help="Reuse cached traces.jsonl; only re-score.")
    p.add_argument("--skip-geval", action="store_true",
                   help="Skip G-Eval pass.")
    p.add_argument("--skip-bertscore", action="store_true",
                   help="Skip BERTScore (downloads ~1.5 GB roberta-large on first run).")
    p.add_argument("--skip-bleu-rouge", action="store_true",
                   help="Skip BLEU / ROUGE-L.")
    p.add_argument("--skip-style", action="store_true",
                   help="Skip TF-IDF + LogReg authorship classifier.")
    p.add_argument("--run-ragas", action="store_true",
                   help="Run optional RAGAS metrics (needs ragas + configured LLM).")
    p.add_argument("--dataset-jsonl", default=str(DEFAULT_DATASET_JSONL),
                   help="Reference corpus source for BERTScore / BLEU / style positives.")
    p.add_argument("--bertscore-model", default=DEFAULT_BERTSCORE_MODEL,
                   help="HF model used by bert-score (default roberta-large).")
    p.add_argument("--refs-per-candidate", type=int, default=DEFAULT_REF_PER_CANDIDATE,
                   help="How many reference samples per generated candidate (best-of-K).")
    return p.parse_args()


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n",
        encoding="utf-8",
    )


def _read_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def main() -> None:
    args = parse_args()
    models = [m.strip() for m in args.models.split(",") if m.strip()]
    if not models:
        raise SystemExit("--models was empty.")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    traces_path = out_dir / "traces.jsonl"
    geval_path = out_dir / "geval.jsonl"
    summary_path = out_dir / "summary.json"

    # 1. Generation pass.
    if args.skip_generate:
        traces = _read_jsonl(traces_path)
        if not traces:
            raise SystemExit(
                f"--skip-generate set but {traces_path} is empty or missing."
            )
        print(f"[eval_rag] reusing {len(traces)} cached traces from {traces_path}")
    else:
        eval_rows = load_eval_set(Path(args.eval_set))
        print(
            f"[eval_rag] eval set: {len(eval_rows)} rows | models: {models} | "
            f"ollama={args.ollama_base}"
        )
        traces = []
        for model in models:
            for i, row in enumerate(eval_rows, 1):
                rid = row.get("id", "-")
                print(f"  [gen] model={model} {i}/{len(eval_rows)} id={rid}")
                try:
                    trace = generate_trace(
                        row,
                        gen_model=model,
                        ollama_base=args.ollama_base,
                        embed_model=args.embed_model,
                        format_specs_path=Path(args.format_specs),
                        persona_path=Path(args.persona),
                        index_path=Path(args.index),
                    )
                    traces.append(trace)
                except Exception as e:
                    print(f"    generate failed: {e!r}")
        _write_jsonl(traces_path, traces)
        print(f"[eval_rag] wrote {traces_path} ({len(traces)} traces)")

    # 2. G-Eval pass.
    if args.skip_geval:
        judge_rows = _read_jsonl(geval_path)
        print(f"[eval_rag] skipped G-Eval; loaded {len(judge_rows)} cached judge rows")
    else:
        print(f"[eval_rag] G-Eval judge={args.judge_model}")
        try:
            author_profile = _geval_author_profile_for_judge(
                load_persona(Path(args.persona)),
            )
        except Exception as e:
            print(f"[eval_rag] WARNING: persona load failed ({e!r}); using minimal author profile")
            author_profile = (
                "Target author: Jason Lemkin — direct B2B/SaaS operator; concrete metrics; "
                "low hype."
            )
        anchor_paths = collect_style_anchor_source_paths(Path(args.dataset_jsonl))
        style_anchors = load_style_anchor_texts(anchor_paths)
        if style_anchors:
            print(
                f"[eval_rag] G-Eval style anchors: {len(style_anchors)} posts from "
                f"{len(anchor_paths)} JSONL source(s)"
            )
        else:
            print(
                "[eval_rag] WARNING: no style-anchor JSONL found; "
                "G-Eval omits real-post excerpt (set --dataset-jsonl or add data/*.jsonl)"
            )
        judge_rows = []
        for i, t in enumerate(traces, 1):
            print(f"  [judge] {i}/{len(traces)} model={t.get('model')} id={t.get('id')}")
            scores: dict | None = None
            try:
                scores = geval_score(
                    t,
                    args.judge_model,
                    args.ollama_base,
                    author_profile=author_profile,
                    style_anchors=style_anchors,
                )
            except Exception as e:
                print(f"    judge failed: {e!r}")
            judge_rows.append({
                "id": t.get("id"),
                "model": t.get("model"),
                "scores": scores,
            })
        _write_jsonl(geval_path, judge_rows)
        print(f"[eval_rag] wrote {geval_path}")

    # 3. Reference corpus + style classifier (shared across all models).
    references: list[str] = []
    style_classifier = None
    need_refs = not (args.skip_bertscore and args.skip_bleu_rouge and args.skip_style)
    if need_refs:
        ref_path = Path(args.dataset_jsonl)
        references = load_reference_corpus(ref_path)
        if references:
            print(
                f"[eval_rag] loaded {len(references)} reference texts from "
                f"{ref_path.name}"
            )
        else:
            print(f"[eval_rag] WARNING: no references loaded from {ref_path}")
        if not args.skip_style and references:
            print("[eval_rag] training TF-IDF + LR authorship classifier...")
            negatives = _negative_corpus()
            style_classifier = fit_style_classifier(references, negatives)
            if style_classifier is None:
                print(
                    "[eval_rag] style classifier unavailable (sklearn missing or "
                    "degenerate corpora); skipping."
                )

    # 4. Aggregate (G-Eval + BERTScore + BLEU/ROUGE-L + style).
    summary = _aggregate(
        traces, judge_rows, models,
        references=references,
        style_classifier=style_classifier,
        do_bertscore=not args.skip_bertscore,
        do_bleu_rouge=not args.skip_bleu_rouge,
        do_style=not args.skip_style,
        bertscore_model=args.bertscore_model,
        refs_per_candidate=args.refs_per_candidate,
    )
    summary["reference_corpus"] = {
        "n_samples": len(references),
        "source": args.dataset_jsonl,
        "refs_per_candidate": args.refs_per_candidate,
    }
    summary["metrics"] = sorted({
        "geval" if not args.skip_geval else None,
        "bertscore" if not args.skip_bertscore else None,
        "bleu_rouge" if not args.skip_bleu_rouge else None,
        "style" if not args.skip_style else None,
        "ragas" if args.run_ragas else None,
    } - {None})

    if args.run_ragas:
        print("[eval_rag] running optional RAGAS pass...")
        for model in models:
            mt = [t for t in traces if t.get("model") == model]
            summary["models"].setdefault(model, {})["ragas"] = ragas_scores(mt)

    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"\n[eval_rag] summary -> {summary_path}\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
