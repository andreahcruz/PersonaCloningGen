"""Is a style distance gap between two runs just a length effect?

The style distance (Kevin's ``host_finetune.voice_distance``) averages rates per 100 words. A
rate measured on a long text is less noisy than on a short one, so longer texts land closer to the
profile. Across real held-out Lemkin blog posts and model drafts, log(word count) and style distance
correlate at about -0.66. Two runs with different draft lengths can therefore differ in "voice"
only because one writes longer.

This tool cuts every draft to the same number of words and repeats the paired comparison. If the
gap shrinks toward zero, the gap was mostly length.

It needs Kevin's code and his reference data (the dataset, split assignments and gold dispositions
that the unified harness uses). Both are on branch ``kevin/298b-integration``. Do not clone the
branch normally: it carries about 480 MB of adapter files in Git LFS. Skip them:

    GIT_LFS_SKIP_SMUDGE=1 git worktree add ../kevin-ws origin/kevin/298b-integration

    python -m eval_tools.length_control --kevin-root ../kevin-ws \
        --a-drafts rag_off_drafts.jsonl --b-drafts rag_on_drafts.jsonl \
        --label-a "RAG off" --label-b "RAG on" --medium blog --out-md length_check.md

Drafts files are JSONL with ``id``, ``medium`` and ``answer`` (the saved ``raw/*.jsonl`` of a run).
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from pathlib import Path
from typing import Callable, Sequence

from eval_tools.bootstrap_ci import summarize_paired

DEFAULT_LENGTHS = (100, 150, 200)


def truncate_words(text: str, n: int) -> str:
    return " ".join((text or "").split()[:n])


def word_count(text: str) -> int:
    return len((text or "").split())


def length_controlled(
    drafts_a: dict[str, str],
    drafts_b: dict[str, str],
    distance: Callable[[str], float | None],
    lengths: Sequence[int] = DEFAULT_LENGTHS,
    *,
    n_boot: int = 5000,
    seed: int = 0,
) -> dict:
    """Paired B minus A style distance at full length and with every draft cut to N words.

    ``drafts_a`` and ``drafts_b`` map topic id to draft text for one medium. ``distance`` returns
    the style distance of a text (lower is closer to the profile) or None.
    """
    topics = sorted(set(drafts_a) & set(drafts_b))
    words_a = [word_count(drafts_a[t]) for t in topics]
    words_b = [word_count(drafts_b[t]) for t in topics]
    out: dict = {
        "n_topics": len(topics),
        "mean_words_a": round(statistics.fmean(words_a), 1) if topics else None,
        "mean_words_b": round(statistics.fmean(words_b), 1) if topics else None,
        "settings": [],
    }
    for label, n in [("full length", None), *[(f"first {k} words", k) for k in lengths]]:
        keep, diffs = [], []
        for topic in topics:
            if n is not None and (word_count(drafts_a[topic]) < n or word_count(drafts_b[topic]) < n):
                continue
            text_a = drafts_a[topic] if n is None else truncate_words(drafts_a[topic], n)
            text_b = drafts_b[topic] if n is None else truncate_words(drafts_b[topic], n)
            d_a, d_b = distance(text_a), distance(text_b)
            if d_a is None or d_b is None:
                continue
            keep.append(topic)
            diffs.append(d_b - d_a)
        summary = summarize_paired(diffs, keep, n_boot=n_boot, n_perm=n_boot, seed=seed)
        summary["setting"] = label
        out["settings"].append(summary)
    return out


def length_correlation(texts_and_distances: Sequence[tuple[str, float]]) -> float | None:
    """Pearson correlation between log(word count) and distance. Negative: longer text scores closer."""
    pts = [(math.log(max(word_count(t), 1)), d) for t, d in texts_and_distances if d is not None]
    if len(pts) < 3:
        return None
    xs, ys = [p[0] for p in pts], [p[1] for p in pts]
    sx, sy = statistics.pstdev(xs), statistics.pstdev(ys)
    if sx == 0 or sy == 0:
        return None
    mx, my = statistics.fmean(xs), statistics.fmean(ys)
    return sum((x - mx) * (y - my) for x, y in pts) / len(pts) / (sx * sy)


def prob_closer(draft_distances: Sequence[float], real_distances: Sequence[float]) -> float:
    """Chance that a random draft is closer to the profile than a random real post (0.5 means equal)."""
    wins = 0.0
    for d in draft_distances:
        for r in real_distances:
            wins += 1.0 if d < r else 0.5 if d == r else 0.0
    return wins / (len(draft_distances) * len(real_distances))


def real_vs_drafts(
    real_texts: Sequence[str],
    drafts: dict[str, str],
    distance: Callable[[str], float | None],
    *,
    n_draws: int = 20,
    n_boot: int = 1000,
    seed: int = 0,
) -> dict:
    """Are drafts closer to the profile than real posts are, and is that just length?

    Full length: every draft against every real post. Length matched: each draft is cut to the
    word count of a random real post, repeated ``n_draws`` times. A value near 0.5 means drafts and
    real posts are equally close. Above 0.5 at full length that falls back to 0.5 once lengths are
    matched means the advantage was length, not voice.
    """
    import random

    real_d = [d for d in (distance(t) for t in real_texts) if d is not None]
    real_w = [word_count(t) for t in real_texts]
    full_d = [d for d in (distance(t) for t in drafts.values()) if d is not None]
    rng = random.Random(seed)
    boots = []
    for _ in range(n_boot):
        boots.append(prob_closer([rng.choice(full_d) for _ in full_d], [rng.choice(real_d) for _ in real_d]))
    boots.sort()
    matched = []
    for rep in range(n_draws):
        r2 = random.Random(seed + rep)
        cut = []
        for text in drafts.values():
            target = r2.choice(real_w)
            d = distance(truncate_words(text, target) if word_count(text) > target else text)
            if d is not None:
                cut.append(d)
        matched.append(prob_closer(cut, real_d))
    return {
        "n_real": len(real_d),
        "n_drafts": len(full_d),
        "median_words_real": statistics.median(real_w),
        "median_words_drafts": statistics.median(word_count(t) for t in drafts.values()),
        "mean_distance_real": round(statistics.fmean(real_d), 4),
        "mean_distance_drafts": round(statistics.fmean(full_d), 4),
        "p_closer_full": round(prob_closer(full_d, real_d), 4),
        "p_closer_full_ci": (round(boots[int(0.025 * n_boot)], 4), round(boots[int(0.975 * n_boot) - 1], 4)),
        "p_closer_matched": round(statistics.fmean(matched), 4),
        "p_closer_matched_range": (round(min(matched), 4), round(max(matched), 4)),
    }


def real_vs_drafts_markdown(label: str, r: dict) -> str:
    return (
        f"| {label} | {r['n_drafts']} | {r['median_words_drafts']:.0f} | {r['mean_distance_drafts']:.3f} | "
        f"{r['p_closer_full']:.2f} ({r['p_closer_full_ci'][0]:.2f} to {r['p_closer_full_ci'][1]:.2f}) | "
        f"{r['p_closer_matched']:.2f} ({r['p_closer_matched_range'][0]:.2f} to {r['p_closer_matched_range'][1]:.2f}) |"
    )


def to_markdown(result: dict, label_a: str, label_b: str, medium: str, correlation: float | None) -> str:
    lines = [
        f"# Style distance and text length ({medium})",
        "",
        f"Difference = {label_b} minus {label_a} in style distance (lower is closer to the Lemkin profile). "
        "Each row cuts every draft to the same number of words, so length cannot explain the gap.",
        "",
        f"Mean draft length: {label_a} {result['mean_words_a']} words, {label_b} {result['mean_words_b']} words.",
    ]
    if correlation is not None:
        lines.append(f"Correlation of log(word count) with style distance over real posts and drafts: {correlation:+.2f}.")
    lines += ["", "| Setting | Pairs | Difference | 95% interval | p |", "|---|---|---|---|---|"]
    for s in result["settings"]:
        if s["mean_diff"] is None:
            lines.append(f"| {s['setting']} | {s['n']} | n/a | n/a | n/a |")
            continue
        interval = "n/a" if s["ci_low"] is None else f"{s['ci_low']:+.3f} to {s['ci_high']:+.3f}"
        p = "n/a" if s["p_value"] is None else f"{s['p_value']:.3f}"
        lines.append(f"| {s['setting']} | {s['n']} | {s['mean_diff']:+.3f} | {interval} | {p} |")
    lines += [
        "",
        "Reading it: if the gap at full length is clear but shrinks and its interval includes zero once drafts are cut "
        "to equal length, most of the gap was length. Cutting also drops the ends of drafts and some topics, so this "
        "is a check, not a replacement score.",
    ]
    return "\n".join(lines) + "\n"


def load_kevin_distance(kevin_root: str | Path, medium: str):
    """Return (distance function, real control texts) using Kevin's own code and reference data."""
    root = str(Path(kevin_root).resolve())
    sys.path.insert(0, root)
    from host_finetune.train_only_index import (  # type: ignore
        DEFAULT_ASSIGNMENTS,
        DEFAULT_DATASET,
        DEFAULT_GOLD_DISPOSITIONS,
        blocked_group_ids,
    )
    from host_finetune.voice_distance import (  # type: ignore
        STYLE_FEATURE_NAMES,
        held_out_reference_texts,
        partition_references,
        reference_profile,
        style_distance,
    )

    def load(path):
        with open(path, encoding="utf-8") as handle:
            return [json.loads(line) for line in handle if line.strip()]

    dataset, assignments, dispositions = load(DEFAULT_DATASET), load(DEFAULT_ASSIGNMENTS), load(DEFAULT_GOLD_DISPOSITIONS)
    held = held_out_reference_texts(dataset, assignments, blocked_group_ids(assignments, dispositions))
    profile_texts, real_texts = partition_references(held[medium])
    profile = reference_profile(profile_texts, STYLE_FEATURE_NAMES)
    return (lambda text: style_distance(text, profile)), real_texts


def _drafts(path: str, medium: str) -> dict[str, str]:
    out = {}
    with open(path, encoding="utf-8-sig") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                if row.get("medium") == medium:
                    out[str(row["id"])] = row.get("answer") or ""
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--kevin-root", required=True, help="Folder with Kevin's host_finetune/ and experiments/ data.")
    p.add_argument("--a-drafts", required=True)
    p.add_argument("--b-drafts", required=True)
    p.add_argument("--label-a", default="A")
    p.add_argument("--label-b", default="B")
    p.add_argument("--medium", default="blog", help="blog, linkedin or x (talk has no reference).")
    p.add_argument("--lengths", type=int, nargs="+", default=list(DEFAULT_LENGTHS))
    p.add_argument("--n-boot", type=int, default=5000)
    p.add_argument("--out-md")
    args = p.parse_args()

    distance, real_texts = load_kevin_distance(args.kevin_root, args.medium)
    a, b = _drafts(args.a_drafts, args.medium), _drafts(args.b_drafts, args.medium)
    if not a or not b:
        raise SystemExit(f"No {args.medium} drafts found in the two files.")
    result = length_controlled(a, b, distance, args.lengths, n_boot=args.n_boot)
    pool = [(t, distance(t)) for t in [*real_texts, *a.values(), *b.values()]]
    markdown = to_markdown(result, args.label_a, args.label_b, args.medium, length_correlation(pool))
    ra, rb = real_vs_drafts(real_texts, a, distance), real_vs_drafts(real_texts, b, distance)
    markdown += (
        f"\n## Drafts against real held-out Lemkin posts ({args.medium})\n\n"
        f"{ra['n_real']} real posts, median {ra['median_words_real']:.0f} words, mean distance {ra['mean_distance_real']:.3f}. "
        "The numbers are the chance that a random draft is closer to the profile than a random real post "
        "(0.50 means equal, above 0.50 means the draft looks closer).\n\n"
        "| Run | Drafts | Median words | Mean distance | Chance closer, full length (95% interval) | Chance closer, cut to real post lengths (range of 20 draws) |\n"
        "|---|---|---|---|---|---|\n"
        f"{real_vs_drafts_markdown(args.label_a, ra)}\n{real_vs_drafts_markdown(args.label_b, rb)}\n\n"
        "If the full length value is above 0.50 and the length matched value is near 0.50, drafts only look "
        "closer than real posts because they are longer.\n"
    )
    if args.out_md:
        Path(args.out_md).write_text(markdown, encoding="utf-8")
    print(markdown)


if __name__ == "__main__":
    main()
