"""Does a difference repeat across seeds?

One run per arm can be luck. When the same comparison was generated with several seeds
(Kevin's repaired against relabel pair exists for seeds 42, 43 and 44), a real difference
should keep its sign. This tool runs the same comparison for each seed and shows them
side by side.

    python -m eval_tools.seeds --rows experiments/EXP-20261004-008-writing-evidence/metrics/rows.jsonl \
        --a-model repaired --b-model relabel \
        --experiments EXP-20261004-003-repaired-vs-relabel EXP-20261004-005-repaired-vs-relabel-seed43 \
                      EXP-20261004-006-repaired-vs-relabel-seed44 \
        --labels 42 43 44 --out-md seeds.md

Reading it: "same sign" means every seed moved the same way. "Interval excludes 0" counts the
seeds where that seed alone gave an interval that excludes zero. A result that keeps its sign
in every seed and is clear in most of them is much more believable than one that flips.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from eval_tools.compare import compare_runs
from eval_tools.rows import LOWER_IS_BETTER, index_rows, load_rows

DEFAULT_METRICS = ("writing_quality", "task_adherence", "stance_consistency", "style_distance", "word_count")


def seed_stability(
    runs: list[tuple[str, dict, dict]],
    metrics: tuple[str, ...] | list[str] = DEFAULT_METRICS,
    scope: str = "all",
    *,
    n_boot: int = 2000,
) -> list[dict]:
    """``runs`` is a list of (label, rows_a, rows_b), one per seed. Differences are B minus A."""
    per_seed = []
    for label, rows_a, rows_b in runs:
        result = compare_runs(rows_a, rows_b, "A", "B", list(metrics), n_boot=n_boot, n_perm=n_boot)
        found = {r["metric"]: r for r in result["views"]["all_drafts"]["results"] if r["scope"] == scope}
        per_seed.append((label, found))
    out = []
    for metric in metrics:
        entries = []
        for label, found in per_seed:
            item = found.get(metric)
            entries.append(
                {
                    "seed": label,
                    "n": item["n"] if item else 0,
                    "mean_diff": item["mean_diff"] if item else None,
                    "ci_low": item["ci_low"] if item else None,
                    "ci_high": item["ci_high"] if item else None,
                }
            )
        diffs = [e["mean_diff"] for e in entries if e["mean_diff"] is not None]
        signs = {(d > 0) - (d < 0) for d in diffs if d != 0}
        excludes = sum(
            1 for e in entries if e["ci_low"] is not None and not (e["ci_low"] <= 0 <= e["ci_high"])
        )
        out.append(
            {
                "metric": metric,
                "scope": scope,
                "lower_is_better": metric in LOWER_IS_BETTER,
                "per_seed": entries,
                "n_seeds": len(diffs),
                "same_sign": len(diffs) > 1 and len(signs) <= 1 and bool(signs),
                "seeds_with_interval_excluding_zero": excludes,
            }
        )
    return out


def to_markdown(results: list[dict], labels: list[str], title: str = "Seed stability") -> str:
    head = " | ".join(f"seed {label} diff (95% interval)" for label in labels)
    lines = [
        f"# {title}",
        "",
        "Each cell is B minus A for that seed. Same sign: every seed moved the same way.",
        "",
        f"| Metric | Scope | {head} | Same sign | Intervals excluding 0 |",
        "|---|---|" + "---|" * len(labels) + "---|---|",
    ]
    for item in results:
        cells = []
        for entry in item["per_seed"]:
            if entry["mean_diff"] is None:
                cells.append("n/a")
            elif entry["ci_low"] is None:
                cells.append(f"{entry['mean_diff']:+.3f}")
            else:
                cells.append(f"{entry['mean_diff']:+.3f} ({entry['ci_low']:+.3f} to {entry['ci_high']:+.3f})")
        lower = " (lower is better)" if item["lower_is_better"] else ""
        lines.append(
            f"| {item['metric']}{lower} | {item['scope']} | " + " | ".join(cells)
            + f" | {'yes' if item['same_sign'] else 'no'} | {item['seeds_with_interval_excluding_zero']} of {item['n_seeds']} |"
        )
    lines += [
        "",
        "A result worth trusting keeps its sign in every seed and has an interval that excludes zero in most of them.",
        "Three seeds is the minimum that shows a pattern; it is not a significance test.",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--rows", required=True, help="One rows.jsonl holding all the experiments.")
    p.add_argument("--a-model", required=True)
    p.add_argument("--b-model", required=True)
    p.add_argument("--experiments", nargs="+", required=True, help="source_experiment of each seed.")
    p.add_argument("--labels", nargs="+", help="Names for the seeds, same order.")
    p.add_argument("--metrics", nargs="*", default=list(DEFAULT_METRICS))
    p.add_argument("--scope", default="all", help="all, blog, linkedin, x or talk")
    p.add_argument("--out-md")
    args = p.parse_args()

    labels = args.labels or [str(i + 1) for i in range(len(args.experiments))]
    if len(labels) != len(args.experiments):
        raise SystemExit("--labels must have one name per experiment.")
    runs = []
    for label, experiment in zip(labels, args.experiments):
        rows_a = index_rows(load_rows(args.rows, source_experiment=experiment, model=args.a_model))
        rows_b = index_rows(load_rows(args.rows, source_experiment=experiment, model=args.b_model))
        if not rows_a or not rows_b:
            raise SystemExit(f"No rows for {experiment} with models {args.a_model!r} and {args.b_model!r}.")
        runs.append((label, rows_a, rows_b))
    markdown = to_markdown(seed_stability(runs, args.metrics, args.scope), labels,
                           f"Seed stability: {args.b_model} against {args.a_model} ({args.scope})")
    if args.out_md:
        Path(args.out_md).write_text(markdown, encoding="utf-8")
    print(markdown)


if __name__ == "__main__":
    main()
