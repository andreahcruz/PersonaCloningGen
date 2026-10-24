"""Compare two saved runs on the same drafts, with intervals and two views.

    python -m eval_tools.compare \
        --a experiments/EXP-20261004-008-writing-evidence/metrics/rows.jsonl \
        --a-experiment EXP-20261004-003-repaired-vs-relabel --a-model relabel --label-a "RAG off" \
        --b experiments/EXP-20261005-003-relabel-rag-final/metrics/rows.jsonl --label-b "RAG on" \
        --out-md report.md --out-json report.json

Run A is the reference. Every difference is B minus A.

Why two views (see docs/evaluation/METRIC_MAP.md, gap G7):

* all drafts: every pair where both runs have a number. No draft is dropped for failing
  a gate, so a run cannot look better by failing more drafts.
* common gates: only pairs where both drafts pass completion, copying and format. The
  grounding gate is left out because only retrieval runs have it.

It reads saved results only. No model and no judge is called.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from eval_tools.bootstrap_ci import holm_adjust, summarize_paired
from eval_tools.rows import (
    COMMON_GATES,
    LOWER_IS_BETTER,
    METRIC_PATHS,
    NEUTRAL_METRICS,
    attach_extra,
    common_gates_pass,
    index_rows,
    load_rows,
    metric_value,
)

DEFAULT_METRICS = (
    "writing_quality",
    "task_adherence",
    "stance_consistency",
    "style_distance",
    "persona_utility",
    "claim_support",
    "evidence_support",
    "word_count",
    "grammar_rate",
    "grammar_deviation",
)
MEDIUMS = ("blog", "linkedin", "x", "talk")


def _mean(values: list[float]) -> float | None:
    return round(sum(values) / len(values), 6) if values else None


P_CLEAR = 0.05


def verdict(summary: dict, lower_is_better: bool, neutral: bool = False) -> str:
    """Plain words for one result.

    A difference is called clear only when the interval excludes zero AND the sign-flip
    p value is below 0.05, so a borderline result is not over-called.
    """
    low, high, diff = summary.get("ci_low"), summary.get("ci_high"), summary.get("mean_diff")
    p = summary.get("p_value")
    if low is None or high is None or diff is None or p is None:
        return "too few pairs"
    if low <= 0 <= high or p >= P_CLEAR:
        return "no clear difference"
    b_higher = diff > 0
    if neutral:
        return "B higher" if b_higher else "B lower"
    return "B better" if (b_higher != lower_is_better) else "B worse"


def arm_summary(rows: dict, keys: list) -> dict:
    """Eligibility and gate failures for one arm, over the paired drafts."""
    n = len(keys)
    out: dict = {"n": n, "gate_failures": {}}
    if not n:
        return out
    eligible = sum(1 for k in keys if rows[k].get("eligible_for_ranking") is True)
    common = sum(1 for k in keys if common_gates_pass(rows[k]))
    out["eligible"] = eligible
    out["eligible_rate"] = round(eligible / n, 4)
    out["common_gates_pass"] = common
    out["common_gates_rate"] = round(common / n, 4)
    for gate in (*COMMON_GATES, "grounding"):
        failed = 0
        applicable = 0
        for k in keys:
            gate_row = (rows[k].get("gates") or {}).get(gate) or {}
            if gate == "grounding" and gate_row.get("applicable") is False:
                continue
            applicable += 1
            if gate_row.get("pass") is not True:
                failed += 1
        out["gate_failures"][gate] = {"failed": failed, "of": applicable}
    return out


def _roles(rows_a: dict, rows_b: dict, medium: str) -> str:
    roles = set()
    for rows in (rows_a, rows_b):
        for (topic, med), row in rows.items():
            if med == medium:
                role = (row.get("persona_style") or {}).get("role")
                if role:
                    roles.add(role)
    if not roles:
        return "unknown"
    return roles.pop() if len(roles) == 1 else "mixed"


def compare_runs(
    rows_a: dict,
    rows_b: dict,
    label_a: str = "A",
    label_b: str = "B",
    metrics: tuple[str, ...] | list[str] | None = None,
    *,
    n_boot: int = 10000,
    n_perm: int = 10000,
    seed: int = 0,
) -> dict:
    """Compare run B against run A on the drafts both runs have."""
    names = list(metrics) if metrics else list(DEFAULT_METRICS)
    unknown = [m for m in names if m not in METRIC_PATHS]
    if unknown:
        raise ValueError(f"Unknown metric(s): {unknown}. Known: {sorted(METRIC_PATHS)}")
    keys = sorted(set(rows_a) & set(rows_b))
    result: dict = {
        "label_a": label_a,
        "label_b": label_b,
        "direction": "B minus A",
        "n_a": len(rows_a),
        "n_b": len(rows_b),
        "n_paired": len(keys),
        "only_in_a": len(set(rows_a) - set(rows_b)),
        "only_in_b": len(set(rows_b) - set(rows_a)),
        "arms": {label_a: arm_summary(rows_a, keys), label_b: arm_summary(rows_b, keys)},
        "style_roles": {m: _roles(rows_a, rows_b, m) for m in MEDIUMS},
        "views": {},
        "skipped_metrics": [],
        "settings": {"n_boot": n_boot, "n_perm": n_perm, "seed": seed, "ci": 0.95},
    }
    common_keys = [k for k in keys if common_gates_pass(rows_a[k]) and common_gates_pass(rows_b[k])]
    views = {"all_drafts": keys, "common_gates": common_keys}

    usable = [m for m in names if any(metric_value(rows_a[k], m) is not None and metric_value(rows_b[k], m) is not None for k in keys)]
    result["skipped_metrics"] = [m for m in names if m not in usable]

    for view_name, view_keys in views.items():
        view: dict = {"n_pairs": len(view_keys), "results": []}
        overall_p: dict[str, float | None] = {}
        for metric in usable:
            for scope in ("all", *MEDIUMS):
                scoped = [k for k in view_keys if scope == "all" or k[1] == scope]
                pairs = []
                for k in scoped:
                    a_val, b_val = metric_value(rows_a[k], metric), metric_value(rows_b[k], metric)
                    if a_val is not None and b_val is not None:
                        pairs.append((k, a_val, b_val))
                if not pairs:
                    continue
                diffs = [b - a for _, a, b in pairs]
                topics = [k[0] for k, _, _ in pairs]
                summary = summarize_paired(diffs, topics, n_boot=n_boot, n_perm=n_perm, seed=seed)
                summary.update(
                    metric=metric,
                    scope=scope,
                    mean_a=_mean([a for _, a, _ in pairs]),
                    mean_b=_mean([b for _, _, b in pairs]),
                    lower_is_better=metric in LOWER_IS_BETTER,
                )
                summary["verdict"] = verdict(summary, summary["lower_is_better"], metric in NEUTRAL_METRICS)
                if scope == "all":
                    overall_p[metric] = summary["p_value"]
                view["results"].append(summary)
        adjusted = holm_adjust(overall_p)
        for item in view["results"]:
            item["p_holm"] = adjusted.get(item["metric"]) if item["scope"] == "all" else None
        result["views"][view_name] = view
    return result


def _fmt(value: float | None, digits: int = 3) -> str:
    return "n/a" if value is None else f"{value:.{digits}f}"


def to_markdown(result: dict) -> str:
    a, b = result["label_a"], result["label_b"]
    lines = [
        f"# Comparison: {b} against {a}",
        "",
        f"Every difference is {result['direction']}. Intervals are 95 percent, resampled by topic. "
        "p values are from a sign-flip test by topic; `p holm` corrects across metrics in the all-scope rows.",
        "",
        f"Paired drafts: {result['n_paired']} (only in {a}: {result['only_in_a']}, only in {b}: {result['only_in_b']}).",
        "",
        "## Gates",
        "",
        "| Run | Drafts | Eligible (all gates) | Pass common gates | Completion fails | Copying fails | Format fails | Grounding fails |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for label in (a, b):
        arm = result["arms"][label]
        if not arm["n"]:
            lines.append(f"| {label} | 0 | | | | | | |")
            continue
        gf = arm["gate_failures"]
        grounding = gf["grounding"]
        lines.append(
            f"| {label} | {arm['n']} | {arm['eligible']} ({arm['eligible_rate']:.0%}) | "
            f"{arm['common_gates_pass']} ({arm['common_gates_rate']:.0%}) | "
            f"{gf['completion']['failed']} | {gf['copying_memorization']['failed']} | {gf['format']['failed']} | "
            f"{grounding['failed']} of {grounding['of']} |"
        )
    lines += [
        "",
        "Grounding is counted only where it applies. A run with retrieval has one extra gate, so its eligible "
        "count is not comparable to a run without retrieval. Use the common gates column.",
        "",
        "Style distance role by medium (selector means it may rank models): "
        + ", ".join(f"{m} {r}" for m, r in result["style_roles"].items()),
    ]
    titles = {
        "all_drafts": "All drafts (no draft dropped)",
        "common_gates": "Drafts that pass the common gates in both runs",
    }
    for view_name, view in result["views"].items():
        lines += ["", f"## {titles[view_name]}", "", f"Pairs in this view: {view['n_pairs']}", ""]
        lines += [
            f"| Metric | Scope | n | {a} mean | {b} mean | Diff | 95% interval | p | p holm | Verdict |",
            "|---|---|---|---|---|---|---|---|---|---|",
        ]
        for item in view["results"]:
            interval = (
                "n/a" if item["ci_low"] is None else f"{item['ci_low']:.3f} to {item['ci_high']:.3f}"
            )
            lower = " (lower is better)" if item["lower_is_better"] else ""
            lines.append(
                f"| {item['metric']}{lower} | {item['scope']} | {item['n']} | {_fmt(item['mean_a'])} | "
                f"{_fmt(item['mean_b'])} | {_fmt(item['mean_diff'])} | {interval} | {_fmt(item['p_value'])} | "
                f"{_fmt(item.get('p_holm'))} | {item['verdict']} |"
            )
    if result["skipped_metrics"]:
        lines += ["", "Skipped (no paired values in the files): " + ", ".join(result["skipped_metrics"])]
    lines += [
        "",
        "How to read the verdict: a difference is called clear only when the interval excludes zero and the p value is "
        "below 0.05. Rows by medium are not corrected for the many comparisons made, so treat a single medium result as "
        "a lead, not a finding. Use `p holm` for the all-scope rows.",
        "",
        "Limits: one run per arm unless more seeds are added, and the 30 gold topics are a development panel. "
        "A result that is not clear here is not proof of no difference.",
    ]
    return "\n".join(lines) + "\n"


def _load_arm(path: str, experiment: str | None, model: str | None, extra: str | None) -> dict:
    rows = index_rows(load_rows(path, source_experiment=experiment, model=model))
    if not rows:
        raise SystemExit(f"No rows matched in {path} (experiment={experiment!r}, model={model!r}).")
    if extra:
        matched = attach_extra(rows, extra)
        print(f"Merged extra values into {matched} of {len(rows)} rows from {extra}")
    return rows


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--a", required=True, help="rows.jsonl of the reference run (A).")
    p.add_argument("--b", required=True, help="rows.jsonl of the run to compare (B).")
    p.add_argument("--a-experiment", help="Keep only rows with this source_experiment.")
    p.add_argument("--b-experiment")
    p.add_argument("--a-model", help="Keep only rows for this model name.")
    p.add_argument("--b-model")
    p.add_argument("--extra-a", help="Side file of per draft numbers for A (e.g. grammar output).")
    p.add_argument("--extra-b")
    p.add_argument("--label-a", default="A")
    p.add_argument("--label-b", default="B")
    p.add_argument("--metrics", nargs="*", help=f"Default: {' '.join(DEFAULT_METRICS)}")
    p.add_argument("--n-boot", type=int, default=10000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out-md")
    p.add_argument("--out-json")
    args = p.parse_args()

    rows_a = _load_arm(args.a, args.a_experiment, args.a_model, args.extra_a)
    rows_b = _load_arm(args.b, args.b_experiment, args.b_model, args.extra_b)
    result = compare_runs(
        rows_a, rows_b, args.label_a, args.label_b, args.metrics, n_boot=args.n_boot, n_perm=args.n_boot, seed=args.seed
    )
    markdown = to_markdown(result)
    if args.out_md:
        Path(args.out_md).write_text(markdown, encoding="utf-8")
    if args.out_json:
        Path(args.out_json).write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(markdown)


if __name__ == "__main__":
    main()
