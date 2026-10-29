"""Draw the three charts used in docs/evaluation/ARCHITECTURE_COMPARISON.md.

Run it from the repository root, after `rebuild_reports.py` has written the combined rows:

    python docs/evaluation/examples/data/architecture/rebuild_reports.py
    python docs/evaluation/figures/make_figures.py

It needs matplotlib (the eval_tools themselves need nothing). Every number in the charts is computed
here from the saved rows with the same functions the reports use, so the charts and the tables
cannot disagree.

The charts:

1. fig1_relabel_rag_on_vs_off.png      Relabel with RAG on minus RAG off, one dot per seed.
2. fig2_relabel_rag_vs_personarag.png  Relabel + RAG minus PersonaRAG, one dot per seed.
3. fig3_draft_length_by_medium.png     Average draft length of each architecture, by medium.

How to read charts 1 and 2: each small panel is one metric. A dot is the average difference between the
two runs for one seed and the line is its 95 percent interval. The dashed vertical line is "no difference".
If the line does not touch it, that seed's interval excludes zero (this is before the correction for how many tests were run; the reports carry the corrected verdicts). If all three dots sit on the same
side, the direction repeated in every seed. The hollow dot is seed 42, which Kevin generated on a different
GPU, so it is a slightly different kind of repeat than seeds 43 and 44.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import MaxNLocator  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))

from eval_tools.rows import index_rows, load_rows, metric_value  # noqa: E402
from eval_tools.seeds import seed_stability  # noqa: E402

COMBINED = ROOT / "docs/evaluation/examples/architecture/combined_rows.jsonl"

# Colours: the first three slots of the validated default palette (light surface). Text never takes a series colour.
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e3e2dd", "#ffffff"
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 10, "axes.edgecolor": GRID, "axes.labelcolor": INK2,
    "xtick.color": INK2, "ytick.color": INK2, "text.color": INK, "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
})

SEEDS = ("42", "43", "44")
# (metric, scope, panel title, words under the axis for the left and right side)
PANELS = [
    ("writing_quality", "all", "Writing quality\n(all drafts)", ("worse", "better")),
    ("task_adherence", "all", "Task adherence\n(all drafts)", ("worse", "better")),
    ("stance_consistency", "all", "Stance consistency\n(all drafts)", ("worse", "better")),
    ("style_distance", "blog", "Voice distance, blog\n(lower is closer to Lemkin)", ("closer", "farther")),
    ("word_count", "blog", "Blog length in words", ("shorter", "longer")),
]


def load_runs(a_model: str, b_model: str) -> list[tuple[str, dict, dict]]:
    runs = []
    for seed in SEEDS:
        rows_a = index_rows(load_rows(COMBINED, source_experiment=f"s{seed}", model=a_model))
        rows_b = index_rows(load_rows(COMBINED, source_experiment=f"s{seed}", model=b_model))
        runs.append((seed, rows_a, rows_b))
    return runs


def difference_figure(runs, title: str, subtitle: str, out: Path) -> None:
    metrics = [p[0] for p in PANELS]
    by_scope = {scope: {r["metric"]: r for r in seed_stability(runs, metrics, scope, n_boot=4000)}
                for scope in ("all", "blog")}
    fig, axes = plt.subplots(1, len(PANELS), figsize=(15, 3.9), sharey=True)
    for ax, (metric, scope, name, (left, right)) in zip(axes, PANELS):
        result = by_scope[scope][metric]
        lows = [e["ci_low"] for e in result["per_seed"]]
        highs = [e["ci_high"] for e in result["per_seed"]]
        span = max(abs(min(lows)), abs(max(highs)))
        for row, entry in enumerate(result["per_seed"]):
            hollow = entry["seed"] == "42"
            ax.plot([entry["ci_low"], entry["ci_high"]], [row, row], color=BLUE, lw=2, solid_capstyle="round", zorder=2)
            ax.plot(entry["mean_diff"], row, "o", ms=9, mfc=SURFACE if hollow else BLUE, mec=BLUE, mew=2, zorder=3)
            text = f"{entry['mean_diff']:+.2f}"
            text = "0.00" if text in ("+0.00", "-0.00") else text   # do not print a sign on "no change"
            ax.annotate(text, (entry["mean_diff"], row), xytext=(0, 10),
                        textcoords="offset points", ha="center", fontsize=9, color=INK)
        ax.axvline(0, color=INK2, lw=1, ls=(0, (4, 3)), zorder=1)
        ax.set_xlim(-span * 1.25, span * 1.25)
        ax.set_ylim(2.6, -0.6)
        ax.set_yticks(range(3), [f"seed {s}" for s in SEEDS])
        ax.xaxis.set_major_locator(MaxNLocator(nbins=4))   # fewer ticks so the narrow panels stay readable
        ax.set_title(name, fontsize=10.5, color=INK, pad=8)
        ax.grid(axis="x", color=GRID, lw=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.tick_params(axis="y", length=0)
        ax.text(0.02, -0.2, f"← {left}", transform=ax.transAxes, ha="left", va="top", fontsize=9, color=INK2)
        ax.text(0.98, -0.2, f"{right} →", transform=ax.transAxes, ha="right", va="top", fontsize=9, color=INK2)
        same = "same direction in all 3 seeds" if result["same_sign"] else "direction changes between seeds"
        clear = result["seeds_with_interval_excluding_zero"]
        ax.text(0.5, -0.42, f"{same}\nclear in {clear} of 3 seeds", transform=ax.transAxes, ha="center", va="top",
                fontsize=8.5, color=INK2)
    fig.suptitle(title, x=0.01, ha="left", fontsize=13, fontweight="bold", color=INK, y=1.04)
    fig.text(0.01, 0.985, subtitle, ha="left", va="top", fontsize=9.5, color=INK2)
    fig.subplots_adjust(top=0.78, bottom=0.30, left=0.06, right=0.99, wspace=0.12)
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print("wrote", out)


def length_figure(out: Path) -> None:
    mediums = ["blog", "linkedin", "x", "talk"]
    labels = ["Blog", "LinkedIn", "X post", "Talk"]

    def mean_words(model: str, seeds: tuple[str, ...]) -> dict[str, float]:
        totals = {m: [] for m in mediums}
        for seed in seeds:
            for row in load_rows(COMBINED, source_experiment=seed, model=model):
                value = metric_value(row, "word_count")
                if value is not None:
                    totals[row["medium"]].append(value)
        return {m: sum(v) / len(v) for m, v in totals.items()}

    arms = [
        ("Base Llama 3.1, no RAG", BLUE, mean_words("base", ("s42",))),
        ("PersonaRAG (base + retrieval)", ORANGE, mean_words("personarag", ("s42",))),
        ("Relabel + RAG (mean of seeds 43, 44)", AQUA, mean_words("rag_on", ("s43", "s44"))),
    ]
    fig, ax = plt.subplots(figsize=(9.5, 4.6))
    width, gap = 0.24, 0.02
    for i, (name, color, values) in enumerate(arms):
        xs = [k + (i - 1) * (width + gap) for k in range(len(mediums))]
        bars = ax.bar(xs, [values[m] for m in mediums], width=width, color=color, label=name, zorder=2)
        for x, m in zip(xs, mediums):
            ax.annotate(f"{values[m]:.0f}", (x, values[m]), xytext=(0, 3), textcoords="offset points",
                        ha="center", fontsize=9, color=INK)
    ax.set_xticks(range(len(mediums)), labels)
    ax.set_ylabel("average words per draft")
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.set_ylim(0, 600)
    # legend sits under the chart so it never covers a bar
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=3, fontsize=9.5)
    ax.set_title("Relabel + RAG writes much shorter drafts than the other two", loc="left", fontsize=12.5,
                 fontweight="bold", color=INK, pad=14)
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print("wrote", out)


def voice_fixed_figure(out: Path) -> None:
    """PUK-28: blog voice distance on the first 100 words only, per architecture and seed, next to real Lemkin posts."""
    import json

    folder = ROOT / "docs/evaluation/examples/data/voice_fixed"
    arms = [
        ("Base Llama 3.1", BLUE, ["base_llama31"]),
        ("PersonaRAG", ORANGE, ["personarag_seed42"]),
        ("Relabel, RAG off", AQUA, [f"relabel_rag_off_seed{s}" for s in (42, 43, 44)]),
        ("Relabel + RAG", AQUA, [f"relabel_rag_on_seed{s}" for s in (42, 43, 44)]),
    ]
    summaries = {name: json.loads((folder / f"{name}.summary.json").read_text(encoding="utf-8")) for _, _, names in arms for name in names}
    real = next(iter(summaries.values()))["real_posts"]["blog"]
    fig, ax = plt.subplots(figsize=(8.5, 3.8))
    for row, (label, color, names) in enumerate(arms):
        values = [summaries[n]["drafts"]["blog"]["mean_distance_fixed"] for n in names]
        ax.plot([min(values), max(values)], [row, row], color=color, lw=2, solid_capstyle="round", zorder=2)
        for value, name in zip(values, names):
            hollow = "rag_off" in name
            ax.plot(value, row, "o", ms=9, mfc=SURFACE if hollow else color, mec=color, mew=2, zorder=3)
        ax.annotate(f"{sum(values) / len(values):.2f}", (max(values), row), xytext=(10, -3), textcoords="offset points", fontsize=10, color=INK)
    ax.axvline(real["mean_distance_fixed"], color=INK2, lw=1.2, ls=(0, (4, 3)), zorder=1)
    ax.annotate(f"real Lemkin posts\n({real['mean_distance_fixed']:.2f}, first {real['fixed_words']} words)", (real["mean_distance_fixed"], -0.55),
                xytext=(6, 0), textcoords="offset points", fontsize=9, color=INK2, va="top")
    ax.set_yticks(range(len(arms)), [a[0] for a in arms])
    ax.set_ylim(len(arms) - 0.4, -0.9)
    ax.set_xlim(0.8, 2.35)
    ax.set_xlabel("voice distance on the first 100 words of a blog draft (lower is closer to Lemkin)")
    ax.grid(axis="x", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.set_title("Relabel matches real posts on voice; PersonaRAG and base do not", loc="left", fontsize=12.5, fontweight="bold", color=INK, pad=14)
    fig.text(0.01, -0.04, "One dot per seed (hollow = RAG off); line spans the seeds. The score uses the same number of words for every draft.",
             fontsize=9, color=INK2)
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print("wrote", out)


def main() -> None:
    if not COMBINED.exists():
        sys.exit("Run docs/evaluation/examples/data/architecture/rebuild_reports.py first.")
    difference_figure(
        load_runs("rag_off", "rag_on"),
        "Relabel adapter: RAG on minus RAG off",
        "Each dot is one seed (hollow = Kevin's seed 42, other GPU). Line = 95% interval. Dashed line = no difference.",
        HERE / "fig1_relabel_rag_on_vs_off.png",
    )
    difference_figure(
        load_runs("personarag", "rag_on"),
        "Relabel + RAG minus PersonaRAG (base Llama 3.1 + retrieval)",
        "PersonaRAG has one run (seed 42) and is compared with each Relabel seed. Line = 95% interval. Dashed line = no difference.",
        HERE / "fig2_relabel_rag_vs_personarag.png",
    )
    length_figure(HERE / "fig3_draft_length_by_medium.png")
    if (ROOT / "docs/evaluation/examples/data/voice_fixed/base_llama31.summary.json").exists():
        voice_fixed_figure(HERE / "fig5_voice_fixed_blog.png")


if __name__ == "__main__":
    main()
