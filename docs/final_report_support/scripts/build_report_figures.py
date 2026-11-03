"""Report figures from saved evaluation rows.

Reads public artifacts only. Does not open the LightRAG snapshot, call a
judge, or write into experiments/ or results/.

Sources:
- results/model-f-handoff-v1/architecture_comparison.json
- results/model-f-handoff-v1/per_medium.json
- results/model-f-handoff-v1/aggregate_summary.json
- results/model-f-handoff-v1/deterministic_audit_summary.json
- results/model-f-handoff-v1/provenance.json
- results/model-f-panel-rows-v1/judge_scores.csv
- results/model-f-panel-rows-v1/deterministic_flags.csv
- docs/evaluation/examples/data/*.jsonl (seed-42 unified_eval.v1 rows)
- experiments/EXP-20261102-001-model-f-unified-panel/diagnostics/descriptive_omissions.json
"""

from __future__ import annotations

import csv
import json
import math
import random
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parents[1]
FIG = OUT / "figures"
TAB = OUT / "tables"
FIG.mkdir(parents=True, exist_ok=True)
TAB.mkdir(parents=True, exist_ok=True)

ARCH_ORDER = [
    "Base",
    "PersonaRAG",
    "Relabel RAG off",
    "Relabel + RAG",
    "Model F",
]
COLORS = {
    "Base": "#4C78A8",
    "PersonaRAG": "#F58518",
    "Relabel RAG off": "#54A24B",
    "Relabel + RAG": "#E45756",
    "Model F": "#72B7B2",
}
MEDIA = ["blog", "linkedin", "x", "talk"]
MEDIA_LABEL = {"blog": "Blog", "linkedin": "LinkedIn", "x": "X", "talk": "Talk"}
ROW_FILES = {
    "Base": ROOT / "docs/evaluation/examples/data/architecture/kevin_base_llama31_rows.jsonl",
    "PersonaRAG": ROOT / "docs/evaluation/examples/data/architecture/colab_personarag_seed42_rows.jsonl",
    "Relabel RAG off": ROOT / "docs/evaluation/examples/data/relabel_rag_off_rows.jsonl",
    "Relabel + RAG": ROOT / "docs/evaluation/examples/data/relabel_rag_on_rows.jsonl",
}


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def mean(values):
    present = [v for v in values if v is not None and not (isinstance(v, float) and math.isnan(v))]
    if not present:
        return None, 0, len(values)
    return sum(present) / len(present), len(present), len(values) - len(present)


def rate(flags):
    present = [v for v in flags if v is not None]
    if not present:
        return None, 0, len(flags)
    return sum(1 for v in present if v) / len(present), len(present), len(flags) - len(present)


def fnum(value, digits=6):
    if value is None:
        return ""
    return f"{value:.{digits}f}"


def write_csv(path: Path, rows: list[dict]):
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def parse_bool(text):
    if text is None or text == "":
        return None
    return text.strip().lower() == "true"


def parse_float(text):
    if text is None or text == "":
        return None
    return float(text)


def load_unified_rows(path: Path, architecture: str):
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        obj = json.loads(line)
        grounding = obj["grounding"]
        applicable = bool(grounding.get("applicable"))
        style = obj["persona_style"].get("distance")
        utility = obj.get("persona_utility")
        rows.append(
            {
                "architecture": architecture,
                "id": obj["id"],
                "medium": obj["medium"],
                "eligible": bool(obj["eligible_for_ranking"]),
                "completion_pass": bool(obj["gates"]["completion"]["pass"]),
                "format_pass": bool(obj["gates"]["format"]["pass"]),
                "copy_pass": bool(obj["gates"]["copying_memorization"]["pass"]),
                "grounding_applicable": applicable,
                "grounding_pass": bool(grounding["pass"]) if applicable else None,
                "claim_support": grounding.get("claim_support_rate") if applicable else None,
                "writing_quality": obj["writing_quality"].get("mean"),
                "task_adherence": obj["task_adherence"].get("score"),
                "style_distance": style,
                "stance_consistency": obj["perspective_fidelity"].get("stance_consistency_rate"),
                "persona_utility": utility if isinstance(utility, (int, float)) else None,
                "word_count": obj["gates"]["format"].get("word_count"),
                "latency_seconds": (obj.get("diagnostics") or {}).get("latency_seconds"),
                "source": str(path.relative_to(ROOT)).replace("\\", "/"),
            }
        )
    return rows


def load_model_f_rows():
    judge_path = ROOT / "results/model-f-panel-rows-v1/judge_scores.csv"
    with judge_path.open(encoding="utf-8", newline="") as handle:
        judge = list(csv.DictReader(handle))
    rows = []
    for obj in judge:
        grounding_pass = parse_bool(obj["grounding_pass"])
        # The comparison file scores grounding on all 120 Model F drafts.
        rows.append(
            {
                "architecture": "Model F",
                "id": obj["question_id"],
                "medium": obj["medium"],
                "eligible": parse_bool(obj["eligible_for_ranking"]),
                "completion_pass": parse_bool(obj["completion_pass"]),
                "format_pass": parse_bool(obj["format_pass"]),
                "copy_pass": parse_bool(obj["copying_pass"]),
                "grounding_applicable": True,
                "grounding_pass": grounding_pass,
                "claim_support": parse_float(obj["grounding_claim_support_rate"]),
                "writing_quality": parse_float(obj["writing_quality"]),
                "task_adherence": parse_float(obj["task_adherence"]),
                "style_distance": parse_float(obj["style_distance"]),
                "stance_consistency": parse_float(obj["stance_consistency"]),
                "persona_utility": parse_float(obj["persona_utility"]),
                "word_count": parse_float(obj["format_word_count"]),
                "latency_seconds": parse_float(obj["latency_seconds"]),
                "source": "results/model-f-panel-rows-v1/judge_scores.csv",
            }
        )
    return rows


def summarize(rows, architecture, medium):
    subset = [row for row in rows if row["architecture"] == architecture and (medium == "all" or row["medium"] == medium)]
    writing_all, writing_n, writing_missing = mean([row["writing_quality"] for row in subset])
    writing_elig, writing_elig_n, _ = mean([row["writing_quality"] for row in subset if row["eligible"]])
    task, task_n, task_missing = mean([row["task_adherence"] for row in subset])
    task_elig, task_elig_n, _ = mean([row["task_adherence"] for row in subset if row["eligible"]])
    style, style_n, style_missing = mean([row["style_distance"] for row in subset])
    stance, stance_n, stance_missing = mean([row["stance_consistency"] for row in subset])
    utility, utility_n, utility_missing = mean([row["persona_utility"] for row in subset])
    words, words_n, words_missing = mean([row["word_count"] for row in subset])
    latency, latency_n, latency_missing = mean([row["latency_seconds"] for row in subset])
    claim, claim_n, claim_missing = mean([row["claim_support"] for row in subset])
    completion, completion_n, _ = rate([row["completion_pass"] for row in subset])
    format_rate, format_n, _ = rate([row["format_pass"] for row in subset])
    copy_rate, copy_n, _ = rate([row["copy_pass"] for row in subset])
    grounding_flags = [row["grounding_pass"] for row in subset if row["grounding_applicable"]]
    grounding, grounding_n, _ = rate(grounding_flags) if grounding_flags else (None, 0, len(subset))
    eligible = sum(1 for row in subset if row["eligible"])
    return {
        "architecture": architecture,
        "medium": medium,
        "drafts": len(subset),
        "eligible": eligible,
        "completion_pass_rate": completion,
        "completion_n": completion_n,
        "completion_failures": sum(1 for row in subset if row["completion_pass"] is False),
        "format_pass_rate": format_rate,
        "format_n": format_n,
        "format_failures": sum(1 for row in subset if row["format_pass"] is False),
        "copy_pass_rate": copy_rate,
        "copy_failures": sum(1 for row in subset if row["copy_pass"] is False),
        "grounding_pass_rate": grounding,
        "grounding_n": grounding_n,
        "grounding_failures": None if grounding is None else sum(1 for flag in grounding_flags if flag is False),
        "writing_quality_all": writing_all,
        "writing_quality_all_n": writing_n,
        "writing_quality_all_missing": writing_missing,
        "writing_quality_eligible": writing_elig,
        "writing_quality_eligible_n": writing_elig_n,
        "task_adherence_all": task,
        "task_adherence_all_n": task_n,
        "task_adherence_eligible": task_elig,
        "task_adherence_eligible_n": task_elig_n,
        "style_distance": style,
        "style_n": style_n,
        "style_missing": style_missing,
        "stance_consistency": stance,
        "stance_n": stance_n,
        "stance_missing": stance_missing,
        "persona_utility": utility,
        "persona_utility_n": utility_n,
        "persona_utility_missing": utility_missing,
        "word_count": words,
        "word_n": words_n,
        "latency_seconds": latency,
        "latency_n": latency_n,
        "latency_missing": latency_missing,
        "claim_support": claim,
        "claim_support_n": claim_n,
        "claim_support_missing": claim_missing,
    }


def check_against_published(summary_rows):
    published = load_json(ROOT / "results/model-f-handoff-v1/architecture_comparison.json")
    notes = []
    by_arch = {row["architecture"]: row for row in summary_rows if row["medium"] == "all"}
    for item in published["rows"]:
        got = by_arch[item["architecture"]]
        checks = [
            ("eligible", got["eligible"], item["eligible"]),
            ("completion", got["completion_pass_rate"], item["gates"]["completion"]["pass_rate"]),
            ("format", got["format_pass_rate"], item["gates"]["format"]["pass_rate"]),
            ("copy", got["copy_pass_rate"], item["gates"]["copying_memorization"]["pass_rate"]),
            ("grounding", got["grounding_pass_rate"], item["gates"]["grounding"]["pass_rate"]),
            ("writing", got["writing_quality_all"], item["metrics"]["writing_quality"]["mean"]),
            ("task", got["task_adherence_all"], item["metrics"]["task_adherence"]["mean"]),
            ("style", got["style_distance"], item["metrics"]["style_distance"]["mean"]),
            ("utility", got["persona_utility"], item["metrics"]["persona_utility"]["mean"]),
            ("words", got["word_count"], item["metrics"]["word_count"]["mean"]),
            ("claim", got["claim_support"], item["metrics"]["claim_support"]["mean"]),
        ]
        for name, left, right in checks:
            if left is None and right is None:
                continue
            if left is None or right is None or abs(left - right) > 5e-4:
                notes.append(f"MISMATCH {item['architecture']} {name}: computed={left} published={right}")
        if not any(note.startswith(f"MISMATCH {item['architecture']}") for note in notes):
            notes.append(f"MATCH {item['architecture']} medium=all against architecture_comparison.json")
    model_f = by_arch["Model F"]
    summary = load_json(ROOT / "results/model-f-handoff-v1/aggregate_summary.json")["model"]
    eligible_writing = summary["writing_quality"]["mean_eligible"]
    if abs(model_f["writing_quality_eligible"] - eligible_writing) > 5e-4:
        notes.append(
            f"MISMATCH Model F eligible writing: computed={model_f['writing_quality_eligible']} summary={eligible_writing}"
        )
    else:
        notes.append(f"MATCH Model F eligible writing quality {eligible_writing} against aggregate_summary.json")
    if model_f["eligible"] != summary["n_eligible"]:
        notes.append(f"MISMATCH eligible count {model_f['eligible']} vs {summary['n_eligible']}")
    return notes


def topic_bootstrap(pairs, key, draws=4000, seed=42):
    """Cluster bootstrap by question id. Difference is Model F minus the other architecture."""
    by_topic = defaultdict(list)
    for pair in pairs:
        if pair["a"][key] is None or pair["b"][key] is None:
            continue
        by_topic[pair["id"]].append(pair["b"][key] - pair["a"][key])
    topics = sorted(by_topic)
    if len(topics) < 2:
        return None
    point_values = [value for topic in topics for value in by_topic[topic]]
    point = sum(point_values) / len(point_values)
    rng = random.Random(seed)
    samples = []
    for _ in range(draws):
        chosen = [rng.choice(topics) for _ in topics]
        values = [value for topic in chosen for value in by_topic[topic]]
        samples.append(sum(values) / len(values))
    samples.sort()
    lo = samples[int(0.025 * (draws - 1))]
    hi = samples[int(0.975 * (draws - 1))]
    return {
        "n_pairs": len(point_values),
        "n_topics": len(topics),
        "mean_difference_f_minus_other": point,
        "ci95_low": lo,
        "ci95_high": hi,
        "method": "topic-cluster bootstrap, 4000 draws, seed 42, percentile interval",
    }


def paired_table(all_rows):
    by_key = {(row["architecture"], row["id"], row["medium"]): row for row in all_rows}
    others = [name for name in ARCH_ORDER if name != "Model F"]
    metrics = ["writing_quality", "task_adherence", "style_distance", "word_count"]
    out = []
    ids = sorted({row["id"] for row in all_rows if row["architecture"] == "Model F"})
    for other in others:
        pairs = []
        for question_id in ids:
            for medium in MEDIA:
                left = by_key.get((other, question_id, medium))
                right = by_key.get(("Model F", question_id, medium))
                if left and right:
                    pairs.append({"id": question_id, "a": left, "b": right})
        for metric in metrics:
            stats = topic_bootstrap(pairs, metric)
            if stats is None:
                continue
            out.append({"comparator": other, "metric": metric, **stats, "interpretation": difference_note(metric)})
    return out


def difference_note(metric):
    if metric == "style_distance":
        return "Negative means Model F is closer to the held-out style profile. Distance is not authorship accuracy."
    if metric == "word_count":
        return "Difference in mean words. Not a quality score."
    return "Positive means Model F is higher on this saved score."


def style_axes():
    plt.rcParams.update(
        {
            "font.size": 11,
            "axes.titlesize": 13,
            "axes.titleweight": "semibold",
            "axes.labelsize": 11,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "savefig.bbox": "tight",
        }
    )


def save(fig, stem):
    fig.savefig(FIG / f"{stem}.png", dpi=160)
    fig.savefig(FIG / f"{stem}.svg")
    plt.close(fig)


def bar_colors(names):
    return [COLORS[name] for name in names]


def annotate_bars(ax, bars, labels):
    for bar, label in zip(bars, labels):
        if label is None:
            continue
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, height, label, ha="center", va="bottom", fontsize=8)


def chart_single(stem, title, ylabel, values, n_labels, source, ylim=None, percent=False):
    style_axes()
    fig, ax = plt.subplots(figsize=(9.2, 5.2))
    names = []
    heights = []
    colors = []
    annotations = []
    for name in ARCH_ORDER:
        value = values[name]
        if value is None:
            continue
        names.append(name)
        heights.append(value * 100 if percent else value)
        colors.append(COLORS[name])
        annotations.append(n_labels[name])
    bars = ax.bar(names, heights, color=colors, width=0.72)
    annotate_bars(ax, bars, annotations)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    if ylim:
        ax.set_ylim(*ylim)
    else:
        top = max(heights) if heights else 1
        ax.set_ylim(0, top * 1.18)
    ax.set_xlabel("Architecture")
    fig.text(0.01, 0.01, source, fontsize=8, color="#444444")
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    save(fig, stem)
    return names


def grouped_bars(stem, title, ylabel, table, value_key, n_key, source, percent=False, architectures=None):
    style_axes()
    architectures = architectures or ARCH_ORDER
    fig, ax = plt.subplots(figsize=(10.4, 5.6))
    width = 0.15
    x0 = list(range(len(MEDIA)))
    plotted = False
    for index, architecture in enumerate(architectures):
        heights = []
        for medium in MEDIA:
            row = next(item for item in table if item["architecture"] == architecture and item["medium"] == medium)
            value = row[value_key]
            heights.append(None if value is None else (value * 100 if percent else value))
        if all(value is None for value in heights):
            continue
        plotted = True
        offset = (index - (len(architectures) - 1) / 2) * width
        plot_heights = [0 if value is None else value for value in heights]
        bars = ax.bar(
            [x + offset for x in x0],
            plot_heights,
            width=width,
            color=COLORS[architecture],
            label=architecture,
        )
        for bar, value, medium in zip(bars, heights, MEDIA):
            if value is None:
                bar.set_height(0)
                bar.set_alpha(0)
                ax.text(bar.get_x() + bar.get_width() / 2, 0.02, "n/a", ha="center", va="bottom", fontsize=7, rotation=90, color="#666666")
    if not plotted:
        plt.close(fig)
        return
    ax.set_xticks(x0, [MEDIA_LABEL[medium] for medium in MEDIA])
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xlabel("Medium")
    ax.legend(frameon=False, ncol=3, fontsize=9)
    fig.text(0.01, 0.01, source + " Blank or n/a means the saved row has no value. It is not zero.", fontsize=8, color="#444444")
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    save(fig, stem)


def box_edge(box, toward):
    x, y, w, h = box
    cx, cy = x + w / 2, y + h / 2
    dx, dy = toward[0] - cx, toward[1] - cy
    if dx == 0 and dy == 0:
        return cx, cy
    scale_x = (w / 2) / abs(dx) if dx else 1e9
    scale_y = (h / 2) / abs(dy) if dy else 1e9
    scale = min(scale_x, scale_y) * 0.92
    return cx + dx * scale, cy + dy * scale


def draw_flow(stem, title, nodes, edges, source):
    style_axes()
    fig, ax = plt.subplots(figsize=(11, 6.2))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")
    ax.set_title(title, loc="left")
    centers = {}
    boxes = {}
    for node in nodes:
        x, y, w, h = node["box"]
        patch = FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.3,rounding_size=0.8",
            facecolor=node.get("color", "#F4F7FB"),
            edgecolor="#1F4E79",
            linewidth=1.2,
        )
        ax.add_patch(patch)
        ax.text(x + w / 2, y + h / 2, node["text"], ha="center", va="center", fontsize=9)
        centers[node["id"]] = (x + w / 2, y + h / 2)
        boxes[node["id"]] = node["box"]
    for start, end in edges:
        x1, y1 = box_edge(boxes[start], centers[end])
        x2, y2 = box_edge(boxes[end], centers[start])
        ax.annotate(
            "",
            xy=(x2, y2),
            xytext=(x1, y1),
            arrowprops={"arrowstyle": "-|>", "color": "#1F4E79", "lw": 1.2, "shrinkA": 0, "shrinkB": 0},
        )
    fig.text(0.01, 0.01, source, fontsize=8, color="#444444")
    save(fig, stem)


def main():
    historical = []
    for architecture, path in ROW_FILES.items():
        historical.extend(load_unified_rows(path, architecture))
    model_f = load_model_f_rows()
    all_rows = historical + model_f
    summary_rows = [summarize(all_rows, architecture, medium) for architecture in ARCH_ORDER for medium in ["all", *MEDIA]]
    notes = check_against_published(summary_rows)
    if any(note.startswith("MISMATCH") for note in notes):
        (TAB / "verification_notes.txt").write_text("\n".join(notes), encoding="utf-8")
        raise SystemExit("Published comparison did not match recomputed rows. No figures written.\n" + "\n".join(notes))

    flat_rows = []
    for row in summary_rows:
        flat_rows.append({key: (fnum(value) if isinstance(value, float) else value) for key, value in row.items()})
    write_csv(TAB / "architecture_metrics_by_medium.csv", flat_rows)
    write_csv(TAB / "paired_model_f_differences.csv", [
        {key: (fnum(value, 4) if isinstance(value, float) else value) for key, value in row.items()}
        for row in paired_table(all_rows)
    ])

    by_all = {row["architecture"]: row for row in summary_rows if row["medium"] == "all"}
    source_arch = (
        "Source: seed-42 unified_eval.v1 rows for Base, PersonaRAG, Relabel RAG off, and Relabel + RAG; "
        "Model F from results/model-f-panel-rows-v1/judge_scores.csv. Denominator is 120 drafts unless the label says otherwise."
    )
    chart_single(
        "01_eligible_drafts",
        "Eligible drafts by architecture (120-draft panel)",
        "Eligible drafts",
        {name: by_all[name]["eligible"] for name in ARCH_ORDER},
        {name: f"{by_all[name]['eligible']}/120" for name in ARCH_ORDER},
        source_arch,
        ylim=(0, 100),
    )
    chart_single(
        "02_completion_pass_rate",
        "Completion pass rate by architecture",
        "Completion pass rate (%)",
        {name: by_all[name]["completion_pass_rate"] for name in ARCH_ORDER},
        {name: f"{by_all[name]['completion_pass_rate']*100:.1f}%" for name in ARCH_ORDER},
        source_arch + " Gate: non-empty sentence-final answer.",
        percent=True,
        ylim=(0, 100),
    )
    retrieval = ["PersonaRAG", "Relabel + RAG", "Model F"]
    chart_single(
        "03_grounding_pass_rate",
        "Judge grounding pass rate, retrieval-enabled architectures",
        "Grounding pass rate (%)",
        {name: by_all[name]["grounding_pass_rate"] if name in retrieval else None for name in ARCH_ORDER},
        {name: f"{by_all[name]['grounding_pass_rate']*100:.1f}%" for name in retrieval},
        source_arch + " Base and Relabel RAG off are omitted because retrieval was off. This is not the deterministic number audit.",
        percent=True,
        ylim=(0, 100),
    )
    chart_single(
        "04_writing_quality_all_drafts",
        "Writing quality on all scored drafts",
        "Mean writing quality (1–5 judge scale)",
        {name: by_all[name]["writing_quality_all"] for name in ARCH_ORDER},
        {name: f"n={by_all[name]['writing_quality_all_n']}" for name in ARCH_ORDER},
        source_arch + " All-draft mean. Model F eligible-only mean is a different number (3.2662).",
        ylim=(0, 5),
    )
    chart_single(
        "04b_writing_quality_eligible",
        "Writing quality on eligible drafts only",
        "Mean writing quality (1–5 judge scale)",
        {name: by_all[name]["writing_quality_eligible"] for name in ARCH_ORDER},
        {name: f"n={by_all[name]['writing_quality_eligible_n']}" for name in ARCH_ORDER},
        source_arch + " Eligible means the saved ranking gates passed. Do not mix this chart with the all-draft chart.",
        ylim=(0, 5),
    )
    chart_single(
        "05_task_adherence",
        "Task adherence on all scored drafts",
        "Mean task-adherence score (0–1)",
        {name: by_all[name]["task_adherence_all"] for name in ARCH_ORDER},
        {name: f"n={by_all[name]['task_adherence_all_n']}" for name in ARCH_ORDER},
        source_arch,
        ylim=(0, 1),
    )
    chart_single(
        "06_style_distance",
        "Style distance (lower is closer to the held-out profile)",
        "Mean style distance",
        {name: by_all[name]["style_distance"] for name in ARCH_ORDER},
        {name: f"n={by_all[name]['style_n']}" for name in ARCH_ORDER},
        source_arch + " n=90 because talk style is missing on every architecture. Not an authorship test.",
        ylim=(0, 1.4),
    )

    grouped_bars(
        "07_writing_quality_by_medium",
        "Writing quality by architecture and medium",
        "Mean writing quality (1–5)",
        summary_rows,
        "writing_quality_all",
        "writing_quality_all_n",
        source_arch + " Each bar is 30 drafts.",
    )
    grouped_bars(
        "08_eligibility_by_medium",
        "Eligible drafts by architecture and medium",
        "Eligible drafts (of 30)",
        summary_rows,
        "eligible",
        "drafts",
        source_arch,
    )
    grouped_bars(
        "09_completion_failures_by_medium",
        "Completion failures by architecture and medium",
        "Completion failures (of 30)",
        summary_rows,
        "completion_failures",
        "drafts",
        source_arch,
    )
    grouped_bars(
        "10_grounding_failures_by_medium",
        "Judge grounding failures by medium",
        "Grounding failures (of 30)",
        summary_rows,
        "grounding_failures",
        "grounding_n",
        source_arch + " Base and Relabel RAG off have no applicable grounding gate.",
        architectures=retrieval,
    )
    grouped_bars(
        "11_mean_words_by_medium",
        "Mean output length by architecture and medium",
        "Mean word count",
        summary_rows,
        "word_count",
        "word_n",
        source_arch + " Word count is the saved format word count, not tokens.",
    )
    grouped_bars(
        "12_format_compliance_by_medium",
        "Format compliance by architecture and medium",
        "Format pass rate (%)",
        summary_rows,
        "format_pass_rate",
        "format_n",
        source_arch,
        percent=True,
    )

    flags_path = ROOT / "results/model-f-panel-rows-v1/deterministic_flags.csv"
    with flags_path.open(encoding="utf-8", newline="") as handle:
        flags = list(csv.DictReader(handle))
    judge_by_key = {(row["id"], row["medium"]): row for row in model_f}
    grid = {(False, False): 0, (False, True): 0, (True, False): 0, (True, True): 0}
    medium_rows = []
    for flag in flags:
        judge = judge_by_key[(flag["question_id"], flag["medium"])]
        grounding_fail = judge["grounding_pass"] is False
        number_fail = parse_bool(flag["numeric_pass"]) is False
        grid[(grounding_fail, number_fail)] += 1
    style_axes()
    fig, ax = plt.subplots(figsize=(7.4, 5.4))
    matrix = [
        [grid[(False, False)], grid[(False, True)]],
        [grid[(True, False)], grid[(True, True)]],
    ]
    image = ax.imshow(matrix, cmap="Blues")
    ax.set_xticks([0, 1], ["Number check passed", "Unsupported number"])
    ax.set_yticks([0, 1], ["Judge grounding passed", "Judge grounding failed"])
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(matrix[i][j]), ha="center", va="center", color="black", fontsize=14)
    ax.set_title("Judge grounding versus deterministic number flags\nModel F, 120 drafts")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04, label="Drafts")
    fig.text(0.01, 0.01, "Source: judge_scores.csv joined to deterministic_flags.csv on question_id and medium. These checks are not the same test.", fontsize=8)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    save(fig, "13_grounding_vs_unsupported_numbers")

    cap_counts = []
    copy_counts = []
    number_counts = []
    for medium in MEDIA:
        subset = [row for row in flags if row["medium"] == medium]
        cap_counts.append(sum(1 for row in subset if parse_bool(row["at_max_tokens"])))
        copy_counts.append(sum(1 for row in subset if parse_bool(row["source_copy_pass"]) is False))
        number_counts.append(sum(1 for row in subset if parse_bool(row["numeric_pass"]) is False))
        medium_rows.append(
            {
                "medium": medium,
                "drafts": len(subset),
                "at_max_tokens": cap_counts[-1],
                "source_copy_fail_12_words": copy_counts[-1],
                "unsupported_numeric_cells": number_counts[-1],
                "mid_sentence_stop": sum(1 for row in subset if parse_bool(row["mid_sentence_stop"])),
                "zero_accepted_claims": sum(1 for row in subset if int(row["accepted_claims"]) == 0),
                "faithfulness_fail": sum(1 for row in subset if parse_bool(row["faithfulness_pass"]) is False),
            }
        )
    write_csv(TAB / "model_f_failures_by_medium.csv", medium_rows)

    def medium_bar(stem, title, ylabel, heights, source):
        style_axes()
        fig, ax = plt.subplots(figsize=(7.6, 4.8))
        labels = [MEDIA_LABEL[medium] for medium in MEDIA]
        bars = ax.bar(labels, heights, color="#72B7B2")
        annotate_bars(ax, bars, [f"{value}/30" for value in heights])
        ax.set_ylim(0, 30)
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.set_xlabel("Medium")
        fig.text(0.01, 0.01, source, fontsize=8)
        fig.tight_layout(rect=(0, 0.05, 1, 1))
        save(fig, stem)

    flag_source = "Source: results/model-f-panel-rows-v1/deterministic_flags.csv. Denominator 30 drafts per medium."
    medium_bar("14_token_cap_by_medium", "Model F outputs at the medium token cap", "Drafts at the recorded cap", cap_counts, flag_source)
    medium_bar("15_source_copy_failures", "Model F retrieved-passage copy failures (12-word threshold)", "Failed drafts", copy_counts, flag_source + " Training-copy pass is a separate gate and is not this chart.")
    medium_bar("18_unsupported_numbers_by_medium", "Model F deterministic unsupported-number flags by medium", "Drafts with an unsupported quantity", number_counts, flag_source + " Not the judge grounding gate.")

    status_counts = defaultdict(int)
    for row in flags:
        status_counts[row["evidence_status"] or "missing"] += 1
    style_axes()
    fig, ax = plt.subplots(figsize=(7.6, 4.8))
    labels = list(status_counts)
    heights = [status_counts[label] for label in labels]
    bars = ax.bar(labels, heights, color="#4C78A8")
    annotate_bars(ax, bars, [str(value) for value in heights])
    ax.set_title("Model F evidence-extraction status (120 drafts)")
    ax.set_ylabel("Drafts")
    ax.set_xlabel("evidence_status on the deterministic audit row")
    fig.text(0.01, 0.01, flag_source + " The same status is repeated across the four media of a topic.", fontsize=8)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    save(fig, "16_evidence_extraction_status")

    omissions = load_json(ROOT / "experiments/EXP-20261102-001-model-f-unified-panel/diagnostics/descriptive_omissions.json")
    events = omissions["omission_events"]
    kind_counts = defaultdict(int)
    for event in events:
        kind_counts[event["kind"]] += 1
    audit = load_json(ROOT / "results/model-f-handoff-v1/deterministic_audit_summary.json")["overall"]
    style_axes()
    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    labels = ["Passage omitted\n(audit cells)", "Graph items omitted\n(audit cells)", "Whole-passage events\n(descriptive file)", "Graph-item events\n(descriptive file)"]
    heights = [
        audit["passage_omitted"]["count"],
        audit["graph_items_omitted"]["count"],
        kind_counts["whole_passage"],
        kind_counts["graph_item"],
    ]
    bars = ax.bar(labels, heights, color=["#E45756", "#F58518", "#4C78A8", "#72B7B2"])
    annotate_bars(ax, bars, [str(value) for value in heights])
    ax.set_ylim(0, max(heights) * 1.25)
    ax.set_title("Recorded retrieval and context omissions, Model F")
    ax.set_ylabel("Count")
    fig.text(
        0.01,
        0.01,
        "Audit counts are cells out of 120. Descriptive events are the omission list in descriptive_omissions.json. The two series use different units.",
        fontsize=8,
    )
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    save(fig, "17_retrieval_omissions")

    provenance = load_json(ROOT / "results/model-f-handoff-v1/provenance.json")
    graph = provenance["graph"]
    style_axes()
    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    coverage_labels = ["Indexed\ncompleted", "Pending", "Interrupted", "Permanent\nfailures"]
    coverage = [2713, 10269, 153, 5]
    bars = ax.bar(coverage_labels, coverage, color=["#54A24B", "#F2C14E", "#F58518", "#E45756"])
    annotate_bars(ax, bars, [f"{value:,}" for value in coverage])
    ax.set_title("Frozen LightRAG snapshot coverage of eligible documents")
    ax.set_ylabel("Documents")
    ax.set_ylim(0, 12000)
    fig.text(
        0.01,
        0.01,
        f"Source: results/model-f-handoff-v1/provenance.json and docs/model_f_architecture.md. Eligible documents 13,140. Nodes {graph['nodes']:,}; edges {graph['edges']:,}. Not a quality score.",
        fontsize=8,
    )
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    save(fig, "20_indexed_vs_eligible_coverage")

    style_axes()
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    bars = ax.bar(["Nodes", "Edges"], [graph["nodes"], graph["edges"]], color=["#4C78A8", "#72B7B2"])
    annotate_bars(ax, bars, [f"{graph['nodes']:,}", f"{graph['edges']:,}"])
    ax.set_title("Frozen snapshot size after shutdown flush")
    ax.set_ylabel("Count")
    ax.set_ylim(0, 60000)
    fig.text(0.01, 0.01, "Source: provenance.json graph object. Node and edge types were not in the public export. Size is not retrieval quality.", fontsize=8)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    save(fig, "19_graph_size")

    latencies = [row["latency_seconds"] for row in model_f if row["latency_seconds"] is not None]
    style_axes()
    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.hist(latencies, bins=15, color="#72B7B2", edgecolor="white")
    ax.axvline(sum(latencies) / len(latencies), color="#E45756", linestyle="--", label=f"Mean {sum(latencies)/len(latencies):.2f} s")
    ax.set_title("Model F generation latency")
    ax.set_xlabel("Generation seconds recorded on the judge row")
    ax.set_ylabel("Drafts")
    ax.legend(frameon=False)
    fig.text(0.01, 0.01, "Source: judge_scores.csv latency_seconds, n=120. This matches the audit generation_seconds mean. Retrieval time is not a separate recorded component.", fontsize=8)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    save(fig, "23_generation_latency")

    by_medium_latency = []
    for medium in MEDIA:
        values = [row["latency_seconds"] for row in model_f if row["medium"] == medium]
        by_medium_latency.append(sum(values) / len(values))
    style_axes()
    fig, ax = plt.subplots(figsize=(7.6, 4.8))
    bars = ax.bar([MEDIA_LABEL[m] for m in MEDIA], by_medium_latency, color="#4C78A8")
    annotate_bars(ax, bars, [f"{value:.1f}s" for value in by_medium_latency])
    ax.set_title("Model F mean generation latency by medium")
    ax.set_ylabel("Mean generation seconds")
    ax.set_xlabel("Medium")
    fig.text(0.01, 0.01, "Source: judge_scores.csv. n=30 per medium. No retrieval-versus-generation split was stored, so this is not an end-to-end bottleneck chart.", fontsize=8)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    save(fig, "24_latency_by_medium")

    draw_flow(
        "21_graph_construction",
        "LightRAG construction used for the frozen Model F snapshot",
        [
            {"id": "src", "text": "Eligible train\ndocuments\n13,140", "box": (3, 62, 18, 22)},
            {"id": "chunk", "text": "Chunk\n400 tokens", "box": (28, 62, 16, 22)},
            {"id": "llm", "text": "Ollama llama3.1\nentity and relation\nextraction", "box": (52, 62, 20, 22)},
            {"id": "graph", "text": "Graph\n27,797 nodes\n47,897 edges", "box": (78, 62, 18, 22)},
            {"id": "embed", "text": "nomic-embed-text\n768-d vectors", "box": (28, 22, 20, 20)},
            {"id": "cov", "text": "2,713 completed\n10,269 pending", "box": (58, 22, 22, 20)},
        ],
        [("src", "chunk"), ("chunk", "llm"), ("llm", "graph"), ("chunk", "embed"), ("graph", "cov")],
        "Source: docs/model_f_architecture.md and results/model-f-handoff-v1/provenance.json. The pilot store was not copied into this snapshot.",
    )
    draw_flow(
        "22_retrieval_generation",
        "Official F-REL retrieval, evidence, and generation path",
        [
            {"id": "topic", "text": "Gold topic", "box": (3, 40, 14, 16)},
            {"id": "query", "text": "LightRAG hybrid\ntop_k 20", "box": (22, 40, 16, 16)},
            {"id": "claims", "text": "Accepted claims\nbudget 4", "box": (43, 58, 16, 16)},
            {"id": "rel", "text": "F-REL lines\npassage-supported", "box": (43, 24, 18, 16)},
            {"id": "window", "text": "Fit to\n2048 tokens", "box": (66, 40, 14, 16)},
            {"id": "gen", "text": "Relabel QLoRA\nLlama 3.1 8B", "box": (84, 40, 14, 16)},
        ],
        [("topic", "query"), ("query", "claims"), ("query", "rel"), ("claims", "window"), ("rel", "window"), ("window", "gen")],
        "Source: docs/model_f_architecture.md. One retrieval is reused across four media. The 120-prompt panel did not select F-REL.",
    )
    draw_flow(
        "25_system_architecture",
        "Repository system paths that exist in this checkout",
        [
            {"id": "data", "text": "Cleaned corpus\nand gold topics", "box": (3, 62, 20, 18)},
            {"id": "train", "text": "Relabel QLoRA\nadapter", "box": (30, 62, 18, 18)},
            {"id": "dense", "text": "Chroma\ntrain-only RAG", "box": (54, 70, 18, 16)},
            {"id": "graph", "text": "LightRAG\nsnapshot", "box": (54, 46, 18, 16)},
            {"id": "eval", "text": "unified_eval.v1\nQwen 14B judge", "box": (78, 62, 18, 18)},
            {"id": "api", "text": "backend/main.py\nolder generate.py", "box": (30, 22, 22, 16)},
        ],
        [("data", "train"), ("train", "dense"), ("train", "graph"), ("dense", "eval"), ("graph", "eval"), ("data", "api")],
        "Source: README.md, docs/architecture/ARCHITECTURE.md, docs/teammate_handoff.md. The React demo is not in this checkout.",
    )
    draw_flow(
        "26_inference_sequence",
        "Model F inference sequence for one official panel cell",
        [
            {"id": "a", "text": "1. Read frozen\nsnapshot", "box": (2, 40, 16, 16)},
            {"id": "b", "text": "2. Hybrid query\nnum_ctx 16384", "box": (22, 40, 16, 16)},
            {"id": "c", "text": "3. Extract claims\nadapter off", "box": (42, 40, 16, 16)},
            {"id": "d", "text": "4. Render F-REL\nand fit window", "box": (62, 40, 16, 16)},
            {"id": "e", "text": "5. Generate\nadapter on", "box": (82, 40, 16, 16)},
        ],
        [("a", "b"), ("b", "c"), ("c", "d"), ("d", "e")],
        "Source: docs/model_f_architecture.md. Official profile F-REL. Frozen pilot selection remains F-SOURCE.",
    )
    draw_flow(
        "27_deployment_dependencies",
        "What a Model F demonstration depends on",
        [
            {"id": "py", "text": "Python 3.11\nhost_finetune venv", "box": (4, 62, 20, 16)},
            {"id": "gpu", "text": "CUDA Llama\n4-bit adapter", "box": (32, 62, 18, 16)},
            {"id": "ollama", "text": "Ollama\nnomic, llama3.1,\nqwen judge", "box": (58, 58, 18, 22)},
            {"id": "files", "text": "Snapshot zip\nand adapter zip\nnot in Git", "box": (18, 24, 22, 18)},
            {"id": "ui", "text": "React service\non another branch", "box": (58, 24, 20, 16)},
        ],
        [("py", "gpu"), ("gpu", "ollama"), ("files", "gpu"), ("ui", "gpu")],
        "Source: README.md and docs/teammate_handoff.md. No response-time target is stored. The React route is not wired to LightRAG in this checkout.",
    )

    (TAB / "verification_notes.txt").write_text("\n".join(notes) + "\n", encoding="utf-8")
    print("\n".join(notes))
    print(f"Wrote figures to {FIG}")
    print(f"Wrote tables to {TAB}")


if __name__ == "__main__":
    main()
