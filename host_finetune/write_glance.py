"""Write the experiment glance from the mixed-medium comparison files."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(
    r"C:\Users\okevi\.cursor\projects\c-Users-okevi-SJSU-DATA298-298P"
    r"\canvases\experiment-glance.canvas.tsx"
)
MODELS = (
    ("llama3.1", "Llama 3.1", "experiments/EXP-20261001-010-repaired-and-baselines"),
    ("lemkin-clone-v5", "Clone v5", "experiments/EXP-20261001-010-repaired-and-baselines"),
    ("lemkin-clone-v6", "Clone v6", "experiments/EXP-20261001-010-repaired-and-baselines"),
    ("cleaned", "Cleaned", "experiments/EXP-20261001-007-mixed-medium-compare"),
    ("balanced", "Balanced", "experiments/EXP-20261001-007-mixed-medium-compare"),
    ("r32", "Rank 32", "experiments/EXP-20261001-007-mixed-medium-compare"),
    ("repaired", "Repaired", "experiments/EXP-20261001-010-repaired-and-baselines"),
)


def load(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _cell(value) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


RUN_LABELS = {
    "EXP-20261001-007-mixed-medium-compare": "Oct 1 mixed",
    "EXP-20261001-010-repaired-and-baselines": "Oct 1 baselines",
    "EXP-20261004-003-repaired-vs-relabel": "Oct 4 retrieval off",
    "EXP-20261004-004-repaired-vs-relabel-retrieval": "Oct 4 retrieval on",
    "EXP-20261004-005-repaired-vs-relabel-seed43": "Oct 4 seed 43",
    "EXP-20261004-006-repaired-vs-relabel-seed44": "Oct 4 seed 44",
}


def _run_label(model: dict) -> str:
    return RUN_LABELS.get(model["source_experiment"], model["source_experiment"])


ADAPTER_LABELS = {
    "llama3.1": "Llama 3.1",
    "lemkin-clone-v5": "Clone v5",
    "lemkin-clone-v6": "Clone v6",
    "cleaned": "Cleaned",
    "balanced": "Balanced",
    "r32": "Rank 32",
    "repaired": "Repaired",
    "relabel": "Relabel",
}
SEED_EXPERIMENTS = frozenset({
    "EXP-20261004-003-repaired-vs-relabel",
    "EXP-20261004-005-repaired-vs-relabel-seed43",
    "EXP-20261004-006-repaired-vs-relabel-seed44",
})
CLAIM_LABELS = ("supported", "unsupported", "contradicted", "not_verifiable", "not_applicable")


def _adapter_name(model: dict) -> str:
    return model["model_id"].split("/", 1)[1]


def _adapter_label(name: str) -> str:
    return ADAPTER_LABELS.get(name, name)


def _retrieval_off(model: dict) -> bool:
    return model["gates"]["grounding_applicable_rows"] == 0


def _stats(values: list[float]) -> dict:
    ordered = sorted(values)
    count = len(ordered)
    mean = sum(ordered) / count
    mid = count // 2
    if count % 2:
        median = ordered[mid]
    else:
        median = (ordered[mid - 1] + ordered[mid]) / 2
    return {
        "n": count,
        "mean": mean,
        "median": median,
        "min": ordered[0],
        "max": ordered[-1],
        "spread": ordered[-1] - ordered[0],
    }


def _eligible_span(models: list[dict]) -> str:
    counts = [model["n_eligible"] for model in models]
    total = models[0]["n"]
    if len(counts) == 1:
        return f"{counts[0]}/{total}"
    return f"{min(counts)}–{max(counts)}/{total}"


def _place(model: dict, value, digits: int = 3) -> str:
    name = _adapter_label(_adapter_name(model))
    return f"{name} · {_run_label(model)} · {value:.{digits}f} · {model['n_eligible']}/{model['n']} eligible"


def _top_distinct(models: list[dict], getter, higher_better: bool, limit: int = 3) -> list[str]:
    """Best run of each adapter, then the top distinct adapters."""
    chosen: dict[str, tuple[float, dict]] = {}
    for model in models:
        value = getter(model)
        if value is None:
            continue
        name = _adapter_name(model)
        current = chosen.get(name)
        if current is None or (value > current[0] if higher_better else value < current[0]):
            chosen[name] = (value, model)
    ranked = sorted(chosen.values(), key=lambda item: item[0], reverse=higher_better)
    places = [_place(model, value) for value, model in ranked[:limit]]
    while len(places) < limit:
        places.append("no further distinct adapter")
    return places


def _aggregate_row(title: str, name: str, models: list[dict]) -> list[str]:
    utilities = [model["persona_utility"] for model in models if model.get("persona_utility") is not None]
    eligible = [float(model["n_eligible"]) for model in models]
    utility = _stats(utilities)
    eligibility = _stats(eligible)
    return [
        _adapter_label(name),
        title,
        str(utility["n"]),
        _cell(utility["mean"]),
        _cell(utility["median"]),
        _cell(utility["min"]),
        _cell(utility["max"]),
        _cell(utility["spread"]) if utility["n"] > 1 else "n/a",
        f"{eligibility['mean']:.1f}/{models[0]['n']}",
        _eligible_span(models),
    ]


def _claim_totals(path: Path) -> dict[tuple[str, str], dict]:
    totals: dict[tuple[str, str], dict] = {}
    if not path.exists():
        return totals
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            grounding = row.get("grounding") or {}
            if not grounding.get("applicable"):
                continue
            key = (row.get("source_experiment"), row.get("model"))
            bucket = totals.setdefault(key, {"claims": 0, **{label: 0 for label in CLAIM_LABELS}})
            counts = grounding.get("counts") or {}
            for label in CLAIM_LABELS:
                bucket[label] += int(counts.get(label) or 0)
            bucket["claims"] += int(grounding.get("n_claims") or 0)
    return totals


def _find_run(models: list[dict], experiment: str, name: str) -> dict | None:
    for model in models:
        if model["source_experiment"] == experiment and _adapter_name(model) == name:
            return model
    return None


def leaderboards(summary: dict, stance_low: float, stance_high: float) -> tuple[list[list[str]], list[list[str]]]:
    models = summary["models"]
    off = [model for model in models if _retrieval_off(model)]
    on = [model for model in models if not _retrieval_off(model)]
    guide = [
        ["Persona Utility", "Weighted blend of voice, stance, writing quality, and task after the gates", "Higher", "0 to 1", "Filled. A gap of a few thousandths is not a decided win."],
        ["Eligible drafts", "Drafts that pass completion, copying, format, and applicable grounding", "Higher", "0 to 120", "Shown with the rate. More eligible drafts is a reliability difference."],
        ["Voice", "How far gate-passing blog drafts sit from held-out Lemkin blog posts", "Lower", "0 is the profile mean", "Blog only. LinkedIn, X, and talk are not in this mean."],
        ["Stance", "Share of advice claims consistent with held-out Lemkin evidence", "Higher", "0 to 1", f"Calibrated, but saturated ({stance_low:.2f} to {stance_high:.2f}). It does not strongly separate adapters."],
        ["Writing quality", "Coherence, fluency, progression, relevance, specificity, usefulness", "Higher", "1 to 5", "Blinded Qwen judge. Calibration passed."],
        ["Task", "Topic coverage plus the requested length band, on gate-passing drafts", "Higher", "0 to 1", "Scored."],
        ["Completion", "Share of drafts that end as a finished sentence", "Higher", "0 to 1", "Early stop of a finished sentence still passes."],
        ["Copying avoidance", "Share of drafts with no 12-word training span outside the prompt", "Higher", "0 to 1", "Almost every run is above 0.99."],
        ["RAG grounding", "Share of verifiable claims supported by saved retrieval passages", "Higher", "0 to 1", "Only retrieval-on runs. A rate near 1.0 still needs the claim counts."],
        ["Brief overlap", "Old gold rubric against the constructed assignment brief", "Higher", "0 to 1", "Diagnostic. A high score is not Lemkin voice."],
        ["BLEU", "Smoothed word-overlap with that same brief", "Higher", "0 to 1", "Diagnostic. These runs sit near 0.01."],
        ["ROUGE-L", "Longest common subsequence overlap with that brief", "Higher", "0 to 1", "Diagnostic."],
        ["BERTScore", "Embedding similarity to a reference", "Higher", "About 0 to 1", "Not computed in this rescore."],
        ["Latency", "Seconds per answer", "Lower", "0 and up", "Not stored on these traces."],
    ]
    def stance_or_distance(model: dict):
        perspective = model["perspective_fidelity"]
        if perspective.get("mean_stance_consistency") is not None:
            return perspective["mean_stance_consistency"]
        return None

    def grounding_value(model: dict):
        grounding = model["grounding"]
        if not grounding.get("applicable"):
            return None
        if grounding.get("mean_claim_support_rate") is not None:
            return grounding["mean_claim_support_rate"]
        return model["gates"].get("grounding_pass_rate")

    rows = [
        ["Persona Utility", "Higher", *_top_distinct(off, lambda model: model.get("persona_utility"), True)],
        ["Eligible rate", "Higher", *_top_distinct(off, lambda model: model["n_eligible"] / model["n"] if model["n"] else None, True)],
        ["Voice", "Lower", *_top_distinct(off, lambda model: model["persona_style"]["mean_distance_eligible"], False)],
        ["Stance", "Higher, saturated", *_top_distinct(off, stance_or_distance, True)],
        ["Writing quality", "Higher", *_top_distinct(off, lambda model: model["writing_quality"].get("mean_eligible"), True)],
        ["Task adherence", "Higher", *_top_distinct(off, lambda model: model["task_adherence"]["mean_eligible"], True)],
        ["Completion", "Higher", *_top_distinct(off, lambda model: model["gates"]["completion_pass_rate"], True)],
        ["Copying avoidance", "Higher", *_top_distinct(off, lambda model: model["gates"]["copying_pass_rate"], True)],
        ["RAG grounding", "Higher, retrieval on only", *_top_distinct(on, grounding_value, True)],
        ["Brief overlap", "Higher, diagnostic", *_top_distinct(models, lambda model: model["diagnostics"]["brief_overlap"]["gold_rubric_overall"], True)],
        ["BLEU", "Higher, diagnostic", *_top_distinct(models, lambda model: model["diagnostics"]["reference_similarity"]["bleu"], True)],
        ["ROUGE-L", "Higher, diagnostic", *_top_distinct(models, lambda model: model["diagnostics"]["reference_similarity"]["rouge_l"], True)],
    ]
    return guide, rows


def unified_tables() -> dict:
    summary_path = ROOT / "experiments" / "EXP-20261004-008-writing-evidence" / "metrics" / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    claims = _claim_totals(summary_path.parent / "rows.jsonl")
    prior = json.loads(
        (ROOT / "experiments" / "EXP-20261004-007-unified-rescore" / "metrics" / "summary.json").read_text(encoding="utf-8")
    )
    models = summary["models"]
    stances = [
        model["perspective_fidelity"]["mean_stance_consistency"]
        for model in models
        if model["perspective_fidelity"].get("mean_stance_consistency") is not None
    ]
    stance_low = min(stances) if stances else 0.0
    stance_high = max(stances) if stances else 0.0
    primary = []
    diagnostic = []
    grounding_rows = []
    for model in models:
        name = _adapter_label(_adapter_name(model))
        run = _run_label(model)
        rate = model["n_eligible"] / model["n"] if model["n"] else None
        primary.append([
            name,
            run,
            _cell(model["persona_utility"]),
            f"{model['n_eligible']}/{model['n']}",
            _cell(rate),
            _cell(model["persona_style"]["mean_distance_eligible"]),
            _cell(model["perspective_fidelity"].get("mean_stance_consistency")),
            _cell(model["writing_quality"]["mean_eligible"]),
            _cell(model["task_adherence"]["mean_eligible"]),
            _cell(model["gates"]["completion_pass_rate"]),
            _cell(model["gates"]["copying_pass_rate"]),
        ])
        diagnostic.append([
            name,
            run,
            _cell(model["diagnostics"]["brief_overlap"]["gold_rubric_overall"]),
            _cell(model["diagnostics"]["reference_similarity"]["bertscore"]),
            _cell(model["diagnostics"]["reference_similarity"]["bleu"]),
            _cell(model["diagnostics"]["reference_similarity"]["rouge_l"]),
            _cell(model["diagnostics"]["latency_seconds"]),
        ])
        bucket = claims.get((model["source_experiment"], _adapter_name(model)))
        if not model["grounding"].get("applicable") or bucket is None:
            grounding_rows.append([name, run, "n/a", "n/a", "n/a", "n/a", "n/a", "n/a", "n/a", "n/a"])
            continue
        verifiable = bucket["supported"] + bucket["unsupported"] + bucket["contradicted"]
        grounding_rows.append([
            name,
            run,
            _cell(model["grounding"].get("mean_claim_support_rate")),
            str(bucket["claims"]),
            str(verifiable),
            str(bucket["supported"]),
            str(bucket["unsupported"]),
            str(bucket["contradicted"]),
            str(bucket["not_verifiable"]),
            str(bucket["not_applicable"]),
        ])
    seed_groups: dict[str, list[dict]] = {}
    other_off: dict[str, list[dict]] = {}
    retrieval_on: dict[str, list[dict]] = {}
    for model in models:
        name = _adapter_name(model)
        if not _retrieval_off(model):
            retrieval_on.setdefault(name, []).append(model)
        elif model["source_experiment"] in SEED_EXPERIMENTS:
            seed_groups.setdefault(name, []).append(model)
        else:
            other_off.setdefault(name, []).append(model)
    adapters = []
    for name in sorted(seed_groups, key=_adapter_label):
        adapters.append(_aggregate_row("Oct 4 retrieval-off seeds", name, seed_groups[name]))
    for name in sorted(other_off, key=_adapter_label):
        adapters.append(_aggregate_row("Retrieval-off run", name, other_off[name]))
    for name in sorted(retrieval_on, key=_adapter_label):
        adapters.append(_aggregate_row("Retrieval on, not mixed into the seed mean", name, retrieval_on[name]))
    provisional = []
    ranked_names = set(seed_groups)
    ranking_pool = []
    for name, group in seed_groups.items():
        utilities = [model["persona_utility"] for model in group if model.get("persona_utility") is not None]
        ranking_pool.append((name, "Oct 4 retrieval-off seeds", group, _stats(utilities)))
    for name, group in other_off.items():
        if name in ranked_names:
            continue
        utilities = [model["persona_utility"] for model in group if model.get("persona_utility") is not None]
        ranking_pool.append((name, "One retrieval-off run", group, _stats(utilities)))
    ranking_pool.sort(key=lambda item: item[3]["median"], reverse=True)
    for index, (name, basis, group, stats) in enumerate(ranking_pool, start=1):
        provisional.append([
            str(index),
            _adapter_label(name),
            basis,
            _cell(stats["median"]),
            _cell(stats["mean"]),
            _cell(stats["spread"]) if stats["n"] > 1 else "n/a",
            f"{_stats([float(model['n_eligible']) for model in group])['mean']:.1f}/{group[0]['n']}",
            _eligible_span(group),
        ])
    repaired = _find_run(models, "EXP-20261001-010-repaired-and-baselines", "repaired")
    relabel = _find_run(models, "EXP-20261004-005-repaired-vs-relabel-seed43", "relabel")
    gap = abs(repaired["persona_utility"] - relabel["persona_utility"])
    intro = (
        f"Repaired on the Oct 1 baselines run has Persona Utility {repaired['persona_utility']:.3f} "
        f"on {repaired['n_eligible']}/{repaired['n']} eligible drafts. "
        f"Relabel seed 43 has Persona Utility {relabel['persona_utility']:.3f} "
        f"on {relabel['n_eligible']}/{relabel['n']} eligible drafts. "
        f"The utility gap is {gap:.3f}. This rescore has no uncertainty estimate for that gap, "
        "so it is not treated as a meaningful difference. Eligibility is the larger practical difference. "
        f"Stance passed calibration and ranges from {stance_low:.3f} to {stance_high:.3f}, "
        "so it is saturated and does not strongly separate adapters. "
        "The provisional table orders distinct adapters by median retrieval-off utility. It is not a final winner."
    )
    calibration = []
    for medium, info in prior["calibration"].items():
        style = info["style_calibration"]
        perspective = info["perspective_calibration"]
        calibration.append([
            medium,
            info["style_role"],
            _cell(style.get("real_mean")),
            _cell(style.get("generic_mean")),
            _cell(style.get("base_mean")),
            info["perspective_role"],
            perspective.get("reason") or "",
        ])
    guide, tops = leaderboards(summary, stance_low, stance_high)
    return {
        "primary": primary,
        "diagnostic": diagnostic,
        "calibration": calibration,
        "guide": guide,
        "tops": tops,
        "provisional": provisional,
        "adapters": adapters,
        "grounding": grounding_rows,
        "intro": intro,
    }


def main() -> None:
    by_name = {}
    for name, _label, exp in MODELS:
        exp_path = ROOT / exp
        by_name[name] = load(exp_path / "raw" / f"{name}.jsonl")
    seen = []
    for row in by_name["repaired"]:
        if row["id"] not in seen:
            seen.append(row["id"])
    items = []
    for gold_id in seen:
        base = next(row for row in by_name["repaired"] if row["id"] == gold_id)
        mediums = {}
        for medium in ("blog", "linkedin", "x", "talk"):
            answers = {}
            words = {}
            prompt = ""
            for name, _label, _exp in MODELS:
                match = next(
                    row for row in by_name[name]
                    if row["id"] == gold_id and row["medium"] == medium
                )
                answers[name] = match["answer"]
                words[name] = len(match["answer"].split())
                prompt = match["prompt"]
            mediums[medium] = {"prompt": prompt, "answers": answers, "words": words}
        items.append({
            "id": gold_id,
            "topic": base["topic"],
            "audience": base.get("audience") or "",
            "goal": base.get("goal") or "",
            "expected": base.get("gold_reference") or "",
            "facts": base.get("expected_facts") or [],
            "mediums": mediums,
        })
    tables = unified_tables()
    primary = tables["primary"]
    diagnostic = tables["diagnostic"]
    calibration = tables["calibration"]
    guide = tables["guide"]
    tops = tables["tops"]
    provisional = tables["provisional"]
    adapters = tables["adapters"]
    grounding = tables["grounding"]
    intro = tables["intro"]
    model_ts = json.dumps(
        [{"id": name, "label": label} for name, label, _exp in MODELS],
        indent=2,
    )
    header = '''import { Button, H1, H2, H3, Row, Select, Stack, Table, Text, useCanvasState } from "cursor/canvas";

type MediumId = "blog" | "linkedin" | "x" | "talk";

const MODELS: { id: string; label: string }[] = ''' + model_ts + ''';

const MEDIUMS: { id: MediumId; label: string }[] = [
  { id: "blog", label: "Blog" },
  { id: "linkedin", label: "LinkedIn" },
  { id: "x", label: "X" },
  { id: "talk", label: "Talk" },
];

const PRIMARY: string[][] = '''
    middle = ''';

const DIAGNOSTIC: string[][] = '''
    diagnostic_end = ''';

const CALIBRATION: string[][] = '''
    guide_mid = ''';

const GUIDE: string[][] = '''
    tops_mid = ''';

const TOPS: string[][] = '''
    provisional_mid = ''';

const PROVISIONAL: string[][] = '''
    adapters_mid = ''';

const ADAPTERS: string[][] = '''
    grounding_mid = ''';

const GROUNDING: string[][] = '''
    footer = r'''

export default function ExperimentGlance() {
  const [goldId, setGoldId] = useCanvasState("goldId", ITEMS[0].id);
  const [medium, setMedium] = useCanvasState<MediumId>("medium", "blog");
  const index = Math.max(0, ITEMS.findIndex((item) => item.id === goldId));
  const item = ITEMS[index];
  const chosen = item.mediums[medium];

  return (
    <Stack gap={16}>
      <Stack gap={6}>
        <H1>Persona evaluation</H1>
        <Text tone="secondary">__INTRO__</Text>
      </Stack>
      <H2>What each number means</H2>
      <Table
        headers={["Metric", "Meaning", "Better", "Range", "This rescore"]}
        rows={GUIDE}
        striped
      />
      <H2>Overall, provisional ranking</H2>
      <Text tone="secondary">
        One row per adapter. Retrieval-off seeds are summarized by the median, not by the
        single highest run. Oct 1 repaired is the highest single utility and is called out
        above; it is not a separate winning adapter. Close medians are not a decided order.
      </Text>
      <Table
        headers={["Scan", "Adapter", "Basis", "Utility median", "Utility mean", "Utility spread", "Eligible mean", "Eligible range"]}
        rows={PROVISIONAL}
        columnAlign={["right", "left", "left", "right", "right", "right", "right", "right"]}
        striped
      />
      <H2>Top three distinct adapters</H2>
      <Text tone="secondary">
        Each place is one adapter, the run that scored best on that metric, the score, and
        how many of 120 drafts passed the gates. Seeds of the same adapter cannot fill all
        three places. Utility places are the highest runs, not a significance test.
        Stance places are saturated.
      </Text>
      <Table
        headers={["Metric", "Better", "1st", "2nd", "3rd"]}
        rows={TOPS}
        striped
      />
      <H2>Persona evaluation, every run</H2>
      <Text tone="secondary">
        Persona Utility, eligibility, voice, stance, writing quality, task, completion, and
        copying. Arrows in the column names are the preferred direction. RAG grounding is
        not in this table.
      </Text>
      <Table
        headers={["Model", "Run", "Persona Utility ↑", "Eligible", "Eligible rate ↑", "Voice ↓", "Stance ↑", "Writing quality ↑", "Task adherence ↑", "Completion ↑", "Copying avoidance ↑"]}
        rows={PRIMARY}
        columnAlign={["left", "left", "right", "right", "right", "right", "right", "right", "right", "right", "right"]}
        striped
      />
      <Text tone="tertiary" size="small">
        Source: EXP-20261004-008 rescore of saved answers. Voice is a z-score on
        gate-passing blog drafts; lower is closer to held-out Lemkin. Stance is 0 to 1
        and saturated. Writing quality is 1 to 5. No answers were regenerated.
      </Text>
      <H2>Adapter summary</H2>
      <Text tone="secondary">
        Repeated Oct 4 retrieval-off seeds are one row, with mean, median, and spread.
        The Oct 1 run and the retrieval-on run stay on their own rows so those conditions
        are not averaged into the seed mean.
      </Text>
      <Table
        headers={["Adapter", "Group", "n", "Utility mean", "Utility median", "Utility min", "Utility max", "Spread", "Eligible mean", "Eligible range"]}
        rows={ADAPTERS}
        columnAlign={["left", "left", "right", "right", "right", "right", "right", "right", "right", "right"]}
        striped
      />
      <H2>Claim-level RAG grounding</H2>
      <Text tone="secondary">
        Retrieval-off runs are n/a. Claim support is the mean of per-draft rates among
        claims the judge treated as verifiable. Verifiable is supported plus unsupported
        plus contradicted. Not verifiable and not applicable are outside that rate, so a
        value near 1.0 can still leave many claims unchecked.
      </Text>
      <Table
        headers={["Model", "Run", "Claim support ↑", "Claims", "Verifiable", "Supported", "Unsupported", "Contradicted", "Not verifiable", "Not applicable"]}
        rows={GROUNDING}
        columnAlign={["left", "left", "right", "right", "right", "right", "right", "right", "right", "right"]}
        striped
      />
      <H2>Reference overlap</H2>
      <Text tone="secondary">
        Brief overlap is the old gold rubric. It is agreement with the constructed
        assignment brief, not proof of Lemkin voice. BLEU and ROUGE-L use that same brief.
      </Text>
      <Table
        headers={["Model", "Run", "Brief overlap", "BERTScore", "BLEU", "ROUGE-L", "Latency"]}
        rows={DIAGNOSTIC}
        columnAlign={["left", "left", "right", "right", "right", "right", "right"]}
        striped
      />
      <H2>Calibration</H2>
      <Table
        headers={["Medium", "Style role", "Real Lemkin", "Generic prose", "Base model", "Framing role", "Framing check"]}
        rows={CALIBRATION}
        striped
      />
      <Text tone="tertiary" size="small">
        Style distances are from EXP-20261004-007 and still decide which mediums enter the
        voice mean. The framing-role column is that older cue-frequency check. Stance in
        the tables above is the EXP-008 evidence check, and it is saturated.
      </Text>
      <Stack gap={6}>
        <H2>Prompt, expected, and actual</H2>
        <Text tone="secondary">
          Same 30 topics and four mediums. Llama 3.1 is the untrained instruct checkpoint.
          Lemkin clone v5 and v6 share the same Ollama weights (id 06b812f9374d), scored
          as two samples. Expected facts are shown here and were not put in the prompt.
        </Text>
      </Stack>
      <H2>Parameters</H2>
      <Table
        headers={["Setting", "Value"]}
        rows={[
          ["Blog prompt", "Write a blog post in the style of Jason Lemkin about: {topic}"],
          ["LinkedIn prompt", "Write a LinkedIn post in the style of Jason Lemkin about: {topic}"],
          ["X prompt", "Write an X post in the style of Jason Lemkin about: {topic}"],
          ["Talk prompt", "Write a talk in the style of Jason Lemkin about: {topic}"],
          ["Context", "2048"],
          ["New tokens", "Blog 768, LinkedIn 320, X 160, Talk 768"],
          ["Temperature", "0.7"],
          ["Top-p", "0.9"],
          ["Repetition penalty", "1.15"],
          ["Sampling", "on"],
          ["Expected facts in the prompt", "no"],
        ]}
        striped
      />
      <H2>Prompt sent</H2>
      <Text weight="semibold">{chosen.prompt}</Text>
      <Text tone="secondary" size="small">
        {item.id}. Audience: {item.audience}. Goal: {item.goal}.
      </Text>
      <Row gap={8} align="center" wrap>
        {MEDIUMS.map((entry) => (
          <Button
            key={entry.id}
            variant={medium === entry.id ? "primary" : "secondary"}
            onClick={() => setMedium(entry.id)}
          >
            {entry.label}
          </Button>
        ))}
      </Row>
      <Row gap={8} align="center" wrap>
        <Button variant="secondary" disabled={index === 0} onClick={() => setGoldId(ITEMS[index - 1].id)}>
          Previous
        </Button>
        <Select
          value={item.id}
          onChange={setGoldId}
          options={ITEMS.map((row) => ({
            value: row.id,
            label: `${row.id}  ${row.topic}`,
          }))}
          style={{ minWidth: 480, flex: 1 }}
        />
        <Button
          variant="secondary"
          disabled={index === ITEMS.length - 1}
          onClick={() => setGoldId(ITEMS[index + 1].id)}
        >
          Next
        </Button>
      </Row>
      <H2>Expected</H2>
      <Text style={{ whiteSpace: "pre-wrap" }}>{item.expected}</Text>
      <H3>Expected facts</H3>
      <Stack gap={4}>
        {item.facts.map((fact, factIndex) => (
          <Text key={`${item.id}-fact-${factIndex}`} tone="secondary" size="small">
            {fact}
          </Text>
        ))}
      </Stack>
      <H2>Actual</H2>
      <Stack gap={12}>
        {MODELS.map((model) => (
          <Stack key={model.id} gap={6}>
            <H3>
              {model.label} · {chosen.words[model.id as keyof typeof chosen.words]} words
            </H3>
            <Text style={{ whiteSpace: "pre-wrap" }}>{chosen.answers[model.id as keyof typeof chosen.answers]}</Text>
          </Stack>
        ))}
      </Stack>
    </Stack>
  );
}
'''
    body = (
        header
        + json.dumps(primary, ensure_ascii=False, indent=2)
        + middle
        + json.dumps(diagnostic, ensure_ascii=False, indent=2)
        + diagnostic_end
        + json.dumps(calibration, ensure_ascii=False, indent=2)
        + guide_mid
        + json.dumps(guide, ensure_ascii=False, indent=2)
        + tops_mid
        + json.dumps(tops, ensure_ascii=False, indent=2)
        + provisional_mid
        + json.dumps(provisional, ensure_ascii=False, indent=2)
        + adapters_mid
        + json.dumps(adapters, ensure_ascii=False, indent=2)
        + grounding_mid
        + json.dumps(grounding, ensure_ascii=False, indent=2)
        + ";\n\nconst ITEMS = "
        + json.dumps(items, ensure_ascii=False)
        + ";\n"
        + footer.replace("__INTRO__", intro)
    )
    OUT.write_text(body, encoding="utf-8")
    print(f"items={len(items)} bytes={OUT.stat().st_size}")


if __name__ == "__main__":
    main()
