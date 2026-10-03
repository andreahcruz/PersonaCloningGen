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


def main() -> None:
    by_name = {}
    summaries = {}
    for name, _label, exp in MODELS:
        exp_path = ROOT / exp
        by_name[name] = load(exp_path / "raw" / f"{name}.jsonl")
        summary_path = exp_path / "metrics" / "summary.json"
        if exp not in summaries:
            summaries[exp] = json.loads(summary_path.read_text(encoding="utf-8"))
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
    score_rows = []
    for name, label, exp in MODELS:
        block = summaries[exp][name]
        by = block["by_medium"]
        score_rows.append([
            label,
            f"{block['gold_rubric']['overall']:.4f}",
            f"{block['paired_bleu']:.4f}",
            f"{block['paired_rouge_l']:.4f}",
            f"{by['blog']['gold_rubric']['overall']:.4f}",
            f"{by['linkedin']['gold_rubric']['overall']:.4f}",
            f"{by['x']['gold_rubric']['overall']:.4f}",
            f"{by['talk']['gold_rubric']['overall']:.4f}",
        ])
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

const SCORES: string[][] = '''
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
        <H1>Prompt, expected, and actual</H1>
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
      <Table
        headers={["Model", "Overall", "BLEU", "ROUGE-L", "Blog", "LinkedIn", "X", "Talk"]}
        rows={SCORES}
        columnAlign={["left", "right", "right", "right", "right", "right", "right", "right"]}
      />
      <Text tone="tertiary" size="small">
        Overall, BLEU, and ROUGE-L are means over 120 prompts. Medium columns are gold-rubric overall. The gold rubric includes format compliance, so a base instruct model can score higher without sounding more like Lemkin.
      </Text>
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
        + json.dumps(score_rows, ensure_ascii=False, indent=2)
        + ";\n\nconst ITEMS = "
        + json.dumps(items, ensure_ascii=False)
        + ";\n"
        + footer
    )
    OUT.write_text(body, encoding="utf-8")
    print(f"items={len(items)} bytes={OUT.stat().st_size}")


if __name__ == "__main__":
    main()
