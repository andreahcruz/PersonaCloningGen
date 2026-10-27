"""Rebuild every number in docs/evaluation/ARCHITECTURE_COMPARISON.md from the saved rows in this folder.

Run it from the repository root:

    python docs/evaluation/examples/data/architecture/rebuild_reports.py

What it does, in plain words:

1. Puts the saved result rows of every run into one file (`combined_rows.jsonl`, written next to the
   reports). The only thing it changes is two labels per row, so that the comparison tools can pair runs up:
   `model` becomes the architecture name (`base`, `personarag`, `rag_off`, `rag_on`) and
   `source_experiment` becomes the seed (`s42`, `s43`, `s44`). The scores are not touched. The original
   names are kept in `orig_model` and `orig_source_experiment`.
2. Runs `eval_tools.seeds` and `eval_tools.compare` on the pairs we care about and writes the reports to
   `docs/evaluation/examples/architecture/`.

Nothing here calls a model or a judge. It only reads files that already exist, so it runs on any laptop in
under a minute. The three Colab runs were scored on an A100 GPU, the seed 42 Relabel runs and the no-RAG
runs are Kevin's saved results (see SOURCES.md for where each file comes from).
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
EXAMPLES = HERE.parent.parent                      # docs/evaluation/examples
OUT = EXAMPLES / "architecture"                    # where the reports go
SEED42_OFF = HERE.parent / "relabel_rag_off_rows.jsonl"   # Kevin's seed 42, RAG off (already in this repo)
SEED42_ON = HERE.parent / "relabel_rag_on_rows.jsonl"     # Kevin's seed 42, RAG on

# (architecture name, seed label) -> file with the 120 scored rows of that run
RUNS = {
    ("rag_off", "s42"): SEED42_OFF,
    ("rag_off", "s43"): HERE / "kevin_relabel_rag_off_seed43_rows.jsonl",
    ("rag_off", "s44"): HERE / "kevin_relabel_rag_off_seed44_rows.jsonl",
    ("rag_on", "s42"): SEED42_ON,
    ("rag_on", "s43"): HERE / "colab_relabel_rag_seed43_rows.jsonl",
    ("rag_on", "s44"): HERE / "colab_relabel_rag_seed44_rows.jsonl",
    ("personarag", "s42"): HERE / "colab_personarag_seed42_rows.jsonl",
    ("base", "s42"): HERE / "kevin_base_llama31_rows.jsonl",
}
# PersonaRAG and the base model were only generated once (seed 42). To compare them with the Relabel runs
# of seeds 43 and 44 we reuse the same run under those seed labels. It is one run, not three.
REUSED = ("personarag", "base")


def load(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def build_combined(out_path: Path) -> int:
    rows: list[dict] = []
    for (arch, seed), path in RUNS.items():
        run = load(path)
        if len(run) != 120:
            sys.exit(f"{path.name}: expected 120 rows, found {len(run)}")
        seeds = [seed] + (["s43", "s44"] if arch in REUSED else [])
        for label in seeds:
            for row in run:
                row = dict(row)
                row["orig_model"] = row["model"]
                row["orig_source_experiment"] = row["source_experiment"]
                row["model"] = arch
                row["source_experiment"] = label
                rows.append(row)
    out_path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    return len(rows)


def tool(*args: str) -> None:
    subprocess.run([sys.executable, "-m", "eval_tools." + args[0], *args[1:]], check=True, stdout=subprocess.DEVNULL)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    combined = OUT / "combined_rows.jsonl"
    print(build_combined(combined), "rows written to", combined)
    rows = str(combined)

    # Does Relabel with RAG on keep its direction against RAG off in every seed?
    for scope in ("all", "blog"):
        tool("seeds", "--rows", rows, "--a-model", "rag_off", "--b-model", "rag_on",
             "--experiments", "s42", "s43", "s44", "--labels", "42", "43", "44", "--scope", scope,
             "--out-md", str(OUT / f"seeds_relabel_rag_off_vs_on_{scope}.md"))
        # Does Relabel + RAG keep its direction against PersonaRAG in every seed?
        tool("seeds", "--rows", rows, "--a-model", "personarag", "--b-model", "rag_on",
             "--experiments", "s42", "s43", "s44", "--labels", "42", "43", "44", "--scope", scope,
             "--out-md", str(OUT / f"seeds_personarag_vs_relabel_rag_{scope}.md"))

    # Full single-seed reports (gates, all mediums, every metric)
    def compare(name, a_model, a_exp, label_a, b_model, b_exp, label_b):
        tool("compare", "--a", rows, "--a-model", a_model, "--a-experiment", a_exp, "--label-a", label_a,
             "--b", rows, "--b-model", b_model, "--b-experiment", b_exp, "--label-b", label_b,
             "--out-md", str(OUT / f"{name}.md"), "--out-json", str(OUT / f"{name}.json"))

    compare("compare_base_vs_personarag", "base", "s42", "Base Llama 3.1 (no RAG)",
            "personarag", "s42", "PersonaRAG (base + retrieval)")
    for seed in ("42", "43", "44"):
        compare(f"compare_personarag_vs_relabel_rag_seed{seed}", "personarag", f"s{seed}", "PersonaRAG (base + retrieval)",
                "rag_on", f"s{seed}", f"Relabel + RAG seed {seed}")
    compare("compare_base_vs_relabel_rag_seed43", "base", "s43", "Base Llama 3.1 (no RAG)",
            "rag_on", "s43", "Relabel + RAG seed 43")
    # Relabel with RAG off against on, one report per seed (these carry the corrected verdicts)
    for seed in ("42", "43", "44"):
        compare(f"compare_relabel_rag_off_vs_on_seed{seed}", "rag_off", f"s{seed}", f"Relabel RAG off seed {seed}",
                "rag_on", f"s{seed}", f"Relabel RAG on seed {seed}")
    print("reports written to", OUT)


if __name__ == "__main__":
    main()
