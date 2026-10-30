"""Rebuild the "what did RAG change and what did it cost" reports for PUK-34 from the slim drafts in this folder.

Run it from the repository root:

    python docs/evaluation/examples/data/rag_effect/rebuild_reports.py

It writes one report per before/after pair to `docs/evaluation/examples/rag_effect/`:

- Relabel adapter, RAG off against RAG on, for seeds 42, 43 and 44.
- Base Llama 3.1 (no retrieval) against PersonaRAG (base + train-only retrieval), seed 42 only.

The slim files keep only the fields `eval_tools.rag_effect` reads (token counts, stop reasons, evidence
seconds, retrieval hits and checks), so they are about 20 to 70 KB each instead of 4 MB. `SOURCES.md` says which
full draft file each one was made from. Nothing here calls a model.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE.parent.parent / "rag_effect"

PAIRS = [
    ("relabel_seed42", "relabel_rag_off_seed42", "relabel_rag_on_seed42", "Relabel RAG off (seed 42)", "Relabel RAG on (seed 42)"),
    ("relabel_seed43", "relabel_rag_off_seed43", "relabel_rag_on_seed43", "Relabel RAG off (seed 43)", "Relabel RAG on (seed 43)"),
    ("relabel_seed44", "relabel_rag_off_seed44", "relabel_rag_on_seed44", "Relabel RAG off (seed 44)", "Relabel RAG on (seed 44)"),
    ("base_vs_personarag", "base_llama31", "personarag_seed42", "Base Llama 3.1 (no retrieval)", "PersonaRAG (base + retrieval)"),
]


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for name, a, b, label_a, label_b in PAIRS:
        subprocess.run(
            [sys.executable, "-m", "eval_tools.rag_effect", "report",
             "--a-drafts", str(HERE / f"{a}.slim.jsonl"), "--b-drafts", str(HERE / f"{b}.slim.jsonl"),
             "--label-a", label_a, "--label-b", label_b,
             "--out-md", str(OUT / f"{name}.md"), "--out-json", str(OUT / f"{name}.json")],
            check=True, stdout=subprocess.DEVNULL,
        )
        print("wrote", OUT / f"{name}.md")


if __name__ == "__main__":
    main()
