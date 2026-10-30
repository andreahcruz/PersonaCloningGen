"""Model E generation: frozen full-corpus Chroma v2, then the Relabel adapter.

The three topics match the pilot generation study. Retrieval finishes
before the generator is loaded.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from host_finetune.compare_adapters import (
    GENERATION_SEED,
    apply_active_rag,
    build_runtime_retriever,
    generate,
    parse_args,
    write_frozen_retrieval,
    load_frozen_retriever,
)

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parents[1]
FROZEN = OUT / "raw" / "frozen_chroma_v2.jsonl"
TRACES = OUT / "raw" / "generation_chroma_v2.json"
ADAPTER = ROOT / "host_finetune" / "output" / "lemkin_lora_relabel"

TOPICS = [
    {
        "id": "q_brex_price",
        "topic": "What price did Capital One announce for acquiring Brex?",
    },
    {
        "id": "q_vp_sales_tension",
        "topic": (
            "Does Jason Lemkin say a great VP of Sales should immediately bring "
            "in more sales executives, or that criticizing a VP of Sales who "
            "cannot do better is counterproductive?"
        ),
    },
    {
        "id": "q_personalized_saas_speaker",
        "topic": "Who said that AI means everything in SaaS needs to be a personalized experience?",
    },
]


def main() -> None:
    sys.argv = [
        "compare_adapters",
        "--retrieval",
        "--retrieval-backend",
        "dense",
        "--rag-version",
        "v2",
        "--k",
        "4",
    ]
    args = parse_args()
    selected = apply_active_rag(args)
    retriever = build_runtime_retriever(args)
    write_frozen_retrieval(FROZEN, [row["topic"] for row in TOPICS], retriever)
    subprocess.run(["ollama", "stop", "llama3.1:latest"], check=False)
    subprocess.run(["ollama", "stop", "nomic-embed-text:latest"], check=False)
    frozen = load_frozen_retriever(FROZEN)
    traces, load_report = generate(
        ADAPTER,
        TOPICS,
        retriever=frozen,
        seed_base=GENERATION_SEED,
    )
    TRACES.write_text(
        json.dumps(
            {
                "load_report": load_report,
                "rag": {key: str(value) for key, value in selected.items()},
                "traces": traces,
            },
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )
    print(f"wrote {TRACES} traces={len(traces)}", flush=True)


if __name__ == "__main__":
    main()
