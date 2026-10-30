"""Regenerate the four paired arms under prompt contract factual-reliability-v1.

Frozen retrieval is read from EXP-20261029-009 and is not rewritten.
"""

from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path

from host_finetune.compare_adapters import GENERATION_SEED, generate, load_frozen_retriever
from host_finetune.evidence_capsule import PROMPT_CONTRACT

ROOT = Path(__file__).resolve().parents[3]
RAW = Path(__file__).resolve().parents[1] / "raw"
PRIOR = ROOT / "experiments" / "EXP-20261029-009-hardened-generation" / "raw"
ADAPTER = ROOT / "host_finetune" / "output" / "lemkin_lora_relabel"

TOPICS = [
    {"id": "q_brex_price", "topic": "What price did Capital One announce for acquiring Brex?"},
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

ARMS = {
    "chroma_v2": "frozen_chroma_v2.jsonl",
    "pilot_dense": "frozen_pilot_dense.jsonl",
    "naive": "frozen_naive.jsonl",
    "mix": "frozen_mix.jsonl",
}


def _stop_ollama() -> None:
    for name in ("llama3.1:latest", "nomic-embed-text:latest", "qwen2.5:14b"):
        subprocess.run(["ollama", "stop", name], check=False)


def generate_arm(name: str, frozen_name: str) -> None:
    destination = RAW / f"generation_{name}.json"
    if destination.is_file():
        print(f"keep existing {destination}", flush=True)
        return
    source = PRIOR / frozen_name
    if not source.is_file():
        raise FileNotFoundError(source)
    _stop_ollama()
    started = time.perf_counter()
    traces, load_report = generate(
        ADAPTER,
        TOPICS,
        retriever=load_frozen_retriever(source),
        seed_base=GENERATION_SEED,
    )
    destination.write_text(
        json.dumps(
            {
                "arm": name,
                "prompt_contract": PROMPT_CONTRACT,
                "frozen_retrieval": str(source.relative_to(ROOT)),
                "load_report": load_report,
                "wall_seconds": round(time.perf_counter() - started, 3),
                "seed": GENERATION_SEED,
                "traces": traces,
            },
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )
    print(f"wrote {destination} traces={len(traces)}", flush=True)


def main() -> None:
    RAW.mkdir(parents=True, exist_ok=True)
    for name, frozen_name in ARMS.items():
        generate_arm(name, frozen_name)


if __name__ == "__main__":
    main()
