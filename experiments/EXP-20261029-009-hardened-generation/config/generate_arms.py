"""Four-arm generation after the evidence-path fixes.

Retrieval is frozen before the Relabel adapter loads. Existing raw files
are left in place.
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

from host_finetune.compare_adapters import (
    GENERATION_SEED,
    apply_active_rag,
    build_runtime_retriever,
    generate,
    load_frozen_retriever,
    parse_args,
    write_frozen_retrieval,
)
from host_finetune.graph_ready import load_jsonl
from host_finetune.lightrag_backend import query_documents_sync
from host_finetune.pilot_retrieval import DEFAULT_DOCS, DEFAULT_IDS, _embed, dense_hits, select_pilot_documents

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parents[1]
RAW = OUT / "raw"
ADAPTER = ROOT / "host_finetune" / "output" / "lemkin_lora_relabel"
WORK = ROOT / "artifacts" / "lightrag" / "model_f" / "storage_pilot160"
BASE = "http://localhost:11434"

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


def _stop_ollama() -> None:
    for name in ("llama3.1:latest", "nomic-embed-text:latest", "qwen2.5:14b"):
        subprocess.run(["ollama", "stop", name], check=False)


def freeze() -> None:
    RAW.mkdir(parents=True, exist_ok=True)
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
    apply_active_rag(args)
    chroma = build_runtime_retriever(args)
    write_frozen_retrieval(RAW / "frozen_chroma_v2.jsonl", [row["topic"] for row in TOPICS], chroma)

    documents = select_pilot_documents(load_jsonl(DEFAULT_DOCS), DEFAULT_IDS)
    texts = [row.get("source_text") or row.get("text") or "" for row in documents]
    matrix = _embed(texts, BASE, "nomic-embed-text")
    pilot_rows = []
    for row in TOPICS:
        payload = dense_hits(row["topic"], documents, matrix, BASE, "nomic-embed-text", 4)
        payload["query"] = row["topic"]
        pilot_rows.append(payload)
        print(f"froze pilot {row['id']} hits={len(payload['hits'])}", flush=True)
    (RAW / "frozen_pilot_dense.jsonl").write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in pilot_rows) + "\n",
        encoding="utf-8",
    )
    for mode in ("naive", "mix"):
        rows = []
        for topic in TOPICS:
            payload = query_documents_sync(
                topic["topic"],
                WORK,
                mode,
                base_url=BASE,
                evidence_budget=4,
                max_per_source=1,
                num_ctx=16384,
            )
            payload["query"] = topic["topic"]
            rows.append(payload)
            print(f"froze {mode} {topic['id']} hits={[hit.get('id') for hit in payload.get('hits') or []]}", flush=True)
        (RAW / f"frozen_{mode}.jsonl").write_text(
            "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
            encoding="utf-8",
        )


def generate_arm(name: str, frozen_name: str) -> None:
    destination = RAW / f"generation_{name}.json"
    if destination.is_file():
        print(f"keep existing {destination}", flush=True)
        return
    _stop_ollama()
    started = time.perf_counter()
    traces, load_report = generate(
        ADAPTER,
        TOPICS,
        retriever=load_frozen_retriever(RAW / frozen_name),
        seed_base=GENERATION_SEED,
    )
    destination.write_text(
        json.dumps(
            {
                "arm": name,
                "load_report": load_report,
                "wall_seconds": round(time.perf_counter() - started, 3),
                "traces": traces,
            },
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )
    print(f"wrote {destination} traces={len(traces)}", flush=True)


def main() -> None:
    freeze()
    generate_arm("chroma_v2", "frozen_chroma_v2.jsonl")
    generate_arm("pilot_dense", "frozen_pilot_dense.jsonl")
    generate_arm("naive", "frozen_naive.jsonl")
    generate_arm("mix", "frozen_mix.jsonl")


if __name__ == "__main__":
    main()
