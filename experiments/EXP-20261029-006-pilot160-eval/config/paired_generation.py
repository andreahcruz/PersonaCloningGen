"""Small paired generation from frozen pilot retrieval.

Uses the existing Relabel generator. Does not open Chroma, does not score
against gold references, and does not send review ids to a retriever.
Retrieval payloads are collected before the generator is loaded.
"""

from __future__ import annotations

import json
from pathlib import Path

from host_finetune.compare_adapters import GENERATION_SEED, generate
from host_finetune.graph_ready import load_jsonl
from host_finetune.lightrag_backend import query_documents_sync
from host_finetune.pilot_retrieval import DEFAULT_DOCS, DEFAULT_IDS, _embed, dense_hits, select_pilot_documents

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parents[1] / "raw"
WORK = ROOT / "artifacts" / "lightrag" / "model_f" / "storage_pilot160"
ADAPTER = ROOT / "host_finetune" / "output" / "lemkin_lora_relabel"
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


def freeze(documents, matrix) -> dict:
    payloads = {}
    for topic in TOPICS:
        dense = dense_hits(topic["topic"], documents, matrix, BASE, "nomic-embed-text", 4)
        naive = query_documents_sync(
            topic["topic"], WORK, "naive", base_url=BASE,
            evidence_budget=4, max_per_source=1, num_ctx=16384,
        )
        mix = query_documents_sync(
            topic["topic"], WORK, "mix", base_url=BASE,
            evidence_budget=4, max_per_source=1, num_ctx=16384,
        )
        payloads[topic["id"]] = {"dense": dense, "naive": naive, "mix": mix}
        print(
            topic["id"],
            {name: [hit.get("id") for hit in payload.get("hits") or []] for name, payload in payloads[topic["id"]].items()},
            flush=True,
        )
    path = OUT / "frozen_generation_retrieval.json"
    path.write_text(json.dumps(payloads, indent=2), encoding="utf-8")
    return payloads


def alias_parent_ids(payload: dict) -> dict:
    """Give each hit the parent document id the evidence prompt asks the model to copy."""
    hits = []
    for hit in payload.get("hits") or []:
        item = dict(hit)
        parent = item.get("parent_graph_document_id") or str(item.get("id") or "").split(":")[0]
        item["id"] = parent
        hits.append(item)
    return {**payload, "hits": hits}


def main() -> None:
    frozen_path = OUT / "frozen_generation_retrieval.json"
    payloads = json.loads(frozen_path.read_text(encoding="utf-8"))
    import subprocess
    subprocess.run(["ollama", "stop", "llama3.1:latest"], check=False)
    for mode in ("naive", "mix"):
        def retriever(topic: str, medium: str, mode=mode) -> dict:
            del medium
            for row in TOPICS:
                if row["topic"] == topic:
                    return alias_parent_ids(payloads[row["id"]][mode])
            raise KeyError(topic)

        print(f"generate {mode} parent-id hits", flush=True)
        traces, load_report = generate(
            ADAPTER,
            TOPICS,
            retriever=retriever,
            seed_base=GENERATION_SEED,
        )
        path = OUT / f"generation_{mode}_parent_ids.json"
        path.write_text(
            json.dumps({"load_report": load_report, "traces": traces, "hit_id": "parent_graph_document_id"}, indent=2),
            encoding="utf-8",
        )
        print(f"wrote {path} traces={len(traces)}", flush=True)


if __name__ == "__main__":
    main()
