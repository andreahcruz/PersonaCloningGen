#!/usr/bin/env python3
"""Run one reproducible PersonaRAG or QLoRA request and save deployment evidence.

Example (on the EC2 host after the DAG succeeds):
  python scripts/capture_personarag_smoke.py \
    --output experiments/EXP-20261003-001-ec2-personarag
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


DEMO_REQUEST = {
    "format": "linkedin_post",
    "topic": "Why Series A is the best place to invest in cloud and SaaS today",
    "audience": "SaaS founders and seed investors",
    "goal": "Explain why Series A is the best entry point in cloud and SaaS right now",
    "cta": "What do you think about investing in Series A?",
    "k": 4,
}

EXPECTED = {
    "personarag": {"model": "llama3.1", "sources": True},
    "qlora": {"model": "lemkin-qlora", "sources": False},
}


def get_json(url: str) -> dict:
    with urlopen(url, timeout=30) as response:  # noqa: S310 -- an operator-supplied local URL
        return json.load(response)


def post_json(url: str, payload: dict) -> dict:
    request = Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urlopen(request, timeout=300) as response:  # noqa: S310 -- an operator-supplied local URL
        return json.load(response)


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--api-base", default="http://localhost:8000")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--method", choices=sorted(EXPECTED), default="personarag")
    args = parser.parse_args()

    api_base = args.api_base.rstrip("/")
    try:
        health = get_json(f"{api_base}/health")
        request_payload = {**DEMO_REQUEST, "method": args.method}
        response = post_json(f"{api_base}/generate", request_payload)
    except (HTTPError, URLError, TimeoutError, json.JSONDecodeError) as exc:
        print(f"PersonaRAG smoke test failed: {exc}", file=sys.stderr)
        return 1

    expected = EXPECTED[args.method]
    if (
        not response.get("draft")
        or response.get("model") != expected["model"]
        or bool(response.get("sources")) != expected["sources"]
        or response.get("method") != args.method
    ):
        print(f"{args.method} response did not match the expected model or source behavior.", file=sys.stderr)
        return 1

    now = datetime.now(UTC).isoformat()
    write_json(args.output / "config" / "request.json", request_payload)
    write_json(args.output / "raw" / "health.json", health)
    write_json(args.output / "raw" / "generate.json", response)
    write_json(
        args.output / "metrics" / "summary.json",
        {
            "recorded_at": now,
            "method": args.method,
            "model": response["model"],
            "latency_ms": response.get("latency_ms"),
            "request_id": response.get("request_id"),
            "source_ids": [source.get("id") for source in response["sources"]],
            "source_count": len(response["sources"]),
        },
    )
    print(f"Saved PersonaRAG evidence to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
