"""Ollama judge client. Llama-family models are refused as judges."""
from __future__ import annotations

import json

LLAMA_FAMILY_MARKERS = ("llama", "lemkin", "alpaca", "vicuna")
# Qwen2.5-14B-Instruct. The local 7B GGUF is a base model and continues the
# draft instead of returning scores, so it is not used as the judge.
DEFAULT_JUDGE_MODEL = "qwen2.5:14b"


def judge_family(model: str) -> str:
    """Return a family name, or raise when the model is in the Llama family."""
    lowered = (model or "").lower()
    if any(marker in lowered for marker in LLAMA_FAMILY_MARKERS):
        raise ValueError(f"judge model {model!r} is in the same family as the Llama candidates")
    for name in ("qwen", "phi", "gemma", "mistral", "deepseek"):
        if name in lowered:
            return name
    return "other"


def extract_json_object(text: str) -> dict | None:
    if not text:
        return None
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        payload = json.loads(text[start:end + 1])
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload, dict) else None


def ollama_json(prompt: str, model: str, base_url: str = "http://127.0.0.1:11434", num_predict: int = 400) -> str:
    """One non-streaming generate call. The prompt must already be blinded."""
    import requests

    judge_family(model)
    response = requests.post(
        f"{base_url.rstrip('/')}/api/chat",
        json={
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "format": "json",
            "keep_alive": "2h",
            "options": {"temperature": 0, "num_predict": num_predict},
        },
        timeout=600,
    )
    response.raise_for_status()
    body = response.json()
    return (body.get("message") or {}).get("content", "").strip()
