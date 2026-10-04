from __future__ import annotations

import pytest

fastapi = pytest.importorskip("fastapi")
pytest.importorskip("starlette")
from fastapi.testclient import TestClient

from backend import main


def test_generate_returns_real_source_metadata(monkeypatch, tmp_path):
    persona = tmp_path / "persona_profile.json"
    persona.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(main, "PERSONA_PATH", persona)
    monkeypatch.setattr(main, "load_format_spec", lambda *_: {"name": "LinkedIn"})
    monkeypatch.setattr(main, "load_persona", lambda *_: {"voice": "direct"})
    monkeypatch.setattr(main, "summarize_persona", lambda *_: "direct")
    monkeypatch.setattr(main, "ollama_embed", lambda *_: [0.1, 0.2])
    monkeypatch.setattr(
        main,
        "retrieve",
        lambda *_: [
            {
                "doc_id": "blog_42",
                "chunk_text": "Series A is the best investment point.",
                "source": "blog",
                "title": "Series A",
                "url": "https://example.test/series-a",
                "date": "2026-01-01",
            },
            {"doc_id": "youtube_7", "chunk_text": "A transcript excerpt.", "source": "youtube_jason"},
        ],
    )
    monkeypatch.setattr(main, "format_context", lambda *_, **__: "context")
    monkeypatch.setattr(main, "build_prompt", lambda *_: "prompt")
    monkeypatch.setattr(main, "ollama_generate", lambda *_: "A grounded draft.")

    response = TestClient(main.app).post(
        "/generate",
        json={
            "format": "linkedin_post",
            "topic": "Series A",
            "audience": "founders",
            "goal": "explain the investment case",
            "k": 2,
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["draft"] == "A grounded draft."
    assert body["model"] == "llama3.1"
    assert body["latency_ms"] >= 0
    assert body["request_id"]
    assert body["sources"] == [
        {
            "id": "blog_42",
            "text": "Series A is the best investment point.",
            "score": None,
            "source_type": "blog",
            "title": "Series A",
            "url": "https://example.test/series-a",
            "date": "2026-01-01",
        },
        {
            "id": "youtube_7",
            "text": "A transcript excerpt.",
            "score": None,
            "source_type": "youtube",
            "title": None,
            "url": None,
            "date": None,
        },
    ]


def test_dependency_health_is_exposed(monkeypatch):
    monkeypatch.setattr(
        main,
        "check_dependencies",
        lambda *_: [("chroma collection", True, "lemkin_content: 15 chunks")],
    )

    response = TestClient(main.app).get("/health/dependencies")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "model": "llama3.1",
        "services": [
            {
                "name": "chroma collection",
                "status": "ok",
                "detail": "lemkin_content: 15 chunks",
            }
        ],
    }
