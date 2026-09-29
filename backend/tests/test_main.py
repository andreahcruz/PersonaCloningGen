import json

import pytest
from fastapi.testclient import TestClient

import main as backend_main
from main import app

client = TestClient(app)


def test_health():
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_list_formats():
    resp = client.get("/formats")
    assert resp.status_code == 200
    formats = resp.json()
    assert "linkedin_post" in formats
    assert "blog_draft" in formats


def test_generate_rejects_unknown_format(monkeypatch, tmp_path):
    persona = tmp_path / "persona_profile.json"
    persona.write_text(json.dumps({"meta": {"source": "Test Persona"}}), encoding="utf-8")
    monkeypatch.setattr(backend_main, "PERSONA_PATH", persona)

    resp = client.post(
        "/generate",
        json={
            "format": "not_a_real_format",
            "topic": "t",
            "audience": "a",
            "goal": "g",
        },
    )
    assert resp.status_code == 400


def test_generate_missing_persona_returns_500(monkeypatch, tmp_path):
    monkeypatch.setattr(backend_main, "PERSONA_PATH", tmp_path / "missing.json")

    resp = client.post(
        "/generate",
        json={
            "format": "linkedin_post",
            "topic": "t",
            "audience": "a",
            "goal": "g",
        },
    )
    assert resp.status_code == 500


def test_generate_happy_path(monkeypatch, tmp_path):
    persona = tmp_path / "persona_profile.json"
    persona.write_text(json.dumps({"meta": {"source": "Test Persona"}}), encoding="utf-8")
    monkeypatch.setattr(backend_main, "PERSONA_PATH", persona)

    monkeypatch.setattr(backend_main, "ollama_embed", lambda text, base, model: [0.0, 0.1])
    monkeypatch.setattr(
        backend_main,
        "retrieve",
        lambda index_path, embed, k, embed_model: [
            {"chunk_text": "Some reference excerpt.", "doc_id": "doc1"}
        ],
    )
    monkeypatch.setattr(
        backend_main, "ollama_generate", lambda prompt, base, model: "Generated draft text."
    )

    resp = client.post(
        "/generate",
        json={
            "format": "linkedin_post",
            "topic": "Hiring a VP Sales",
            "audience": "B2B founders",
            "goal": "share a takeaway",
            "cta": "none",
            "k": 5,
        },
    )
    assert resp.status_code == 200
    assert resp.json() == {"draft": "Generated draft text."}


def test_generate_upstream_failure_returns_502(monkeypatch, tmp_path):
    persona = tmp_path / "persona_profile.json"
    persona.write_text(json.dumps({"meta": {"source": "Test Persona"}}), encoding="utf-8")
    monkeypatch.setattr(backend_main, "PERSONA_PATH", persona)

    def _boom(*args, **kwargs):
        raise ConnectionError("Ollama unreachable")

    monkeypatch.setattr(backend_main, "ollama_embed", _boom)

    resp = client.post(
        "/generate",
        json={
            "format": "linkedin_post",
            "topic": "t",
            "audience": "a",
            "goal": "g",
        },
    )
    assert resp.status_code == 502
