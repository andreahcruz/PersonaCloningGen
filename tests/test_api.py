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
    assert body["method"] == "personarag"
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
    monkeypatch.setattr(
        main.requests,
        "get",
        lambda *_args, **_kwargs: type(
            "Response", (), {"json": lambda self: {"models": [{"name": "lemkin-qlora:latest"}]}}
        )(),
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
            },
            {"name": "qlora model", "status": "ok", "detail": "lemkin-qlora"},
        ],
    }


def test_qlora_generation_bypasses_retrieval(monkeypatch):
    calls = []

    def fake_chat(prompt, _base, model, **kwargs):
        calls.append((prompt, model, kwargs))
        return f"{model}: {prompt}"

    monkeypatch.setattr(main, "load_format_spec", lambda *_: {"output_type": "markdown"})
    monkeypatch.setattr(main, "ollama_chat", fake_chat)
    monkeypatch.setattr(
        main,
        "retrieve",
        lambda *_: pytest.fail("direct QLoRA must not retrieve Chroma source chunks"),
    )
    monkeypatch.setattr(
        main,
        "ollama_embed",
        lambda *_: pytest.fail("direct QLoRA must not embed the content brief"),
    )

    response = TestClient(main.app).post(
        "/generate",
        json={
            "format": "linkedin_post",
            "topic": "Hiring your first salesperson",
            "audience": "B2B SaaS founders",
            "goal": "Share one takeaway",
            "cta": "none",
            "k": 4,
            "method": "qlora",
        },
    )

    assert response.status_code == 200
    body = response.json()
    sentence = (
        "About 80 words. The first line is a claim, not a headline. No hashtags. "
        "Write a LinkedIn post in the style of Jason Lemkin about Hiring your first salesperson."
    )
    assert body["method"] == "qlora"
    assert body["model"] == "lemkin-qlora"
    assert body["sources"] == []
    assert body["draft"] == f"lemkin-qlora: {sentence}"
    assert calls == [
        (
            sentence,
            "lemkin-qlora",
            {
                "options": {
                    "temperature": 0,
                    "seed": 42,
                    "top_p": 0.9,
                    "num_ctx": 4096,
                    "stop": ["<|eot_id|>"],
                    "repeat_penalty": 1.1,
                    "num_predict": 320,
                },
                "keep_alive": 0,
            },
        )
    ]


@pytest.mark.parametrize(
    ("format_name", "sentence", "max_tokens", "repeat_penalty"),
    [
        (
            "linkedin_post",
            "About 80 words. The first line is a claim, not a headline. No hashtags. "
            "Write a LinkedIn post in the style of Jason Lemkin about Hiring your first salesperson.",
            320,
            1.1,
        ),
        (
            "blog_draft",
            "Write a blog post in the style of Jason Lemkin about Hiring your first salesperson. "
            "Write the post itself in about 250 words, with no title line.",
            500,
            1.1,
        ),
        (
            "x_thread",
            "Write one X post in the style of Jason Lemkin. Exactly 3 sentences, under 60 words, "
            "no headline, and no hashtags. Topic: Hiring your first salesperson",
            120,
            1.1,
        ),
        (
            "youtube_script",
            "Write a spoken talk in the style of Jason Lemkin about Hiring your first salesperson. "
            "Use short spoken sentences to founders, about 160 words, with no title and no headings.",
            400,
            1.15,
        ),
    ],
)
def test_direct_qlora_prompt_matches_kevin_medium_grid(format_name, sentence, max_tokens, repeat_penalty):
    assert main.direct_qlora_prompt(format_name, "Hiring your first salesperson") == sentence
    assert main.DIRECT_QLORA_MAX_TOKENS[format_name] == max_tokens
    assert main.direct_qlora_options(format_name)["repeat_penalty"] == repeat_penalty
    assert main.direct_qlora_options(format_name)["num_predict"] == max_tokens
