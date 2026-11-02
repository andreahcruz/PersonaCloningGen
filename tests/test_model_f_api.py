from __future__ import annotations

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient

from backend import main
from backend.model_f import model_f_status


def test_model_f_is_disabled_by_default(monkeypatch):
    monkeypatch.delenv("MODEL_F_ENABLED", raising=False)
    status = model_f_status()

    assert status.enabled is False
    assert status.ready is False
    assert "disabled" in status.detail.lower()


def test_model_f_rejects_live_index_directory(monkeypatch, tmp_path):
    live_index = tmp_path / "storage_expand_v1"
    live_index.mkdir()
    monkeypatch.setenv("MODEL_F_ENABLED", "true")
    monkeypatch.setenv("LIGHTRAG_WORKDIR", str(live_index))

    status = model_f_status()

    assert status.ready is False
    assert "live" in status.detail.lower()


def test_model_f_rejects_locked_snapshot(monkeypatch, tmp_path):
    snapshot = tmp_path / "verified-snapshot"
    snapshot.mkdir()
    (snapshot / "indexing.lock").touch()
    monkeypatch.setenv("MODEL_F_ENABLED", "true")
    monkeypatch.setenv("LIGHTRAG_WORKDIR", str(snapshot))

    status = model_f_status()

    assert status.ready is False
    assert "locked" in status.detail.lower()


def test_model_f_status_endpoint_is_safe_when_disabled(monkeypatch):
    monkeypatch.setenv("MODEL_F_ENABLED", "false")

    response = TestClient(main.app).get("/model-f/status")

    assert response.status_code == 200
    assert response.json()["enabled"] is False
    assert response.json()["ready"] is False


def test_model_f_generation_returns_controlled_unavailable_response(monkeypatch):
    monkeypatch.setenv("MODEL_F_ENABLED", "false")

    response = TestClient(main.app).post(
        "/generate",
        json={
            "format": "linkedin_post",
            "topic": "Hiring your first salesperson",
            "audience": "B2B SaaS founders",
            "goal": "Share one takeaway",
            "method": "model_f",
            "retrieval_backend": "lightrag",
            "model_f_context": "source",
        },
    )

    assert response.status_code == 503
    assert "Model F unavailable" in response.json()["detail"]


def test_model_f_requires_lightrag_backend():
    response = TestClient(main.app).post(
        "/generate",
        json={
            "format": "linkedin_post",
            "topic": "Hiring your first salesperson",
            "audience": "B2B SaaS founders",
            "goal": "Share one takeaway",
            "method": "model_f",
        },
    )

    assert response.status_code == 400
    assert "requires retrieval_backend" in response.json()["detail"]


def test_existing_methods_cannot_accidentally_request_lightrag():
    response = TestClient(main.app).post(
        "/generate",
        json={
            "format": "linkedin_post",
            "topic": "Hiring your first salesperson",
            "audience": "B2B SaaS founders",
            "goal": "Share one takeaway",
            "method": "personarag",
            "retrieval_backend": "lightrag",
        },
    )

    assert response.status_code == 400
    assert "only through the Model F" in response.json()["detail"]
