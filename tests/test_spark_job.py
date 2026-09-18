import pytest
import requests

pytest.importorskip("pyspark")
pytest.importorskip("bs4")

import clean_and_embed as job  # noqa: E402


class FakeResp:
    def __init__(self, status=200, payload=None, text=""):
        self.status_code = status
        self._payload = payload
        self.text = text
        self.ok = status < 400

    def json(self):
        if self._payload is None:
            raise ValueError("no json")
        return self._payload

    def raise_for_status(self):
        if not self.ok:
            raise requests.exceptions.HTTPError(f"HTTP {self.status_code}")


def _post(monkeypatch, result):
    def fake(url, json=None, timeout=None):
        if isinstance(result, Exception):
            raise result
        return result

    monkeypatch.setattr(job.req, "post", fake)


def test_preflight_passes_when_ollama_embeds(monkeypatch):
    _post(monkeypatch, FakeResp(200, {"embeddings": [[0.1, 0.2]]}))
    job.preflight_ollama()


def test_preflight_fails_fast_when_ollama_is_down(monkeypatch):
    _post(monkeypatch, requests.exceptions.ConnectionError("refused"))
    with pytest.raises(RuntimeError) as exc:
        job.preflight_ollama()
    assert "cannot reach" in str(exc.value) and "host.docker.internal" in str(exc.value)


def test_preflight_tells_you_to_pull_a_missing_model(monkeypatch):
    _post(monkeypatch, FakeResp(404, {"error": "model not found"}, text="not found"))
    with pytest.raises(RuntimeError, match=r"ollama pull nomic-embed-text"):
        job.preflight_ollama()


@pytest.mark.parametrize(
    "resp",
    [FakeResp(500, None, text="boom"), FakeResp(200, {"embeddings": []}), FakeResp(200, {"x": 1}), FakeResp(200, None)],
)
def test_preflight_rejects_unusable_responses(monkeypatch, resp):
    _post(monkeypatch, resp)
    with pytest.raises(RuntimeError, match="Ollama preflight failed"):
        job.preflight_ollama()


@pytest.mark.parametrize(
    "total,with_emb,ok",
    [(100, 100, True), (100, 90, True), (100, 89, False), (100, 0, False)],
)
def test_coverage_gate_runs_before_anything_is_written(total, with_emb, ok):
    if ok:
        job.check_embedding_coverage(total, with_emb, max_missing_fraction=0.10)
    else:
        with pytest.raises(RuntimeError, match="Not writing chunks/"):
            job.check_embedding_coverage(total, with_emb, max_missing_fraction=0.10)


def test_coverage_gate_rejects_an_empty_input():
    with pytest.raises(RuntimeError, match="No chunk rows"):
        job.check_embedding_coverage(0, 0)


def test_embed_text_logs_a_warning_instead_of_printing_when_ollama_fails(monkeypatch, caplog):
    monkeypatch.setattr(job, "OLLAMA_EMBED_MAX_RETRIES", 2)
    monkeypatch.setattr(job.time, "sleep", lambda s: None)
    _post(monkeypatch, requests.exceptions.ConnectionError("refused"))
    with caplog.at_level("WARNING", logger="lemkin.spark.embed"):
        assert job.embed_text("some chunk") == []
    assert "Embedding failed after 2 tries" in caplog.text


def test_embed_text_returns_the_vector_on_success(monkeypatch):
    monkeypatch.setattr(job, "OLLAMA_EMBED_DELAY_SEC", 0)
    _post(monkeypatch, FakeResp(200, {"embeddings": [[1.0, 2.0]]}))
    assert job.embed_text("some chunk") == [1.0, 2.0]
