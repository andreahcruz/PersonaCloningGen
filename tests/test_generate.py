import json
import logging
from types import SimpleNamespace

import pytest
import requests

import generate
from generate import (
    ConfigError,
    EmbeddingError,
    GenerationError,
    LemkinError,
    ModelNotFoundError,
    OllamaError,
    OllamaUnavailableError,
    RetrievalError,
)

BASE = "http://ollama.test:11434"


class FakeResp:
    def __init__(self, status=200, payload=None, text=""):
        self.status_code = status
        self._payload = payload
        self.text = text or (json.dumps(payload) if payload is not None else "")
        self.ok = status < 400

    def json(self):
        if self._payload is None:
            raise ValueError("no json body")
        return self._payload


@pytest.fixture
def post(monkeypatch):
    """Replace requests.post with a scripted sequence; returns the recorder."""

    class Recorder:
        calls = 0
        script: list = []

    rec = Recorder()
    monkeypatch.setattr(generate.time, "sleep", lambda s: None)
    monkeypatch.setattr(generate, "_EMBED_RETRIES", 2)
    monkeypatch.setattr(generate, "_GEN_RETRIES", 1)

    def fake_post(url, json=None, timeout=None):
        rec.calls += 1
        step = rec.script[min(rec.calls, len(rec.script)) - 1]
        if isinstance(step, Exception):
            raise step
        return step

    monkeypatch.setattr(generate.requests, "post", fake_post)
    return rec


# ── Ollama transport ──────────────────────────────────────────────────────
def test_embed_connection_error_is_retried_then_typed(post):
    post.script = [requests.exceptions.ConnectionError("refused")]
    with pytest.raises(OllamaUnavailableError) as exc:
        generate.ollama_embed("hi", BASE, "nomic-embed-text")
    assert post.calls == 3  # 1 try + 2 retries
    assert "ollama serve" in exc.value.hint
    assert isinstance(exc.value.__cause__, requests.exceptions.ConnectionError)


def test_embed_recovers_after_a_transient_failure(post):
    post.script = [
        requests.exceptions.ConnectionError("blip"),
        FakeResp(200, {"embeddings": [[0.1, 0.2, 0.3]]}),
    ]
    assert generate.ollama_embed("hi", BASE, "m") == [0.1, 0.2, 0.3]
    assert post.calls == 2


def test_generate_timeout_is_not_retried(post):
    post.script = [requests.exceptions.Timeout("slow")]
    with pytest.raises(OllamaUnavailableError) as exc:
        generate.ollama_generate("prompt", BASE, "llama3.1")
    assert post.calls == 1
    assert "timed out" in str(exc.value)
    assert "OLLAMA_GENERATE_TIMEOUT" in exc.value.hint


def test_generate_connection_error_is_retried_once(post):
    post.script = [requests.exceptions.ConnectionError("refused")]
    with pytest.raises(OllamaUnavailableError):
        generate.ollama_generate("prompt", BASE, "llama3.1")
    assert post.calls == 2


def test_missing_model_gets_a_pull_hint_and_no_retry(post):
    post.script = [FakeResp(404, {"error": "model 'lemkin-clone' not found"})]
    with pytest.raises(ModelNotFoundError) as exc:
        generate.ollama_generate("prompt", BASE, "lemkin-clone")
    assert post.calls == 1
    assert "ollama pull lemkin-clone" in exc.value.hint


def test_server_error_is_retried_then_reported_with_body(post):
    post.script = [FakeResp(500, {"error": "out of memory"})]
    with pytest.raises(OllamaError) as exc:
        generate.ollama_generate("prompt", BASE, "llama3.1")
    assert post.calls == 2
    assert "HTTP 500" in str(exc.value) and "out of memory" in str(exc.value)


def test_client_error_is_not_retried(post):
    post.script = [FakeResp(400, {"error": "bad request"})]
    with pytest.raises(OllamaError):
        generate.ollama_embed("hi", BASE, "m")
    assert post.calls == 1


def test_unusable_embedding_payloads_raise_embedding_error(post):
    for resp in (FakeResp(200, {"embeddings": []}), FakeResp(200, {"nope": 1}), FakeResp(200, None, text="<html>")):
        post.script = [resp]
        post.calls = 0
        with pytest.raises(EmbeddingError):
            generate.ollama_embed("hi", BASE, "m")


def test_generate_invalid_json_raises_generation_error(post):
    post.script = [FakeResp(200, None, text="<html>")]
    with pytest.raises(GenerationError):
        generate.ollama_generate("prompt", BASE, "m")


def test_generate_returns_stripped_text(post):
    post.script = [FakeResp(200, {"response": "  hello  \n"})]
    assert generate.ollama_generate("prompt", BASE, "m") == "hello"


def test_all_expected_errors_are_runtime_errors():
    # eval_rag and older callers catch RuntimeError / Exception; that must keep working.
    for cls in (ConfigError, OllamaError, OllamaUnavailableError, ModelNotFoundError,
                EmbeddingError, RetrievalError, GenerationError):
        assert issubclass(cls, LemkinError) and issubclass(cls, RuntimeError)


# ── Draft validation ──────────────────────────────────────────────────────
def test_validate_draft_rejects_empty_and_degenerate_output():
    with pytest.raises(GenerationError, match="empty"):
        generate.validate_draft("   ", "lemkin-clone")
    with pytest.raises(GenerationError, match="degenerate") as exc:
        generate.validate_draft("vibe " * 100, "lemkin-clone")
    assert "diagnose_pipeline" in exc.value.hint


def test_validate_draft_passes_normal_text_through():
    text = "Hire your first seller earlier than feels comfortable."
    assert generate.validate_draft(f"  {text}\n", "m") == text


# ── Chroma ────────────────────────────────────────────────────────────────
@pytest.fixture
def http_chroma(monkeypatch):
    monkeypatch.setenv("CHROMA_USE_HTTP", "true")
    monkeypatch.setenv("CHROMA_HOST", "chroma")
    monkeypatch.setenv("CHROMA_PORT", "8000")


def test_chroma_tenant_error_gets_the_volume_hint(monkeypatch, http_chroma, tmp_path):
    def boom(**kwargs):
        raise ValueError("Could not connect to tenant default_tenant. Are you sure it exists?")

    monkeypatch.setattr(generate.chromadb, "HttpClient", boom)
    with pytest.raises(RetrievalError) as exc:
        generate.retrieve(tmp_path, [0.1], 3, "nomic-embed-text")
    assert "chroma-data" in exc.value.hint and "1.5.5" in exc.value.hint


def test_chroma_unreachable_names_the_target(monkeypatch, http_chroma, tmp_path):
    def boom(**kwargs):
        raise ConnectionError("Connection refused")

    monkeypatch.setattr(generate.chromadb, "HttpClient", boom)
    with pytest.raises(RetrievalError) as exc:
        generate.retrieve(tmp_path, [0.1], 3, "m")
    assert "chroma:8000" in str(exc.value)


def test_missing_collection_lists_existing_ones(monkeypatch, http_chroma, tmp_path):
    client = SimpleNamespace(
        get_collection=lambda name: (_ for _ in ()).throw(generate.NotFoundError("nope")),
        list_collections=lambda: [SimpleNamespace(name="other")],
    )
    monkeypatch.setattr(generate.chromadb, "HttpClient", lambda **kw: client)
    with pytest.raises(RetrievalError) as exc:
        generate.retrieve(tmp_path, [0.1], 3, "m")
    assert "load_to_chroma" in exc.value.hint and "other" in exc.value.hint


def test_dimension_mismatch_is_explained(monkeypatch, http_chroma, tmp_path):
    def query(**kwargs):
        raise Exception("Embedding dimension 768 does not match collection dimensionality 384")

    collection = SimpleNamespace(query=query)
    monkeypatch.setattr(
        generate.chromadb, "HttpClient", lambda **kw: SimpleNamespace(get_collection=lambda n: collection)
    )
    with pytest.raises(RetrievalError) as exc:
        generate.retrieve(tmp_path, [0.1] * 768, 3, "some-other-model")
    assert "some-other-model" in exc.value.hint


def test_retrieve_returns_chunks_with_metadata(monkeypatch, http_chroma, tmp_path):
    collection = SimpleNamespace(
        query=lambda **kw: {"documents": [["chunk one"]], "metadatas": [[{"title": "T"}]]}
    )
    monkeypatch.setattr(
        generate.chromadb, "HttpClient", lambda **kw: SimpleNamespace(get_collection=lambda n: collection)
    )
    assert generate.retrieve(tmp_path, [0.1], 1, "m") == [{"chunk_text": "chunk one", "title": "T"}]


# ── Config files ──────────────────────────────────────────────────────────
def test_config_loaders_raise_config_error(tmp_path):
    bad_yaml = tmp_path / "specs.yaml"
    bad_yaml.write_text("a: [unclosed", encoding="utf-8")
    with pytest.raises(ConfigError, match="not valid YAML"):
        generate.load_format_spec(bad_yaml, "a")

    ok_yaml = tmp_path / "ok.yaml"
    ok_yaml.write_text("linkedin_post: {output_type: markdown}\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="Unknown format: nope"):
        generate.load_format_spec(ok_yaml, "nope")
    with pytest.raises(ConfigError, match="Cannot read"):
        generate.load_format_spec(tmp_path / "missing.yaml", "a")

    bad_json = tmp_path / "persona.json"
    bad_json.write_text("{not json", encoding="utf-8")
    with pytest.raises(ConfigError) as exc:
        generate.load_persona(bad_json)
    assert "extract_persona" in exc.value.hint
    with pytest.raises(ConfigError) as exc:
        generate.load_persona(tmp_path / "missing.json")
    assert "extract_persona" in exc.value.hint


# ── Shared pipeline ───────────────────────────────────────────────────────
@pytest.fixture
def rag_files(tmp_path):
    persona = tmp_path / "persona.json"
    persona.write_text(json.dumps({"meta": {"source": "test persona"}}), encoding="utf-8")
    specs = generate.Path(generate.__file__).parent / "formats" / "format_specs.yaml"
    return SimpleNamespace(persona=persona, specs=specs, index=tmp_path / "index")


def _run(rag_files, **overrides):
    kwargs = dict(
        format_name="linkedin_post", topic="Hiring", audience="founders", goal="one takeaway",
        cta="none", k=3, index_path=rag_files.index, persona_path=rag_files.persona,
        format_specs_path=rag_files.specs, ollama_base=BASE, embed_model="nomic-embed-text",
        gen_model="llama3.1",
    )
    kwargs.update(overrides)
    return generate.run_rag_pipeline(**kwargs)


def test_pipeline_happy_path_logs_stages_with_a_request_id(monkeypatch, rag_files, caplog):
    monkeypatch.setattr(generate, "ollama_embed", lambda *a: [0.1, 0.2])
    monkeypatch.setattr(generate, "retrieve", lambda *a: [{"chunk_text": "some context", "doc_id": "d1"}])
    monkeypatch.setattr(generate, "ollama_generate", lambda *a: "A perfectly normal draft about hiring.")
    caplog.set_level(logging.INFO, logger="lemkin")

    assert _run(rag_files) == "A perfectly normal draft about hiring."

    messages = [r.getMessage() for r in caplog.records if r.name == "lemkin.rag"]
    assert any("embedded query" in m for m in messages)
    assert any("retrieved 1 chunks" in m for m in messages)
    assert any(m.rstrip().endswith("s") and "done in" in m for m in messages)
    ids = {m.split("]")[0] for m in messages if m.startswith("[")}
    assert len(ids) == 1, "every line of one run should share one request id"


def test_pipeline_empty_retrieval_is_an_error_not_a_context_free_draft(monkeypatch, rag_files, caplog):
    monkeypatch.setattr(generate, "ollama_embed", lambda *a: [0.1])
    monkeypatch.setattr(generate, "retrieve", lambda *a: [])
    monkeypatch.setattr(generate, "ollama_generate", lambda *a: pytest.fail("must not generate without context"))
    with pytest.raises(RetrievalError, match="no chunks"):
        _run(rag_files)
    assert any(r.levelno == logging.ERROR and "RetrievalError" in r.getMessage() for r in caplog.records)


def test_pipeline_rejects_a_degenerate_model(monkeypatch, rag_files):
    monkeypatch.setattr(generate, "ollama_embed", lambda *a: [0.1])
    monkeypatch.setattr(generate, "retrieve", lambda *a: [{"chunk_text": "ctx"}])
    monkeypatch.setattr(generate, "ollama_generate", lambda *a: "vibe " * 100)
    with pytest.raises(GenerationError, match="degenerate"):
        _run(rag_files, gen_model="lemkin-clone")


def test_pipeline_logs_a_traceback_for_unexpected_errors(monkeypatch, rag_files, caplog):
    monkeypatch.setattr(generate, "ollama_embed", lambda *a: (_ for _ in ()).throw(KeyError("surprise")))
    with pytest.raises(KeyError):
        _run(rag_files)
    assert any(r.exc_info for r in caplog.records if r.name == "lemkin.rag")


def test_pipeline_bad_format_is_a_config_error(monkeypatch, rag_files):
    with pytest.raises(ConfigError, match="Unknown format"):
        _run(rag_files, format_name="not_a_format")


# ── Preflight ─────────────────────────────────────────────────────────────
def test_check_dependencies_reports_each_problem_and_never_raises(monkeypatch, rag_files):
    def down(url, timeout=None):
        raise requests.exceptions.ConnectionError("refused")

    monkeypatch.setattr(generate.requests, "get", down)
    monkeypatch.setattr(
        generate, "_get_collection",
        lambda p: (_ for _ in ()).throw(RetrievalError("Cannot connect to Chroma (chroma:8000): refused")),
    )
    rows = generate.check_dependencies(
        BASE, "nomic-embed-text", "llama3.1", rag_files.persona, rag_files.specs, rag_files.index
    )
    status = {name: ok for name, ok, _ in rows}
    assert status["format specs"] and status["persona profile"]
    assert status["ollama"] is False and status["chroma collection"] is False
    assert "embed model" not in status  # models are only checked once Ollama answers


def test_check_dependencies_matches_models_with_and_without_latest_tag(monkeypatch, rag_files):
    tags = {"models": [{"name": "nomic-embed-text:latest"}, {"name": "lemkin-clone:latest"}]}

    class Ok:
        def raise_for_status(self):
            pass

        def json(self):
            return tags

    monkeypatch.setattr(generate.requests, "get", lambda url, timeout=None: Ok())
    monkeypatch.setattr(generate, "_get_collection", lambda p: SimpleNamespace(count=lambda: 0))
    rows = {n: (ok, d) for n, ok, d in generate.check_dependencies(
        BASE, "nomic-embed-text", "llama3.1", rag_files.persona, rag_files.specs, rag_files.index
    )}
    assert rows["embed model"][0] is True
    assert rows["generate model"][0] is False and "llama3.1" in rows["generate model"][1]
    assert rows["chroma collection"][0] is False  # 0 chunks = empty collection


def test_configure_logging_is_idempotent():
    log = logging.getLogger("lemkin")
    before = (list(log.handlers), log.propagate, getattr(log, "_lemkin_configured", False))
    try:
        log._lemkin_configured = False  # type: ignore[attr-defined]
        generate.configure_logging()
        generate.configure_logging()
        assert len(log.handlers) == len(before[0]) + 1
    finally:
        log.handlers[:] = before[0]
        log.propagate = True
        log._lemkin_configured = before[2]  # type: ignore[attr-defined]
