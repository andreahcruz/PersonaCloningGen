"""Drive the real Streamlit script with Streamlit's AppTest and check what the user sees."""
from pathlib import Path

import pytest

pytest.importorskip("streamlit.testing.v1")
from streamlit.testing.v1 import AppTest  # noqa: E402

import generate  # noqa: E402

APP = str(Path(__file__).resolve().parents[1] / "user_interface.py")


@pytest.fixture
def app(monkeypatch):
    # CHROMA_USE_HTTP skips the local-index existence check; Ollama's model list is stubbed
    # so the sidebar does not try the network.
    monkeypatch.setenv("CHROMA_USE_HTTP", "true")
    monkeypatch.setattr("requests.get", lambda *a, **k: (_ for _ in ()).throw(ConnectionError("no ollama")))
    at = AppTest.from_file(APP, default_timeout=30)
    at.run()
    assert not at.exception
    return at


def _fill_and_generate(at):
    at.sidebar.text_input[0].set_value("Hiring a VP Sales")
    at.sidebar.text_input[1].set_value("B2B founders")
    at.sidebar.text_input[2].set_value("one takeaway")
    generate_btn = next(b for b in at.sidebar.button if b.label == "Generate")
    generate_btn.click().run()
    return at


def test_typed_error_shows_the_message_and_the_fix(app, monkeypatch):
    def fail(**kwargs):
        raise generate.OllamaUnavailableError("Cannot connect to Ollama at http://x.", hint="Start it with `ollama serve`.")

    monkeypatch.setattr(generate, "run_rag_pipeline", fail)
    at = _fill_and_generate(app)
    assert any("Cannot connect to Ollama" in e.value for e in at.error)
    assert any("ollama serve" in i.value for i in at.info)


def test_unexpected_error_is_shown_without_crashing_and_logged(app, monkeypatch, caplog):
    def fail(**kwargs):
        raise KeyError("surprise")

    monkeypatch.setattr(generate, "run_rag_pipeline", fail)
    at = _fill_and_generate(app)
    assert any("Unexpected error" in e.value for e in at.error)
    assert any("docker compose logs streamlit" in i.value for i in at.info)
    assert any(r.exc_info for r in caplog.records if r.name == "lemkin.ui")


def test_successful_generation_shows_the_draft(app, monkeypatch):
    monkeypatch.setattr(generate, "run_rag_pipeline", lambda **kw: "A short, clear draft.")
    at = _fill_and_generate(app)
    assert not at.error
    assert any("A short, clear draft." in m.value for m in at.markdown)


def test_system_check_button_reports_each_check(app, monkeypatch):
    monkeypatch.setattr(
        generate, "check_dependencies",
        lambda *a, **k: [("ollama", False, "cannot reach http://x"), ("persona profile", True, "data/persona_profile.json")],
    )
    check_btn = next(b for b in app.sidebar.button if b.label == "Run system check")
    at = check_btn.click().run()
    assert any("ollama: cannot reach" in e.value for e in at.error)
    assert any("persona profile" in s.value for s in at.success)
