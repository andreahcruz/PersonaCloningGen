import logging
import subprocess
import sys

import pytest

from host_finetune import register_ollama as ro
from host_finetune import watcher


# ── watcher ───────────────────────────────────────────────────────────────
def test_failed_step_logs_exit_code_and_a_hint(caplog):
    caplog.set_level(logging.INFO, logger="lemkin.watcher")
    with pytest.raises(RuntimeError, match="finetune failed \\(exit 3\\)"):
        watcher._run("finetune", [sys.executable, "-c", "import sys; sys.exit(3)"])
    errors = [r.getMessage() for r in caplog.records if r.levelno == logging.ERROR]
    assert errors and "exit 3" in errors[0] and "RESUME_FROM_CHECKPOINT" in errors[0]


def test_successful_step_logs_its_duration(caplog):
    caplog.set_level(logging.INFO, logger="lemkin.watcher")
    watcher._run("register_ollama", [sys.executable, "-c", "pass"])
    assert "register_ollama finished in" in caplog.text


def test_once_mode_exits_nonzero_and_logs_a_traceback_on_failure(monkeypatch, caplog):
    monkeypatch.setattr(sys, "argv", ["watcher", "--once"])
    monkeypatch.setattr(watcher, "_configure_logging", lambda: None)

    def boom(force=False):
        raise RuntimeError("merge blew up")

    monkeypatch.setattr(watcher, "run_pipeline_once", boom)
    with pytest.raises(SystemExit) as exc:
        watcher.main()
    assert exc.value.code == 1
    assert any(r.exc_info for r in caplog.records if r.name == "lemkin.watcher")


def test_daemon_backs_off_after_repeated_failures_and_resets_on_success(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["watcher", "--interval", "60"])
    monkeypatch.setattr(watcher, "_configure_logging", lambda: None)
    outcomes = iter([RuntimeError("x"), RuntimeError("x"), RuntimeError("x"), False, RuntimeError("x")])

    def fake_run(force=False):
        step = next(outcomes, KeyboardInterrupt())
        if isinstance(step, BaseException):
            raise step
        return step

    waits = []
    monkeypatch.setattr(watcher, "run_pipeline_once", fake_run)
    monkeypatch.setattr(watcher.time, "sleep", waits.append)
    watcher.main()  # returns when the fake raises KeyboardInterrupt
    # failures 1,2,3 -> 120, 240, 480; a quiet poll resets to the interval; next failure starts over.
    assert waits == [120, 240, 480, 60, 120]


def test_backoff_is_capped(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["watcher", "--interval", "1000"])
    monkeypatch.setattr(watcher, "_configure_logging", lambda: None)
    outcomes = iter([RuntimeError("x")] * 4)

    def fake_run(force=False):
        step = next(outcomes, KeyboardInterrupt())
        if isinstance(step, BaseException):
            raise step
        return step

    waits = []
    monkeypatch.setattr(watcher, "run_pipeline_once", fake_run)
    monkeypatch.setattr(watcher.time, "sleep", waits.append)
    watcher.main()
    assert max(waits) == watcher.MAX_BACKOFF_SECONDS


# ── register_ollama smoke test ────────────────────────────────────────────
@pytest.fixture
def ollama(monkeypatch, tmp_path):
    gguf = tmp_path / "model.Q4_K_M.gguf"
    gguf.write_bytes(b"gguf")
    modelfile = tmp_path / "Modelfile"
    modelfile.write_text("FROM x\n", encoding="utf-8")
    monkeypatch.setattr(ro, "have_ollama", lambda: True)
    monkeypatch.setattr(ro, "find_gguf", lambda: gguf)
    monkeypatch.setattr(ro, "render_modelfile", lambda p: modelfile)
    monkeypatch.delenv("SKIP_OLLAMA_SMOKE_TEST", raising=False)

    state = {"list_rc": 0, "run_rc": 0, "run_out": "ARR is your annual recurring revenue.", "cmds": []}

    def fake_run(cmd, **kwargs):
        state["cmds"].append(cmd[1])
        rc, out = {
            "list": (state["list_rc"], ""),
            "create": (0, ""),
            "run": (state["run_rc"], state["run_out"]),
        }[cmd[1]]
        return subprocess.CompletedProcess(cmd, rc, stdout=out, stderr="server said no" if rc else "")

    monkeypatch.setattr(ro.subprocess, "run", fake_run)
    return state


def test_register_succeeds_with_a_sane_answer(ollama):
    ro.main()
    assert ollama["cmds"] == ["list", "create", "run"]


def test_register_fails_loudly_on_the_known_gibberish_export(ollama):
    ollama["run_out"] = "vibe " * 20
    with pytest.raises(RuntimeError, match="degenerate"):
        ro.main()


def test_register_fails_on_an_empty_answer(ollama):
    ollama["run_out"] = "   "
    with pytest.raises(RuntimeError, match="empty or degenerate"):
        ro.main()


def test_register_fails_if_the_model_cannot_run_at_all(ollama):
    ollama["run_rc"] = 1
    with pytest.raises(RuntimeError, match="cannot generate"):
        ro.main()


def test_register_checks_the_ollama_server_before_the_slow_create(ollama):
    ollama["list_rc"] = 1
    with pytest.raises(RuntimeError, match="not reachable"):
        ro.main()
    assert ollama["cmds"] == ["list"]
