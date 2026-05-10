"""Register the merged GGUF as a custom Ollama model named ``lemkin-clone``.

Steps:
1. Find the GGUF written by ``merge_and_export.py``.
2. Render a temporary Modelfile with ``FROM`` pointing at that GGUF (absolute
   path — Ollama on Windows sometimes resolves relative paths from the wrong cwd).
3. ``ollama create lemkin-clone -f <tmp_modelfile>``.
4. Smoke-test with a short prompt.

Run:
    python -m host_finetune.register_ollama
"""
from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from host_finetune.config import GGUF_DIR, GGUF_SIDECAR_DIR, MODELFILE_PATH, OLLAMA_MODEL_NAME


def find_gguf() -> Path:
    candidates: list[Path] = []
    for root in (GGUF_DIR, GGUF_SIDECAR_DIR):
        if root.is_dir():
            candidates.extend(root.glob("*.gguf"))
    if not candidates:
        raise FileNotFoundError(
            f"No *.gguf files under {GGUF_DIR} or {GGUF_SIDECAR_DIR}. "
            "Run `python -m host_finetune.merge_and_export` first."
        )
    # Prefer Q4_K_M if present, else newest by mtime.
    q4 = [c for c in candidates if "q4_k_m" in c.name.lower()]
    return (q4[-1] if q4 else max(candidates, key=lambda p: p.stat().st_mtime))


def render_modelfile(gguf_path: Path) -> Path:
    if not MODELFILE_PATH.is_file():
        raise FileNotFoundError(f"Template Modelfile missing: {MODELFILE_PATH}")
    template = MODELFILE_PATH.read_text(encoding="utf-8")
    new_lines = []
    replaced = False
    for line in template.splitlines():
        if line.lstrip().startswith("FROM ") and not replaced:
            new_lines.append(f'FROM "{gguf_path.as_posix()}"')
            replaced = True
        else:
            new_lines.append(line)
    if not replaced:
        new_lines.insert(0, f'FROM "{gguf_path.as_posix()}"')
    rendered = "\n".join(new_lines) + "\n"

    tmp_dir = Path(tempfile.mkdtemp(prefix="lemkin_modelfile_"))
    tmp_path = tmp_dir / "Modelfile"
    tmp_path.write_text(rendered, encoding="utf-8")
    return tmp_path


def have_ollama() -> bool:
    return shutil.which("ollama") is not None


def main() -> None:
    if not have_ollama():
        raise RuntimeError(
            "`ollama` CLI not on PATH. Install Ollama (https://ollama.com) and ensure "
            "`ollama serve` is running before re-running this script."
        )

    gguf_path = find_gguf()
    print(f"Using GGUF: {gguf_path}  ({gguf_path.stat().st_size / 1e9:.2f} GB)")

    modelfile = render_modelfile(gguf_path)
    print(f"Rendered Modelfile -> {modelfile}")

    cmd = ["ollama", "create", OLLAMA_MODEL_NAME, "-f", str(modelfile)]
    print("Running:", " ".join(cmd))
    proc = subprocess.run(cmd, check=False)
    if proc.returncode != 0:
        raise RuntimeError(
            f"`ollama create` failed (exit {proc.returncode}). Check `ollama serve` is running."
        )

    print("\nSmoke test...")
    test_cmd = ["ollama", "run", OLLAMA_MODEL_NAME, "Write one sentence about ARR."]
    test_proc = subprocess.run(test_cmd, check=False, capture_output=True, text=True, encoding="utf-8")
    if test_proc.returncode == 0:
        print("--- model output ---")
        print((test_proc.stdout or "").strip())
        print("--------------------")
    else:
        print(f"Smoke test exited {test_proc.returncode}; stderr:\n{test_proc.stderr}")

    print(f"\nModel '{OLLAMA_MODEL_NAME}' is ready. Pick it in Streamlit's Generate model dropdown.")


if __name__ == "__main__":
    main()
