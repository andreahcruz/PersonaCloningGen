"""Merge the LoRA adapter into the base model and export GGUF for Ollama.

Loads ``host_finetune/output/lemkin_lora/`` on top of ``HF_MODEL_NAME`` and
writes a merged GGUF (default Q4_K_M) to ``host_finetune/output/lemkin-clone/``.

Run directly:
    python -m host_finetune.merge_and_export

Unsloth's ``save_pretrained_gguf`` calls llama.cpp under the hood; the first
run downloads/builds llama.cpp into the unsloth cache (a few minutes), then
subsequent runs are fast.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

from unsloth import FastLanguageModel  # noqa: E402  (must precede transformers)

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from host_finetune.config import (
    GGUF_DIR,
    GGUF_QUANT,
    HF_MODEL_NAME,
    LORA_ADAPTER_PATH,
    MAX_SEQ_LENGTH,
)


OPENSSL_INSTALL_HELP = """
OpenSSL Win64 for llama.cpp / CMake:
  • The **Light** winget/npm-style build only ships `openssl.exe` — CMake still fails
    because **`include/` and `lib/` are missing.**
  • Install the **full Win64 installer** from Shining Light (NOT "Light"):
       https://slproweb.com/products/Win32OpenSSL.html
    Under "Win64 OpenSSL" download an EXE whose name does **NOT** contain `Light`,
    e.g. `Win64OpenSSL-3_*.exe` vs `Win64OpenSSL-*_Light.exe`.
  • Accept the default path: **C:\\Program Files\\OpenSSL-Win64**
  • Optionally copy **lib\\*.dll** into **bin\\** when the wizard offers it (recommended).
  • Fully quit Cursor, reopen a terminal, then run `merge_and_export` again.

You also need **Visual Studio Build Tools** → workload **Desktop development with C++**.
"""


def _openssl_install_has_dev_libs(root: Path) -> bool:
    """CMake needs headers + import libs — Light installers omit these."""
    if not (root / "include" / "openssl" / "ssl.h").is_file():
        return False
    lib_root = root / "lib"
    if not lib_root.is_dir():
        return False
    if any(lib_root.glob("*.lib")) or any(lib_root.glob("*.LIB")):
        return True
    vc_md = lib_root / "VC" / "x64" / "MD"
    vc_mt = lib_root / "VC" / "x64" / "MT"
    return (vc_md.is_dir() and any(vc_md.glob("*.lib"))) or (
        vc_mt.is_dir() and any(vc_mt.glob("*.lib"))
    )


def _ensure_windows_openssl_on_path() -> None:
    """Unsloth's llama.cpp install checks ``shutil.which('openssl')``. Cursor/PowerShell
    often inherits a different PATH than cmd.exe, so we prepend common install dirs.
    Also set CMake variables so **non-Light** installs (headers + libs) are found.
    """
    if os.name != "nt":
        return
    candidates = [
        Path(r"C:\Program Files\OpenSSL-Win64"),
        Path(r"C:\Program Files (x86)\OpenSSL-Win64"),
        Path(r"C:\OpenSSL-Win64"),
        Path(os.environ.get("LOCALAPPDATA", "")) / "Programs" / "OpenSSL-Win64",
    ]
    for root in candidates:
        if not root.is_dir():
            continue
        bin_dir = root / "bin"
        exe = bin_dir / "openssl.exe"
        if not exe.is_file():
            continue
        path = os.environ.get("PATH", "")
        bin_s = str(bin_dir)
        if bin_s not in path:
            os.environ["PATH"] = bin_s + os.pathsep + path

        root_s = str(root)
        os.environ.setdefault("OPENSSL_ROOT_DIR", root_s)
        cmake_prefix = os.environ.get("CMAKE_PREFIX_PATH", "")
        if root_s not in cmake_prefix.split(os.pathsep):
            os.environ["CMAKE_PREFIX_PATH"] = (
                root_s + (os.pathsep + cmake_prefix if cmake_prefix else "")
            )

        dev_ok = _openssl_install_has_dev_libs(root)
        print(
            f"[merge_and_export] OpenSSL binaries: {exe}\n"
            f"[merge_and_export] OPENSSL_ROOT_DIR={root_s}\n"
            f"[merge_and_export] OpenSSL headers+libs present (CMake): {dev_ok}"
        )

        if not dev_ok:
            raise RuntimeError(
                "Installed OpenSSL is missing **development files** "
                "(no `include/openssl/ssl.h` / `lib\\*.lib`). "
                "**Win64 Light** installers cause this.\n"
                + OPENSSL_INSTALL_HELP
            )

        return

    raise RuntimeError(
        "Could not find `openssl.exe` under common Win64 paths. "
        "Install Win64 OpenSSL (full installer) from slproweb, then reopen Cursor.\n"
        + OPENSSL_INSTALL_HELP
    )


def main() -> None:
    _ensure_windows_openssl_on_path()

    if not LORA_ADAPTER_PATH.is_dir() or not any(LORA_ADAPTER_PATH.iterdir()):
        raise FileNotFoundError(
            f"No adapter at {LORA_ADAPTER_PATH}. Run `python -m host_finetune.finetune` "
            "(or set LORA_ADAPTER_PATH to a checkpoint-* folder)."
        )

    GGUF_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading base {HF_MODEL_NAME} + adapter {LORA_ADAPTER_PATH} ...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=str(LORA_ADAPTER_PATH),
        max_seq_length=MAX_SEQ_LENGTH,
        dtype=None,
        load_in_4bit=True,
    )

    print(f"Exporting GGUF (quant={GGUF_QUANT}) -> {GGUF_DIR}")
    # Unsloth handles: dequantize 4-bit -> merge LoRA -> save merged HF model ->
    # invoke llama.cpp converter -> quantize to the requested format.
    model.save_pretrained_gguf(
        str(GGUF_DIR),
        tokenizer,
        quantization_method=GGUF_QUANT,
    )

    ggufs = sorted(GGUF_DIR.glob("*.gguf"))
    if not ggufs:
        raise RuntimeError(
            f"GGUF export finished but no *.gguf file was written under {GGUF_DIR}."
        )
    print(f"Wrote: {ggufs[-1]}  ({ggufs[-1].stat().st_size / 1e9:.2f} GB)")


if __name__ == "__main__":
    main()
