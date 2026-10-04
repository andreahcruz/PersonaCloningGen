"""Score the repaired and relabel adapters after EXP-005 has saved.

This does not load Unsloth unless the train log contains the adapter save
line for ``lemkin_lora_relabel``. Retrieval stays off. Voice judging is a
later call to ``eval_rag`` on the saved traces; BLEU and ROUGE are logged
only.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TRAIN_LOG = ROOT / "experiments" / "EXP-20261003-005-relabel-qlora" / "raw" / "train.log"
OUT_DIR = ROOT / "experiments" / "EXP-20261004-001-relabel-vs-repaired"
ADAPTER_MARK = "lemkin_lora_relabel"


def train_log_finished(text: str, adapter_name: str = ADAPTER_MARK) -> bool:
    """True only after finetune.py has saved this adapter."""
    return "Saving LoRA adapter ->" in text and adapter_name in text


def main() -> None:
    text = TRAIN_LOG.read_text(encoding="utf-8", errors="replace") if TRAIN_LOG.is_file() else ""
    if not train_log_finished(text):
        print(
            "EXP-005 has not saved lemkin_lora_relabel. Not loading a model.",
            file=sys.stderr,
        )
        sys.exit(2)
    from host_finetune.compare_adapters import main as compare_main

    sys.argv = [
        "compare_adapters",
        "--adapters",
        "repaired-vs-relabel",
        "--out-dir",
        str(OUT_DIR),
    ]
    compare_main()


if __name__ == "__main__":
    main()
