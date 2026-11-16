"""Small, reproducible base-vs-Relabel smoke test on Kevin's held-out rows.

The comparison intentionally disables retrieval and uses the same quantized
base, prompt, generation settings, and seed for both conditions. It accepts
either a converted LoRA adapter or Kevin's merged GGUF. It is not a replacement
for the full evaluation suite or the original Unsloth runtime.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
from pathlib import Path

DATASET_PATH = "experiments/EXP-20261003-004-relabel-continuations/raw/dataset.jsonl"
SPLITS_PATH = "experiments/EXP-20261003-004-relabel-continuations/raw/split_assignments.jsonl"
EXPECTED_DATASET_SHA256 = "1231835ecc5dcb608fefa0357d325affb2afbdc3aeade2a63b1dc42b2920897b"
DEFAULT_ROWS = (97, 11832, 14565, 26520)  # blog, LinkedIn, X, YouTube


def git_blob(ref: str, path: str) -> bytes:
    return subprocess.check_output(["git", "show", f"{ref}:{path}"])


def response_from_cli(stdout: str, instruction: str) -> str:
    marker = f"> {instruction}\n"
    if marker not in stdout:
        raise RuntimeError("Could not locate the prompt in llama CLI output")
    answer = stdout.split(marker, 1)[1]
    answer = re.split(r"\n\[ Prompt:", answer, maxsplit=1)[0]
    return answer.strip()


def generate(
    binary: Path, base: Path, adapter: Path | None, instruction: str, max_tokens: int
) -> str:
    cmd = [
        str(binary), "cli", "-m", str(base), "-ngl", "99", "-c", "2048",
        "-n", str(max_tokens), "--temp", "0.7", "--top-p", "0.9",
        "--repeat-penalty", "1.15", "-s", "42", "-p", instruction,
        "--single-turn", "--simple-io", "--no-display-prompt",
    ]
    if adapter is not None:
        cmd.extend(["--lora", str(adapter)])
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    return response_from_cli(result.stdout, instruction)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--llama-bin", type=Path, required=True)
    parser.add_argument("--base-gguf", type=Path, required=True)
    artifact = parser.add_mutually_exclusive_group(required=True)
    artifact.add_argument("--adapter-gguf", type=Path)
    artifact.add_argument("--fine-tuned-gguf", type=Path)
    parser.add_argument("--git-ref", default="origin/kevin/298b-integration")
    parser.add_argument("--rows", type=int, nargs="+", default=DEFAULT_ROWS)
    parser.add_argument("--max-tokens", type=int, default=384)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    dataset = git_blob(args.git_ref, DATASET_PATH)
    digest = hashlib.sha256(dataset).hexdigest()
    if digest != EXPECTED_DATASET_SHA256:
        raise RuntimeError(f"Frozen dataset hash mismatch: {digest}")
    rows = [json.loads(line) for line in dataset.splitlines()]
    assignments = [
        json.loads(line) for line in git_blob(args.git_ref, SPLITS_PATH).splitlines()
    ]
    splits = {row["row_index"]: row for row in assignments if "row_index" in row}

    results = []
    for index in args.rows:
        if not 0 <= index < len(rows) or splits[index]["split"] != "test":
            raise ValueError(f"Row {index} is not in the frozen test split")
        instruction = rows[index]["instruction"]
        fine_tuned_model = args.fine_tuned_gguf or args.base_gguf
        fine_tuned_adapter = None if args.fine_tuned_gguf else args.adapter_gguf
        result = {
            "row_index": index,
            "source": splits[index]["source_platform"],
            "instruction": instruction,
            "reference": rows[index]["output"],
            "base": generate(
                args.llama_bin, args.base_gguf, None, instruction, args.max_tokens
            ),
            "relabel": generate(
                args.llama_bin, fine_tuned_model, fine_tuned_adapter, instruction,
                args.max_tokens,
            ),
        }
        results.append(result)
        print(f"Completed held-out row {index} ({result['source']})", flush=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(
            {
                "dataset_sha256": digest,
                "base_gguf": str(args.base_gguf),
                "adapter_gguf": str(args.adapter_gguf) if args.adapter_gguf else None,
                "fine_tuned_gguf": str(args.fine_tuned_gguf) if args.fine_tuned_gguf else None,
                "generation": {"tokens": args.max_tokens, "temperature": 0.7, "top_p": 0.9,
                               "repeat_penalty": 1.15, "seed": 42, "retrieval": False},
                "results": results,
            },
            ensure_ascii=False,
            indent=2,
        ) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
