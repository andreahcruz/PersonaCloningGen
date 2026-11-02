"""Offline evaluation for saved Model E and Model F predictions.

Run from the repository root. This command does not query LightRAG and does
not load a generator.

    host_finetune\\.venv\\Scripts\\python.exe -X utf8 -m host_finetune.eval_bench ^
        --predictions path\\to\\predictions.jsonl ^
        --out path\\to\\report_dir ^
        --experiment-id EXP-DRY-001
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from host_finetune.eval_bench.report import refuse_live_store, write_report
from host_finetune.eval_bench.runner import paired_faithfulness, score_batch


def load_predictions(path: Path) -> list[dict]:
    refuse_live_store(path)
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--pair-left", default=None)
    parser.add_argument("--pair-right", default=None)
    args = parser.parse_args(argv)
    refuse_live_store(args.predictions)
    refuse_live_store(args.out)
    batch = score_batch(load_predictions(args.predictions))
    paired = None
    if args.pair_left or args.pair_right:
        if not args.pair_left or not args.pair_right:
            raise SystemExit("pass both --pair-left and --pair-right")
        paired = paired_faithfulness(batch["results"], args.pair_left, args.pair_right)
    write_report(args.out, batch, experiment_id=args.experiment_id, paired=paired)
    print(
        f"scored {len(batch['results'])} predictions, {len(batch['failures'])} failure rows, wrote {args.out}",
        flush=True,
    )


if __name__ == "__main__":
    main()
