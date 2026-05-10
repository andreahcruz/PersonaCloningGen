"""Host-side watcher: poll MinIO for a new training/_READY sentinel and run the
finetune -> merge -> register chain when one appears.

Default mode: poll forever (use Ctrl-C to stop).
With ``--once``: check once and exit (good for cron / Task Scheduler).
With ``--force``: ignore the cached last_run timestamp and run unconditionally.

Run:
    python -m host_finetune.watcher           # daemon (poll every 60s)
    python -m host_finetune.watcher --once    # one shot
    python -m host_finetune.watcher --force   # retrain even if sentinel hasn't moved
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from host_finetune.config import (
    DATA_DIR,
    DATASET_KEY,
    DATASET_LOCAL,
    LAST_RUN_FILE,
    SENTINEL_KEY,
)
from host_finetune.s3util import download_to, minio_client, read_sentinel

POLL_SECONDS = 60


def _load_last_run() -> str | None:
    if LAST_RUN_FILE.is_file():
        try:
            return json.loads(LAST_RUN_FILE.read_text(encoding="utf-8")).get("ready_at")
        except (json.JSONDecodeError, OSError):
            return None
    return None


def _save_last_run(sentinel: dict) -> None:
    LAST_RUN_FILE.write_text(json.dumps(sentinel, indent=2), encoding="utf-8")


def _run(label: str, cmd: list[str]) -> None:
    print(f"\n=== {label}: {' '.join(cmd)} ===")
    proc = subprocess.run(cmd, check=False)
    if proc.returncode != 0:
        raise RuntimeError(f"{label} failed (exit {proc.returncode}).")


def run_pipeline_once(force: bool = False) -> bool:
    """Returns True if a run was performed, False if nothing to do."""
    s3 = minio_client()
    sentinel = read_sentinel(s3, SENTINEL_KEY)
    if sentinel is None:
        print(f"No sentinel at s3://.../{SENTINEL_KEY} yet. Trigger the Airflow DAG first.")
        return False

    last = _load_last_run()
    if not force and last == sentinel.get("ready_at"):
        # Nothing new since the last successful run.
        return False

    print(f"New training data detected (ready_at={sentinel.get('ready_at')}, "
          f"rows={sentinel.get('approx_rows')}, bytes={sentinel.get('dataset_bytes')}).")
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {DATASET_KEY} -> {DATASET_LOCAL}")
    download_to(s3, DATASET_KEY, DATASET_LOCAL)

    py = sys.executable
    _run("finetune", [py, "-m", "host_finetune.finetune"])
    _run("merge_and_export", [py, "-m", "host_finetune.merge_and_export"])
    _run("register_ollama", [py, "-m", "host_finetune.register_ollama"])

    _save_last_run(sentinel)
    print("\nPipeline complete. lemkin-clone is registered with Ollama.")
    return True


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--once", action="store_true", help="Check once and exit.")
    p.add_argument("--force", action="store_true", help="Ignore cached last_run; always retrain.")
    p.add_argument("--interval", type=int, default=POLL_SECONDS,
                   help=f"Poll interval in seconds (default {POLL_SECONDS}).")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.once:
        ran = run_pipeline_once(force=args.force)
        sys.exit(0 if ran or not args.force else 1)

    print(f"Watching MinIO every {args.interval}s. Ctrl-C to stop.")
    force = args.force
    while True:
        try:
            ran = run_pipeline_once(force=force)
            force = False  # only honor --force once; subsequent loops use sentinel diff
            if not ran:
                print(".", end="", flush=True)
        except KeyboardInterrupt:
            print("\nStopped.")
            return
        except Exception as exc:  # noqa: BLE001
            print(f"\nPipeline error: {exc}. Will retry in {args.interval}s.")
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
