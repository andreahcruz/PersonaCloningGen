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
import logging
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
    OUTPUT_DIR,
    SENTINEL_KEY,
    SKIP_DATASET_CLEAN,
)
from host_finetune.s3util import download_to, minio_client, read_sentinel

POLL_SECONDS = 60
MAX_BACKOFF_SECONDS = 3600

logger = logging.getLogger("lemkin.watcher")

# Where to look first when a step fails; the step's own output is just above the error.
_FAILURE_HINTS = {
    "finetune": (
        "If this is CUDA out-of-memory see README 'Fine-tune troubleshooting'. "
        "Resume a cut-off run with RESUME_FROM_CHECKPOINT=1."
    ),
    "merge_and_export": (
        "Check free disk space (tens of GB) and that llama.cpp exists (LLAMA_CPP_DIR). "
        "Re-run only this step: python -m host_finetune.merge_and_export"
    ),
    "register_ollama": (
        "Is `ollama serve` running? Re-run only this step: python -m host_finetune.register_ollama"
    ),
}


def _configure_logging() -> None:
    """Console plus host_finetune/output/watcher.log, so an unattended run leaves a record."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(OUTPUT_DIR / "watcher.log", encoding="utf-8"),
        ],
    )


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
    logger.info("=== %s: %s ===", label, " ".join(cmd))
    started = time.perf_counter()
    proc = subprocess.run(cmd, check=False)
    elapsed = time.perf_counter() - started
    if proc.returncode != 0:
        logger.error(
            "%s failed (exit %d) after %.0fs. %s",
            label, proc.returncode, elapsed, _FAILURE_HINTS.get(label, ""),
        )
        raise RuntimeError(f"{label} failed (exit {proc.returncode}).")
    logger.info("%s finished in %.0fs", label, elapsed)


def run_pipeline_once(force: bool = False) -> bool:
    """Returns True if a run was performed, False if nothing to do."""
    s3 = minio_client()
    sentinel = read_sentinel(s3, SENTINEL_KEY)
    if sentinel is None:
        logger.info("No sentinel at s3://.../%s yet. Trigger the Airflow DAG first.", SENTINEL_KEY)
        return False

    last = _load_last_run()
    if not force and last == sentinel.get("ready_at"):
        # Nothing new since the last successful run.
        return False

    logger.info(
        "New training data detected (ready_at=%s, rows=%s, bytes=%s).",
        sentinel.get("ready_at"), sentinel.get("approx_rows"), sentinel.get("dataset_bytes"),
    )
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    logger.info("Downloading %s -> %s", DATASET_KEY, DATASET_LOCAL)
    download_to(s3, DATASET_KEY, DATASET_LOCAL)
    if DATASET_LOCAL.stat().st_size == 0:
        raise RuntimeError(f"Downloaded dataset {DATASET_LOCAL} is empty; not starting a GPU run.")

    from host_finetune.dataset_prepare import prepare_dataset_file

    prep = prepare_dataset_file(DATASET_LOCAL)
    if prep.get("chunk_expand"):
        ce = prep["chunk_expand"]
        logger.info(
            "Chunk expand: rows %s -> %s, long_docs_split=%s",
            ce.get("rows_in"), ce.get("rows_out"), ce.get("split_from_long"),
        )
    if prep.get("clean"):
        cl = prep["clean"]
        logger.info(
            "Cleaned dataset: rows %s -> %s, dropped_junk=%s truncated=%s",
            cl.get("rows_in"), cl.get("rows_out"), cl.get("dropped_junk"), cl.get("truncated_outputs"),
        )
        if not cl.get("rows_out"):
            raise RuntimeError("clean_dataset removed every row; refusing to fine-tune on nothing.")
    elif SKIP_DATASET_CLEAN:
        logger.info("SKIP_DATASET_CLEAN: skipped clean_dataset.")

    py = sys.executable
    _run("finetune", [py, "-m", "host_finetune.finetune"])
    _run("merge_and_export", [py, "-m", "host_finetune.merge_and_export"])
    _run("register_ollama", [py, "-m", "host_finetune.register_ollama"])

    _save_last_run(sentinel)
    logger.info("Pipeline complete. lemkin-clone is registered with Ollama.")
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
    _configure_logging()
    if args.once:
        try:
            ran = run_pipeline_once(force=args.force)
        except Exception:  # noqa: BLE001 - logged with traceback, then non-zero exit for schedulers
            logger.exception("Pipeline failed")
            sys.exit(1)
        sys.exit(0 if ran or not args.force else 1)

    logger.info("Watching MinIO every %ds. Ctrl-C to stop.", args.interval)
    force = args.force
    failures = 0
    while True:
        wait = args.interval
        try:
            ran = run_pipeline_once(force=force)
            force = False  # only honor --force once; subsequent loops use sentinel diff
            failures = 0
            if not ran:
                print(".", end="", flush=True)
        except KeyboardInterrupt:
            print("\nStopped.")
            return
        except Exception:  # noqa: BLE001
            # A failed run is retried, but with backoff: an out-of-memory fine-tune would
            # otherwise restart every poll and never stop.
            failures += 1
            wait = min(args.interval * 2 ** failures, MAX_BACKOFF_SECONDS)
            logger.exception(
                "Pipeline error (%d in a row). Will retry in %ds.", failures, wait
            )
        time.sleep(wait)


if __name__ == "__main__":
    main()
