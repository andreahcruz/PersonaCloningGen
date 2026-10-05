"""Shadow DAG for the canonical Python oracle through RAG governance.

This does not replace ``lemkin_content_pipeline``. It does not embed, and it
does not write ``lemkin_content`` or ``lemkin_train_only``.
"""

from __future__ import annotations

import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from airflow import DAG
from airflow.operators.python import PythonOperator

from host_finetune.spark_v2 import (
    STAGING_ROOT,
    assert_raw_hashes,
    iter_raw_records,
    load_balanced,
    load_cleaned,
    load_fit,
    load_nopromo,
    load_relabel,
    load_split,
    stage_balance,
    stage_clean,
    stage_fit,
    stage_nopromo,
    stage_rag,
    stage_recorded_parity,
    stage_relabel,
    stage_split,
    stage_summary,
    stage_validate,
)

DAG_ID = "lemkin_canonical_v2_shadow"
# Embedding and Chroma publication stay out of the graph.
FUTURE_STAGES = (
    "embeddings",
    "chroma_publish",
)
TASK_ORDER = (
    "validate_raw",
    "canonical_clean",
    "clean_parity",
    "build_nopromo",
    "nopromo_parity",
    "canonical_fit",
    "fit_parity",
    "balance",
    "balance_parity",
    "split",
    "split_parity",
    "relabel",
    "relabel_parity",
    "rag_governance",
    "rag_parity",
    "parity_summary",
)


def _staging(context) -> tuple[Path, str]:
    run_id = str(context.get("run_id") or "manual")
    root = Path(os.environ.get("LEMKIN_V2_STAGING", str(STAGING_ROOT)))
    return root / run_id, run_id


def _executor() -> str:
    return os.environ.get("LEMKIN_V2_CLEAN_EXECUTOR", "ordered")


def _slices() -> int:
    return int(os.environ.get("LEMKIN_V2_CLEAN_SLICES", "4"))


def validate_raw(**context):
    staging, run_id = _staging(context)
    stage_validate(staging, run_id)


def canonical_clean(**context):
    staging, run_id = _staging(context)
    assert_raw_hashes()
    stage_clean(staging, run_id, iter_raw_records(), executor=_executor(), slices=_slices())


def clean_parity(**context):
    staging, run_id = _staging(context)
    stage_recorded_parity(staging, run_id, "canonical_clean", "clean_parity")


def build_nopromo(**context):
    staging, run_id = _staging(context)
    stage_nopromo(staging, run_id, load_cleaned(staging))


def nopromo_parity(**context):
    staging, run_id = _staging(context)
    stage_recorded_parity(staging, run_id, "build_nopromo", "nopromo_parity")


def canonical_fit(**context):
    staging, run_id = _staging(context)
    stage_fit(staging, run_id, load_nopromo(staging))


def fit_parity(**context):
    staging, run_id = _staging(context)
    stage_recorded_parity(staging, run_id, "canonical_fit", "fit_parity")


def balance(**context):
    staging, run_id = _staging(context)
    stage_balance(staging, run_id, load_fit(staging))


def balance_parity(**context):
    staging, run_id = _staging(context)
    stage_recorded_parity(staging, run_id, "balance", "balance_parity")


def split(**context):
    staging, run_id = _staging(context)
    stage_split(staging, run_id, load_balanced(staging))


def split_parity(**context):
    staging, run_id = _staging(context)
    stage_recorded_parity(staging, run_id, "split", "split_parity")


def relabel(**context):
    staging, run_id = _staging(context)
    stage_relabel(staging, run_id, load_balanced(staging), load_split(staging))


def relabel_parity(**context):
    staging, run_id = _staging(context)
    stage_recorded_parity(staging, run_id, "relabel", "relabel_parity")


def rag_governance(**context):
    staging, run_id = _staging(context)
    rows, assignments = load_relabel(staging)
    stage_rag(staging, run_id, rows, assignments)


def rag_parity(**context):
    staging, run_id = _staging(context)
    stage_recorded_parity(staging, run_id, "rag_governance", "rag_parity")


def parity_summary(**context):
    staging, run_id = _staging(context)
    stage_summary(staging, run_id)


TASK_CALLABLES = {
    "validate_raw": validate_raw,
    "canonical_clean": canonical_clean,
    "clean_parity": clean_parity,
    "build_nopromo": build_nopromo,
    "nopromo_parity": nopromo_parity,
    "canonical_fit": canonical_fit,
    "fit_parity": fit_parity,
    "balance": balance,
    "balance_parity": balance_parity,
    "split": split,
    "split_parity": split_parity,
    "relabel": relabel,
    "relabel_parity": relabel_parity,
    "rag_governance": rag_governance,
    "rag_parity": rag_parity,
    "parity_summary": parity_summary,
}


with DAG(
    dag_id=DAG_ID,
    description="Shadow pipeline: raw through RAG governance, with hard parity gates.",
    start_date=datetime(2026, 1, 1),
    schedule=None,
    catchup=False,
    tags=["lemkin", "shadow", "v2"],
    default_args={
        "retries": 0,
        "retry_delay": timedelta(minutes=1),
    },
) as dag:
    previous = None
    for task_id in TASK_ORDER:
        options = {}
        if task_id in {"canonical_fit", "relabel", "split"}:
            options["execution_timeout"] = timedelta(hours=2)
        operator = PythonOperator(
            task_id=task_id,
            python_callable=TASK_CALLABLES[task_id],
            **options,
        )
        if previous is not None:
            previous >> operator
        previous = operator
