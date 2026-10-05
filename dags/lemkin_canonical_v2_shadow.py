"""Shadow DAG for the canonical Python oracle through the fit stage.

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

from host_finetune.spark_v2 import STAGING_ROOT, run_shadow, stage_summary

DAG_ID = "lemkin_canonical_v2_shadow"
# Later stages stay out of the graph until their parity gates exist.
FUTURE_STAGES = (
    "balance",
    "split",
    "relabel",
    "rag_governance",
    "embeddings",
    "chroma_publish",
)


def _staging(context) -> tuple[Path, str]:
    run_id = str(context.get("run_id") or "manual")
    root = Path(os.environ.get("LEMKIN_V2_STAGING", str(STAGING_ROOT)))
    return root / run_id, run_id


def validate_raw(**context):
    staging, run_id = _staging(context)
    run_shadow(staging, run_id, through="validate_raw")


def canonical_clean(**context):
    staging, run_id = _staging(context)
    run_shadow(staging, run_id, through="canonical_clean", executor=os.environ.get("LEMKIN_V2_CLEAN_EXECUTOR", "ordered"))


def build_nopromo(**context):
    staging, run_id = _staging(context)
    run_shadow(staging, run_id, through="build_nopromo")


def canonical_fit(**context):
    staging, run_id = _staging(context)
    run_shadow(staging, run_id, through="canonical_fit")


def parity_summary(**context):
    staging, run_id = _staging(context)
    stage_summary(staging, run_id)


with DAG(
    dag_id=DAG_ID,
    description="Shadow pipeline: raw to canonical fit, with hard parity gates.",
    start_date=datetime(2026, 1, 1),
    schedule=None,
    catchup=False,
    tags=["lemkin", "shadow", "v2"],
    default_args={
        "retries": 0,
        "retry_delay": timedelta(minutes=1),
    },
) as dag:
    validate_task = PythonOperator(task_id="validate_raw", python_callable=validate_raw)
    clean_task = PythonOperator(task_id="canonical_clean", python_callable=canonical_clean)
    nopromo_task = PythonOperator(task_id="build_nopromo", python_callable=build_nopromo)
    fit_task = PythonOperator(
        task_id="canonical_fit",
        python_callable=canonical_fit,
        execution_timeout=timedelta(hours=2),
    )
    summary_task = PythonOperator(task_id="parity_summary", python_callable=parity_summary)
    validate_task >> clean_task >> nopromo_task >> fit_task >> summary_task
