"""
Phase 7 — Airflow DAG orchestrating the Jason Lemkin persona pipeline.

Task graph:
  [collect_blog, collect_lemkin_yt, collect_saastr_yt]
      -> spark_process -> load_duckdb -> build_persona_profile
      -> build_chromadb -> validate

Important for local Mac / laptops
----------------------------------
This DAG:
1. Sets ``max_active_tasks=1`` so only one task runs at a time.
2. Uses BashOperator + venv python so each step is a fresh, small interpreter
   instead of in-process callables inside Airflow's heavy worker.

Optional env (if .venv is not at repo root):
  export PERSONA_PIPELINE_VENV_PYTHON=/path/to/python
"""
from __future__ import annotations

import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

REPO_ROOT = str(Path(__file__).resolve().parents[2])
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from airflow import DAG
from airflow.operators.bash import BashOperator

from persona_pipeline.config.config import DB_PATH, SCHEDULE

_VENV_PY = os.environ.get("PERSONA_PIPELINE_VENV_PYTHON", str(Path(REPO_ROOT) / ".venv" / "bin" / "python"))

default_args = {
    "owner": "persona_pipeline",
    "retries": 1,
    "retry_delay": timedelta(minutes=10),
}


def _venv_cmd(script_relative: str) -> str:
    """Run a project script with PYTHONPATH=repo root and the project venv python."""
    script = str(Path(REPO_ROOT) / script_relative)
    return (
        f'export PYTHONPATH="{REPO_ROOT}" && '
        f'export PYTHONUNBUFFERED=1 && '
        f'exec "{_VENV_PY}" "{script}"'
    )


with DAG(
    dag_id="lemkin_persona_pipeline",
    default_args=default_args,
    description="Weekly pipeline: collect -> process -> profile -> vectorise Jason Lemkin content",
    schedule=SCHEDULE,
    start_date=datetime(2024, 1, 1),
    catchup=False,
    max_active_tasks=1,
    max_active_runs=1,
    tags=["persona", "lemkin", "saastr"],
) as dag:

    collect_blog = BashOperator(
        task_id="collect_blog",
        bash_command=_venv_cmd("persona_pipeline/collectors/collect_blog.py"),
    )
    collect_lemkin_yt = BashOperator(
        task_id="collect_lemkin_yt",
        bash_command=_venv_cmd("persona_pipeline/collectors/collect_lemkin_yt.py"),
    )
    collect_saastr_yt = BashOperator(
        task_id="collect_saastr_yt",
        bash_command=_venv_cmd("persona_pipeline/collectors/collect_saastr_yt.py"),
    )

    spark_process = BashOperator(
        task_id="spark_process",
        bash_command=_venv_cmd("persona_pipeline/jobs/spark_process.py"),
    )

    load_duckdb = BashOperator(
        task_id="load_duckdb",
        bash_command=_venv_cmd("persona_pipeline/jobs/load_duckdb.py"),
    )

    build_persona_profile = BashOperator(
        task_id="build_persona_profile",
        bash_command=_venv_cmd("persona_pipeline/jobs/build_persona_profile.py"),
    )

    build_chromadb = BashOperator(
        task_id="build_chromadb",
        bash_command=_venv_cmd("persona_pipeline/jobs/build_chromadb.py"),
    )

    validate = BashOperator(
        task_id="validate",
        bash_command=(
            f'export PYTHONPATH="{REPO_ROOT}" && '
            f'"{_VENV_PY}" -c "'
            f"import duckdb; "
            f"con = duckdb.connect(r'''{DB_PATH}''', read_only=True); "
            f"n = con.execute('SELECT count(*) FROM all_posts WHERE is_b2b = true').fetchone()[0]; "
            f"con.close(); "
            f"assert n >= 20, f'Only {{n}} B2B posts — expected >= 20'; "
            f"print(f'Validation passed: {{n}} B2B posts')"
            f'"'
        ),
    )

    [collect_blog, collect_lemkin_yt, collect_saastr_yt] >> spark_process >> load_duckdb
    load_duckdb >> build_persona_profile >> build_chromadb >> validate
