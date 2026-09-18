"""Shared test setup.

The real app runs in Docker with chromadb, boto3 and Airflow installed. These tests only
exercise our own error handling, so when one of those packages is missing locally we
register a tiny stand-in module instead. The real package is always used when present.
"""
import importlib
import logging
import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT, ROOT / "spark_jobs", ROOT / "dags"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))


def _importable(name: str) -> bool:
    try:
        importlib.import_module(name)
        return True
    except ImportError:
        return False


def _module(name: str, **attrs) -> types.ModuleType:
    mod = types.ModuleType(name)
    mod.__dict__.update(attrs)
    sys.modules[name] = mod
    return mod


if not _importable("chromadb"):
    class _NotFoundError(Exception):
        pass

    class _Settings:
        def __init__(self, **kwargs):
            pass

    def _unavailable(*args, **kwargs):
        raise RuntimeError("chromadb stub: tests must monkeypatch HttpClient / PersistentClient")

    chroma = _module("chromadb", HttpClient=_unavailable, PersistentClient=_unavailable)
    chroma.config = _module(
        "chromadb.config", DEFAULT_DATABASE="default_database", DEFAULT_TENANT="default_tenant",
        Settings=_Settings,
    )
    chroma.errors = _module("chromadb.errors", NotFoundError=_NotFoundError)

if not _importable("boto3"):
    _module("boto3", client=lambda *a, **k: None)
    botocore = _module("botocore")
    botocore.client = _module("botocore.client", BaseClient=object)
    botocore.config = _module("botocore.config", Config=lambda **k: k)

if not _importable("airflow"):
    class _Op:
        def __init__(self, *args, **kwargs):
            self.kwargs = kwargs

        def __rshift__(self, other):
            return other

    class _DAG:
        def __init__(self, *args, **kwargs):
            self.kwargs = kwargs

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    airflow = _module("airflow", DAG=_DAG)
    airflow.operators = _module("airflow.operators")
    airflow.operators.python = _module("airflow.operators.python", PythonOperator=_Op)
    airflow.providers = _module("airflow.providers")
    airflow.providers.apache = _module("airflow.providers.apache")
    airflow.providers.apache.spark = _module("airflow.providers.apache.spark")
    airflow.providers.apache.spark.operators = _module("airflow.providers.apache.spark.operators")
    airflow.providers.apache.spark.operators.spark_submit = _module(
        "airflow.providers.apache.spark.operators.spark_submit", SparkSubmitOperator=_Op
    )


@pytest.fixture(autouse=True)
def _capture_lemkin_and_airflow_logs():
    """Let caplog see our loggers even if a test (or real Airflow) turned propagation off."""
    names = ("lemkin", "airflow.task")
    saved = {n: logging.getLogger(n).propagate for n in names}
    for n in names:
        logging.getLogger(n).propagate = True
    yield
    for n, value in saved.items():
        logging.getLogger(n).propagate = value
