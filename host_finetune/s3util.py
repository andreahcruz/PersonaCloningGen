"""Tiny boto3 helpers shared by watcher / merge / register scripts."""
from __future__ import annotations

import json
import logging
from typing import Optional

import boto3
from botocore.client import BaseClient
from botocore.config import Config

from host_finetune.config import (
    BUCKET_PROCESSED,
    MINIO_ACCESS_KEY,
    MINIO_ENDPOINT,
    MINIO_SECRET_KEY,
)

logger = logging.getLogger("lemkin.s3")


def minio_client() -> BaseClient:
    return boto3.client(
        "s3",
        endpoint_url=MINIO_ENDPOINT,
        aws_access_key_id=MINIO_ACCESS_KEY,
        aws_secret_access_key=MINIO_SECRET_KEY,
        # Retry transient MinIO errors instead of failing a multi-hour pipeline run.
        config=Config(
            retries={"max_attempts": 5, "mode": "standard"},
            connect_timeout=10,
            read_timeout=60,
        ),
    )


def head_object(s3: BaseClient, key: str, bucket: str = BUCKET_PROCESSED) -> Optional[dict]:
    try:
        return s3.head_object(Bucket=bucket, Key=key)
    except s3.exceptions.ClientError:
        return None


def read_sentinel(s3: BaseClient, key: str, bucket: str = BUCKET_PROCESSED) -> Optional[dict]:
    """Return the parsed sentinel JSON, or None if not present / unreadable."""
    try:
        body = s3.get_object(Bucket=bucket, Key=key)["Body"].read().decode("utf-8")
    except s3.exceptions.NoSuchKey:
        return None
    except s3.exceptions.ClientError as e:
        # Not "sentinel missing": credentials, permissions or bucket problems land here.
        logger.warning("Could not read s3://%s/%s: %s", bucket, key, e)
        return None
    try:
        return json.loads(body)
    except json.JSONDecodeError as e:
        logger.warning("Sentinel s3://%s/%s is not valid JSON: %s", bucket, key, e)
        return None


def download_to(s3: BaseClient, key: str, dest_path, bucket: str = BUCKET_PROCESSED) -> None:
    try:
        s3.download_file(bucket, key, str(dest_path))
    except Exception as e:  # noqa: BLE001 - re-raised with context
        raise RuntimeError(
            f"Could not download s3://{bucket}/{key}: {e}. Is MinIO up (`docker compose ps minio`) "
            "and did the Airflow DAG finish mark_training_ready?"
        ) from e
