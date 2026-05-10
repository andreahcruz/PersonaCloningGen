"""Tiny boto3 helpers shared by watcher / merge / register scripts."""
from __future__ import annotations

import json
from typing import Optional

import boto3
from botocore.client import BaseClient

from host_finetune.config import (
    BUCKET_PROCESSED,
    MINIO_ACCESS_KEY,
    MINIO_ENDPOINT,
    MINIO_SECRET_KEY,
)


def minio_client() -> BaseClient:
    return boto3.client(
        "s3",
        endpoint_url=MINIO_ENDPOINT,
        aws_access_key_id=MINIO_ACCESS_KEY,
        aws_secret_access_key=MINIO_SECRET_KEY,
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
    except s3.exceptions.ClientError:
        return None
    try:
        return json.loads(body)
    except json.JSONDecodeError:
        return None


def download_to(s3: BaseClient, key: str, dest_path, bucket: str = BUCKET_PROCESSED) -> None:
    s3.download_file(bucket, key, str(dest_path))
