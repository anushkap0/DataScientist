"""
Shared object-storage client.

Every microservice is stateless and horizontally scalable, so nothing is
written to local disk that another service needs later. Instead, all
intermediate artifacts (raw data, cleaned data, feature sets, trained
models, reports) are written to a MinIO (S3-compatible) bucket keyed by
`run_id`. This is what lets the orchestrator fan work out to N replicas
of any given service without caring which pod handles which request.
"""
import io
import json
import os
import pickle
from typing import Any

import boto3
import pandas as pd
from botocore.client import Config

MINIO_ENDPOINT = os.getenv("MINIO_ENDPOINT", "http://minio:9000")
MINIO_ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY", "minioadmin")
MINIO_SECRET_KEY = os.getenv("MINIO_SECRET_KEY", "minioadmin")
BUCKET = os.getenv("MINIO_BUCKET", "ads-artifacts")


def _client():
    return boto3.client(
        "s3",
        endpoint_url=MINIO_ENDPOINT,
        aws_access_key_id=MINIO_ACCESS_KEY,
        aws_secret_access_key=MINIO_SECRET_KEY,
        config=Config(signature_version="s3v4"),
        region_name="us-east-1",
    )


def ensure_bucket() -> None:
    c = _client()
    existing = [b["Name"] for b in c.list_buckets().get("Buckets", [])]
    if BUCKET not in existing:
        c.create_bucket(Bucket=BUCKET)


def put_bytes(key: str, data: bytes) -> str:
    ensure_bucket()
    _client().put_object(Bucket=BUCKET, Key=key, Body=data)
    return f"s3://{BUCKET}/{key}"


def get_bytes(key: str) -> bytes:
    obj = _client().get_object(Bucket=BUCKET, Key=key)
    return obj["Body"].read()


def put_json(key: str, obj: Any) -> str:
    return put_bytes(key, json.dumps(obj, default=str, indent=2).encode("utf-8"))


def get_json(key: str) -> Any:
    return json.loads(get_bytes(key).decode("utf-8"))


def put_dataframe(key: str, df: pd.DataFrame) -> str:
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    return put_bytes(key, buf.getvalue().encode("utf-8"))


def get_dataframe(key: str) -> pd.DataFrame:
    return pd.read_csv(io.BytesIO(get_bytes(key)))


def put_pickle(key: str, obj: Any) -> str:
    return put_bytes(key, pickle.dumps(obj))


def get_pickle(key: str) -> Any:
    return pickle.loads(get_bytes(key))


def run_key(run_id: str, name: str) -> str:
    """Namespaced key so every artifact from one pipeline run lives together."""
    return f"runs/{run_id}/{name}"
