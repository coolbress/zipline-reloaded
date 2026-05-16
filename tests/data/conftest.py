"""Fixtures shared by tests under tests/data/.

Provides ArcticDB fixtures backed by a local MinIO (S3-compatible) server.

Why MinIO and not LMDB
----------------------
Production target is S3 / object storage. The pixi env (conda-forge arcticdb +
pyarrow) shares ``aws-sdk-cpp`` between arcticdb_ext and libarrow, so the
``aws_mem_acquire`` NULL-allocator abort observed under PyPI wheels does not
trigger (ArcticDB GH #848, aws-sdk-cpp GH #2699).

If the local MinIO binary is missing, the session-scope fixture skips so tests
that don't need it (ABC compliance, estimator math) still run.
"""
from __future__ import annotations

import os
import random
import socket
import subprocess
import time
import uuid
from pathlib import Path

import pytest

# Path to a local MinIO server binary. Override via FYAN_MINIO_BIN env var.
MINIO_BIN = os.environ.get("FYAN_MINIO_BIN", "/tmp/minio_spike_bin/minio")
MINIO_USER = "minioadmin"
MINIO_PASS = "minioadmin"


def _free_port(lo: int = 19500, hi: int = 19999, attempts: int = 50) -> int:
    """Pick a free TCP port in the given range. Avoids collision with other dev services."""
    for _ in range(attempts):
        port = random.randint(lo, hi)
        with socket.socket() as s:
            try:
                s.bind(("127.0.0.1", port))
                return port
            except OSError:
                continue
    raise RuntimeError(f"no free port found in {lo}-{hi}")


@pytest.fixture(scope="session")
def minio_server(tmp_path_factory):
    """Spawn a MinIO subprocess on a random port for the whole test session.

    Skips cleanly if the MinIO binary isn't installed at the expected path.
    """
    if not Path(MINIO_BIN).exists():
        pytest.skip(
            f"MinIO binary not at {MINIO_BIN}; set FYAN_MINIO_BIN to override. "
            f"Skipping ArcticDB-on-S3 tests."
        )

    data_dir = tmp_path_factory.mktemp("minio_data")
    port = _free_port()

    env = {
        **os.environ,
        "MINIO_ROOT_USER": MINIO_USER,
        "MINIO_ROOT_PASSWORD": MINIO_PASS,
    }
    proc = subprocess.Popen(
        [MINIO_BIN, "server", "--quiet", "--address", f":{port}", str(data_dir)],
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    # Wait until the server is accepting connections (up to ~10s).
    ready = False
    for _ in range(100):
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.1):
                ready = True
                break
        except OSError:
            time.sleep(0.1)

    if not ready:
        proc.terminate()
        pytest.fail(f"MinIO failed to start on port {port} within 10s")

    yield port

    proc.terminate()
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()


@pytest.fixture
def arctic_uri(minio_server):
    """Create a fresh S3 bucket per test and return an ArcticDB URI for it.

    The bucket is wiped + deleted on test teardown so successive tests are
    isolated even within the same session.
    """
    import boto3

    port = minio_server
    bucket = f"test-{uuid.uuid4().hex[:8]}"

    s3 = boto3.client(
        "s3",
        endpoint_url=f"http://127.0.0.1:{port}",
        aws_access_key_id=MINIO_USER,
        aws_secret_access_key=MINIO_PASS,
        region_name="us-east-1",
    )
    s3.create_bucket(Bucket=bucket)

    uri = (
        f"s3://127.0.0.1:{bucket}"
        f"?access={MINIO_USER}&secret={MINIO_PASS}"
        f"&port={port}&ssl=False&use_virtual_addressing=False"
    )
    yield uri

    # Best-effort cleanup; ignore errors so test failures aren't masked.
    try:
        listing = s3.list_objects_v2(Bucket=bucket).get("Contents", [])
        for obj in listing:
            s3.delete_object(Bucket=bucket, Key=obj["Key"])
        s3.delete_bucket(Bucket=bucket)
    except Exception:
        pass
