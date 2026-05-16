"""Import-order guard for ArcticDB / libarrow AWS SDK symbol collision.

This module must be imported BEFORE any module that imports pyarrow.fs
or instantiates pyarrow's S3FileSystem. Importing arcticdb first ensures
its statically-linked AWS SDK claims the dynamic symbols before libarrow's
vendored aws-c-common loads.

Verified failure mode if reversed (Phase 0 spike, 2026-05-16):
    Fatal error condition occurred in aws-c-common/source/allocator.c:202:
    allocator != ((void*)0)
    → process abort (exit code 134)

This module is a no-op when arcticdb is not installed.
"""
from __future__ import annotations

try:
    import arcticdb  # noqa: F401  side effect: claim AWS SDK symbols
except ImportError:
    # arcticdb is optional — zipline core works without it.
    pass
