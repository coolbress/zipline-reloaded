"""Protocol for adjustment readers.

Defines the structural type used everywhere an adjustment reader is
passed (``DataPortal``, ``history_loader``, ``ledger``, pipeline loaders).
The two concrete implementations
(:class:`zipline.data.adjustments.SQLiteAdjustmentReader` and
:class:`zipline.data.arctic_adjustments.ArcticAdjustmentReader`) satisfy
this Protocol structurally — no inheritance required.

Adding a third backend (e.g. Parquet, in-memory) only needs the methods
declared here.
"""
from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

import pandas as pd


@runtime_checkable
class AdjustmentReader(Protocol):
    """Structural interface for objects that read adjustment data.

    The Cython hot-path (``load_adjustments_from_sqlite`` /
    ``_load_adjustments_from_arctic``) lives behind ``load_adjustments``.
    Everything else is a per-sid or per-date convenience used by
    DataPortal, the ledger, and history loaders.
    """

    # --- Bulk loaders (Pipeline) -----------------------------------------

    def load_adjustments(
        self,
        dates: pd.DatetimeIndex,
        assets: pd.Index,
        should_include_splits: bool,
        should_include_mergers: bool,
        should_include_dividends: bool,
        adjustment_type: str,
    ) -> dict: ...

    def load_pricing_adjustments(
        self,
        columns: list[str],
        dates: pd.DatetimeIndex,
        assets: pd.Index,
    ) -> list: ...

    # --- Per-sid history queries (history_loader) ------------------------

    def get_adjustments_for_sid(
        self, table_name: str, sid: int
    ) -> list[list[Any]]: ...

    # --- Ledger queries --------------------------------------------------

    def get_dividends_with_ex_date(
        self, assets, date: pd.Timestamp, asset_finder
    ) -> list: ...

    def get_stock_dividends_with_ex_date(
        self, assets, date: pd.Timestamp, asset_finder
    ) -> list: ...

    # --- DataPortal session queries --------------------------------------

    def get_splits(
        self, assets, dt: pd.Timestamp
    ) -> list[tuple[int, float]]: ...

    def get_stock_dividends(
        self, sid: int, trading_days: pd.DatetimeIndex
    ) -> list[dict]: ...
