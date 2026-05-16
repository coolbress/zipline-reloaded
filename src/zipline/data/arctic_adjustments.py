"""ArcticDB-backed adjustment reader for Zipline.

This module owns the **read** side of the adjustment pipeline. The companion
writer lives in ``fyan-collect-compute-sdk`` (`fyan.compute.adjustments`).
zipline does not write adjustment data — fyan computes and populates the
Arctic libraries during bundle build.

Library layout (5 libraries, one Arctic symbol ``"data"`` per library)::

    {library_prefix}splits          DataFrame[sid, effective_date, ratio]
    {library_prefix}mergers         DataFrame[sid, effective_date, ratio]
    {library_prefix}dividends       DataFrame[sid, effective_date, ratio]
                                    ← fyan computes ratio = 1 - amount/prev_close
    {library_prefix}dividend_payouts        DataFrame[sid, declared_date,
                                                      ex_date, record_date,
                                                      pay_date, amount]
    {library_prefix}stock_dividend_payouts  DataFrame[sid, declared_date,
                                                      ex_date, record_date,
                                                      pay_date, payment_sid,
                                                      ratio]

Date columns are stored as ``pd.Timestamp`` (datetime64[ns], tz-naive UTC).
The reader normalises to UTC and converts to int seconds where needed to
match the SQLite reader's contract (``load_adjustments_from_sqlite``).

Default ``library_prefix`` is ``"adj_"``.
"""
from __future__ import annotations

# Import-order guard: arcticdb must be registered before pyarrow is loaded.
from zipline.data import _arctic_init  # noqa: F401

import logging
from collections import namedtuple
from typing import Optional

import numpy as np
import pandas as pd

from zipline.lib.adjustment import Float64Multiply
from zipline.utils.memoize import lazyval
from zipline.utils.pandas_utils import timedelta_to_integral_seconds

log = logging.getLogger(__name__)

__all__ = ["ArcticAdjustmentReader", "Dividend", "StockDividend"]

# Mirrors zipline.data.adjustments — re-exported here so callers don't need
# to import from two modules.
Dividend = namedtuple("Dividend", ["asset", "amount", "pay_date"])
StockDividend = namedtuple(
    "StockDividend", ["asset", "payment_asset", "ratio", "pay_date"]
)

EPOCH = pd.Timestamp(0, tz="UTC")

# Table names in the SQLite contract — kept identical for parity.
_RATIO_TABLES = ("splits", "mergers", "dividends")
_PAYOUT_TABLES = ("dividend_payouts", "stock_dividend_payouts")
_ALL_TABLES = _RATIO_TABLES + _PAYOUT_TABLES

# Single Arctic symbol per library carrying the full table DataFrame.
_DATA_SYMBOL = "data"


def _to_int_seconds(ts) -> int:
    """Convert a pd.Timestamp (or naive datetime) to int seconds since epoch (UTC)."""
    t = pd.Timestamp(ts)
    if t.tzinfo is None:
        t = t.tz_localize("UTC")
    return int(t.value // 1_000_000_000)


def _column_seconds(series: pd.Series) -> np.ndarray:
    """Convert a Series of timestamps (tz-naive or tz-aware) to int64 seconds-since-epoch array."""
    if series.empty:
        return np.array([], dtype=np.int64)
    s = pd.to_datetime(series)
    if getattr(s.dt, "tz", None) is None:
        s = s.dt.tz_localize("UTC")
    return (s.view("int64") // 1_000_000_000).astype(np.int64).to_numpy()


class ArcticAdjustmentReader:
    """ArcticDB-backed reader for adjustment data.

    Drop-in replacement for ``zipline.data.adjustments.SQLiteAdjustmentReader``.
    All public methods produce the same outputs.

    Parameters
    ----------
    arctic_uri : str
        ArcticDB URI, e.g. ``"s3://127.0.0.1:fyanbkt?access=..."`` or
        ``"lmdb:///path/to/db"``.
    library_prefix : str, default "adj_"
        Prefix for the five adjustment libraries.
    """

    def __init__(self, arctic_uri: str, library_prefix: str = "adj_"):
        from arcticdb import Arctic  # local — optional dep

        self._arctic = Arctic(arctic_uri)
        self._library_prefix = library_prefix

        # Tables are loaded lazily; cache the DataFrame after first read.
        self._cache: dict[str, Optional[pd.DataFrame]] = {}

    # ------------------------------------------------------------------
    # Library access
    # ------------------------------------------------------------------

    def _read_table(self, name: str) -> pd.DataFrame:
        """Return the DataFrame for *name* (e.g. "splits"), cached after first read.

        Returns an empty DataFrame with the appropriate columns if the library
        or symbol is missing — matches SQLite reader's "no rows" semantics.
        """
        if name in self._cache:
            cached = self._cache[name]
            return cached if cached is not None else _empty_table(name)

        lib_name = f"{self._library_prefix}{name}"
        try:
            lib = self._arctic.get_library(lib_name, create_if_missing=False)
            item = lib.read(_DATA_SYMBOL)
            df = item.data
        except Exception as exc:
            log.debug("ArcticAdjustmentReader: %s not found (%s)", lib_name, exc)
            self._cache[name] = None
            return _empty_table(name)

        # Drop the index if present so columns are easier to filter; downstream
        # uses explicit columns (effective_date / ex_date / sid).
        if df.index.name is not None or not isinstance(df.index, pd.RangeIndex):
            df = df.reset_index(drop=False) if df.index.name else df.reset_index(drop=True)

        self._cache[name] = df
        return df

    def close(self) -> None:
        """No-op (Arctic doesn't require explicit close)."""
        self._cache.clear()

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        self.close()

    # ------------------------------------------------------------------
    # Public API — matches SQLiteAdjustmentReader signatures
    # ------------------------------------------------------------------

    def load_adjustments(
        self,
        dates,
        assets,
        should_include_splits: bool,
        should_include_mergers: bool,
        should_include_dividends: bool,
        adjustment_type: str,
    ) -> dict:
        """See ``SQLiteAdjustmentReader.load_adjustments``."""
        if adjustment_type not in ("price", "volume", "all"):
            raise ValueError(
                f"{adjustment_type!r} is not a valid adjustment type."
                " Valid: 'price', 'volume', 'all'."
            )

        if getattr(dates, "tz", None) is None:
            dates = dates.tz_localize("UTC")

        return _load_adjustments_from_arctic(
            self,
            dates,
            assets,
            should_include_splits,
            should_include_mergers,
            should_include_dividends,
            adjustment_type,
        )

    def load_pricing_adjustments(self, columns, dates, assets) -> list:
        """See ``SQLiteAdjustmentReader.load_pricing_adjustments``."""
        if "volume" not in set(columns):
            adjustment_type = "price"
        elif len(set(columns)) == 1:
            adjustment_type = "volume"
        else:
            adjustment_type = "all"

        adjustments = self.load_adjustments(
            dates,
            assets,
            should_include_splits=True,
            should_include_mergers=True,
            should_include_dividends=True,
            adjustment_type=adjustment_type,
        )
        price_adjustments = adjustments.get("price")
        volume_adjustments = adjustments.get("volume")

        return [
            volume_adjustments if column == "volume" else price_adjustments
            for column in columns
        ]

    def get_adjustments_for_sid(self, table_name: str, sid: int) -> list:
        """See ``SQLiteAdjustmentReader.get_adjustments_for_sid``.

        Returns list of ``[pd.Timestamp, ratio]`` pairs.
        """
        if table_name not in _RATIO_TABLES:
            raise ValueError(
                f"Unknown table {table_name!r}; expected one of {_RATIO_TABLES}"
            )
        df = self._read_table(table_name)
        if df.empty:
            return []
        rows = df[df["sid"] == int(sid)]
        out = []
        for _, row in rows.iterrows():
            ts = pd.Timestamp(row["effective_date"])
            if ts.tzinfo is None:
                ts = ts.tz_localize("UTC")
            out.append([ts, float(row["ratio"])])
        return out

    def get_dividends_with_ex_date(
        self, assets, date, asset_finder
    ) -> list:
        """See ``SQLiteAdjustmentReader.get_dividends_with_ex_date``."""
        df = self._read_table("dividend_payouts")
        if df.empty:
            return []
        ex_seconds = _to_int_seconds(date)
        row_seconds = _column_seconds(df["ex_date"])
        sid_set = set(int(a) for a in assets)
        mask = (row_seconds == ex_seconds) & df["sid"].astype(int).isin(sid_set).to_numpy()
        rows = df[mask]
        out = []
        for _, r in rows.iterrows():
            pay_ts = pd.Timestamp(r["pay_date"])
            if pay_ts.tzinfo is None:
                pay_ts = pay_ts.tz_localize("UTC")
            out.append(
                Dividend(
                    asset=asset_finder.retrieve_asset(int(r["sid"])),
                    amount=float(r["amount"]),
                    pay_date=pay_ts,
                )
            )
        return out

    def get_stock_dividends_with_ex_date(
        self, assets, date, asset_finder
    ) -> list:
        """See ``SQLiteAdjustmentReader.get_stock_dividends_with_ex_date``."""
        df = self._read_table("stock_dividend_payouts")
        if df.empty:
            return []
        ex_seconds = _to_int_seconds(date)
        row_seconds = _column_seconds(df["ex_date"])
        sid_set = set(int(a) for a in assets)
        mask = (row_seconds == ex_seconds) & df["sid"].astype(int).isin(sid_set).to_numpy()
        rows = df[mask]
        out = []
        for _, r in rows.iterrows():
            pay_ts = pd.Timestamp(r["pay_date"])
            if pay_ts.tzinfo is None:
                pay_ts = pay_ts.tz_localize("UTC")
            out.append(
                StockDividend(
                    asset=asset_finder.retrieve_asset(int(r["sid"])),
                    payment_asset=asset_finder.retrieve_asset(int(r["payment_sid"])),
                    ratio=float(r["ratio"]),
                    pay_date=pay_ts,
                )
            )
        return out

    def unpack_db_to_component_dfs(self, convert_dates: bool = False) -> dict:
        """Return DataFrames for all known tables. Mirrors SQLite reader.

        When ``convert_dates`` is True, date columns are returned as pd.Timestamp
        (already the native type in Arctic — this flag is for SQLite parity).
        """
        out = {}
        for name in _ALL_TABLES:
            df = self._read_table(name)
            if not convert_dates:
                # SQLite reader returns int seconds when convert_dates=False.
                df = df.copy()
                date_cols = _DATE_COLUMNS_BY_TABLE.get(name, ())
                for col in date_cols:
                    if col in df.columns:
                        df[col] = _column_seconds(df[col])
            out[name] = df
        return out

    def get_df_from_table(self, table_name: str, convert_dates: bool = False) -> pd.DataFrame:
        if table_name not in _ALL_TABLES:
            raise ValueError(
                f"Unknown table {table_name!r}; expected one of {_ALL_TABLES}"
            )
        df = self._read_table(table_name)
        if not convert_dates:
            df = df.copy()
            for col in _DATE_COLUMNS_BY_TABLE.get(table_name, ()):
                if col in df.columns:
                    df[col] = _column_seconds(df[col])
        return df


# ──────────────────────────────────────────────────────────────────────────────
# Hot-path implementation — Python equivalent of _adjustments.pyx
# ──────────────────────────────────────────────────────────────────────────────

_DATE_COLUMNS_BY_TABLE = {
    "splits": ("effective_date",),
    "mergers": ("effective_date",),
    "dividends": ("effective_date",),
    "dividend_payouts": ("declared_date", "ex_date", "pay_date", "record_date"),
    "stock_dividend_payouts": ("declared_date", "ex_date", "pay_date", "record_date"),
}


def _empty_table(name: str) -> pd.DataFrame:
    """Empty DataFrame with the correct schema for *name*."""
    if name in ("splits", "mergers", "dividends"):
        return pd.DataFrame(
            {
                "sid": pd.Series([], dtype="int64"),
                "effective_date": pd.Series([], dtype="datetime64[ns]"),
                "ratio": pd.Series([], dtype="float64"),
            }
        )
    if name == "dividend_payouts":
        return pd.DataFrame(
            {
                "sid": pd.Series([], dtype="int64"),
                "declared_date": pd.Series([], dtype="datetime64[ns]"),
                "ex_date": pd.Series([], dtype="datetime64[ns]"),
                "record_date": pd.Series([], dtype="datetime64[ns]"),
                "pay_date": pd.Series([], dtype="datetime64[ns]"),
                "amount": pd.Series([], dtype="float64"),
            }
        )
    if name == "stock_dividend_payouts":
        return pd.DataFrame(
            {
                "sid": pd.Series([], dtype="int64"),
                "declared_date": pd.Series([], dtype="datetime64[ns]"),
                "ex_date": pd.Series([], dtype="datetime64[ns]"),
                "record_date": pd.Series([], dtype="datetime64[ns]"),
                "pay_date": pd.Series([], dtype="datetime64[ns]"),
                "payment_sid": pd.Series([], dtype="int64"),
                "ratio": pd.Series([], dtype="float64"),
            }
        )
    raise ValueError(f"unknown adjustment table: {name!r}")


def _filtered_rows(
    df: pd.DataFrame,
    sid_set: set,
    date_col: str,
    start_seconds: int,
    end_seconds: int,
) -> list:
    """Return ``[(sid, ratio, effective_seconds), ...]`` for rows matching filters.

    Mirrors the per-table SQL query in ``_adjustments.pyx``.
    """
    if df.empty:
        return []
    date_seconds = _column_seconds(df[date_col])
    sid_arr = df["sid"].astype(np.int64).to_numpy()
    ratio_arr = df["ratio"].astype(np.float64).to_numpy()
    mask = (
        (date_seconds >= start_seconds)
        & (date_seconds <= end_seconds)
        & np.isin(sid_arr, np.fromiter(sid_set, dtype=np.int64, count=len(sid_set)))
    )
    return list(zip(sid_arr[mask].tolist(), ratio_arr[mask].tolist(), date_seconds[mask].tolist()))


def _load_adjustments_from_arctic(
    reader: ArcticAdjustmentReader,
    dates,  # pd.DatetimeIndex with tz=UTC
    assets,  # pd.Int64Index-like
    should_include_splits: bool,
    should_include_mergers: bool,
    should_include_dividends: bool,
    adjustment_type: str,
) -> dict:
    """Python equivalent of ``load_adjustments_from_sqlite`` (Cython).

    Output shape and semantics are identical:
      {'price': {date_loc: [Float64Multiply, ...]},
       'volume': {date_loc: [Float64Multiply, ...]}}

    Volume adjustments are split-derived only — mergers/dividends are
    price-only. Volume ratios are ``1.0 / split_ratio``.
    """
    should_include_price = adjustment_type in ("all", "price")
    should_include_volume = adjustment_type in ("all", "volume")

    # Mirror Cython: if not price, mergers/dividends are skipped entirely.
    if not should_include_price:
        should_include_mergers = False
        should_include_dividends = False

    start_seconds = int((dates[0] - EPOCH).total_seconds())
    end_seconds = int((dates[-1] - EPOCH).total_seconds())

    # Date-index lookup. SQLite version uses `dates.values.astype('datetime64[s]')`.
    dates_seconds = (dates.view("int64") // 1_000_000_000).astype(np.int64)
    date_ixs: dict[int, int] = {int(dt): i for i, dt in enumerate(dates_seconds)}

    # Asset-index cache (sid → location in `assets`).
    # `assets` is expected to support `.get_loc()` (Int64Index / pd.Index).
    asset_ixs: dict[int, int] = {}

    def asset_ix(sid: int) -> int:
        if sid not in asset_ixs:
            asset_ixs[sid] = assets.get_loc(sid)
        return asset_ixs[sid]

    def lookup_dt(eff_seconds: int) -> int:
        if eff_seconds in date_ixs:
            return date_ixs[eff_seconds]
        # SQLite uses searchsorted(side='right'); replicate.
        loc = int(np.searchsorted(dates_seconds, eff_seconds, side="right"))
        date_ixs[eff_seconds] = loc
        return loc

    sid_set = set(int(a) for a in assets)

    # Fetch + filter each table.
    splits = (
        _filtered_rows(
            reader._read_table("splits"), sid_set, "effective_date",
            start_seconds, end_seconds,
        )
        if should_include_splits
        else []
    )
    mergers = (
        _filtered_rows(
            reader._read_table("mergers"), sid_set, "effective_date",
            start_seconds, end_seconds,
        )
        if should_include_mergers
        else []
    )
    dividends = (
        _filtered_rows(
            reader._read_table("dividends"), sid_set, "effective_date",
            start_seconds, end_seconds,
        )
        if should_include_dividends
        else []
    )

    price_adjustments: dict[int, list] = {}
    volume_adjustments: dict[int, list] = {}

    # splits affect prices and volumes (inverse).
    for sid, ratio, eff_seconds in splits:
        if eff_seconds < start_seconds:
            continue
        date_loc = lookup_dt(eff_seconds)
        a_ix = asset_ix(sid)
        if should_include_price:
            price_adjustments.setdefault(date_loc, []).append(
                Float64Multiply(0, date_loc, a_ix, a_ix, ratio)
            )
        if should_include_volume:
            volume_adjustments.setdefault(date_loc, []).append(
                Float64Multiply(0, date_loc, a_ix, a_ix, 1.0 / ratio)
            )

    # mergers and dividends affect prices only.
    for sid, ratio, eff_seconds in mergers + dividends:
        if eff_seconds < start_seconds:
            continue
        date_loc = lookup_dt(eff_seconds)
        a_ix = asset_ix(sid)
        price_adjustments.setdefault(date_loc, []).append(
            Float64Multiply(0, date_loc, a_ix, a_ix, ratio)
        )

    result: dict = {}
    if should_include_price:
        result["price"] = price_adjustments
    if should_include_volume:
        result["volume"] = volume_adjustments
    return result
