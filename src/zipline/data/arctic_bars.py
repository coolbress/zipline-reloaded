"""
ArcticDB-backed bar readers for Zipline.

Public API
----------
ArcticDailyBarReader   – implements CurrencyAwareSessionBarReader
ArcticMinuteBarReader  – implements MinuteBarReader
estimate_preload_size_mb – helper for capacity planning

arcticdb is an *optional* dependency — imported lazily inside methods so that
zipline's core functionality remains available without it.
"""
from __future__ import annotations

# Import-order guard: arcticdb must be registered before pyarrow is loaded.
from zipline.data import _arctic_init  # noqa: F401

import logging
import time
from typing import Optional

import numpy as np
import pandas as pd

from zipline.data.bar_reader import NoDataOnDate, NoDataBeforeDate, NoDataAfterDate
from zipline.data.session_bars import CurrencyAwareSessionBarReader
from zipline.data.bcolz_minute_bars import MinuteBarReader
from zipline.utils.calendar_utils import get_calendar
from zipline.utils.memoize import lazyval

log = logging.getLogger(__name__)

__all__ = [
    "ArcticDailyBarReader",
    "ArcticMinuteBarReader",
    "estimate_preload_size_mb",
]

# ---------------------------------------------------------------------------
# Cache size thresholds
# ---------------------------------------------------------------------------

_DEFAULT_FULL_THRESHOLD_MB: float = 2_000.0
_DEFAULT_CHUNKED_THRESHOLD_MB: float = 16_000.0

_DEFAULT_COLUMNS = ["open", "high", "low", "close", "volume"]


# ---------------------------------------------------------------------------
# Public helper
# ---------------------------------------------------------------------------


def estimate_preload_size_mb(
    n_symbols: int,
    start_dt: pd.Timestamp,
    end_dt: pd.Timestamp,
    n_columns: int,
    freq: str,  # "session" or "minute"
) -> float:
    """Estimate RAM required to preload a date-range for *n_symbols* symbols.

    The formula counts one float64 per (period, column) plus one int64 for the
    timestamp — (1 + n_columns) * 8 bytes per period.

    Parameters
    ----------
    n_symbols : int
    start_dt, end_dt : pd.Timestamp
    n_columns : int
    freq : str
        ``"session"`` for daily bars, ``"minute"`` for minute bars.

    Returns
    -------
    float
        Estimated megabytes (base-10).
    """
    days = (pd.Timestamp(end_dt) - pd.Timestamp(start_dt)).days + 1
    n_periods = days * (1440 if freq == "minute" else 1)
    bytes_per_period = (1 + n_columns) * 8  # ts_ns int64 + data cols float64
    return n_symbols * n_periods * bytes_per_period / 1e6


# ---------------------------------------------------------------------------
# Cache implementations (private)
# ---------------------------------------------------------------------------


class _FullPreloadCache:
    """In-memory cache for a fixed date window (≤ _DEFAULT_FULL_THRESHOLD_MB).

    Layout per symbol::

        _data[sym] = (ts_ns: np.ndarray[int64], cols: dict[str, np.ndarray[float64]])

    ``ts_ns`` is sorted ascending.
    """

    def __init__(self, lib, start_dt, end_dt, symbols, columns):
        from arcticdb import ReadRequest  # local — optional dep

        t0 = time.perf_counter()

        start_ts = pd.Timestamp(start_dt)
        end_ts = pd.Timestamp(end_dt)
        if start_ts.tzinfo is None:
            start_ts = start_ts.tz_localize("UTC")
        if end_ts.tzinfo is None:
            end_ts = end_ts.tz_localize("UTC")

        requests = [
            ReadRequest(sym, date_range=(start_ts, end_ts), columns=columns)
            for sym in symbols
        ]

        results = lib.read_batch(requests)

        self._data: dict[str, tuple[np.ndarray, dict[str, np.ndarray]]] = {}
        total_bytes = 0

        for sym, res in zip(symbols, results):
            if isinstance(res, Exception):
                log.debug("_FullPreloadCache: skipping %s — %s", sym, res)
                continue
            df: pd.DataFrame = res.data
            if df is None or df.empty:
                continue
            # Normalise index to UTC
            idx = df.index
            if idx.tzinfo is None:
                idx = idx.tz_localize("UTC")
            ts_ns = idx.astype(np.int64).to_numpy()
            sort_order = np.argsort(ts_ns)
            ts_ns = ts_ns[sort_order]
            col_arrays: dict[str, np.ndarray] = {}
            for col in columns:
                if col in df.columns:
                    arr = df[col].to_numpy(dtype=np.float64, na_value=np.nan)
                    col_arrays[col] = arr[sort_order]
                    total_bytes += arr.nbytes
            total_bytes += ts_ns.nbytes
            self._data[sym] = (ts_ns, col_arrays)

        self.load_ms = (time.perf_counter() - t0) * 1000.0
        self.cache_mb = total_bytes / 1e6
        log.debug(
            "_FullPreloadCache: loaded %d symbols in %.1f ms (%.1f MB)",
            len(self._data),
            self.load_ms,
            self.cache_mb,
        )

    def get_value(self, sym: str, dt, field: str) -> float:
        if sym not in self._data:
            return float("nan")
        ts_ns, col_arrays = self._data[sym]
        if field not in col_arrays:
            return float("nan")
        dt_ns = int(pd.Timestamp(dt).value)
        idx = np.searchsorted(ts_ns, dt_ns)
        if idx >= len(ts_ns) or ts_ns[idx] != dt_ns:
            raise NoDataOnDate(f"No data for sym={sym} dt={dt}")
        return float(col_arrays[field][idx])


class _ChunkedPreloadCache:
    """Sliding-window cache that loads one chunk at a time (≤ _DEFAULT_CHUNKED_THRESHOLD_MB).

    Lazy: no data is fetched at ``__init__``.  A new chunk is loaded whenever
    the requested ``dt`` falls outside the current window.

    The window start is aligned to the first day of the month containing ``dt``;
    the window end is ``window_months`` months later.
    """

    def __init__(self, lib, symbols, columns, window_months: int = 3):
        self._lib = lib
        self._symbols = symbols
        self._columns = columns
        self._window = pd.DateOffset(months=window_months)
        self._current: Optional[_FullPreloadCache] = None
        self._current_start: Optional[pd.Timestamp] = None
        self._current_end: Optional[pd.Timestamp] = None

    def _ensure_chunk_for(self, dt) -> None:
        ts = pd.Timestamp(dt)
        if ts.tzinfo is None:
            ts = ts.tz_localize("UTC")
        if (
            self._current is not None
            and self._current_start is not None
            and self._current_end is not None
            and self._current_start <= ts <= self._current_end
        ):
            return  # cache hit

        # Align chunk start to first of month
        chunk_start = ts.normalize().replace(day=1)
        chunk_end = chunk_start + self._window

        log.debug(
            "_ChunkedPreloadCache: loading chunk %s – %s", chunk_start, chunk_end
        )
        self._current = _FullPreloadCache(
            self._lib, chunk_start, chunk_end, self._symbols, self._columns
        )
        self._current_start = chunk_start
        self._current_end = chunk_end

    def get_value(self, sym: str, dt, field: str) -> float:
        self._ensure_chunk_for(dt)
        return self._current.get_value(sym, dt, field)  # type: ignore[union-attr]


# ---------------------------------------------------------------------------
# Shared reader implementation (private)
# ---------------------------------------------------------------------------


class _ArcticReaderImpl:
    """Frequency-agnostic ArcticDB reader logic.

    Concrete subclasses must set ``data_frequency`` and inherit from the
    appropriate zipline ABC (``CurrencyAwareSessionBarReader`` /
    ``MinuteBarReader``).

    Parameters
    ----------
    arctic_uri : str
        ArcticDB URI, e.g. ``"lmdb:///tmp/mylib"`` or ``"s3://bucket"``.
    library_name : str
        Name of the ArcticDB library that holds bar data.
    calendar_name : str
        Trading-calendar name (e.g. ``"XNYS"``).  May be overridden by bundle
        metadata stored at the ``"__bundle__"`` key.
    asset_finder : optional
        Zipline ``AssetFinder`` instance — used for sid→symbol lookup when
        ``sid_to_symbol`` is not provided.
    sid_to_symbol : dict[int, str] | None
        Explicit mapping from integer sid to ArcticDB symbol string.
    start_session_ns : int | None
        Fallback first-session timestamp (nanoseconds since epoch, UTC).
    end_session_ns : int | None
        Fallback last-session timestamp (nanoseconds since epoch, UTC).
    """

    def __init__(
        self,
        arctic_uri: str,
        library_name: str,
        calendar_name: str,
        *,
        asset_finder=None,
        sid_to_symbol: Optional[dict] = None,
        start_session_ns: Optional[int] = None,
        end_session_ns: Optional[int] = None,
    ):
        from arcticdb import Arctic  # local — optional dep

        self._arctic = Arctic(arctic_uri)
        self._lib = self._arctic.get_library(library_name)
        self._library_name = library_name

        # Resolve bundle metadata — prefer what's stored in the library.
        meta = self._read_bundle_meta()

        self._calendar_name: str = meta.get("calendar_name", calendar_name)

        start_ns = meta.get("start_session_ns", start_session_ns)
        end_ns = meta.get("end_session_ns", end_session_ns)

        self._start_session_ns: Optional[int] = int(start_ns) if start_ns is not None else None
        self._end_session_ns: Optional[int] = int(end_ns) if end_ns is not None else None

        self._asset_finder = asset_finder
        self._sid_to_symbol: dict = dict(sid_to_symbol) if sid_to_symbol else {}

        # Cache (set by prepare_for_backtest)
        self._cache: Optional[object] = None  # _FullPreloadCache | _ChunkedPreloadCache | None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _read_bundle_meta(self) -> dict:
        """Read ``__bundle__`` metadata from the library, return {} on miss."""
        try:
            item = self._lib.read_metadata("__bundle__")
            meta = item.metadata
            return meta if isinstance(meta, dict) else {}
        except Exception:
            return {}

    def _sym_for_sid(self, sid: int) -> str:
        """Return the ArcticDB symbol string for *sid*."""
        if sid in self._sid_to_symbol:
            return self._sid_to_symbol[sid]
        if self._asset_finder is not None:
            try:
                asset = self._asset_finder.retrieve_asset(sid)
                sym = asset.symbol
                self._sid_to_symbol[sid] = sym
                return sym
            except Exception:
                pass
        # Best-effort fallback
        return str(sid)

    def _raw_get_value(self, sym: str, dt, field: str) -> float:
        """Single-point fetch directly from ArcticDB (no cache)."""
        ts = pd.Timestamp(dt)
        if ts.tzinfo is None:
            ts = ts.tz_localize("UTC")
        try:
            item = self._lib.read(sym, date_range=(ts, ts), columns=[field])
            df: pd.DataFrame = item.data
            if df is None or df.empty:
                return float("nan")
            return float(df[field].iloc[0])
        except Exception:
            return float("nan")

    # ------------------------------------------------------------------
    # Cache management
    # ------------------------------------------------------------------

    def prepare_for_backtest(
        self,
        start_dt: pd.Timestamp,
        end_dt: pd.Timestamp,
        sids: list,
        columns: Optional[list] = None,
        *,
        full_threshold_mb: float = _DEFAULT_FULL_THRESHOLD_MB,
        chunked_threshold_mb: float = _DEFAULT_CHUNKED_THRESHOLD_MB,
    ) -> None:
        """Preload data into cache for the given window.

        Dispatches to the appropriate cache strategy based on estimated size:

        - ``< full_threshold_mb``    → :class:`_FullPreloadCache`
        - ``< chunked_threshold_mb`` → :class:`_ChunkedPreloadCache` (lazy, 3-month window)
        - ``≥ chunked_threshold_mb`` → no cache (raw mode)

        Parameters
        ----------
        start_dt, end_dt : pd.Timestamp
        sids : list[int]
        columns : list[str] | None
            Defaults to ``["open", "high", "low", "close", "volume"]``.
        full_threshold_mb, chunked_threshold_mb : float
            Override the default size thresholds.
        """
        if columns is None:
            columns = _DEFAULT_COLUMNS

        symbols = [self._sym_for_sid(s) for s in sids]
        freq = self.data_frequency  # "session" or "minute"

        est_mb = estimate_preload_size_mb(
            len(sids), start_dt, end_dt, len(columns), freq
        )
        log.info(
            "prepare_for_backtest: estimated %.1f MB for %d syms × %s – %s (%s)",
            est_mb,
            len(sids),
            start_dt,
            end_dt,
            freq,
        )

        if est_mb < full_threshold_mb:
            log.info("prepare_for_backtest: using _FullPreloadCache")
            self._cache = _FullPreloadCache(
                self._lib, start_dt, end_dt, symbols, columns
            )
        elif est_mb < chunked_threshold_mb:
            log.info("prepare_for_backtest: using _ChunkedPreloadCache (3-month window)")
            self._cache = _ChunkedPreloadCache(self._lib, symbols, columns)
        else:
            log.info(
                "prepare_for_backtest: %.1f MB exceeds threshold — raw mode (no cache)",
                est_mb,
            )
            self._cache = None

    def clear_cache(self) -> None:
        """Drop any preloaded cache, reverting to raw (per-request) mode."""
        self._cache = None

    # ------------------------------------------------------------------
    # BarReader ABC implementation
    # ------------------------------------------------------------------

    @property
    def last_available_dt(self) -> pd.Timestamp:
        if self._end_session_ns is not None:
            return pd.Timestamp(self._end_session_ns, unit="ns", tz="UTC")
        raise AttributeError(
            "last_available_dt: no end_session_ns in bundle metadata or constructor"
        )

    @property
    def first_trading_day(self) -> pd.Timestamp:
        if self._start_session_ns is not None:
            return pd.Timestamp(self._start_session_ns, unit="ns", tz="UTC")
        raise AttributeError(
            "first_trading_day: no start_session_ns in bundle metadata or constructor"
        )

    @lazyval
    def trading_calendar(self):
        return get_calendar(self._calendar_name)

    def get_value(self, sid, dt, field: str) -> float:
        sym = self._sym_for_sid(int(sid))
        if self._cache is not None:
            return self._cache.get_value(sym, dt, field)
        return self._raw_get_value(sym, dt, field)

    def get_last_traded_dt(self, asset, dt) -> pd.Timestamp:
        """Return the latest timestamp ≤ *dt* where *asset* had volume > 0.

        Returns ``pd.NaT`` if the symbol is unknown or has no qualifying rows.
        """
        sid = int(asset)
        sym = self._sym_for_sid(sid)
        ts = pd.Timestamp(dt)
        if ts.tzinfo is None:
            ts = ts.tz_localize("UTC")

        # Determine earliest reasonable start for the backward scan.
        if self._start_session_ns is not None:
            scan_start = pd.Timestamp(self._start_session_ns, unit="ns", tz="UTC")
        else:
            scan_start = ts - pd.DateOffset(years=10)

        try:
            item = self._lib.read(
                sym,
                date_range=(scan_start, ts),
                columns=["volume"],
            )
            df: pd.DataFrame = item.data
        except Exception:
            return pd.NaT

        if df is None or df.empty:
            return pd.NaT

        vol = df["volume"]
        nonzero = vol[vol > 0]
        if nonzero.empty:
            return pd.NaT

        last_idx = nonzero.index[-1]
        last_ts = pd.Timestamp(last_idx)
        if last_ts.tzinfo is None:
            last_ts = last_ts.tz_localize("UTC")
        return last_ts

    def load_raw_arrays(self, columns, start_date, end_date, assets) -> list:
        """Fetch OHLCV arrays shaped ``(n_periods, n_assets)`` from ArcticDB.

        Uses ``lib.read_batch`` for parallel symbol retrieval.  Missing symbols
        or gaps produce ``NaN`` values in the output arrays.

        Parameters
        ----------
        columns : list[str]
        start_date, end_date : pd.Timestamp
        assets : list[int]

        Returns
        -------
        list[np.ndarray]
            One array per column, shape ``(n_periods, n_assets)``, dtype float64.
        """
        from arcticdb import ReadRequest  # local — optional dep

        start_ts = pd.Timestamp(start_date)
        end_ts = pd.Timestamp(end_date)
        if start_ts.tzinfo is None:
            start_ts = start_ts.tz_localize("UTC")
        if end_ts.tzinfo is None:
            end_ts = end_ts.tz_localize("UTC")

        syms = [self._sym_for_sid(int(a)) for a in assets]

        requests = [
            ReadRequest(sym, date_range=(start_ts, end_ts), columns=list(columns))
            for sym in syms
        ]
        results = self._lib.read_batch(requests)

        # Build the union time-grid from all returned DataFrames.
        all_ts_ns: set = set()
        dfs: list = []
        for res in results:
            if isinstance(res, Exception):
                dfs.append(None)
                continue
            df: pd.DataFrame = res.data
            if df is None or df.empty:
                dfs.append(None)
                continue
            idx = df.index
            if idx.tzinfo is None:
                idx = idx.tz_localize("UTC")
            ts_ns_vals = idx.astype(np.int64)
            all_ts_ns.update(ts_ns_vals.tolist())
            # Store normalised df with UTC index
            df = df.copy()
            df.index = idx
            dfs.append(df)

        if not all_ts_ns:
            # No data at all — return empty arrays.
            return [np.empty((0, len(assets)), dtype=np.float64) for _ in columns]

        periods_ns = np.array(sorted(all_ts_ns), dtype=np.int64)
        n_periods = len(periods_ns)
        n_assets = len(assets)

        out_arrays: list = []
        for col in columns:
            out = np.full((n_periods, n_assets), np.nan, dtype=np.float64)
            for a_idx, df in enumerate(dfs):
                if df is None or col not in df.columns:
                    continue
                ts_ns_col = df.index.astype(np.int64).to_numpy()
                col_vals = df[col].to_numpy(dtype=np.float64, na_value=np.nan)
                period_indices = np.searchsorted(periods_ns, ts_ns_col)
                # Guard against out-of-range (shouldn't happen but be safe)
                valid = period_indices < n_periods
                out[period_indices[valid], a_idx] = col_vals[valid]
            out_arrays.append(out)

        return out_arrays


# ---------------------------------------------------------------------------
# Concrete public subclasses
# ---------------------------------------------------------------------------


class ArcticDailyBarReader(_ArcticReaderImpl, CurrencyAwareSessionBarReader):
    """ArcticDB-backed daily (session) bar reader.

    Implements :class:`~zipline.data.session_bars.CurrencyAwareSessionBarReader`.

    Parameters
    ----------
    arctic_uri : str
    library_name : str
    calendar_name : str
    asset_finder : optional
    sid_to_symbol : dict[int, str] | None
    start_session_ns : int | None
    end_session_ns : int | None
    """

    @property
    def data_frequency(self) -> str:
        return "session"

    @property
    def sessions(self) -> pd.DatetimeIndex:
        """All sessions covered by this bundle."""
        cal = self.trading_calendar
        # exchange_calendars 4.6+ requires tz-naive timestamps for sessions_in_range.
        first = self.first_trading_day
        last = self.last_available_dt
        if getattr(first, "tzinfo", None) is not None:
            first = first.tz_localize(None)
        if getattr(last, "tzinfo", None) is not None:
            last = last.tz_localize(None)
        return cal.sessions_in_range(first, last)

    def currency_codes(self, sids) -> np.ndarray:
        """Return ISO-4217 currency codes for *sids* (always ``"USD"``)."""
        return np.array(["USD"] * len(sids), dtype=object)


class ArcticMinuteBarReader(_ArcticReaderImpl, MinuteBarReader):
    """ArcticDB-backed minute bar reader.

    Implements :class:`~zipline.data.bcolz_minute_bars.MinuteBarReader`.

    Parameters
    ----------
    arctic_uri : str
    library_name : str
    calendar_name : str
    asset_finder : optional
    sid_to_symbol : dict[int, str] | None
    start_session_ns : int | None
    end_session_ns : int | None
    """

    @property
    def data_frequency(self) -> str:
        return "minute"
