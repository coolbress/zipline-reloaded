"""Arrow IPC per-symbol minute bar reader for Fyan compute bundles.

Reads Fyan compute orchestrator minute bundles directly — no ingest step.
Bundle layout identical to daily bundles; bar_frequency='minute' in meta.json.

per-symbol .arrow schema::

    ts_ns   : int64   UTC nanoseconds (intraday, one row per trading minute)
    open    : float64
    high    : float64
    low     : float64
    close   : float64
    volume  : float64
    <feature columns>: float64  (optional)
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.ipc as pa_ipc

from zipline.data.bar_reader import NoDataOnDate
from zipline.data.bcolz_minute_bars import MinuteBarReader
from zipline.utils.calendar_utils import get_calendar
from zipline.utils.memoize import lazyval


class ArrowMinuteBarReader(MinuteBarReader):
    """MinuteBarReader backed by per-symbol Arrow IPC files.

    Reads Fyan compute orchestrator minute bundles directly — no ingest step.
    The bundle root must contain meta.json with bar_frequency='minute'.

    Parameters
    ----------
    bundle_dir : Path-like
        Root of the Arrow bundle (contains meta.json, index.arrow, by-symbol/).
    asset_finder : zipline.assets.AssetFinder, optional
    sid_to_symbol : dict[int, str], optional
        Direct sid → symbol mapping. Takes precedence over asset_finder.
    """

    def __init__(
        self,
        bundle_dir,
        asset_finder=None,
        sid_to_symbol: Optional[dict] = None,
    ):
        self._bundle_dir = Path(bundle_dir)

        with (self._bundle_dir / "meta.json").open() as fh:
            meta = json.load(fh)

        if meta.get("status") != "committed":
            raise ValueError(
                f"Arrow bundle at {bundle_dir!r} has status={meta.get('status')!r};"
                " expected 'committed'."
            )
        if meta.get("bar_frequency") != "minute":
            raise ValueError(
                f"ArrowMinuteBarReader expects bar_frequency='minute',"
                f" got {meta.get('bar_frequency')!r}."
            )

        self._meta = meta
        self._calendar_name: str = meta["calendar_name"]
        self._start_session_ns: int = int(meta["start_session_ns"])
        self._end_session_ns: int = int(meta["end_session_ns"])

        idx_path = self._bundle_dir / "index.arrow"
        with pa.memory_map(str(idx_path)) as mmap:
            reader = pa_ipc.open_file(mmap)
            idx_tbl = reader.read_all()
        self._known_symbols: frozenset = frozenset(
            idx_tbl.column("symbol").to_pylist()
        )

        self._asset_finder = asset_finder
        self._sid_to_symbol_map: Optional[dict] = sid_to_symbol
        self._table_cache: dict = {}

    # ------------------------------------------------------------------
    # BarReader abstract implementations
    # ------------------------------------------------------------------

    @property
    def data_frequency(self):
        return "minute"

    @lazyval
    def trading_calendar(self):
        return get_calendar(self._calendar_name)

    @property
    def first_trading_day(self):
        return pd.Timestamp(self._start_session_ns)

    @lazyval
    def last_available_dt(self):
        # end_session_ns may be a last-minute timestamp (e.g. 23:59 for 24/7 bundles)
        # or midnight for daily-aligned bundles; normalize to midnight either way.
        end_session = pd.Timestamp(self._end_session_ns).normalize()
        return self.trading_calendar.session_last_minute(end_session)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _resolve_sid(self, sid: int) -> str:
        if self._sid_to_symbol_map is not None:
            sym = self._sid_to_symbol_map.get(int(sid))
            if sym is not None:
                return sym
        if self._asset_finder is not None:
            asset = self._asset_finder.retrieve_asset(int(sid))
            return asset.symbol
        raise KeyError(
            f"Cannot resolve sid {sid}: provide sid_to_symbol or asset_finder."
        )

    def _get_table(self, symbol: str) -> pa.Table:
        if symbol not in self._table_cache:
            path = self._bundle_dir / "by-symbol" / f"{symbol}.arrow"
            mmap = pa.memory_map(str(path))
            reader = pa_ipc.open_file(mmap)
            tbl = reader.read_all()
            self._table_cache[symbol] = (mmap, tbl)
        return self._table_cache[symbol][1]

    def _ts_arr(self, symbol: str) -> np.ndarray:
        return self._get_table(symbol).column("ts_ns").to_numpy()

    # ------------------------------------------------------------------
    # MinuteBarReader interface
    # ------------------------------------------------------------------

    def load_raw_arrays(self, columns, start_dt, end_dt, assets):
        """Load minute data for a timestamp/asset window.

        Parameters
        ----------
        columns : list of str
        start_dt : pd.Timestamp  (trading minute, UTC)
        end_dt   : pd.Timestamp  (trading minute, UTC)
        assets   : list of int

        Returns
        -------
        list[np.ndarray]
            One array per column, shape (n_minutes, n_assets), dtype float64.
            Minutes with no data for a given asset are NaN.
        """
        # zipline's calendar uses side="right": minutes_in_range is (start, end].
        # Subtract 1 ns from start to make it [start, end] inclusive.
        adjusted_start = start_dt - pd.Timedelta(nanoseconds=1)
        minutes = self.trading_calendar.minutes_in_range(adjusted_start, end_dt)
        n_minutes = len(minutes)
        n_assets = len(assets)
        # .asi8 is nanoseconds — matches ts_ns stored in Arrow files.
        minute_ns: np.ndarray = minutes.asi8

        results = [
            np.full((n_minutes, n_assets), np.nan, dtype=np.float64)
            for _ in columns
        ]

        for asset_idx, sid in enumerate(assets):
            try:
                symbol = self._resolve_sid(sid)
            except (KeyError, Exception):
                continue

            arrow_path = self._bundle_dir / "by-symbol" / f"{symbol}.arrow"
            if not arrow_path.exists():
                continue

            try:
                ts_arr = self._ts_arr(symbol)
                tbl = self._get_table(symbol)
            except Exception:
                continue

            if len(ts_arr) == 0:
                continue

            row_indices = np.searchsorted(ts_arr, minute_ns)
            row_indices = np.clip(row_indices, 0, len(ts_arr) - 1)
            valid_mask: np.ndarray = ts_arr[row_indices] == minute_ns

            if not valid_mask.any():
                continue

            schema_names = set(tbl.schema.names)
            for col_idx, col in enumerate(columns):
                if col not in schema_names:
                    continue
                col_arr = tbl.column(col).to_numpy(zero_copy_only=False)
                results[col_idx][valid_mask, asset_idx] = col_arr[row_indices[valid_mask]]

        return results

    def get_value(self, sid, dt, field):
        """Return scalar value for (sid, dt, field) where dt is a minute timestamp."""
        try:
            symbol = self._resolve_sid(sid)
        except (KeyError, Exception) as exc:
            raise NoDataOnDate(f"No data for sid={sid}") from exc

        ts_arr = self._ts_arr(symbol)
        dt_ns = int(pd.Timestamp(dt).value)

        idx = int(np.searchsorted(ts_arr, dt_ns))
        if idx >= len(ts_arr) or int(ts_arr[idx]) != dt_ns:
            raise NoDataOnDate(f"No data at dt={dt} for sid={sid}")

        tbl = self._get_table(symbol)
        if field not in tbl.schema.names:
            return np.nan

        value = tbl.column(field)[idx].as_py()
        return np.nan if value is None else float(value)

    def get_last_traded_dt(self, asset, dt):
        """Latest minute ≤ dt with non-zero volume, or pd.NaT."""
        try:
            sid = asset.sid if hasattr(asset, "sid") else int(asset)
            symbol = self._resolve_sid(sid)
        except (KeyError, Exception):
            return pd.NaT

        try:
            ts_arr = self._ts_arr(symbol)
            tbl = self._get_table(symbol)
        except Exception:
            return pd.NaT

        if "volume" not in tbl.schema.names or len(ts_arr) == 0:
            return pd.NaT

        vol_arr = tbl.column("volume").to_numpy(zero_copy_only=False)
        dt_ns = int(pd.Timestamp(dt).value)

        # rightmost index where ts_arr[idx] <= dt_ns
        idx = int(np.searchsorted(ts_arr, dt_ns, side="right")) - 1

        while idx >= 0:
            v = vol_arr[idx]
            if not np.isnan(v) and v != 0:
                return pd.Timestamp(int(ts_arr[idx]), unit="ns", tz="UTC")
            idx -= 1

        return pd.NaT
