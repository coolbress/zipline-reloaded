"""Arrow IPC per-symbol daily bar reader for Fyan compute bundles.

Reads Fyan compute orchestrator output directly — no zipline ingest step.

Bundle layout::

    <bundle_root>/
      by-symbol/<SYM>.arrow   Arrow IPC File (pa_ipc.new_file)
      index.arrow              symbol index
      meta.json                calendar_name, start/end_session_ns, status

per-symbol .arrow schema::

    ts_ns   : int64   UTC nanoseconds (midnight, sorted, IPO-trimmed)
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

from zipline.data.bar_reader import NoDataAfterDate, NoDataBeforeDate, NoDataOnDate
from zipline.data.session_bars import CurrencyAwareSessionBarReader
from zipline.utils.calendar_utils import get_calendar
from zipline.utils.memoize import lazyval


class ArrowDailyBarReader(CurrencyAwareSessionBarReader):
    """CurrencyAwareSessionBarReader backed by per-symbol Arrow IPC files.

    Reads Fyan compute orchestrator output directly — no zipline ingest step.

    Parameters
    ----------
    bundle_dir : Path-like
        Root of the Arrow bundle (contains meta.json, index.arrow, by-symbol/).
    asset_finder : zipline.assets.AssetFinder, optional
        Used to resolve sid → symbol when sid_to_symbol is not provided.
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

        meta_path = self._bundle_dir / "meta.json"
        with meta_path.open() as fh:
            meta = json.load(fh)

        if meta.get("status") != "committed":
            raise ValueError(
                f"Arrow bundle at {bundle_dir!r} has status={meta.get('status')!r};"
                " expected 'committed'."
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

        # Cache: symbol → (MemoryMappedFile, pa.Table)
        # Keep mmap alongside table so Arrow doesn't free the underlying buffer.
        self._table_cache: dict = {}

    # ------------------------------------------------------------------
    # SessionBarReader / BarReader abstract implementations
    # ------------------------------------------------------------------

    @lazyval
    def sessions(self):
        cal = get_calendar(self._calendar_name)
        start = pd.Timestamp(self._start_session_ns)
        end = pd.Timestamp(self._end_session_ns)
        return cal.sessions_in_range(start, end)

    @lazyval
    def first_trading_day(self):
        return pd.Timestamp(self._start_session_ns)

    @lazyval
    def trading_calendar(self):
        return get_calendar(self._calendar_name)

    @property
    def last_available_dt(self):
        return self.sessions[-1]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _resolve_sid(self, sid: int) -> str:
        """Resolve numeric sid to symbol string."""
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
        """Return memory-mapped pa.Table for *symbol*, caching result."""
        if symbol not in self._table_cache:
            path = self._bundle_dir / "by-symbol" / f"{symbol}.arrow"
            mmap = pa.memory_map(str(path))
            reader = pa_ipc.open_file(mmap)
            tbl = reader.read_all()
            self._table_cache[symbol] = (mmap, tbl)
        return self._table_cache[symbol][1]

    def _ts_arr(self, symbol: str) -> np.ndarray:
        """ts_ns column as int64 numpy array (for searchsorted)."""
        return self._get_table(symbol).column("ts_ns").to_numpy()

    # ------------------------------------------------------------------
    # CurrencyAwareSessionBarReader interface
    # ------------------------------------------------------------------

    def load_raw_arrays(self, columns, start_date, end_date, assets):
        """Load data for a date/asset window.

        Returns
        -------
        list[np.ndarray]
            One array per column, shape (n_sessions, n_assets), dtype float64.
            Sessions with no data for a given asset are NaN.
        """
        try:
            start_idx = self.sessions.get_loc(start_date)
        except KeyError as exc:
            raise NoDataOnDate(start_date) from exc
        try:
            end_idx = self.sessions.get_loc(end_date)
        except KeyError as exc:
            raise NoDataOnDate(end_date) from exc

        session_slice = self.sessions[start_idx : end_idx + 1]
        n_sessions = len(session_slice)
        n_assets = len(assets)

        # UTC nanoseconds — matches ts_ns stored in Arrow files.
        session_ns: np.ndarray = session_slice.asi8

        results = [
            np.full((n_sessions, n_assets), np.nan, dtype=np.float64)
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

            row_indices = np.searchsorted(ts_arr, session_ns)
            row_indices = np.clip(row_indices, 0, len(ts_arr) - 1)
            valid_mask: np.ndarray = ts_arr[row_indices] == session_ns

            if not valid_mask.any():
                continue

            valid_row_indices = row_indices[valid_mask]
            schema_names = set(tbl.schema.names)

            for col_idx, col in enumerate(columns):
                if col not in schema_names:
                    continue
                col_arr = tbl.column(col).to_numpy(zero_copy_only=False)
                results[col_idx][valid_mask, asset_idx] = col_arr[valid_row_indices]

        return results

    def get_value(self, sid, dt, field):
        """Return scalar value for (sid, dt, field). NaN if field missing."""
        ix = self.sid_day_index(sid, dt)
        symbol = self._resolve_sid(sid)
        tbl = self._get_table(symbol)

        if field not in tbl.schema.names:
            return np.nan

        value = tbl.column(field)[ix].as_py()
        return np.nan if value is None else float(value)

    def sid_day_index(self, sid, day):
        """Per-symbol row index for (sid, day).

        Raises
        ------
        NoDataOnDate
            If *day* is not a session in this bundle's calendar.
        NoDataBeforeDate
            If *day* is before the symbol's first data row.
        NoDataAfterDate
            If *day* is after the symbol's last data row.
        """
        try:
            self.sessions.get_loc(day)
        except Exception as exc:
            raise NoDataOnDate(
                f"day={day} is outside of calendar={self._calendar_name}"
            ) from exc

        symbol = self._resolve_sid(sid)
        ts_arr = self._ts_arr(symbol)
        day_ns = int(pd.Timestamp(day).value)

        idx = int(np.searchsorted(ts_arr, day_ns))

        if idx >= len(ts_arr) or int(ts_arr[idx]) != day_ns:
            if len(ts_arr) == 0 or day_ns < int(ts_arr[0]):
                raise NoDataBeforeDate(
                    f"No data on or before day={day} for sid={sid}"
                )
            raise NoDataAfterDate(
                f"No data on or after day={day} for sid={sid}"
            )
        return idx

    def get_last_traded_dt(self, asset, day):
        """Latest session ≤ day with non-zero volume, or pd.NaT."""
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

        try:
            day_idx = self.sessions.get_loc(day)
        except KeyError:
            return pd.NaT

        while day_idx >= 0:
            search_day = self.sessions[day_idx]
            search_ns = int(search_day.value)

            row = int(np.searchsorted(ts_arr, search_ns))
            if row < len(ts_arr) and int(ts_arr[row]) == search_ns:
                v = vol_arr[row]
                if not np.isnan(v) and v != 0:
                    return search_day

            day_idx -= 1

        return pd.NaT

    def currency_codes(self, sids):
        out = []
        for sid in sids:
            try:
                symbol = self._resolve_sid(sid)
                out.append("USD" if symbol in self._known_symbols else None)
            except (KeyError, Exception):
                out.append(None)
        return np.array(out, dtype=object)
