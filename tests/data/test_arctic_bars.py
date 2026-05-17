"""Tests for ArcticDailyBarReader / ArcticMinuteBarReader.

ABC + pure-function tests run anywhere.
Reader behavior tests depend on the ``arctic_uri`` fixture (conftest.py),
which skips when the local MinIO binary is missing.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from zipline.data.arctic_bars import (
    ArcticDailyBarReader,
    ArcticMinuteBarReader,
    _ChunkedPreloadCache,
    _FullPreloadCache,
    estimate_preload_size_mb,
)
from zipline.data.bar_reader import NoDataOnDate
from zipline.data.bcolz_minute_bars import MinuteBarReader
from zipline.data.session_bars import CurrencyAwareSessionBarReader


# ──────────────────────────────────────────────────────────────────────────────
# Sample data generators (synthetic — no external data dependency)
# ──────────────────────────────────────────────────────────────────────────────

def _make_daily_df(start: str = "2024-01-02", end: str = "2024-01-31", seed: int = 0):
    rng = np.random.default_rng(seed)
    idx = pd.date_range(start, end, freq="B")
    n = len(idx)
    return pd.DataFrame(
        {
            "open":   rng.uniform(100, 200, n),
            "high":   rng.uniform(200, 300, n),
            "low":    rng.uniform(50, 100, n),
            "close":  rng.uniform(100, 200, n),
            "volume": rng.uniform(1e6, 1e7, n),
        },
        index=idx,
    )


def _make_minute_df(
    start: str = "2024-01-02 00:00",
    end: str = "2024-01-02 23:59",
    seed: int = 0,
):
    rng = np.random.default_rng(seed)
    idx = pd.date_range(start, end, freq="1min")
    n = len(idx)
    return pd.DataFrame(
        {
            "open":   rng.uniform(100, 200, n),
            "high":   rng.uniform(200, 300, n),
            "low":    rng.uniform(50, 100, n),
            "close":  rng.uniform(100, 200, n),
            "volume": rng.uniform(1e6, 1e7, n),
        },
        index=idx,
    )


def _setup_lib(
    arctic_uri: str,
    symbols_data: dict,
    calendar: str,
    bar_frequency: str,
    library_name: str = "bars",
):
    """Write per-symbol DataFrames + bundle metadata. Returns (lib, sid_to_symbol)."""
    from arcticdb import Arctic

    ac = Arctic(arctic_uri)
    lib = ac.get_library(library_name, create_if_missing=True)
    sid_to_symbol = {}
    for sid, (sym, df) in enumerate(symbols_data.items()):
        lib.write(sym, df)
        sid_to_symbol[sid] = sym
    all_idx = pd.concat(list(symbols_data.values())).index
    lib.write_metadata(
        "__bundle__",
        {
            "calendar_name": calendar,
            "start_session_ns": int(pd.Timestamp(all_idx.min()).value),
            "end_session_ns": int(pd.Timestamp(all_idx.max()).value),
            "bar_frequency": bar_frequency,
        },
    )
    return lib, sid_to_symbol


# ──────────────────────────────────────────────────────────────────────────────
# ABC compliance + pure-function tests  (no MinIO required)
# ──────────────────────────────────────────────────────────────────────────────

def test_arctic_daily_abc_compliance():
    assert issubclass(ArcticDailyBarReader, CurrencyAwareSessionBarReader)


def test_arctic_minute_abc_compliance():
    assert issubclass(ArcticMinuteBarReader, MinuteBarReader)


def test_data_frequency_daily_is_property():
    """data_frequency must be a property defined on the daily subclass."""
    attr = ArcticDailyBarReader.__dict__.get("data_frequency")
    assert isinstance(attr, property), "data_frequency should be a property"


def test_data_frequency_minute_is_property():
    attr = ArcticMinuteBarReader.__dict__.get("data_frequency")
    assert isinstance(attr, property), "data_frequency should be a property"


def test_estimate_preload_size_mb_small():
    """10 syms × 1y minute × 5 cols ≈ 252 MB."""
    mb = estimate_preload_size_mb(
        n_symbols=10,
        start_dt=pd.Timestamp("2024-01-01"),
        end_dt=pd.Timestamp("2024-12-31"),
        n_columns=5,
        freq="minute",
    )
    assert 200 < mb < 400, f"expected ~252 MB, got {mb:.1f}"


def test_estimate_preload_size_mb_large():
    """1000 syms × 5y minute × 5 cols → ≥ 100_000 MB."""
    mb = estimate_preload_size_mb(
        n_symbols=1000,
        start_dt=pd.Timestamp("2020-01-01"),
        end_dt=pd.Timestamp("2024-12-31"),
        n_columns=5,
        freq="minute",
    )
    assert mb > 100_000, f"expected > 100 GB, got {mb:.1f}"


def test_estimate_preload_size_mb_daily_small():
    """100 syms × 1y daily × 5 cols should be tiny (well under 10 MB)."""
    mb = estimate_preload_size_mb(
        n_symbols=100,
        start_dt=pd.Timestamp("2024-01-01"),
        end_dt=pd.Timestamp("2024-12-31"),
        n_columns=5,
        freq="session",
    )
    assert mb < 10, f"expected < 10 MB for daily small workload, got {mb:.1f}"


# ──────────────────────────────────────────────────────────────────────────────
# Reader behavior tests  (require MinIO via arctic_uri fixture)
# ──────────────────────────────────────────────────────────────────────────────

def test_daily_reader_get_value_round_trip(arctic_uri):
    df = _make_daily_df()
    _, sid_to_symbol = _setup_lib(arctic_uri, {"AAPL": df}, "XNYS", "daily")

    reader = ArcticDailyBarReader(
        arctic_uri, "bars", "XNYS", sid_to_symbol=sid_to_symbol,
    )

    dt = df.index[5]
    expected = float(df.loc[dt, "close"])
    result = reader.get_value(0, dt, "close")
    assert abs(result - expected) < 1e-6


def test_minute_reader_get_value_round_trip(arctic_uri):
    df = _make_minute_df()
    _, sid_to_symbol = _setup_lib(arctic_uri, {"AAPL": df}, "24/7", "minute")

    reader = ArcticMinuteBarReader(
        arctic_uri, "bars", "24/7", sid_to_symbol=sid_to_symbol,
    )

    dt = df.index[10]
    expected = float(df.loc[dt, "close"])
    result = reader.get_value(0, dt, "close")
    assert abs(result - expected) < 1e-6


def test_get_value_unknown_sid_returns_nan(arctic_uri):
    df = _make_daily_df()
    _, sid_to_symbol = _setup_lib(arctic_uri, {"AAPL": df}, "XNYS", "daily")
    reader = ArcticDailyBarReader(
        arctic_uri, "bars", "XNYS", sid_to_symbol=sid_to_symbol,
    )
    # sid 999 is not in sid_to_symbol → falls back to str(999) which won't exist in lib
    result = reader.get_value(999, df.index[0], "close")
    assert np.isnan(result)


def test_get_value_missing_field_returns_nan(arctic_uri):
    df = _make_daily_df()
    _, sid_to_symbol = _setup_lib(arctic_uri, {"AAPL": df}, "XNYS", "daily")
    reader = ArcticDailyBarReader(
        arctic_uri, "bars", "XNYS", sid_to_symbol=sid_to_symbol,
    )
    result = reader.get_value(0, df.index[0], "nonexistent_field")
    assert np.isnan(result)


def test_get_last_traded_dt_finds_non_zero_volume(arctic_uri):
    df = _make_daily_df()
    df = df.copy()
    # Zero out the volume for the last 3 rows
    df.iloc[-3:, df.columns.get_loc("volume")] = 0.0
    expected_last = df.index[-4]

    _, sid_to_symbol = _setup_lib(arctic_uri, {"AAPL": df}, "XNYS", "daily")
    reader = ArcticDailyBarReader(
        arctic_uri, "bars", "XNYS", sid_to_symbol=sid_to_symbol,
    )

    result = reader.get_last_traded_dt(0, df.index[-1])
    # Compare wall-clock day; reader localizes index to UTC, expected is tz-naive
    assert result.tz_localize(None) == expected_last


def test_get_last_traded_dt_returns_nat_when_no_data(arctic_uri):
    df = _make_daily_df()
    _, sid_to_symbol = _setup_lib(arctic_uri, {"AAPL": df}, "XNYS", "daily")
    reader = ArcticDailyBarReader(
        arctic_uri, "bars", "XNYS", sid_to_symbol=sid_to_symbol,
    )
    # sid 999 → unknown symbol → read fails → NaT
    result = reader.get_last_traded_dt(999, df.index[-1])
    assert pd.isnull(result)


def test_sessions_property_handles_tz_aware_metadata(arctic_uri):
    """sessions must tolerate tz-aware first_trading_day/last_available_dt by stripping tz
    before passing to exchange_calendars (4.6+ rejects tz-aware).
    """
    df = _make_daily_df(start="2024-01-02", end="2024-01-31")
    _, sid_to_symbol = _setup_lib(arctic_uri, {"AAPL": df}, "XNYS", "daily")
    reader = ArcticDailyBarReader(
        arctic_uri, "bars", "XNYS", sid_to_symbol=sid_to_symbol,
    )
    # Should not raise — the regression was AttributeError on
    # 'datetime.timezone' object has no attribute 'key'.
    sessions = reader.sessions
    assert len(sessions) > 0


def test_currency_codes_returns_usd(arctic_uri):
    df = _make_daily_df()
    _, sid_to_symbol = _setup_lib(
        arctic_uri, {"AAPL": df, "MSFT": df, "GOOG": df}, "XNYS", "daily",
    )
    reader = ArcticDailyBarReader(
        arctic_uri, "bars", "XNYS", sid_to_symbol=sid_to_symbol,
    )
    result = reader.currency_codes([0, 1, 2])
    assert list(result) == ["USD", "USD", "USD"]


def test_load_raw_arrays_shape(arctic_uri):
    df0 = _make_daily_df(seed=0)
    df1 = _make_daily_df(seed=1)
    _, sid_to_symbol = _setup_lib(
        arctic_uri, {"AAPL": df0, "MSFT": df1}, "XNYS", "daily",
    )
    reader = ArcticDailyBarReader(
        arctic_uri, "bars", "XNYS", sid_to_symbol=sid_to_symbol,
    )

    start_dt = df0.index[2]
    end_dt = df0.index[7]
    n_periods = sum(1 for d in df0.index if start_dt <= d <= end_dt)

    arrays = reader.load_raw_arrays(["close", "volume"], start_dt, end_dt, [0, 1])
    assert len(arrays) == 2
    for arr in arrays:
        assert arr.shape == (n_periods, 2)


def test_load_raw_arrays_values_match(arctic_uri):
    df = _make_daily_df(seed=42)
    _, sid_to_symbol = _setup_lib(arctic_uri, {"AAPL": df}, "XNYS", "daily")
    reader = ArcticDailyBarReader(
        arctic_uri, "bars", "XNYS", sid_to_symbol=sid_to_symbol,
    )

    start_dt = df.index[0]
    end_dt = df.index[4]
    mask = (df.index >= start_dt) & (df.index <= end_dt)
    expected_close = df.loc[mask, "close"].values

    arrays = reader.load_raw_arrays(["close"], start_dt, end_dt, [0])
    np.testing.assert_allclose(arrays[0][:, 0], expected_close, rtol=1e-6)


# ──────────────────────────────────────────────────────────────────────────────
# Cache behavior tests
# ──────────────────────────────────────────────────────────────────────────────

def test_prepare_for_backtest_full_preload_dispatch(arctic_uri):
    df = _make_daily_df()
    _, sid_to_symbol = _setup_lib(arctic_uri, {"AAPL": df}, "XNYS", "daily")
    reader = ArcticDailyBarReader(
        arctic_uri, "bars", "XNYS", sid_to_symbol=sid_to_symbol,
    )
    reader.prepare_for_backtest(
        df.index[0], df.index[-1], [0], columns=["close", "volume"],
        full_threshold_mb=10_000,
        chunked_threshold_mb=10_000,
    )
    assert isinstance(reader._cache, _FullPreloadCache)


def test_prepare_for_backtest_chunked_dispatch(arctic_uri):
    """Force chunked by setting full_threshold below the workload's estimated size.

    1 sym × ~22 business days × 1 col ≈ 0.0004 MB. full_threshold=0.0001 forces
    skip-full; chunked_threshold=1e9 keeps it in chunked range.
    """
    df = _make_daily_df()
    _, sid_to_symbol = _setup_lib(arctic_uri, {"AAPL": df}, "XNYS", "daily")
    reader = ArcticDailyBarReader(
        arctic_uri, "bars", "XNYS", sid_to_symbol=sid_to_symbol,
    )
    reader.prepare_for_backtest(
        df.index[0], df.index[-1], [0], columns=["close"],
        full_threshold_mb=0.0001,
        chunked_threshold_mb=1e9,
    )
    assert isinstance(reader._cache, _ChunkedPreloadCache)


def test_prepare_for_backtest_raw_dispatch(arctic_uri):
    """Force raw mode (no cache) by setting both thresholds below the estimate."""
    df = _make_daily_df()
    _, sid_to_symbol = _setup_lib(arctic_uri, {"AAPL": df}, "XNYS", "daily")
    reader = ArcticDailyBarReader(
        arctic_uri, "bars", "XNYS", sid_to_symbol=sid_to_symbol,
    )
    reader.prepare_for_backtest(
        df.index[0], df.index[-1], [0], columns=["close"],
        full_threshold_mb=0.00001,
        chunked_threshold_mb=0.0001,
    )
    assert reader._cache is None


def test_get_value_uses_cache_after_prepare(arctic_uri):
    df = _make_daily_df(seed=7)
    _, sid_to_symbol = _setup_lib(arctic_uri, {"AAPL": df}, "XNYS", "daily")
    reader = ArcticDailyBarReader(
        arctic_uri, "bars", "XNYS", sid_to_symbol=sid_to_symbol,
    )

    dt = df.index[3]
    expected = float(df.loc[dt, "close"])

    reader.prepare_for_backtest(
        df.index[0], df.index[-1], [0], columns=["close"],
        full_threshold_mb=10_000,
        chunked_threshold_mb=10_000,
    )
    assert reader._cache is not None

    result = reader.get_value(0, dt, "close")
    assert abs(result - expected) < 1e-6


def test_get_value_out_of_window_after_preload_raises(arctic_uri):
    """After preload of a narrow window, accessing a dt outside it must raise NoDataOnDate."""
    df = _make_daily_df(start="2024-01-02", end="2024-01-31")
    _, sid_to_symbol = _setup_lib(arctic_uri, {"AAPL": df}, "XNYS", "daily")
    reader = ArcticDailyBarReader(
        arctic_uri, "bars", "XNYS", sid_to_symbol=sid_to_symbol,
    )

    # Preload only the first 5 business days
    preload_end = df.index[4]
    reader.prepare_for_backtest(
        df.index[0], preload_end, [0], columns=["close"],
        full_threshold_mb=10_000,
        chunked_threshold_mb=10_000,
    )
    assert isinstance(reader._cache, _FullPreloadCache)

    out_of_window_dt = df.index[10]
    with pytest.raises(NoDataOnDate):
        reader.get_value(0, out_of_window_dt, "close")


def test_get_value_pre_listing_returns_nan_within_window(arctic_uri):
    """Pre-IPO dt inside the preload window must return NaN, not raise.

    Regression for the case where a newly-listed asset's data starts mid-window:
    bcolz returns NaN for missing rows within the bundle range, so the cache
    contract must match — otherwise any backtest touching an IPO crashes.
    """
    # AAPL "lists" on Jan 15, but the preload window is Jan 2 – Jan 31.
    full_range = _make_daily_df(start="2024-01-02", end="2024-01-31")
    df = full_range.loc["2024-01-15":]
    _, sid_to_symbol = _setup_lib(arctic_uri, {"AAPL": df}, "XNYS", "daily")
    reader = ArcticDailyBarReader(
        arctic_uri, "bars", "XNYS", sid_to_symbol=sid_to_symbol,
    )

    window_start = pd.Timestamp("2024-01-02", tz="UTC")
    window_end = pd.Timestamp("2024-01-31", tz="UTC")
    reader.prepare_for_backtest(
        window_start, window_end, [0], columns=["close"],
        full_threshold_mb=10_000,
        chunked_threshold_mb=10_000,
    )

    pre_listing_dt = pd.Timestamp("2024-01-03", tz="UTC")  # in window, pre-IPO
    assert np.isnan(reader.get_value(0, pre_listing_dt, "close"))

    # Sanity: a real listed day still returns a value.
    listed_dt = df.index[0]
    assert not np.isnan(reader.get_value(0, listed_dt, "close"))

    # Sanity: out-of-window still raises.
    with pytest.raises(NoDataOnDate):
        reader.get_value(0, pd.Timestamp("2024-02-15", tz="UTC"), "close")


def test_load_raw_arrays_raises_on_uncached_column(arctic_uri):
    """If load_raw_arrays asks for a column that prepare_for_backtest didn't
    cache, raise ValueError — silently returning all-NaN is indistinguishable
    from a legitimate missing-data result.
    """
    df = _make_daily_df()
    _, sid_to_symbol = _setup_lib(arctic_uri, {"AAPL": df}, "XNYS", "daily")
    reader = ArcticDailyBarReader(
        arctic_uri, "bars", "XNYS", sid_to_symbol=sid_to_symbol,
    )
    reader.prepare_for_backtest(
        df.index[0], df.index[-1], [0], columns=["close"],
        full_threshold_mb=10_000,
        chunked_threshold_mb=10_000,
    )

    with pytest.raises(ValueError, match="not preloaded"):
        reader.load_raw_arrays(
            ["close", "volume"], df.index[0], df.index[-1], [0],
        )


def test_clear_cache_resets_cache(arctic_uri):
    df = _make_daily_df()
    _, sid_to_symbol = _setup_lib(arctic_uri, {"AAPL": df}, "XNYS", "daily")
    reader = ArcticDailyBarReader(
        arctic_uri, "bars", "XNYS", sid_to_symbol=sid_to_symbol,
    )
    reader.prepare_for_backtest(
        df.index[0], df.index[-1], [0], columns=["close"],
        full_threshold_mb=10_000,
        chunked_threshold_mb=10_000,
    )
    assert reader._cache is not None

    reader.clear_cache()
    assert reader._cache is None


def test_load_raw_arrays_uses_cache_after_prepare(arctic_uri):
    """load_raw_arrays must hit the in-memory _FullPreloadCache when one exists
    (cache-aware fast path) and return numerically identical results to the
    raw Arctic path.
    """
    df0 = _make_daily_df(seed=11)
    df1 = _make_daily_df(seed=12)
    _, sid_to_symbol = _setup_lib(
        arctic_uri, {"AAPL": df0, "MSFT": df1}, "XNYS", "daily",
    )
    reader = ArcticDailyBarReader(
        arctic_uri, "bars", "XNYS", sid_to_symbol=sid_to_symbol,
    )

    start = df0.index[0]
    end = df0.index[-1]

    raw_arrays = reader.load_raw_arrays(["close", "volume"], start, end, [0, 1])

    reader.prepare_for_backtest(
        start, end, [0, 1], columns=["open", "high", "low", "close", "volume"],
        full_threshold_mb=10_000,
        chunked_threshold_mb=10_000,
    )
    assert isinstance(reader._cache, _FullPreloadCache)
    cached_arrays = reader.load_raw_arrays(["close", "volume"], start, end, [0, 1])

    assert len(raw_arrays) == len(cached_arrays) == 2
    for raw, cached in zip(raw_arrays, cached_arrays):
        assert raw.shape == cached.shape
        np.testing.assert_allclose(raw, cached, rtol=1e-9, equal_nan=True)


def test_chunked_cache_rollover(arctic_uri):
    """5 months of data with chunked dispatch — accessing far-apart dates should
    advance the window's `_current_start`.
    """
    df = _make_daily_df(start="2024-01-02", end="2024-05-31")
    _, sid_to_symbol = _setup_lib(arctic_uri, {"AAPL": df}, "XNYS", "daily")
    reader = ArcticDailyBarReader(
        arctic_uri, "bars", "XNYS", sid_to_symbol=sid_to_symbol,
    )
    reader.prepare_for_backtest(
        df.index[0], df.index[-1], [0], columns=["close"],
        full_threshold_mb=0.0001,    # force chunked (5mo × 1col ≈ 0.002 MB)
        chunked_threshold_mb=1e9,
    )
    cache = reader._cache
    assert isinstance(cache, _ChunkedPreloadCache)

    dt_jan = df.index[5]   # early January
    reader.get_value(0, dt_jan, "close")
    chunk_start_jan = cache._current_start

    # Default window is 3 months: Jan → covers Jan/Feb/Mar.
    # May falls outside that window, triggering a reload.
    dt_may = df.index[-5]   # late May
    reader.get_value(0, dt_may, "close")
    chunk_start_may = cache._current_start

    assert chunk_start_may > chunk_start_jan
