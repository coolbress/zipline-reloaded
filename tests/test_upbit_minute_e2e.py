"""
End-to-end test: synthetic Upbit 1-min data → duckdbdir ingest → bundle read → pipeline.

Stages verified:
  1. STAGING  — synthetic fyan_computed.duckdb with ohlcv_minute / assets tables
  2. INGEST   — BcolzMinuteBarWriter produces a bcolz minute bundle on disk
  3. READ     — BcolzMinuteBarReader roundtrip: sampled value matches synthetic data
  4. PIPELINE — run_algorithm (minute) iterates a full day, no NaN, all prices > 0

Run:
    /Users/coolbress/zipline-reloaded/.venv/bin/pytest tests/test_upbit_minute_e2e.py -v -s
"""

from __future__ import annotations

import datetime
import os

import duckdb
import numpy as np
import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BUNDLE_NAME = "upbit_1m_e2e"
SCHEMA_NAME = "upbit_test"
MARKETS = ["KRW-BTC", "KRW-ETH"]

# 2-day window: 2025-01-06 and 2025-01-07 (Mon + Tue — always valid for 24/7)
START_DT = pd.Timestamp("2025-01-06 00:00:00", tz="UTC")
END_DT   = pd.Timestamp("2025-01-07 23:59:00", tz="UTC")

# ---------------------------------------------------------------------------
# Synthetic data helpers
# ---------------------------------------------------------------------------


def _make_ohlcv(market: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    """Generate 1-min synthetic OHLCV rows for one market."""
    minutes = pd.date_range(start, end, freq="1min")
    n = len(minutes)
    rng = np.random.default_rng(seed=abs(hash(market)) % (2**31))
    close = 50_000.0 + rng.standard_normal(n).cumsum() * 100
    spread = np.abs(rng.standard_normal(n)) * 50 + 10.0
    return pd.DataFrame({
        "symbol": market,
        # UTC-naive TIMESTAMP — _pricing_iter normalises to tz-aware UTC
        "date": minutes.tz_localize(None),
        "open":   close - spread / 4,
        "high":   close + spread,
        "low":    close - spread,
        "close":  close,
        "volume": np.abs(rng.standard_normal(n)) * 1_000 + 5_000,
    })


def _create_fyan_computed(db_path: str) -> pd.DataFrame:
    """Write staging DB and return the combined OHLCV frame for assertions."""
    frames = [_make_ohlcv(m, START_DT, END_DT) for m in MARKETS]
    ohlcv = pd.concat(frames, ignore_index=True)

    by_sym = ohlcv.groupby("symbol")["date"]
    assets = pd.DataFrame({
        "symbol":          MARKETS,
        "exchange":        "UPBIT",
        "start_date":      [by_sym.min()[m].date() for m in MARKETS],
        "end_date":        [by_sym.max()[m].date() for m in MARKETS],
        "auto_close_date": [(by_sym.max()[m] + datetime.timedelta(days=1)).date()
                            for m in MARKETS],
    })

    conn = duckdb.connect(db_path)
    conn.execute(f"CREATE SCHEMA {SCHEMA_NAME}")
    conn.execute(f"CREATE TABLE {SCHEMA_NAME}.ohlcv_minute AS SELECT * FROM ohlcv")
    conn.execute(f"CREATE TABLE {SCHEMA_NAME}.assets      AS SELECT * FROM assets")
    for tbl in ("splits", "mergers"):
        conn.execute(f"""
            CREATE TABLE {SCHEMA_NAME}.{tbl} (
                symbol VARCHAR, effective_date DATE, ratio DOUBLE
            )""")
    conn.execute(f"""
        CREATE TABLE {SCHEMA_NAME}.dividends (
            symbol VARCHAR, ex_date DATE, declared_date DATE,
            record_date DATE, pay_date DATE, amount DOUBLE
        )""")
    conn.close()
    return ohlcv


# ---------------------------------------------------------------------------
# Module-scoped fixture: ingest once, share across all tests
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def bundle_env(tmp_path_factory):
    root   = tmp_path_factory.mktemp("zipline_root")
    db_path = str(root / "fyan_computed.duckdb")

    ohlcv_df = _create_fyan_computed(db_path)

    environ = {
        **os.environ,
        "ZIPLINE_ROOT":     str(root),
        "FYAN_BUNDLE_NAME": SCHEMA_NAME,
    }

    from zipline.data import bundles
    from zipline.data.bundles.duckdbdir import duckdb_equities

    bundles.register(
        BUNDLE_NAME,
        duckdb_equities(
            db_path,
            tframes=("minute",),
            minute_table="ohlcv_minute",
        ),
        calendar_name="24/7",
        minutes_per_day=1440,
    )

    print(f"\n[INGEST] registering bundle '{BUNDLE_NAME}' (24/7, 1440 min/day) …")
    bundles.ingest(BUNDLE_NAME, environ=environ, show_progress=True)
    print(f"[INGEST] done → {root / 'data' / BUNDLE_NAME}")

    return {"environ": environ, "ohlcv_df": ohlcv_df, "root": root}


# ---------------------------------------------------------------------------
# Test 1: bundle files on disk
# ---------------------------------------------------------------------------


def test_ingest_creates_bundle_dir(bundle_env):
    """Ingest must create a timestamped bundle directory under ZIPLINE_ROOT/data."""
    bundle_dir = bundle_env["root"] / "data" / BUNDLE_NAME
    assert bundle_dir.exists(), f"Bundle dir missing: {bundle_dir}"
    ts_dirs = [d for d in bundle_dir.iterdir() if d.is_dir()]
    assert ts_dirs, "No timestamped ingest dir found"
    # Expect minute_equities.bcolz inside
    minute_bcolz = ts_dirs[-1] / "minute_equities.bcolz"
    assert minute_bcolz.exists(), f"minute_equities.bcolz missing at {ts_dirs[-1]}"
    print(f"\n[OK] bundle dir: {ts_dirs[-1]}")


# ---------------------------------------------------------------------------
# Test 2: BcolzMinuteBarReader roundtrip
# ---------------------------------------------------------------------------


def test_minute_reader_roundtrip(bundle_env):
    """Values read back from bcolz must match synthetic source for a sample of minutes."""
    from zipline.data import bundles

    bd = bundles.load(BUNDLE_NAME, environ=bundle_env["environ"])
    reader = bd.equity_minute_bar_reader

    # KRW-BTC is sid=0 (MARKETS sorted → KRW-BTC < KRW-ETH alphabetically)
    sid_btc = 0
    ohlcv_df = bundle_env["ohlcv_df"]
    btc_rows = (
        ohlcv_df[ohlcv_df["symbol"] == "KRW-BTC"]
        .sort_values("date")
        .reset_index(drop=True)
    )

    # Sample 5 evenly-spaced minutes across day 1
    sample_indices = [0, 287, 575, 863, 1151]   # ~every 4 hours
    for i in sample_indices:
        row    = btc_rows.iloc[i]
        dt_utc = pd.Timestamp(row["date"]).tz_localize("UTC")

        for field in ("open", "high", "low", "close", "volume"):
            expected = float(row[field])
            got      = reader.get_value(sid_btc, dt_utc, field)
            assert not np.isnan(got), f"{field} is NaN at {dt_utc}"
            # bcolz stores OHLC as scaled uint32 (×1000); allow 0.2% tolerance
            assert abs(got - expected) / max(abs(expected), 1e-9) < 0.002, (
                f"{field} mismatch at {dt_utc}: expected {expected:.4f}, got {got:.4f}"
            )
            print(f"  {dt_utc} [{field}]  expected={expected:.2f}  got={got:.2f}  ✓")


# ---------------------------------------------------------------------------
# Test 3: load_raw_arrays bulk read (both assets)
# ---------------------------------------------------------------------------


def test_load_raw_arrays_both_sids(bundle_env):
    """load_raw_arrays must return (n_minutes, 2) array for both KRW-BTC and KRW-ETH."""
    from zipline.data import bundles

    bd     = bundles.load(BUNDLE_NAME, environ=bundle_env["environ"])
    reader = bd.equity_minute_bar_reader

    # First 60 minutes of day 1
    start = pd.Timestamp("2025-01-06 00:00:00", tz="UTC")
    end   = pd.Timestamp("2025-01-06 00:59:00", tz="UTC")
    arrays = reader.load_raw_arrays(["close"], start, end, [0, 1])

    close_array = arrays[0]           # shape: (60, 2) — minutes × sids
    assert close_array.shape == (60, 2), f"Expected (60, 2), got {close_array.shape}"
    assert not np.any(np.isnan(close_array)), "NaN values in close array"
    assert np.all(close_array > 0), "Non-positive close prices"
    print(f"\n[OK] load_raw_arrays close[0:60, 0:2] — "
          f"BTC mean={close_array[:,0].mean():.0f}  "
          f"ETH mean={close_array[:,1].mean():.0f}")


# ---------------------------------------------------------------------------
# Test 4: run_algorithm (minute) — simple momentum signal
# ---------------------------------------------------------------------------


def test_run_algorithm_minute_pipeline(bundle_env):
    """run_algorithm with data_frequency='minute' must iterate 1440 minutes without error."""
    from zipline import run_algorithm
    from zipline.utils.calendar_utils import get_calendar as zget_calendar

    environ = bundle_env["environ"]

    # Zero-return benchmark over the test sessions (avoids SPY lookup failure).
    # exchange_calendars ≥4.3 with pandas 2.x: sessions are tz-naive — keep as-is.
    # run_algorithm start/end must also be tz-naive (exchange_calendars .key bug on
    # stdlib datetime.timezone objects when pandas returns datetime.timezone.utc).
    # IMPORTANT: use zipline's get_calendar (side="right") so trading_calendar matches
    # the reader's cached instance — ec.get_calendar has a different cache key.
    cal = zget_calendar("24/7")
    sessions = cal.sessions_in_range("2025-01-06", "2025-01-07")   # tz-naive index
    benchmark_returns = pd.Series(0.0, index=sessions)

    collected: dict[str, list] = {"prices": []}

    def initialize(context):
        context.btc = context.sid(0)    # KRW-BTC (sid=0 per sorted order)
        context.eth = context.sid(1)    # KRW-ETH

    def handle_data(context, data):
        btc_close = data.current(context.btc, "close")
        eth_close = data.current(context.eth, "close")
        collected["prices"].append((btc_close, eth_close))

    def analyze(context, perf):
        pass  # assertions done post-run in the test body

    # tz-naive timestamps: zipline converts to calendar timezone internally.
    # trading_calendar must be passed explicitly so run_algorithm builds the
    # DataPortal with 24/7 session minutes (1440/day) instead of defaulting
    # to NYSE (390/day).
    print(f"\n[PIPELINE] running minute algo over 2025-01-06 …")
    run_algorithm(
        start=pd.Timestamp("2025-01-06"),
        end=pd.Timestamp("2025-01-06"),
        initialize=initialize,
        handle_data=handle_data,
        analyze=analyze,
        data_frequency="minute",
        bundle=BUNDLE_NAME,
        environ=environ,
        capital_base=1_000_000,
        benchmark_returns=benchmark_returns,
        trading_calendar=cal,
    )

    prices = collected["prices"]
    assert len(prices) == 1440, f"Expected 1440 minute bars, got {len(prices)}"

    btc_prices = [p[0] for p in prices]
    eth_prices = [p[1] for p in prices]
    assert all(not np.isnan(p) for p in btc_prices), "NaN in BTC close prices"
    assert all(not np.isnan(p) for p in eth_prices), "NaN in ETH close prices"
    assert all(p > 0 for p in btc_prices), "Non-positive BTC close"
    assert all(p > 0 for p in eth_prices), "Non-positive ETH close"

    print(f"[OK] 1440 minute bars — "
          f"BTC close range [{min(btc_prices):.0f}, {max(btc_prices):.0f}] | "
          f"ETH close range [{min(eth_prices):.0f}, {max(eth_prices):.0f}]")
