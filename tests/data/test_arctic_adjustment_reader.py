"""Tests for ``ArcticAdjustmentReader``.

Strategy
--------
Seed the same synthetic data into both a SQLite database (via the existing
``SQLiteAdjustmentWriter``) and into ArcticDB libraries (via direct
``lib.write``), then assert the two readers produce **bit-exact** identical
outputs from ``load_adjustments`` and the helper methods.

The writer side does NOT live in zipline (it's owned by fyan); tests write
to Arctic directly to validate the reader.

Tests depend on the ``arctic_uri`` fixture (MinIO via conftest.py). Pure-logic
tests (the Cython-parity helper, schema validation) run regardless.
"""
from __future__ import annotations

import sqlite3
import tempfile
from collections import namedtuple
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from zipline.data.adjustments import SQLiteAdjustmentReader, SQLiteAdjustmentWriter
from zipline.data.arctic_adjustments import (
    ArcticAdjustmentReader,
    _DATA_SYMBOL,
    _load_adjustments_from_arctic,
)
from zipline.lib.adjustment import Float64Multiply


# ──────────────────────────────────────────────────────────────────────────────
# Synthetic data shared across SQLite + Arctic seeding
# ──────────────────────────────────────────────────────────────────────────────


def _make_splits() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sid": [1, 2],
            "effective_date": pd.to_datetime(["2024-01-15", "2024-03-10"]),
            "ratio": [0.5, 0.25],   # 2:1 and 4:1 splits
        }
    )


def _make_mergers() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sid": [3],
            "effective_date": pd.to_datetime(["2024-02-20"]),
            "ratio": [0.7],
        }
    )


def _make_dividend_ratios() -> pd.DataFrame:
    """Pre-computed dividend ratios (what fyan would write after calc_dividend_ratios).

    The SQLite reader's `dividends` table holds ratios (not raw amounts).
    """
    return pd.DataFrame(
        {
            "sid": [1, 1],
            "effective_date": pd.to_datetime(["2024-02-05", "2024-04-05"]),
            "ratio": [0.998, 0.997],
        }
    )


def _make_dividend_payouts() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sid": [1, 1],
            "declared_date": pd.to_datetime(["2024-01-25", "2024-03-25"]),
            "ex_date": pd.to_datetime(["2024-02-05", "2024-04-05"]),
            "record_date": pd.to_datetime(["2024-02-08", "2024-04-08"]),
            "pay_date": pd.to_datetime(["2024-02-15", "2024-04-15"]),
            "amount": [0.24, 0.25],
        }
    )


def _make_stock_dividend_payouts() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sid": [2],
            "declared_date": pd.to_datetime(["2024-02-25"]),
            "ex_date": pd.to_datetime(["2024-03-05"]),
            "record_date": pd.to_datetime(["2024-03-08"]),
            "pay_date": pd.to_datetime(["2024-03-15"]),
            "payment_sid": [99],
            "ratio": [0.05],
        }
    )


# ──────────────────────────────────────────────────────────────────────────────
# Seeders
# ──────────────────────────────────────────────────────────────────────────────


def _seed_arctic(arctic_uri: str, library_prefix: str = "adj_") -> None:
    """Write the same DataFrames into the 5 Arctic libraries."""
    from arcticdb import Arctic

    ac = Arctic(arctic_uri)
    tables = {
        "splits": _make_splits(),
        "mergers": _make_mergers(),
        "dividends": _make_dividend_ratios(),
        "dividend_payouts": _make_dividend_payouts(),
        "stock_dividend_payouts": _make_stock_dividend_payouts(),
    }
    for name, df in tables.items():
        lib = ac.get_library(f"{library_prefix}{name}", create_if_missing=True)
        lib.write(_DATA_SYMBOL, df, prune_previous_versions=True)


def _seed_sqlite(sqlite_path: Path) -> None:
    """Write the same DataFrames into a SQLite adjustments.db via raw SQL.

    Bypasses SQLiteAdjustmentWriter because it would re-compute dividend ratios
    from a bar reader; we want the SAME `dividends` table contents as Arctic.
    """
    splits = _make_splits().copy()
    splits["effective_date"] = (
        splits["effective_date"].view("int64") // 1_000_000_000
    ).astype(np.int64)

    mergers = _make_mergers().copy()
    mergers["effective_date"] = (
        mergers["effective_date"].view("int64") // 1_000_000_000
    ).astype(np.int64)

    dividends = _make_dividend_ratios().copy()
    dividends["effective_date"] = (
        dividends["effective_date"].view("int64") // 1_000_000_000
    ).astype(np.int64)

    div_payouts = _make_dividend_payouts().copy()
    for col in ("declared_date", "ex_date", "record_date", "pay_date"):
        div_payouts[col] = (div_payouts[col].view("int64") // 1_000_000_000).astype(np.int64)

    stock_div_payouts = _make_stock_dividend_payouts().copy()
    for col in ("declared_date", "ex_date", "record_date", "pay_date"):
        stock_div_payouts[col] = (
            stock_div_payouts[col].view("int64") // 1_000_000_000
        ).astype(np.int64)

    conn = sqlite3.connect(str(sqlite_path))
    try:
        splits.to_sql("splits", conn, if_exists="replace", index=False)
        mergers.to_sql("mergers", conn, if_exists="replace", index=False)
        dividends.to_sql("dividends", conn, if_exists="replace", index=False)
        div_payouts.to_sql("dividend_payouts", conn, if_exists="replace", index=False)
        stock_div_payouts.to_sql(
            "stock_dividend_payouts", conn, if_exists="replace", index=False
        )
        conn.commit()
    finally:
        conn.close()


# ──────────────────────────────────────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────────────────────────────────────


@pytest.fixture
def sqlite_db(tmp_path):
    db = tmp_path / "adjustments.db"
    _seed_sqlite(db)
    return db


@pytest.fixture
def seeded_arctic(arctic_uri):
    _seed_arctic(arctic_uri)
    return arctic_uri


# ──────────────────────────────────────────────────────────────────────────────
# Pure-logic tests (no MinIO required)
# ──────────────────────────────────────────────────────────────────────────────


def test_filtered_rows_empty():
    from zipline.data.arctic_adjustments import _empty_table, _filtered_rows

    out = _filtered_rows(_empty_table("splits"), {1, 2}, "effective_date", 0, 1)
    assert out == []


def test_empty_table_schemas():
    from zipline.data.arctic_adjustments import _empty_table

    for name in (
        "splits",
        "mergers",
        "dividends",
        "dividend_payouts",
        "stock_dividend_payouts",
    ):
        df = _empty_table(name)
        assert "sid" in df.columns


# ──────────────────────────────────────────────────────────────────────────────
# Parity tests vs SQLiteAdjustmentReader (require MinIO)
# ──────────────────────────────────────────────────────────────────────────────


def _dates_for_test() -> pd.DatetimeIndex:
    # SQLite reader calls dates.tz_localize("UTC") unconditionally, so pass tz-naive.
    return pd.date_range("2024-01-01", "2024-05-01", freq="D")


def _assets_for_test() -> pd.Index:
    return pd.Index([1, 2, 3, 99], dtype=np.int64)


def _adj_key(adj):
    """Hashable key for a Float64Multiply (matches its __richcmp__)."""
    return (adj.first_row, adj.last_row, adj.first_col, adj.last_col, adj.value)


def _adjustments_equal(a: dict, b: dict) -> bool:
    """Compare two ``load_adjustments`` outputs ignoring list order within date_loc."""
    if set(a.keys()) != set(b.keys()):
        return False
    for kind in a:
        keys_a, keys_b = set(a[kind].keys()), set(b[kind].keys())
        if keys_a != keys_b:
            return False
        for date_loc in keys_a:
            xs = sorted(_adj_key(x) for x in a[kind][date_loc])
            ys = sorted(_adj_key(y) for y in b[kind][date_loc])
            if xs != ys:
                return False
    return True


def test_load_adjustments_parity_all(sqlite_db, seeded_arctic):
    """Bit-exact parity between SQLite reader and Arctic reader for adjustment_type='all'."""
    dates = _dates_for_test()
    assets = _assets_for_test()

    sql_reader = SQLiteAdjustmentReader(str(sqlite_db))
    arc_reader = ArcticAdjustmentReader(seeded_arctic)

    sql_result = sql_reader.load_adjustments(
        dates, assets,
        should_include_splits=True,
        should_include_mergers=True,
        should_include_dividends=True,
        adjustment_type="all",
    )
    arc_result = arc_reader.load_adjustments(
        dates, assets,
        should_include_splits=True,
        should_include_mergers=True,
        should_include_dividends=True,
        adjustment_type="all",
    )
    assert _adjustments_equal(sql_result, arc_result), (
        f"SQLite={sql_result!r}\n\nArctic={arc_result!r}"
    )


def test_load_adjustments_parity_price_only(sqlite_db, seeded_arctic):
    dates = _dates_for_test()
    assets = _assets_for_test()

    sql_reader = SQLiteAdjustmentReader(str(sqlite_db))
    arc_reader = ArcticAdjustmentReader(seeded_arctic)

    for kw in [
        dict(should_include_splits=True, should_include_mergers=True,
             should_include_dividends=True, adjustment_type="price"),
        dict(should_include_splits=True, should_include_mergers=False,
             should_include_dividends=False, adjustment_type="price"),
    ]:
        sql_result = sql_reader.load_adjustments(dates, assets, **kw)
        arc_result = arc_reader.load_adjustments(dates, assets, **kw)
        assert _adjustments_equal(sql_result, arc_result), kw


def test_load_adjustments_parity_volume_only(sqlite_db, seeded_arctic):
    dates = _dates_for_test()
    assets = _assets_for_test()

    sql_reader = SQLiteAdjustmentReader(str(sqlite_db))
    arc_reader = ArcticAdjustmentReader(seeded_arctic)

    sql_result = sql_reader.load_adjustments(
        dates, assets,
        should_include_splits=True,
        should_include_mergers=True,
        should_include_dividends=True,
        adjustment_type="volume",
    )
    arc_result = arc_reader.load_adjustments(
        dates, assets,
        should_include_splits=True,
        should_include_mergers=True,
        should_include_dividends=True,
        adjustment_type="volume",
    )
    assert _adjustments_equal(sql_result, arc_result)


def test_load_pricing_adjustments_parity(sqlite_db, seeded_arctic):
    dates = _dates_for_test()
    assets = _assets_for_test()

    sql_reader = SQLiteAdjustmentReader(str(sqlite_db))
    arc_reader = ArcticAdjustmentReader(seeded_arctic)

    sql_list = sql_reader.load_pricing_adjustments(["open", "close"], dates, assets)
    arc_list = arc_reader.load_pricing_adjustments(["open", "close"], dates, assets)
    assert len(sql_list) == len(arc_list)
    # Both should produce price-only dicts for non-volume columns.
    for sql_dict, arc_dict in zip(sql_list, arc_list):
        assert _adjustments_equal({"x": sql_dict}, {"x": arc_dict})


def test_get_adjustments_for_sid_splits(sqlite_db, seeded_arctic):
    sql_reader = SQLiteAdjustmentReader(str(sqlite_db))
    arc_reader = ArcticAdjustmentReader(seeded_arctic)

    sql_rows = sql_reader.get_adjustments_for_sid("splits", 1)
    arc_rows = arc_reader.get_adjustments_for_sid("splits", 1)
    # SQLite returns [Timestamp, ratio]; sort by ts for stable comparison.
    sql_rows.sort(key=lambda r: r[0])
    arc_rows.sort(key=lambda r: r[0])
    assert len(sql_rows) == len(arc_rows) == 1
    sql_ts, sql_r = sql_rows[0]
    arc_ts, arc_r = arc_rows[0]
    # Normalise timestamps to UTC for comparison
    if sql_ts.tzinfo is None:
        sql_ts = sql_ts.tz_localize("UTC")
    if arc_ts.tzinfo is None:
        arc_ts = arc_ts.tz_localize("UTC")
    assert sql_ts == arc_ts
    assert sql_r == pytest.approx(arc_r)


def test_get_adjustments_for_sid_unknown_sid_empty(seeded_arctic):
    arc_reader = ArcticAdjustmentReader(seeded_arctic)
    assert arc_reader.get_adjustments_for_sid("splits", 9999) == []


def test_get_adjustments_for_sid_invalid_table_raises(seeded_arctic):
    arc_reader = ArcticAdjustmentReader(seeded_arctic)
    with pytest.raises(ValueError):
        arc_reader.get_adjustments_for_sid("not_a_table", 1)


def test_get_dividends_with_ex_date(seeded_arctic):
    """Compare returned Dividends against the seed data."""
    arc_reader = ArcticAdjustmentReader(seeded_arctic)

    # Lightweight mock asset_finder.
    Asset = namedtuple("Asset", ["sid", "symbol"])
    class _Finder:
        def retrieve_asset(self, sid):
            return Asset(sid=int(sid), symbol=f"SID{sid}")

    ex = pd.Timestamp("2024-02-05", tz="UTC")
    out = arc_reader.get_dividends_with_ex_date([1, 2], ex, _Finder())
    assert len(out) == 1
    assert out[0].asset.sid == 1
    assert out[0].amount == pytest.approx(0.24)
    assert out[0].pay_date == pd.Timestamp("2024-02-15", tz="UTC")


def test_get_dividends_with_ex_date_no_match(seeded_arctic):
    arc_reader = ArcticAdjustmentReader(seeded_arctic)
    class _Finder:
        def retrieve_asset(self, sid):
            raise AssertionError("should not be called")
    out = arc_reader.get_dividends_with_ex_date(
        [1, 2], pd.Timestamp("2099-01-01", tz="UTC"), _Finder()
    )
    assert out == []


def test_get_stock_dividends_with_ex_date(seeded_arctic):
    arc_reader = ArcticAdjustmentReader(seeded_arctic)
    Asset = namedtuple("Asset", ["sid", "symbol"])
    class _Finder:
        def retrieve_asset(self, sid):
            return Asset(sid=int(sid), symbol=f"SID{sid}")

    out = arc_reader.get_stock_dividends_with_ex_date(
        [2], pd.Timestamp("2024-03-05", tz="UTC"), _Finder()
    )
    assert len(out) == 1
    assert out[0].asset.sid == 2
    assert out[0].payment_asset.sid == 99
    assert out[0].ratio == pytest.approx(0.05)
    assert out[0].pay_date == pd.Timestamp("2024-03-15", tz="UTC")


def test_unpack_db_to_component_dfs(seeded_arctic):
    arc_reader = ArcticAdjustmentReader(seeded_arctic)
    dfs = arc_reader.unpack_db_to_component_dfs(convert_dates=True)
    assert set(dfs.keys()) == {
        "splits", "mergers", "dividends",
        "dividend_payouts", "stock_dividend_payouts",
    }
    assert len(dfs["splits"]) == 2
    assert len(dfs["dividend_payouts"]) == 2


def test_empty_library_returns_empty_results(arctic_uri):
    """Reader on an empty Arctic instance returns empty results (no errors)."""
    arc_reader = ArcticAdjustmentReader(arctic_uri)
    dates = _dates_for_test()
    assets = _assets_for_test()
    result = arc_reader.load_adjustments(
        dates, assets, True, True, True, "all"
    )
    assert result == {"price": {}, "volume": {}}
