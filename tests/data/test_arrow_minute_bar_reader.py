"""Tests for ArrowMinuteBarReader."""
import json

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.ipc as pa_ipc
import pytest

from zipline.data.arrow_minute_bars import ArrowMinuteBarReader
from zipline.data.bar_reader import NoDataOnDate


# Two NYSE trading sessions, two minutes per session used as fixture data.
# zipline uses side="right" calendar: first trading minute = 14:31 UTC (not 14:30).
# The bar labeled 14:31 represents the 14:30–14:31 candle (right-edge convention).
_S1_OPEN = pd.Timestamp("2021-01-04 14:31", tz="UTC")
_S1_OPEN1 = pd.Timestamp("2021-01-04 14:32", tz="UTC")
_S2_OPEN = pd.Timestamp("2021-01-05 14:31", tz="UTC")
_S2_OPEN1 = pd.Timestamp("2021-01-05 14:32", tz="UTC")

_MINUTES_NS = [
    _S1_OPEN.value,
    _S1_OPEN1.value,
    _S2_OPEN.value,
    _S2_OPEN1.value,
]

_START_SESSION_NS = pd.Timestamp("2021-01-04").value
_END_SESSION_NS = pd.Timestamp("2021-01-05").value

_META = {
    "calendar_name": "XNYS",
    "start_session_ns": _START_SESSION_NS,
    "end_session_ns": _END_SESSION_NS,
    "status": "committed",
    "bar_frequency": "minute",
}


def _write_arrow(path, schema, data: dict):
    tbl = pa.table(data, schema=schema)
    with pa.OSFile(str(path), "wb") as f:
        writer = pa_ipc.new_file(f, tbl.schema)
        writer.write_table(tbl)
        writer.close()


def _make_mock_bundle(tmp_path, symbol_data: dict[str, dict], meta: dict = None):
    by_sym = tmp_path / "by-symbol"
    by_sym.mkdir()

    index_rows = {
        "symbol": [], "first_ts_ns": [], "last_ts_ns": [],
        "row_count": [], "file_mtime_ns": [],
    }

    for sym, cols in symbol_data.items():
        ts = cols["ts_ns"]
        n = len(ts)

        base_fields = [
            pa.field("ts_ns", pa.int64()),
            pa.field("open", pa.float64()),
            pa.field("high", pa.float64()),
            pa.field("low", pa.float64()),
            pa.field("close", pa.float64()),
            pa.field("volume", pa.float64()),
        ]
        data = {
            "ts_ns": pa.array(ts, pa.int64()),
            "open": pa.array(cols.get("open", [100.0] * n), pa.float64()),
            "high": pa.array(cols.get("high", [105.0] * n), pa.float64()),
            "low": pa.array(cols.get("low", [95.0] * n), pa.float64()),
            "close": pa.array(cols.get("close", [102.0] * n), pa.float64()),
            "volume": pa.array(cols.get("volume", [1_000_000.0] * n), pa.float64()),
        }
        extra_fields = []
        for key, vals in cols.items():
            if key not in {"ts_ns", "open", "high", "low", "close", "volume"}:
                data[key] = pa.array(vals, pa.float64())
                extra_fields.append(pa.field(key, pa.float64()))

        schema = pa.schema(base_fields + extra_fields)
        _write_arrow(by_sym / f"{sym}.arrow", schema, data)

        index_rows["symbol"].append(sym)
        index_rows["first_ts_ns"].append(ts[0])
        index_rows["last_ts_ns"].append(ts[-1])
        index_rows["row_count"].append(n)
        index_rows["file_mtime_ns"].append(0)

    idx_schema = pa.schema([
        pa.field("symbol", pa.utf8()),
        pa.field("first_ts_ns", pa.int64()),
        pa.field("last_ts_ns", pa.int64()),
        pa.field("row_count", pa.int64()),
        pa.field("file_mtime_ns", pa.int64()),
    ])
    _write_arrow(tmp_path / "index.arrow", idx_schema, {
        k: pa.array(v, idx_schema.field(k).type) for k, v in index_rows.items()
    })

    bundle_meta = {**_META, **(meta or {})}
    (tmp_path / "meta.json").write_text(json.dumps(bundle_meta))
    return tmp_path


@pytest.fixture
def bundle(tmp_path):
    return _make_mock_bundle(
        tmp_path,
        {
            "AAPL": {
                "ts_ns": _MINUTES_NS,
                "open": [100.0, 101.0, 102.0, 103.0],
                "close": [110.0, 111.0, 112.0, 113.0],
                "volume": [1000.0, 2000.0, 3000.0, 4000.0],
            },
            "MSFT": {
                "ts_ns": _MINUTES_NS,
                "open": [200.0, 201.0, 202.0, 203.0],
                "close": [210.0, 211.0, 212.0, 213.0],
                "volume": [500.0, 600.0, 700.0, 800.0],
            },
        },
    )


@pytest.fixture
def reader(bundle):
    return ArrowMinuteBarReader(bundle, sid_to_symbol={0: "AAPL", 1: "MSFT"})


# ------------------------------------------------------------------
# data_frequency / calendar properties
# ------------------------------------------------------------------

def test_data_frequency(reader):
    assert reader.data_frequency == "minute"


def test_first_trading_day(reader):
    assert reader.first_trading_day == pd.Timestamp(_START_SESSION_NS)


def test_last_available_dt_is_session_close(reader):
    dt = reader.last_available_dt
    assert isinstance(dt, pd.Timestamp)
    # Must be after the last data point.
    assert dt > pd.Timestamp(_END_SESSION_NS, tz="UTC")
    # Must be UTC-aware.
    assert dt.tzinfo is not None


# ------------------------------------------------------------------
# load_raw_arrays
# ------------------------------------------------------------------

def test_load_raw_arrays_shape(reader):
    result = reader.load_raw_arrays(["open", "close"], _S1_OPEN, _S1_OPEN1, [0, 1])
    assert len(result) == 2
    assert result[0].shape == (2, 2)
    assert result[1].shape == (2, 2)


def test_load_raw_arrays_values(reader):
    opens, closes = reader.load_raw_arrays(
        ["open", "close"], _S1_OPEN, _S1_OPEN1, [0, 1]
    )
    np.testing.assert_array_equal(opens[:, 0], [100.0, 101.0])   # AAPL
    np.testing.assert_array_equal(opens[:, 1], [200.0, 201.0])   # MSFT
    np.testing.assert_array_equal(closes[:, 0], [110.0, 111.0])
    np.testing.assert_array_equal(closes[:, 1], [210.0, 211.0])


def test_load_raw_arrays_cross_session(reader):
    """Range spanning two sessions: full minute count, with NaN for gap."""
    result = reader.load_raw_arrays(["open"], _S1_OPEN, _S2_OPEN1, [0])
    # S1_OPEN (14:31 Jan 4) → S2_OPEN1 (14:32 Jan 5):
    # Session 1: 14:31–21:00 = 390 min; session 2: 14:31–14:32 = 2 min → 392 total.
    # Session gap (non-trading hours) is excluded by the calendar.
    assert result[0].shape[0] == 392
    # Only 4 rows have data (the 4 mock minutes); the rest are NaN.
    assert np.sum(~np.isnan(result[0][:, 0])) == 4


def test_load_raw_arrays_missing_column(reader):
    result = reader.load_raw_arrays(["return_1d"], _S1_OPEN, _S1_OPEN1, [0, 1])
    assert result[0].shape == (2, 2)
    assert np.all(np.isnan(result[0]))


def test_load_raw_arrays_unknown_sid_nan(reader):
    result = reader.load_raw_arrays(["open"], _S1_OPEN, _S1_OPEN1, [99])
    assert np.all(np.isnan(result[0]))


def test_load_raw_arrays_partial_symbol_coverage(tmp_path):
    """AAPL has data only in session 2; session 1 rows → NaN."""
    bd = _make_mock_bundle(
        tmp_path,
        {
            "AAPL": {
                "ts_ns": [_S2_OPEN.value, _S2_OPEN1.value],
                "open": [200.0, 201.0],
            },
        },
    )
    r = ArrowMinuteBarReader(bd, sid_to_symbol={0: "AAPL"})
    (opens,) = r.load_raw_arrays(["open"], _S1_OPEN, _S2_OPEN1, [0])
    # S1_OPEN→S2_OPEN1 spans 392 trading minutes (390 in session 1, 2 in session 2).
    assert opens.shape == (392, 1)
    # All session-1 minutes (first 390 rows) are NaN.
    assert np.all(np.isnan(opens[:390, 0]))
    # Session-2 data is in the last 2 rows.
    np.testing.assert_array_equal(opens[-2:, 0], [200.0, 201.0])


# ------------------------------------------------------------------
# get_value
# ------------------------------------------------------------------

def test_get_value(reader):
    assert reader.get_value(0, _S1_OPEN, "open") == 100.0
    assert reader.get_value(1, _S1_OPEN1, "close") == 211.0


def test_get_value_missing_field_returns_nan(reader):
    assert np.isnan(reader.get_value(0, _S1_OPEN, "nonexistent"))


def test_get_value_invalid_minute_raises(reader):
    not_a_trading_minute = pd.Timestamp("2021-01-04 00:00", tz="UTC")
    with pytest.raises(NoDataOnDate):
        reader.get_value(0, not_a_trading_minute, "open")


# ------------------------------------------------------------------
# get_last_traded_dt
# ------------------------------------------------------------------

def test_get_last_traded_dt(tmp_path):
    """Returns last minute with non-zero volume at or before dt."""
    bd = _make_mock_bundle(
        tmp_path,
        {
            "AAPL": {
                "ts_ns": _MINUTES_NS,
                "volume": [1000.0, 0.0, 500.0, 0.0],
            },
        },
    )
    r = ArrowMinuteBarReader(bd, sid_to_symbol={0: "AAPL"})

    class FakeAsset:
        sid = 0

    # At _S1_OPEN1, volume=0; last traded should be _S1_OPEN.
    last = r.get_last_traded_dt(FakeAsset(), _S1_OPEN1)
    assert last == _S1_OPEN

    # At _S2_OPEN1, volume=0; last traded should be _S2_OPEN.
    last2 = r.get_last_traded_dt(FakeAsset(), _S2_OPEN1)
    assert last2 == _S2_OPEN


def test_get_last_traded_dt_all_zero_volume(tmp_path):
    bd = _make_mock_bundle(
        tmp_path,
        {"AAPL": {"ts_ns": _MINUTES_NS, "volume": [0.0, 0.0, 0.0, 0.0]}},
    )
    r = ArrowMinuteBarReader(bd, sid_to_symbol={0: "AAPL"})

    class FakeAsset:
        sid = 0

    assert r.get_last_traded_dt(FakeAsset(), _S2_OPEN1) is pd.NaT


# ------------------------------------------------------------------
# status and bar_frequency validation
# ------------------------------------------------------------------

def test_status_not_committed_raises(tmp_path):
    _make_mock_bundle(tmp_path, {"AAPL": {"ts_ns": _MINUTES_NS}},
                      meta={"status": "pending"})
    with pytest.raises(ValueError, match="committed"):
        ArrowMinuteBarReader(tmp_path, sid_to_symbol={0: "AAPL"})


def test_wrong_bar_frequency_raises(tmp_path):
    _make_mock_bundle(tmp_path, {"AAPL": {"ts_ns": _MINUTES_NS}},
                      meta={"bar_frequency": "daily"})
    with pytest.raises(ValueError, match="minute"):
        ArrowMinuteBarReader(tmp_path, sid_to_symbol={0: "AAPL"})
