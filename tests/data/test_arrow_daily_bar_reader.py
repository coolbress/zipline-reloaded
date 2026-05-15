"""Tests for ArrowDailyBarReader."""
import json

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.ipc as pa_ipc
import pytest

from zipline.data.arrow_daily_bars import ArrowDailyBarReader
from zipline.data.bar_reader import NoDataAfterDate, NoDataBeforeDate


# Three consecutive NYSE sessions used across all tests.
_SESSIONS_NS = [
    pd.Timestamp("2021-01-04").value,
    pd.Timestamp("2021-01-05").value,
    pd.Timestamp("2021-01-06").value,
]

_META = {
    "calendar_name": "XNYS",
    "start_session_ns": _SESSIONS_NS[0],
    "end_session_ns": _SESSIONS_NS[-1],
    "status": "committed",
    "bar_frequency": "daily",
}


def _write_arrow(path, schema, data: dict):
    tbl = pa.table(data, schema=schema)
    with pa.OSFile(str(path), "wb") as f:
        writer = pa_ipc.new_file(f, tbl.schema)
        writer.write_table(tbl)
        writer.close()


def _make_mock_bundle(tmp_path, symbol_data: dict[str, dict], meta: dict = None):
    """Create a minimal Arrow bundle under tmp_path.

    Parameters
    ----------
    symbol_data:
        {symbol: {"ts_ns": [...], "open": [...], "close": [...], ...}}
    meta:
        Override default _META fields.
    """
    by_sym = tmp_path / "by-symbol"
    by_sym.mkdir()

    sym_schema = pa.schema([
        pa.field("ts_ns", pa.int64()),
        pa.field("open", pa.float64()),
        pa.field("high", pa.float64()),
        pa.field("low", pa.float64()),
        pa.field("close", pa.float64()),
        pa.field("volume", pa.float64()),
    ])

    index_rows = {"symbol": [], "first_ts_ns": [], "last_ts_ns": [], "row_count": [], "file_mtime_ns": []}

    for sym, cols in symbol_data.items():
        ts = cols["ts_ns"]
        n = len(ts)
        data = {
            "ts_ns": pa.array(ts, pa.int64()),
            "open": pa.array(cols.get("open", [100.0] * n), pa.float64()),
            "high": pa.array(cols.get("high", [105.0] * n), pa.float64()),
            "low": pa.array(cols.get("low", [95.0] * n), pa.float64()),
            "close": pa.array(cols.get("close", [102.0] * n), pa.float64()),
            "volume": pa.array(cols.get("volume", [1_000_000.0] * n), pa.float64()),
        }
        # Add optional feature columns if present
        extra_schema_fields = [
            pa.field("ts_ns", pa.int64()),
            pa.field("open", pa.float64()),
            pa.field("high", pa.float64()),
            pa.field("low", pa.float64()),
            pa.field("close", pa.float64()),
            pa.field("volume", pa.float64()),
        ]
        for key, vals in cols.items():
            if key not in {"ts_ns", "open", "high", "low", "close", "volume"}:
                data[key] = pa.array(vals, pa.float64())
                extra_schema_fields.append(pa.field(key, pa.float64()))
        schema = pa.schema(extra_schema_fields)
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
                "ts_ns": _SESSIONS_NS,
                "open": [100.0, 101.0, 102.0],
                "close": [103.0, 104.0, 105.0],
            },
            "MSFT": {
                "ts_ns": _SESSIONS_NS,
                "open": [200.0, 201.0, 202.0],
                "close": [203.0, 204.0, 205.0],
            },
        },
    )


@pytest.fixture
def reader(bundle):
    return ArrowDailyBarReader(bundle, sid_to_symbol={0: "AAPL", 1: "MSFT"})


# ------------------------------------------------------------------
# sessions / calendar properties
# ------------------------------------------------------------------

def test_sessions_length(reader):
    assert len(reader.sessions) == 3


def test_first_and_last_trading_day(reader):
    assert reader.first_trading_day == pd.Timestamp(_SESSIONS_NS[0])
    assert reader.last_available_dt == pd.Timestamp(_SESSIONS_NS[-1])


# ------------------------------------------------------------------
# load_raw_arrays
# ------------------------------------------------------------------

def test_load_raw_arrays_shape(reader):
    start = pd.Timestamp(_SESSIONS_NS[0])
    end = pd.Timestamp(_SESSIONS_NS[-1])
    result = reader.load_raw_arrays(["open", "close"], start, end, [0, 1])
    assert len(result) == 2
    assert result[0].shape == (3, 2)
    assert result[1].shape == (3, 2)


def test_load_raw_arrays_values(reader):
    start = pd.Timestamp(_SESSIONS_NS[0])
    end = pd.Timestamp(_SESSIONS_NS[-1])
    opens, closes = reader.load_raw_arrays(["open", "close"], start, end, [0, 1])

    np.testing.assert_array_equal(opens[:, 0], [100.0, 101.0, 102.0])  # AAPL
    np.testing.assert_array_equal(opens[:, 1], [200.0, 201.0, 202.0])  # MSFT
    np.testing.assert_array_equal(closes[:, 0], [103.0, 104.0, 105.0])
    np.testing.assert_array_equal(closes[:, 1], [203.0, 204.0, 205.0])


def test_load_raw_arrays_missing_column(reader):
    """Requesting an absent feature column returns NaN — no exception."""
    start = pd.Timestamp(_SESSIONS_NS[0])
    end = pd.Timestamp(_SESSIONS_NS[-1])
    result = reader.load_raw_arrays(["return_1d"], start, end, [0, 1])
    assert result[0].shape == (3, 2)
    assert np.all(np.isnan(result[0]))


def test_load_raw_arrays_optional_feature(tmp_path):
    """Optional feature column present in one symbol → correct values, NaN for other."""
    bd = _make_mock_bundle(
        tmp_path,
        {
            "AAPL": {
                "ts_ns": _SESSIONS_NS,
                "open": [100.0] * 3,
                "return_1d": [0.01, 0.02, 0.03],
            },
            "MSFT": {
                "ts_ns": _SESSIONS_NS,
                "open": [200.0] * 3,
                # no return_1d
            },
        },
    )
    r = ArrowDailyBarReader(bd, sid_to_symbol={0: "AAPL", 1: "MSFT"})
    start = pd.Timestamp(_SESSIONS_NS[0])
    end = pd.Timestamp(_SESSIONS_NS[-1])
    (ret,) = r.load_raw_arrays(["return_1d"], start, end, [0, 1])

    np.testing.assert_array_almost_equal(ret[:, 0], [0.01, 0.02, 0.03])
    assert np.all(np.isnan(ret[:, 1]))


# ------------------------------------------------------------------
# get_value / sid_day_index
# ------------------------------------------------------------------

def test_get_value(reader):
    day = pd.Timestamp(_SESSIONS_NS[1])  # middle session
    assert reader.get_value(0, day, "open") == 101.0   # AAPL
    assert reader.get_value(1, day, "close") == 204.0  # MSFT


def test_get_value_missing_field_returns_nan(reader):
    day = pd.Timestamp(_SESSIONS_NS[0])
    assert np.isnan(reader.get_value(0, day, "nonexistent_field"))


def test_sid_day_index_before_data(tmp_path):
    bd = _make_mock_bundle(
        tmp_path,
        {
            "AAPL": {"ts_ns": _SESSIONS_NS[1:]},  # starts from session 1
        },
        meta={"start_session_ns": _SESSIONS_NS[0], "end_session_ns": _SESSIONS_NS[-1]},
    )
    r = ArrowDailyBarReader(bd, sid_to_symbol={0: "AAPL"})
    with pytest.raises(NoDataBeforeDate):
        r.sid_day_index(0, pd.Timestamp(_SESSIONS_NS[0]))


def test_sid_day_index_after_data(tmp_path):
    bd = _make_mock_bundle(
        tmp_path,
        {
            "AAPL": {"ts_ns": _SESSIONS_NS[:2]},  # ends at session 1
        },
        meta={"start_session_ns": _SESSIONS_NS[0], "end_session_ns": _SESSIONS_NS[-1]},
    )
    r = ArrowDailyBarReader(bd, sid_to_symbol={0: "AAPL"})
    with pytest.raises(NoDataAfterDate):
        r.sid_day_index(0, pd.Timestamp(_SESSIONS_NS[2]))


# ------------------------------------------------------------------
# get_last_traded_dt
# ------------------------------------------------------------------

def test_get_last_traded_dt(tmp_path):
    """Returns the most recent session with non-zero volume."""
    bd = _make_mock_bundle(
        tmp_path,
        {
            "AAPL": {
                "ts_ns": _SESSIONS_NS,
                "volume": [1_000.0, 0.0, 500.0],
            },
        },
    )
    r = ArrowDailyBarReader(bd, sid_to_symbol={0: "AAPL"})

    class FakeAsset:
        sid = 0

    last = r.get_last_traded_dt(FakeAsset(), pd.Timestamp(_SESSIONS_NS[1]))
    assert last == pd.Timestamp(_SESSIONS_NS[0])


# ------------------------------------------------------------------
# currency_codes
# ------------------------------------------------------------------

def test_currency_codes_known_sids(reader):
    codes = reader.currency_codes(np.array([0, 1]))
    assert list(codes) == ["USD", "USD"]


def test_currency_codes_unknown_sid(reader):
    codes = reader.currency_codes(np.array([99]))
    assert codes[0] is None


# ------------------------------------------------------------------
# status validation
# ------------------------------------------------------------------

def test_status_not_committed_raises(tmp_path):
    _make_mock_bundle(tmp_path, {"AAPL": {"ts_ns": _SESSIONS_NS}},
                      meta={"status": "pending"})
    with pytest.raises(ValueError, match="committed"):
        ArrowDailyBarReader(tmp_path, sid_to_symbol={0: "AAPL"})


# ------------------------------------------------------------------
# unknown sid → NaN (no exception)
# ------------------------------------------------------------------

def test_unknown_sid_returns_nan(reader):
    start = pd.Timestamp(_SESSIONS_NS[0])
    end = pd.Timestamp(_SESSIONS_NS[-1])
    result = reader.load_raw_arrays(["open"], start, end, [99])
    assert np.all(np.isnan(result[0]))
