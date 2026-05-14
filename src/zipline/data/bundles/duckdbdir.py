"""DuckDB-backed equities bundle ingester for Fyan.

Replaces hdfdir.py.  Reads directly from fyan_computed.duckdb staging tables
(materialised by Compute Engine R1–R6) — no intermediate file format.

Bundle = timeframe: each registered bundle covers exactly one timeframe (daily
OR minute).  Both timeframes use the same output table name ``ohlcv_features``
inside the bundle schema — the ``date`` column type (DATE vs TIMESTAMP UTC)
distinguishes them.  Mixing daily and minute in a single bundle is not supported.

Registration examples (in Fyan's engine/bundle.py via _isolation.py):

    # Daily US equities — NYSE calendar
    bundles.register(
        "sharadar_daily",
        duckdb_equities("/path/to/fyan_computed.duckdb"),
        calendar_name="NYSE",
    )

    # Minute US equities (NYSE, 390 min/day)
    bundles.register(
        "alpaca_1m",
        duckdb_equities(
            "/path/to/fyan_computed.duckdb",
            tframes=("minute",),
        ),
        calendar_name="NYSE",
        minutes_per_day=390,
    )

    # Minute crypto (Upbit) — 24/7 AlwaysOpenCalendar, 1440 min/day
    bundles.register(
        "upbit_1m",
        duckdb_equities(
            "/path/to/fyan_computed.duckdb",
            tframes=("minute",),
        ),
        calendar_name="24/7",
        minutes_per_day=1440,
    )

If duckdb_path is omitted the ingester falls back to the FYAN_COMPUTED_PATH
env-var, which the Compute Engine sets on the subprocess.  FYAN_BUNDLE_NAME
selects the staging schema (<bundle_name>.*) to read.

Staging table contract (both timeframes)
-----------------------------------------
<bundle>.ohlcv_features — columns: symbol, date, open, high, low, close, volume
                           (all float64), + optional feature cols (daily only).
  Daily:  date is DATE.
  Minute: date is TIMESTAMP UTC (UTC-naive or UTC-aware; ingester normalises both).
The canonical column name "date" is used for the time axis regardless of grain.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from typing import Final, Generator

# Set FYAN_PROFILE=1 to emit {"type":"ingest_timing",...} summary on stdout.
_PROFILE: bool = os.environ.get("FYAN_PROFILE") == "1"

import duckdb
import numpy as np
import pandas as pd
from zipline.utils.cli import maybe_show_progress

logger = logging.getLogger(__name__)

_OHLCV_COLS: frozenset[str] = frozenset(
    {"symbol", "date", "open", "high", "low", "close", "volume"}
)

_UINT32_MAX: int = int(np.iinfo(np.uint32).max)  # 4_294_967_295

# DuckDB base types that map to pandas integer dtypes (pd.api.types.is_integer_dtype → True).
_DUCKDB_INTEGER_DTYPES: frozenset[str] = frozenset({
    "TINYINT", "INT1", "SMALLINT", "INT2", "INTEGER", "INT4", "INT",
    "BIGINT", "INT8", "HUGEINT", "UBIGINT", "UINTEGER", "USMALLINT",
    "UTINYINT", "SIGNED",
})
# DuckDB base types that represent string/categorical data.
# ENUM type: DESCRIBE returns e.g. "ENUM('a','b')" → split("(")[0] = "ENUM" ✓
_DUCKDB_CATEGORICAL_DTYPES: frozenset[str] = frozenset({
    "VARCHAR", "TEXT", "CHAR", "BPCHAR", "STRING", "ENUM",
})


# ---------------------------------------------------------------------------
# Public factory
# ---------------------------------------------------------------------------


def duckdb_equities(
    duckdb_path: str | None = None,
    tframes: tuple[str, ...] = ("daily",),
):
    """Curried factory.  Returns the bound ingest callable zipline will invoke.

    Parameters
    ----------
    duckdb_path:
        Absolute path to fyan_computed.duckdb.  Falls back to
        ``os.environ["FYAN_COMPUTED_PATH"]`` if not supplied.
    tframes:
        One of ``("daily",)`` or ``("minute",)``.  A bundle covers exactly one
        timeframe.  For minute, also set ``minutes_per_day`` on
        ``bundles.register()`` to match the calendar (390 for NYSE, 1440 for 24/7).
    """

    def ingest(
        environ,
        asset_db_writer,
        minute_bar_writer,
        daily_bar_writer,
        adjustment_writer,
        calendar,
        start_session,
        end_session,
        cache,
        show_progress,
        output_dir,
    ) -> None:
        path = duckdb_path or environ.get("FYAN_COMPUTED_PATH")
        if not path:
            raise ValueError(
                "duckdb_path not provided and FYAN_COMPUTED_PATH is not set"
            )
        if not os.path.isfile(path):
            raise ValueError(
                f"fyan_computed.duckdb not found at: {path}"
            )

        bundle_name = environ.get("FYAN_BUNDLE_NAME")
        if not bundle_name:
            raise ValueError("FYAN_BUNDLE_NAME environment variable is not set")

        with duckdb.connect(path, read_only=True) as conn:
            _ingest_bundle(
                conn=conn,
                bundle_name=bundle_name,
                tframes=tframes,
                calendar=calendar,
                daily_bar_writer=daily_bar_writer,
                minute_bar_writer=minute_bar_writer,
                asset_db_writer=asset_db_writer,
                adjustment_writer=adjustment_writer,
                show_progress=show_progress,
            )

    return ingest


# ---------------------------------------------------------------------------
# Core ingestion logic
# ---------------------------------------------------------------------------


_SUPPORTED_TFRAMES: frozenset[str] = frozenset({"daily", "minute"})

_OHLCV_WRITE_COLS: Final = ("open", "high", "low", "close", "volume", "day", "id")

# Batch sizes for _pricing_iter per timeframe.
# daily:  64 symbols/query → ~4.4x speedup vs per-symbol; peak RAM ~161 MB at 50 features, 25yr history.
# minute: 1  (per-symbol)  → full-history minute bars are ~24 MB/symbol; N=8 already hits 189 MB.
_DAILY_BATCH_SIZE: Final = 64
_MINUTE_BATCH_SIZE: Final = 1


def _compute_feature_metadata(
    conn: duckdb.DuckDBPyConnection,
    schema: str,
    table: str,
    feature_cols: list[str],
) -> dict:
    """Compute feature metadata via DuckDB aggregate queries.

    Replaces the two-pass itertools.tee + BcolzDailyBarWriter._unify_feature_metadata()
    pattern.  A single aggregate query covers all numeric columns; one DISTINCT query
    runs per categorical column.  The returned dict matches _unify_feature_metadata()'s
    output format exactly so BcolzDailyBarWriter.write() is unchanged beyond accepting
    it as a parameter.

    Equivalence guarantees vs the tee / Python path:
    - is_integer: DuckDB schema enforces a single dtype per column, so the global
      dtype check is equivalent to the per-symbol AND aggregation.
    - min_value: global MIN across all rows equals min-of-per-symbol-mins.
    - outlier filter (abs(v) > UINT32_MAX for int, > UINT32_MAX/1000 for float):
      reproduced via a CASE expression; NULLs are skipped by MIN() automatically.
    - categorical unique_values: DISTINCT is equivalent to the union of per-symbol
      np.unique sets.
    """
    if not feature_cols:
        return {}

    describe_rows = conn.execute(f"DESCRIBE {schema}.{table}").fetchall()
    col_dtypes: dict[str, str] = {row[0]: row[1].upper() for row in describe_rows}

    # Pre-compute base types once; raise immediately on unsupported compound types
    # (arrays like INTEGER[], multi-word types like TIMESTAMP WITH TIME ZONE) so
    # callers get a clear error rather than silent mis-classification.
    base_types: dict[str, str] = {}
    for col in feature_cols:
        dtype = col_dtypes[col]  # KeyError → col not in DESCRIBE; caller bug, not ours
        if "[]" in dtype or (" WITH " in dtype and "TIME" in dtype):
            raise NotImplementedError(
                f"Feature column {col!r} has unsupported DuckDB type {dtype!r}. "
                "Array and timezone-qualified types are not supported as feature columns."
            )
        base_types[col] = dtype.split("(")[0].strip()

    numeric_cols: list[str] = []
    categorical_cols: list[str] = []
    for col in feature_cols:
        if base_types[col] in _DUCKDB_CATEGORICAL_DTYPES:
            categorical_cols.append(col)
        else:
            numeric_cols.append(col)

    metadata: dict[str, dict] = {}

    if numeric_cols:
        is_integer: dict[str, bool] = {
            col: base_types[col] in _DUCKDB_INTEGER_DTYPES
            for col in numeric_cols
        }
        # One aggregate query: global MIN per column with outlier filter.
        # CASE … END returns NULL for out-of-range values; MIN() ignores NULLs.
        # Read results by positional index — aliases omitted intentionally.
        min_clauses = [
            'MIN(CASE WHEN ABS("{col}") <= {threshold} THEN "{col}" END)'.format(
                col=col,
                threshold=_UINT32_MAX if is_integer[col] else _UINT32_MAX / 1000.0,
            )
            for col in numeric_cols
        ]
        row = conn.execute(f'SELECT {", ".join(min_clauses)} FROM {schema}.{table}').fetchone()
        for i, col in enumerate(numeric_cols):
            min_val = row[i]
            negative_offset = float(-min_val) if (min_val is not None and min_val < 0) else 0.0
            metadata[col] = {
                "dtype": "uint32",
                "semantic_dtype": "numeric",
                "scale_with_thousand": not is_integer[col],
                "negative_offset": negative_offset,
            }

    for col in categorical_cols:
        rows = conn.execute(
            f'SELECT DISTINCT "{col}" FROM {schema}.{table}'
            f' WHERE "{col}" IS NOT NULL ORDER BY "{col}"'
        ).fetchall()
        unique_values = [str(r[0]) for r in rows]
        encoding_map = {val: idx + 1 for idx, val in enumerate(unique_values)}
        decoding_map: dict[int, str | None] = {idx + 1: val for idx, val in enumerate(unique_values)}
        decoding_map[0] = None
        metadata[col] = {
            "dtype": "uint32",
            "semantic_dtype": "categorical",
            "encoding_map": encoding_map,
            "decoding_map": decoding_map,
        }

    return metadata


def _ingest_bundle(
    conn: duckdb.DuckDBPyConnection,
    bundle_name: str,
    tframes: tuple[str, ...],
    calendar,
    daily_bar_writer,
    minute_bar_writer,
    asset_db_writer,
    adjustment_writer,
    show_progress: bool,
) -> None:
    schema = bundle_name

    unknown = [t for t in tframes if t not in _SUPPORTED_TFRAMES]
    if unknown:
        raise NotImplementedError(
            f"duckdbdir: unsupported tframes {unknown!r}; supported: {sorted(_SUPPORTED_TFRAMES)}"
        )

    # Both daily and minute bundles read from ohlcv_features.
    # The date column type (DATE vs TIMESTAMP UTC) distinguishes the timeframe.
    describe_rows = conn.execute(f"DESCRIBE {schema}.ohlcv_features").fetchall()
    all_cols = [row[0] for row in describe_rows]
    feature_cols = [c for c in all_cols if c not in _OHLCV_COLS]
    # Strip compute-identity columns — not data features for zipline.
    feature_cols = [c for c in feature_cols if c not in {"config_hash", "created_by"}]

    # Stable sid assignment: sorted symbol order, same as asset metadata below.
    symbol_rows = conn.execute(
        f"SELECT DISTINCT symbol FROM {schema}.ohlcv_features ORDER BY symbol"
    ).fetchall()
    symbols: list[str] = [r[0] for r in symbol_rows]
    if not symbols:
        raise ValueError(f"No symbols found in {schema}.ohlcv_features")

    symbol_to_sid: dict[str, int] = {sym: i for i, sym in enumerate(symbols)}

    # 1. Write pricing data — dispatch per tframe.
    writers = {"daily": daily_bar_writer, "minute": minute_bar_writer}
    pricing_iters = {
        "daily": lambda: _pricing_iter(
            conn, schema, "ohlcv_features", symbols, feature_cols, show_progress,
            label="Loading daily pricing data: ",
            batch_size=_DAILY_BATCH_SIZE,
        ),
        "minute": lambda: _pricing_iter(
            conn, schema, "ohlcv_features", symbols, [], show_progress,
            label="Loading minute pricing data: ",
            batch_size=_MINUTE_BATCH_SIZE,
        ),
    }
    for tframe in tframes:
        if tframe == "daily" and feature_cols:
            feature_metadata = _compute_feature_metadata(
                conn, schema, "ohlcv_features", feature_cols
            )
            all_column_names = list(_OHLCV_WRITE_COLS) + sorted(feature_cols)
        else:
            feature_metadata = {}
            all_column_names = list(_OHLCV_WRITE_COLS)
        if tframe == "daily":
            writers[tframe].write(
                pricing_iters[tframe](),
                show_progress=show_progress,
                feature_metadata=feature_metadata,
                all_column_names=all_column_names,
            )
        else:
            # BcolzMinuteBarWriter.write() does not accept feature_metadata /
            # all_column_names — minute bundles carry no feature cols.
            writers[tframe].write(
                pricing_iters[tframe](),
                show_progress=show_progress,
            )

    # 2. Write asset metadata.
    _write_assets(conn, schema, symbols, symbol_to_sid, calendar, asset_db_writer)

    # 3. Write adjustments (crypto bundles will have empty tables — handled gracefully).
    splits_df = _read_ratio_table(conn, schema, "splits", symbol_to_sid)
    mergers_df = _read_ratio_table(conn, schema, "mergers", symbol_to_sid)
    dividends_df = _read_dividends(conn, schema, symbol_to_sid)
    adjustment_writer.write(
        splits=splits_df,
        mergers=mergers_df,
        dividends=dividends_df,
    )


# ---------------------------------------------------------------------------
# Pricing iterator
# ---------------------------------------------------------------------------


def _emit_batch(
    conn: duckdb.DuckDBPyConnection,
    schema: str,
    table: str,
    cols_sql: str,
    batch: list[tuple[int, str]],
) -> Generator[tuple[int, pd.DataFrame], None, None]:
    """Fetch a batch of symbols in one query and yield (sid, df) per symbol.

    One IN-list query replaces N individual queries, cutting DuckDB round-trip
    overhead significantly (~4.4x for N=64 on daily data).  Results are split
    by symbol via groupby before yielding so the caller interface is identical
    to the per-symbol path.

    When FYAN_PROFILE=1 the DuckDB fetch wall-time is emitted as a JSON event
    so the caller can separate I/O cost from bcolz write cost.
    """
    batch_syms = [sym for _, sym in batch]
    if not batch_syms:
        return
    ph = ", ".join("?" * len(batch_syms))

    t_fetch = time.perf_counter()
    df_all = conn.execute(
        f"SELECT symbol, {cols_sql} FROM {schema}.{table}"
        f" WHERE symbol IN ({ph}) ORDER BY symbol, date",
        batch_syms,
    ).df()
    fetch_ms = (time.perf_counter() - t_fetch) * 1000

    if _PROFILE:
        print(json.dumps({
            "type": "batch_fetch",
            "batch_size": len(batch_syms),
            "rows": len(df_all),
            "fetch_ms": round(fetch_ms, 1),
        }), flush=True, file=sys.stdout)

    sym_groups = {
        sym: grp.drop(columns=["symbol"])
        for sym, grp in df_all.groupby("symbol", sort=False)
    }
    for sid, sym in batch:
        df = sym_groups.get(sym)
        if df is None or df.empty:
            logger.warning("No pricing data for symbol %s (sid %d) in %s", sym, sid, table)
            continue
        dates = pd.to_datetime(df["date"])
        df["date"] = (
            dates.dt.tz_convert("UTC") if dates.dt.tz is not None else dates.dt.tz_localize("UTC")
        )
        yield sid, df.set_index("date")


def _pricing_iter(
    conn: duckdb.DuckDBPyConnection,
    schema: str,
    table: str,
    symbols: list[str],
    feature_cols: list[str],
    show_progress: bool,
    label: str = "Loading pricing data: ",
    batch_size: int = 1,
) -> Generator[tuple[int, pd.DataFrame], None, None]:
    """Yield (sid, df) for each symbol from <schema>.<table>.

    Works for both daily (DATE "date" column) and minute (TIMESTAMP UTC "date"
    column) bundles — both read from ohlcv_features; the "date" column name is
    canonical regardless of grain (transformer R6 contract).

    When batch_size > 1, symbols are fetched in batches via IN-list queries
    (_emit_batch), reducing DuckDB round-trip overhead (~4.4x at batch_size=64
    on daily data).  batch_size=1 (minute default) preserves the per-symbol
    path to keep peak RSS bounded (~24 MB/symbol for full-history minute bars).
    """
    cols_sql = "date, open, high, low, close, volume"
    if feature_cols:
        cols_sql += ", " + ", ".join(feature_cols)

    total = len(symbols)
    # Emit structured JSON progress so the Fyan orchestrator (subprocess.PIPE, not a TTY)
    # can forward per-symbol progress to the web UI via SSE.  click.progressbar is also
    # active when show_progress=True on a real TTY — both paths coexist without conflict.
    print(json.dumps({"type": "ingest_progress", "current": 0, "total": total, "symbol": ""}),
          flush=True, file=sys.stdout)

    with maybe_show_progress(symbols, show_progress, label=label) as it:
        if batch_size <= 1:
            for sid, symbol in enumerate(it):
                df = conn.execute(
                    f"SELECT {cols_sql} FROM {schema}.{table}"
                    " WHERE symbol = ? ORDER BY date",
                    [symbol],
                ).df()
                if df.empty:
                    logger.warning(
                        "No pricing data for symbol %s (sid %d) in %s", symbol, sid, table
                    )
                    continue
                # Normalise date column to tz-aware UTC regardless of what DuckDB returns.
                # Handles both DATE (daily) and TIMESTAMP (minute) column types.
                dates = pd.to_datetime(df["date"])
                df["date"] = (
                    dates.dt.tz_convert("UTC")
                    if dates.dt.tz is not None
                    else dates.dt.tz_localize("UTC")
                )
                if sid % 50 == 0 or sid == total - 1:
                    print(json.dumps({"type": "ingest_progress", "current": sid + 1,
                                      "total": total, "symbol": symbol}),
                          flush=True, file=sys.stdout)
                yield sid, df.set_index("date")
        else:
            pending: list[tuple[int, str]] = []
            t_iter_start = time.perf_counter()
            total_batch_ms: float = 0.0  # wall-time of each yield-from block (fetch + write)
            n_batches: int = 0

            def _run_batch(batch: list[tuple[int, str]]) -> Generator:
                nonlocal total_batch_ms, n_batches
                t0 = time.perf_counter()
                yield from _emit_batch(conn, schema, table, cols_sql, batch)
                # Time measured AFTER all yields are consumed by the caller (bcolz writes done).
                total_batch_ms += (time.perf_counter() - t0) * 1000
                n_batches += 1

            for sid, symbol in enumerate(it):
                pending.append((sid, symbol))
                if len(pending) >= batch_size:
                    yield from _run_batch(pending)
                    last_sid, last_sym = pending[-1]
                    print(json.dumps({"type": "ingest_progress", "current": last_sid + 1,
                                      "total": total, "symbol": last_sym}),
                          flush=True, file=sys.stdout)
                    pending = []
            if pending:
                yield from _run_batch(pending)
                last_sid, last_sym = pending[-1]
                print(json.dumps({"type": "ingest_progress", "current": last_sid + 1,
                                  "total": total, "symbol": last_sym}),
                      flush=True, file=sys.stdout)

            if _PROFILE and n_batches:
                total_elapsed_ms = (time.perf_counter() - t_iter_start) * 1000
                # fetch_ms sum comes from batch_fetch events; total_batch_ms ≈ fetch + write.
                # overhead = total_elapsed - total_batch_ms (progress prints, groupby splits, etc.)
                print(json.dumps({
                    "type": "ingest_timing_summary",
                    "total_symbols": total,
                    "n_batches": n_batches,
                    "batch_size": batch_size,
                    "total_elapsed_ms": round(total_elapsed_ms, 0),
                    "total_batch_ms": round(total_batch_ms, 0),
                    "avg_batch_ms": round(total_batch_ms / n_batches, 1),
                }), flush=True, file=sys.stdout)


# ---------------------------------------------------------------------------
# Asset metadata writer
# ---------------------------------------------------------------------------


def _write_assets(
    conn: duckdb.DuckDBPyConnection,
    schema: str,
    symbols: list[str],
    symbol_to_sid: dict[str, int],
    calendar,
    asset_db_writer,
) -> None:
    assets_raw: pd.DataFrame = conn.execute(
        f"""
        SELECT symbol, exchange, start_date, end_date, auto_close_date
        FROM {schema}.assets
        ORDER BY symbol
        """
    ).df()

    if assets_raw.empty:
        raise ValueError(f"No rows found in {schema}.assets")

    # Vectorised construction — reindex into symbols order (same stable sid order).
    asset_lookup = assets_raw.set_index("symbol").reindex(symbols)
    # DuckDB DATE columns come back as datetime64[us]; asset_db_writer._dt_to_epoch_ns
    # calls .view(int64) which must see nanoseconds — upcast explicitly.
    start_dates = pd.to_datetime(asset_lookup["start_date"]).dt.normalize().dt.as_unit("ns")
    end_dates = pd.to_datetime(asset_lookup["end_date"]).dt.normalize().dt.as_unit("ns")

    equities = pd.DataFrame({
        "start_date": start_dates.values,
        "end_date": end_dates.values,
        "auto_close_date": (end_dates + pd.Timedelta(days=1)).values,
        "symbol": symbols,
        "exchange": asset_lookup["exchange"].values,
    })

    # Derive exchange label and country code from the first asset row.
    exchange_label = str(equities["exchange"].iloc[0])
    country_code = getattr(calendar, "country_code", "US")

    exchanges = pd.DataFrame(
        data=[[exchange_label, exchange_label, country_code]],
        columns=["exchange", "canonical_name", "country_code"],
    )

    asset_db_writer.write(equities=equities, exchanges=exchanges)


# ---------------------------------------------------------------------------
# Adjustment readers
# ---------------------------------------------------------------------------


def _read_ratio_table(
    conn: duckdb.DuckDBPyConnection,
    schema: str,
    table: str,
    symbol_to_sid: dict[str, int],
) -> pd.DataFrame:
    """Read splits or mergers table and return the shape SQLiteAdjustmentWriter expects."""
    date_col = "effective_date"
    empty = pd.DataFrame({
        "sid": pd.Series(dtype="int64"),
        "ratio": pd.Series(dtype="float64"),
        date_col: pd.Series(dtype="int64"),
    })

    try:
        rows: pd.DataFrame = conn.execute(
            f"SELECT symbol, ratio, {date_col} FROM {schema}.{table} ORDER BY {date_col}"
        ).df()
    except duckdb.CatalogException:
        return empty

    if rows.empty:
        return empty

    rows = rows[rows["symbol"].isin(symbol_to_sid)]
    if rows.empty:
        return empty
    rows["sid"] = rows["symbol"].map(symbol_to_sid).astype("int64")
    rows[date_col] = _to_epoch_seconds(rows[date_col])
    return rows[["sid", "ratio", date_col]].reset_index(drop=True)


def _read_dividends(
    conn: duckdb.DuckDBPyConnection,
    schema: str,
    symbol_to_sid: dict[str, int],
) -> pd.DataFrame:
    """Read dividends table; ratio is NOT stored — zipline derives it from close prices."""
    empty = pd.DataFrame({
        "sid": pd.Series(dtype="int64"),
        "amount": pd.Series(dtype="float64"),
        "ex_date": pd.Series(dtype="int64"),
        "declared_date": pd.Series(dtype="int64"),
        "record_date": pd.Series(dtype="int64"),
        "pay_date": pd.Series(dtype="int64"),
    })

    try:
        rows: pd.DataFrame = conn.execute(
            f"""
            SELECT symbol, ex_date, declared_date, record_date, pay_date, amount
            FROM {schema}.dividends
            ORDER BY ex_date
            """
        ).df()
    except duckdb.CatalogException:
        return empty

    if rows.empty:
        return empty

    rows = rows[rows["symbol"].isin(symbol_to_sid)]
    if rows.empty:
        return empty
    rows["sid"] = rows["symbol"].map(symbol_to_sid).astype("int64")
    for dcol in ("ex_date", "declared_date", "record_date", "pay_date"):
        rows[dcol] = _to_epoch_seconds(rows[dcol], fill_null=0)

    return rows[
        ["sid", "amount", "ex_date", "declared_date", "record_date", "pay_date"]
    ].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Date helpers
# ---------------------------------------------------------------------------


def _to_epoch_seconds(series: pd.Series, fill_null: int = 0) -> pd.Series:
    """Convert a date column to integer epoch seconds (NaT / NULL → fill_null).

    SQLiteAdjustmentWriter.write_frame stores effective_date as epoch seconds,
    and get_adjustments_for_sid reads it back as pd.Timestamp(val, unit="s").
    Must be seconds, not days.
    """
    dt = pd.to_datetime(series, errors="coerce").dt.normalize()
    secs = (dt - pd.Timestamp("1970-01-01")) / pd.Timedelta("1s")
    return secs.fillna(fill_null).astype("int64")
