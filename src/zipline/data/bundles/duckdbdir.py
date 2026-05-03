"""DuckDB-backed equities bundle ingester for Fyan.

Replaces hdfdir.py.  Reads directly from fyan_computed.duckdb staging tables
(materialised by Compute Engine R1–R6) — no intermediate file format.

Registration examples (in Fyan's engine/bundle.py via _isolation.py):

    # Daily US equities — NYSE calendar
    bundles.register(
        "us_equities",
        duckdb_equities("/path/to/fyan_computed.duckdb"),
        calendar_name="NYSE",
    )

    # Daily crypto (Upbit) — 24/7 AlwaysOpenCalendar
    bundles.register(
        "upbit_daily",
        duckdb_equities("/path/to/fyan_computed.duckdb"),
        calendar_name="24/7",
    )

    # Minute crypto (Upbit 1-min) — 24/7, 1440 min/day
    bundles.register(
        "upbit_1m",
        duckdb_equities(
            "/path/to/fyan_computed.duckdb",
            tframes=("daily", "minute"),
            minute_table="ohlcv_minute",
        ),
        calendar_name="24/7",
        minutes_per_day=1440,   # AlwaysOpenCalendar: 1440 min/session, NOT 390
    )

If duckdb_path is omitted the ingester falls back to the FYAN_COMPUTED_PATH
env-var, which the Compute Engine sets on the subprocess.  FYAN_BUNDLE_NAME
selects the staging schema (<bundle_name>.*) to read.

Staging table contracts
-----------------------
Daily:  <bundle>.ohlcv_features  — columns: symbol, date (DATE), open, high, low,
                                    close, volume (all float64), + optional feature cols
Minute: <bundle>.ohlcv_minute    — columns: symbol, date (TIMESTAMP UTC), open, high,
                                    low, close, volume (all float64); no feature cols
Both tables use the canonical column name "date" for the time axis regardless of grain.
Timezone: minute timestamps must be UTC-naive or UTC-aware; the ingester normalises both.
"""

from __future__ import annotations

import logging
import os
from typing import Generator

import duckdb
import numpy as np
import pandas as pd
from zipline.utils.cli import maybe_show_progress

logger = logging.getLogger(__name__)

_OHLCV_COLS: frozenset[str] = frozenset(
    {"symbol", "date", "open", "high", "low", "close", "volume"}
)


# ---------------------------------------------------------------------------
# Public factory
# ---------------------------------------------------------------------------


def duckdb_equities(
    duckdb_path: str | None = None,
    tframes: tuple[str, ...] = ("daily",),
    calendar_name: str = "NYSE",
    minute_table: str = "ohlcv_minute",
):
    """Curried factory.  Returns the bound ingest callable zipline will invoke.

    Parameters
    ----------
    duckdb_path:
        Absolute path to fyan_computed.duckdb.  Falls back to
        ``os.environ["FYAN_COMPUTED_PATH"]`` if not supplied.
    tframes:
        Subset of ``{"daily", "minute"}`` to ingest.  Default is ``("daily",)``.
        For minute support register with ``minutes_per_day`` matching the calendar
        (e.g. 1440 for ``"24/7"``, 390 for ``"NYSE"``).
    calendar_name:
        Trading calendar (forwarded to bundles.register at registration site).
    minute_table:
        Name of the staging table holding minute OHLCV inside the bundle schema.
        Default ``"ohlcv_minute"``.  Ignored when ``"minute"`` not in *tframes*.
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
                minute_table=minute_table,
                show_progress=show_progress,
            )

    return ingest


# ---------------------------------------------------------------------------
# Core ingestion logic
# ---------------------------------------------------------------------------


_SUPPORTED_TFRAMES: frozenset[str] = frozenset({"daily", "minute"})


def _ingest_bundle(
    conn: duckdb.DuckDBPyConnection,
    bundle_name: str,
    tframes: tuple[str, ...],
    calendar,
    daily_bar_writer,
    minute_bar_writer,
    asset_db_writer,
    adjustment_writer,
    minute_table: str,
    show_progress: bool,
) -> None:
    schema = bundle_name

    unknown = [t for t in tframes if t not in _SUPPORTED_TFRAMES]
    if unknown:
        raise NotImplementedError(
            f"duckdbdir: unsupported tframes {unknown!r}; supported: {sorted(_SUPPORTED_TFRAMES)}"
        )

    # Primary table drives symbol discovery and feature-column enumeration.
    # When both daily and minute are requested, daily is primary (features live there).
    primary_table = "ohlcv_features" if "daily" in tframes else minute_table

    describe_rows = conn.execute(f"DESCRIBE {schema}.{primary_table}").fetchall()
    all_cols = [row[0] for row in describe_rows]
    feature_cols = [c for c in all_cols if c not in _OHLCV_COLS]

    # Stable sid assignment: sorted symbol order, same as asset metadata below.
    symbol_rows = conn.execute(
        f"SELECT DISTINCT symbol FROM {schema}.{primary_table} ORDER BY symbol"
    ).fetchall()
    symbols: list[str] = [r[0] for r in symbol_rows]
    if not symbols:
        raise ValueError(f"No symbols found in {schema}.{primary_table}")

    symbol_to_sid: dict[str, int] = {sym: i for i, sym in enumerate(symbols)}

    # 1. Write pricing data — dispatch per tframe.
    # Both daily and minute share _pricing_iter; the table name and feature_cols differ.
    # Minute staging table contract: (symbol, date TIMESTAMP UTC, open, high, low, close, volume)
    # "date" column name is canonical regardless of grain (same as transformer R6 output).
    writers = {"daily": daily_bar_writer, "minute": minute_bar_writer}
    pricing_iters = {
        "daily": lambda: _pricing_iter(
            conn, schema, "ohlcv_features", symbols, feature_cols, show_progress,
            label="Loading daily pricing data: ",
        ),
        "minute": lambda: _pricing_iter(
            conn, schema, minute_table, symbols, [], show_progress,
            label="Loading minute pricing data: ",
        ),
    }
    for tframe in tframes:
        writers[tframe].write(pricing_iters[tframe](), show_progress=show_progress)

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


def _pricing_iter(
    conn: duckdb.DuckDBPyConnection,
    schema: str,
    table: str,
    symbols: list[str],
    feature_cols: list[str],
    show_progress: bool,
    label: str = "Loading pricing data: ",
) -> Generator[tuple[int, pd.DataFrame], None, None]:
    """Yield (sid, df) for each symbol from <schema>.<table>.

    Works for both daily (ohlcv_features, DATE "date" column) and minute
    (ohlcv_minute, TIMESTAMP UTC "date" column) staging tables — the "date"
    column name is canonical regardless of grain (transformer R6 contract).

    One query per symbol keeps peak RSS bounded to a single symbol's rows
    regardless of universe size.  DuckDB evaluates the WHERE clause via zone
    maps on sorted data so each query is fast even on a 200 M-row file.
    """
    cols_sql = "date, open, high, low, close, volume"
    if feature_cols:
        cols_sql += ", " + ", ".join(feature_cols)

    with maybe_show_progress(symbols, show_progress, label=label) as it:
        for sid, symbol in enumerate(it):
            df = conn.execute(
                f"SELECT {cols_sql} FROM {schema}.{table}"
                " WHERE symbol = ? ORDER BY date",
                [symbol],
            ).df()
            if df.empty:
                logger.warning("No pricing data for symbol %s (sid %d) in %s", symbol, sid, table)
                continue
            # Normalise date column to tz-aware UTC regardless of what DuckDB returns.
            # Handles both DATE (daily) and TIMESTAMP (minute) column types.
            dates = pd.to_datetime(df["date"])
            if dates.dt.tz is not None:
                df["date"] = dates.dt.tz_convert("UTC")
            else:
                df["date"] = dates.dt.tz_localize("UTC")
            df = df.set_index("date")
            yield sid, df


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
    start_dates = pd.to_datetime(asset_lookup["start_date"]).dt.normalize()
    end_dates = pd.to_datetime(asset_lookup["end_date"]).dt.normalize()

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
    rows[date_col] = _to_epoch_days(rows[date_col])
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
        rows[dcol] = _to_epoch_days(rows[dcol], fill_null=0)

    return rows[
        ["sid", "amount", "ex_date", "declared_date", "record_date", "pay_date"]
    ].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Date helpers
# ---------------------------------------------------------------------------


def _to_epoch_days(series: pd.Series, fill_null: int = 0) -> pd.Series:
    """Convert a date column to integer epoch days (NaT / NULL → fill_null)."""
    dt = pd.to_datetime(series, errors="coerce").dt.normalize()
    days = (dt - pd.Timestamp("1970-01-01")) / pd.Timedelta("1D")
    return days.fillna(fill_null).astype("int64")
