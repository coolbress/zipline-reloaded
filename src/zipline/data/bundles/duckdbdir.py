"""DuckDB-backed equities bundle ingester for Fyan.

Replaces hdfdir.py.  Reads directly from fyan_computed.duckdb staging tables
(materialised by Compute Engine R1–R6) — no intermediate file format.

Registration (in Fyan's engine/bundle.py via _isolation.py):
    bundles.register(
        "us_equities",
        duckdb_equities("/path/to/fyan_computed.duckdb"),
        calendar_name="NYSE",
    )

If duckdb_path is omitted the ingester falls back to the FYAN_COMPUTED_PATH
env-var, which the Compute Engine sets on the subprocess.  FYAN_BUNDLE_NAME
selects the staging schema (<bundle_name>.*) to read.
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
):
    """Curried factory.  Returns the bound ingest callable zipline will invoke.

    Parameters
    ----------
    duckdb_path:
        Absolute path to fyan_computed.duckdb.  Falls back to
        ``os.environ["FYAN_COMPUTED_PATH"]`` if not supplied.
    tframes:
        Time-frame names.  Only "daily" is supported in v1.
    calendar_name:
        Trading calendar (forwarded to bundles.register at registration site).
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
                asset_db_writer=asset_db_writer,
                adjustment_writer=adjustment_writer,
                show_progress=show_progress,
            )

    return ingest


# ---------------------------------------------------------------------------
# Core ingestion logic
# ---------------------------------------------------------------------------


def _ingest_bundle(
    conn: duckdb.DuckDBPyConnection,
    bundle_name: str,
    tframes: tuple[str, ...],
    calendar,
    daily_bar_writer,
    asset_db_writer,
    adjustment_writer,
    show_progress: bool,
) -> None:
    schema = bundle_name

    # Discover feature columns (non-OHLCV) from staging table schema.
    describe_rows = conn.execute(f"DESCRIBE {schema}.ohlcv_features").fetchall()
    all_cols = [row[0] for row in describe_rows]
    feature_cols = [c for c in all_cols if c not in _OHLCV_COLS]

    # Stable sid assignment: sorted symbol order, same as asset metadata below.
    symbol_rows = conn.execute(
        f"SELECT DISTINCT symbol FROM {schema}.ohlcv_features ORDER BY symbol"
    ).fetchall()
    symbols: list[str] = [r[0] for r in symbol_rows]
    if not symbols:
        raise ValueError(f"No symbols found in {schema}.ohlcv_features")

    symbol_to_sid: dict[str, int] = {sym: i for i, sym in enumerate(symbols)}

    # 1. Write pricing data (bcolz pre-write).
    for tframe in tframes:
        writer = daily_bar_writer  # minute not supported in v1
        writer.write(
            _pricing_iter(conn, schema, symbols, feature_cols, show_progress),
            show_progress=show_progress,
        )

    # 2. Write asset metadata.
    _write_assets(conn, schema, symbols, symbol_to_sid, calendar, asset_db_writer)

    # 3. Write adjustments.
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
    symbols: list[str],
    feature_cols: list[str],
    show_progress: bool,
) -> Generator[tuple[int, pd.DataFrame], None, None]:
    """Yield (sid, df) for each symbol from <schema>.ohlcv_features."""
    extra = (", " + ", ".join(feature_cols)) if feature_cols else ""
    with maybe_show_progress(
        symbols, show_progress, label="Loading daily pricing data: "
    ) as it:
        for sid, symbol in enumerate(it):
            df: pd.DataFrame = conn.execute(
                f"""
                SELECT date, open, high, low, close, volume{extra}
                FROM {schema}.ohlcv_features
                WHERE symbol = ?
                ORDER BY date
                """,
                [symbol],
            ).df()

            if df.empty:
                logger.warning("No pricing data for symbol %s (sid %d)", symbol, sid)
                continue

            df["date"] = pd.to_datetime(df["date"]).dt.tz_localize("UTC")
            df.set_index("date", inplace=True)
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

    dtype = [
        ("start_date", "datetime64[ns]"),
        ("end_date", "datetime64[ns]"),
        ("auto_close_date", "datetime64[ns]"),
        ("symbol", "object"),
        ("exchange", "object"),
    ]
    equities = pd.DataFrame(np.empty(len(symbols), dtype=dtype))

    asset_lookup = assets_raw.set_index("symbol")
    for sid, symbol in enumerate(symbols):
        row = asset_lookup.loc[symbol]
        end_dt = pd.Timestamp(row["end_date"])
        equities.iloc[sid] = (
            pd.Timestamp(row["start_date"]).tz_localize(None),
            end_dt.tz_localize(None),
            (end_dt + pd.Timedelta(days=1)).tz_localize(None),
            symbol,
            row["exchange"],
        )

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
    dt = pd.to_datetime(series, errors="coerce")
    days = (dt - pd.Timestamp("1970-01-01")) / pd.Timedelta("1D")
    return days.fillna(fill_null).astype("int64")
