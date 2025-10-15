"""
Module for building a complete dataset from a single parquet file using Polars.
All symbols data should be contained in one parquet file with a 'symbol' column.
"""

import os
import sys
import logging
import numpy as np
import pandas as pd
import polars as pl
from pathlib import Path
from zipline.utils.calendar_utils import register_calendar_alias
from zipline.utils.cli import maybe_show_progress

from . import core as bundles

handler = logging.StreamHandler()
logger = logging.getLogger(__name__)
logger.handlers.append(handler)
logger.setLevel(logging.INFO)


def parquetdir_equities(tframes=None, parquet_path=None, symbol_column='symbol'):
    """
    Generate an ingest function for custom data bundle using Polars + single Parquet file
    This function can be used in ~/.zipline/extension.py
    to register bundle with custom parameters, e.g. with
    a custom trading calendar.

    Parameters
    ----------
    tframes: tuple, optional
        The data time frames, supported timeframes: 'daily' and 'minute'
    parquet_path : string, optional, default: PARQUET_BUNDLE_PATH environment variable
        The path to the single parquet file containing all symbols data.
        Expected structure:
        - Must contain a 'symbol' column with symbol identifiers
        - Must contain OHLCV columns: 'open', 'high', 'low', 'close', 'volume'
        - Must contain a datetime column (first column or specified)
        - Optional: 'split', 'dividend' columns for corporate actions
    symbol_column : string, optional, default: 'symbol'
        Column name containing symbol identifiers

    Returns
    -------
    ingest : callable
        The bundle ingest function

    Examples
    --------
    This code should be added to ~/.zipline/extension.py
    .. code-block:: python
       from zipline.data.bundles import parquetdir_equities, register
       register('custom-parquetdir-bundle',
                parquetdir_equities(["daily", "minute"],
                '/full/path/to/all_symbols.parquet',
                symbol_column='symbol'))
    """

    return ParquetDIRBundle(tframes, parquet_path, symbol_column).ingest


class ParquetDIRBundle:
    """
    Wrapper class to call parquetdir_bundle with provided
    list of time frames and a path to the single parquet file
    """

    def __init__(self, tframes=None, parquet_path=None, symbol_column='symbol'):
        self.tframes = tframes
        self.parquet_path = parquet_path
        self.symbol_column = symbol_column

    def ingest(
        self,
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
    ):
        parquetdir_bundle(
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
            self.tframes,
            self.parquet_path,
            self.symbol_column,
        )


@bundles.register("parquetdir")
def parquetdir_bundle(
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
    tframes=None,
    parquet_path=None,
    symbol_column='symbol',
):
    """
    Build a zipline data bundle from a single parquet file using Polars.
    """
    if not parquet_path:
        parquet_path = environ.get("PARQUET_BUNDLE_PATH")
        if not parquet_path:
            raise ValueError("PARQUET_BUNDLE_PATH environment variable is not set")

    if not os.path.isfile(parquet_path):
        raise ValueError("%s is not a file" % parquet_path)

    if not tframes:
        tframes = ['daily']  # 기본값으로 daily 설정

    # Initialize dividend and split DataFrames with specific dtypes to avoid pandas FutureWarnings
    # and ensure compatibility with pd.concat() even when empty. (판다스 2.1+에서는 빈 데이터프레임을 pd.concat() 할 때 dtype 을 명시적으로 지정해야 함)
    divs_splits = {
        "divs": pd.DataFrame({
            "sid": pd.Series(dtype='int64'),
            "amount": pd.Series(dtype='float64'),
            "ex_date": pd.Series(dtype='int64'),
            "record_date": pd.Series(dtype='int64'),
            "declared_date": pd.Series(dtype='int64'),
            "pay_date": pd.Series(dtype='int64'),
        }),
        "splits": pd.DataFrame({
            "sid": pd.Series(dtype='int64'),
            "ratio": pd.Series(dtype='float64'),
            "effective_date": pd.Series(dtype='int64'),
        }),
    }
    
    # Read the single parquet file using Polars
    try:
        logger.info(f"Reading parquet file: {parquet_path}")
        df_polars = pl.read_parquet(parquet_path)
        df_all = df_polars.to_pandas()
        
        # Check if symbol column exists
        if symbol_column not in df_all.columns:
            raise ValueError(f"Symbol column '{symbol_column}' not found in parquet file")
        
        # Get unique symbols
        symbols = sorted(df_all[symbol_column].unique())
        if not symbols:
            raise ValueError("No symbols found in parquet file")
        
        logger.info(f"Found {len(symbols)} symbols: {symbols[:5]}{'...' if len(symbols) > 5 else ''}")
        
    except Exception as e:
        logger.error(f"Error reading parquet file {parquet_path}: {e}")
        raise ValueError(f"Error reading parquet file {parquet_path}: {e}")

    # Process each timeframe
    dtype = [
        ("start_date", "datetime64[ns]"),
        ("end_date", "datetime64[ns]"),
        ("auto_close_date", "datetime64[ns]"),
        ("symbol", "object"),
    ]
    metadata = pd.DataFrame(np.empty(len(symbols), dtype=dtype))

    for tframe in tframes:
        writer = daily_bar_writer if tframe == 'daily' else minute_bar_writer
        writer.write(
            _parquet_pricing_iter(df_all, symbols, tframe, symbol_column, metadata, divs_splits, show_progress),
            show_progress=show_progress,
        )

    metadata["exchange"] = "PARQUET" # exchange 값 설정해도 asset db 에는 적용안되는데 원인은 모르겠음..

    asset_db_writer.write(equities=metadata)

    divs_splits["divs"]["sid"] = divs_splits["divs"]["sid"].astype(int)
    divs_splits["splits"]["sid"] = divs_splits["splits"]["sid"].astype(int)
    adjustment_writer.write(
        splits=divs_splits["splits"], dividends=divs_splits["divs"]
    )


def _parquet_pricing_iter(df_all, symbols, tframe, symbol_column, metadata, divs_splits, show_progress):
    with maybe_show_progress(
        symbols, show_progress, label=f"Loading {tframe} pricing data: "
    ) as it:
        for sid, symbol in enumerate(it):
            logger.debug(f"{symbol}: sid {sid}")
            
            # Filter data for current symbol from parquet file
            df_symbol = df_all[df_all[symbol_column] == symbol].copy()
            
            if df_symbol.empty:
                logger.warning(f"No data found for symbol {symbol}")
                continue
            
            # Set datetime column as index (assume first column is datetime if not explicitly set)
            datetime_col = df_symbol.columns[0]
            if datetime_col == symbol_column:
                datetime_col = df_symbol.columns[1]  # Use second column if first is symbol
            
            df = df_symbol.set_index(datetime_col)
            
            # Remove symbol column from the data
            if symbol_column in df.columns:
                df = df.drop(columns=[symbol_column])
            
            # Convert index to datetime and sort, then set UTC timezone
            if not isinstance(df.index, pd.DatetimeIndex):
                df.index = pd.to_datetime(df.index)
            df = df.sort_index()
            df.index = df.index.tz_localize('UTC')  # zipline은 UTC timezone을 사용

            start_date = df.index[0]
            end_date = df.index[-1]

            ac_date = end_date + pd.Timedelta(days=1)
            metadata.iloc[sid] = (
                start_date.tz_convert(None),
                end_date.tz_convert(None),
                ac_date.tz_convert(None),
                symbol
            )

            _process_corporate_actions(df, sid, divs_splits)

            yield sid, df

def _process_corporate_actions(df, sid, divs_splits):
    if "split" in df.columns:
        tmp = 1.0 / df[df["split"] != 1.0]["split"]
        split = pd.DataFrame(
            data=tmp.index.tolist(), columns=["effective_date"]
        )
        split["ratio"] = tmp.tolist()
        split["sid"] = sid

        splits = divs_splits["splits"]
        index = pd.Index(
            range(splits.shape[0], splits.shape[0] + split.shape[0])
        )
        split.set_index(index, inplace=True)
        divs_splits["splits"] = pd.concat([splits, split], axis=0)

    if "dividend" in df.columns:
        # ex_date   amount  sid record_date declared_date pay_date
        tmp = df[df["dividend"] != 0.0]["dividend"]
        div = pd.DataFrame(data=tmp.index.tolist(), columns=["ex_date"])
        div["record_date"] = pd.NaT
        div["declared_date"] = pd.NaT
        div["pay_date"] = pd.NaT
        div["amount"] = tmp.tolist()
        div["sid"] = sid

        divs = divs_splits["divs"]
        ind = pd.Index(range(divs.shape[0], divs.shape[0] + div.shape[0]))
        div.set_index(ind, inplace=True)
        divs_splits["divs"] = pd.concat([divs, div], axis=0)


# Register the calendar
# register_calendar_alias("PARQUETDIR", "NYSE")