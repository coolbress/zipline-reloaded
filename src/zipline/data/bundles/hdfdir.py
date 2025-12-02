import os 
import logging 
import numpy as np 
import pandas as pd 
import h5py
from zipline.data.bundles import core as bundles 
from zipline.data.bundles.core import BundleData
from zipline.utils.calendar_utils import register_calendar_alias
from zipline.utils.cli import maybe_show_progress

logger = logging.getLogger(__name__)
handler = logging.StreamHandler()
logger.addHandler(handler)
logger.setLevel(logging.INFO)


def hdf5_equities(tframes=None, hdf5_path=None):
    """
    Generate an ingest function for custom HDF5 data bundle
    This function can be used in ~/.zipline/extension.py
    to register bundle with custom parameters.

    Parameters
    ----------
    tframes: tuple, optional
        The data time frames, supported timeframes: 'daily' and 'minute'
    hdf5_path : string, optional, default: HDF5_BUNDLE_PATH environment variable
        The path to the HDF5 file containing the pricing data

    Returns
    -------
    ingest : callable
        The bundle ingest function

    HDF5 File Structure
    --------------------
    The HDF5 file must be organized with the following structure:
    Each symbol should be a group at the root level, containing:
    
    Required datasets:
    - open: float64 array of opening prices
    - high: float64 array of high prices
    - low: float64 array of low prices
    - close: float64 array of closing prices
    - volume: float64 or uint64 array of trading volumes
    - dates: datetime64 array of trading dates
    
    Optional group:
    - corporate_actions/: group containing splits and/or dividends
      - splits: DataFrame with columns (effective_date, ratio)
      - dividends: DataFrame with columns (ex_date, pay_date, amount, ...)
    
    Optional custom feature datasets:
    - Any dataset that is not OHLCV or corporate_actions will be treated
      as a custom feature. Feature data must have the same length as OHLCV dates.
      Examples: pe, pbr, revenue, sector, industry, etc.
    
    Example HDF5 structure:
    .. code-block:: python
    /AAPL/
      ├── open          (float64 array, length N)
      ├── high          (float64 array, length N)
      ├── low           (float64 array, length N)
      ├── close         (float64 array, length N)
      ├── volume        (float64/uint64 array, length N)
      ├── dates         (datetime64 array, length N)
      ├── pe            (float64 array, length N)  # Custom feature
      ├── pbr           (float64 array, length N)  # Custom feature
      ├── sector        (string array, length N)   # Custom feature (categorical)
      └── corporate_actions/
          ├── splits    (DataFrame with effective_date, ratio)
          └── dividends  (DataFrame with ex_date, pay_date, amount, ...)
    
    /MSFT/
      ├── open
      ├── high
      ├── low
      ├── close
      ├── volume
      ├── dates
      └── ...
    
    Notes
    -----
    - All OHLCV datasets must have the same length (N = number of trading days)
    - Custom feature datasets must have the same length as OHLCV dates
    - Within each symbol group, dataset names must be unique (HDF5 requirement)
    - Custom features are automatically detected and added to the bundle
    - Corporate actions (splits, dividends) are processed separately and stored
      in adjustments.sqlite, not in the main pricing bundle

    Examples
    --------
    This code should be added to ~/.zipline/extension.py
    .. code-block:: python
    from zipline.data.bundles import register
    from zipline.data.bundles.hdfdir import hdf5_equities
    register('custom-hdf5-bundle',
            hdf5_equities(["daily", "minute"],
            '/full/path/to/your/hdf5file.h5'))
    """
    return HDF5Bundle(tframes, hdf5_path).ingest


class HDF5Bundle:
    def __init__(self, tframes=None, hdf5_path=None):
        self.tframes = tframes or ['daily']
        self.hdf5_path = os.path.expanduser(hdf5_path) if hdf5_path else None

    def ingest(self,
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
               output_dir):
        hdf5_bundle(environ,
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
                    self.hdf5_path)

@bundles.register("hdfdir")
def hdf5_bundle(environ,
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
                hdf5_path=None):
    if not hdf5_path:
        hdf5_path = environ.get("HDF5_BUNDLE_PATH")
        if not hdf5_path:
            raise ValueError("HDF5_BUNDLE_PATH environment variable is not set")

    if not os.path.isfile(hdf5_path):
        raise ValueError(f"{hdf5_path} is not a file")

    if not tframes:
        tframes = ['daily']

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
    
    with h5py.File(hdf5_path, 'r') as hdf:
        symbols = sorted(hdf.keys())
        if not symbols:
            raise ValueError("No symbols found in HDF5 file")

        dtype = [
            ("start_date", "datetime64[ns]"),
            ("end_date", "datetime64[ns]"),
            ("auto_close_date", "datetime64[ns]"),
            ("symbol", "object"),
        ]
        metadata = pd.DataFrame(np.empty(len(symbols), dtype=dtype))

        for tframe in tframes:
            writer = daily_bar_writer if tframe == 'daily' else minute_bar_writer
            
            # _hdf5_pricing_iter now yields (sid, df) where df contains OHLCV + Features
            # Features are automatically detected and added to DataFrame as columns
            # (OHLCV and corporate_actions are excluded, everything else is a feature)
            pricing_iter = _hdf5_pricing_iter(
                hdf, symbols, tframe, metadata, divs_splits, show_progress
            )
            
            # Pass generator directly to writer (original Zipline pattern)
            writer.write(
                pricing_iter,
                show_progress=show_progress,
            )

        metadata["exchange"] = "HDF5"
        
        # Determine country_code from calendar name
        # Common calendar names and their country codes
        calendar_name = calendar.name
        country_code_map = {
            "XNYS": "US",  # NYSE
            "NASDAQ": "US",  # NASDAQ
            "XKRX": "KR",  # Korea Exchange
            "XASX": "AU",  # Australian Securities Exchange
            "XTSE": "CA",  # Toronto Stock Exchange
            "XHKG": "HK",  # Hong Kong Exchange
            "XTKS": "JP",  # Tokyo Stock Exchange
            "XSHG": "CN",  # Shanghai Stock Exchange
            "XPAR": "FR",  # Euronext Paris
            "XLON": "GB",  # London Stock Exchange
            "XFRA": "DE",  # Frankfurt Stock Exchange
            "XMIL": "IT",  # Milan Stock Exchange
            "XAMS": "NL",  # Amsterdam Stock Exchange
            "XBRU": "BE",  # Brussels Stock Exchange
            "XSWX": "CH",  # Swiss Exchange
            "XOSL": "NO",  # Oslo Stock Exchange
            "XSTO": "SE",  # Stockholm Stock Exchange
            "XCSE": "DK",  # Copenhagen Stock Exchange
            "XHEL": "FI",  # Helsinki Stock Exchange
            "XMAD": "ES",  # Madrid Stock Exchange
            "XWBO": "AT",  # Vienna Stock Exchange
            "XBUD": "HU",  # Budapest Stock Exchange
            "XPRA": "CZ",  # Prague Stock Exchange
            "XBOM": "IN",  # Bombay Stock Exchange
            "XNSE": "IN",  # National Stock Exchange of India
            "XIDX": "ID",  # Indonesia Stock Exchange
            "XKLS": "MY",  # Bursa Malaysia
            "XSGX": "SG",  # Singapore Exchange
            "XTAI": "TW",  # Taiwan Stock Exchange
            "XBOG": "CO",  # Bogota Stock Exchange
            "XSGO": "CL",  # Santiago Stock Exchange
            "XMEX": "MX",  # Mexican Stock Exchange
            "XBUE": "AR",  # Buenos Aires Stock Exchange
            "BVMF": "BR",  # B3 (Brazil)
            "XLIM": "PE",  # Lima Stock Exchange
            "XNZE": "NZ",  # New Zealand Exchange
            "XJSE": "ZA",  # Johannesburg Stock Exchange
        }
        
        # Get country_code from calendar name, default to "US" if not found
        country_code = country_code_map.get(calendar_name, "US")
        
        # Create exchanges DataFrame with proper country_code
        exchanges = pd.DataFrame(
            data=[["HDF5", "HDF5", country_code]],
            columns=["exchange", "canonical_name", "country_code"],
        )

        asset_db_writer.write(equities=metadata, exchanges=exchanges)

        divs_splits["divs"]["sid"] = divs_splits["divs"]["sid"].astype(int)
        divs_splits["splits"]["sid"] = divs_splits["splits"]["sid"].astype(int)
        adjustment_writer.write(
            splits=divs_splits["splits"], dividends=divs_splits["divs"]
        )
        
def _hdf5_pricing_iter(hdf, symbols, tframe, metadata, divs_splits, show_progress):
    """Iterator that yields (sid, DataFrame) tuples for each symbol.
    
    Parameters
    ----------
    hdf : h5py.File
        Open HDF5 file
    symbols : list[str]
        List of symbol names
    tframe : str
        Time frame ('daily' or 'minute')
    metadata : pd.DataFrame
        Metadata DataFrame to update
    divs_splits : dict
        Dictionary to store dividends and splits
    show_progress : bool
        Whether to show progress bar
    
    Yields
    ------
    tuple
        (sid, df) where:
        - sid: int, asset identifier
        - df: pd.DataFrame, OHLCV data + Features (automatically detected)
    
    Notes
    -----
    Features are automatically detected from HDF5 file structure:
    - OHLCV columns: 'open', 'high', 'low', 'close', 'volume', 'dates', 'day' (excluded)
    - Corporate actions: 'corporate_actions' group (excluded)
    - Everything else: Custom features (added to DataFrame as columns)
    """
    # Define reserved keys (OHLCV and corporate_actions)
    ohlcv_keys = {'open', 'high', 'low', 'close', 'volume', 'dates', 'day'}
    
    with maybe_show_progress(
        symbols, show_progress, label=f"Loading {tframe} pricing data: "
    ) as it:
        for sid, symbol in enumerate(it):
            logger.debug(f"{symbol}: sid {sid}")
            grp = hdf[symbol]

            # Collect all data (OHLCV + Features) in a single pass
            # Process all keys in one loop
            data_dict = {}
            dates = None
            
            for key in grp.keys():
                if key == 'dates':
                    # Extract dates first (needed for index and validation)
                    dates = pd.to_datetime(grp['dates'][:]).tz_localize('UTC')
                elif key in ohlcv_keys:
                    # OHLCV columns
                    data_dict[key] = grp[key][:]
                elif key == 'corporate_actions':
                    # Corporate actions (splits, dividends) are processed separately
                    # and stored in separate DataFrames, then passed to adjustment_writer
                    # (not included in the OHLCV + Features DataFrame)
                    _process_corporate_actions(grp, sid, divs_splits)
                else:
                    # Everything else is a feature
                    if isinstance(grp[key], h5py.Dataset):
                        feature_values = grp[key][:]
                        # Feature data is assumed to be already aligned with OHLCV dates
                        # (pre-processed to calendar sessions before saving to HDF5)
                        # Missing dates will be NaN, which will be converted to 0 in writer
                        # and restored to NaN in reader
                        data_dict[key] = feature_values
            
            # Create DataFrame with all columns (OHLCV + Features) at once
            if dates is None:
                raise ValueError(f"No 'dates' dataset found for symbol '{symbol}'")
            
            df = pd.DataFrame(data_dict, index=dates)

            start_date = df.index[0]
            end_date = df.index[-1]

            ac_date = end_date + pd.Timedelta(days=1)
            metadata.iloc[sid] = (
                start_date.tz_convert(None),
                end_date.tz_convert(None),
                ac_date.tz_convert(None),
                symbol
            )

            # Yield (sid, df) where df contains OHLCV + Features
            yield sid, df

def _process_corporate_actions(grp, sid, divs_splits):
    if 'corporate_actions' in grp:
        ca_grp = grp['corporate_actions']
        if 'splits' in ca_grp:
            splits = pd.DataFrame(ca_grp['splits'][:])
            splits['sid'] = sid
            if 'ratio' in splits.columns:
                splits['ratio'] = 1.0 / splits['ratio'] # 역수로 만들어줘야 zipline 에서 그에 맞게 계산함 
            divs_splits['splits'] = pd.concat([divs_splits['splits'], splits], ignore_index=True)
        
        if 'dividends' in ca_grp:
            divs = pd.DataFrame(ca_grp['dividends'][:])
            divs['sid'] = sid
            # 애초에 넘겨받을 때 데이터가 없는 경우 받지 않기 때문에, 여기서 따로 체크할 필요는 없음
            divs_splits['divs'] = pd.concat([divs_splits['divs'], divs], ignore_index=True)


# Register the calendar
# register_calendar_alias("HDF5", "NYSE")