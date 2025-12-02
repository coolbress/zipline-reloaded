# Copyright 2015 Quantopian, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import warnings
from functools import partial
from itertools import tee

with warnings.catch_warnings():  # noqa
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    from bcolz import carray, ctable
    import numpy as np

import logging

import pandas as pd

from zipline.data.bar_reader import NoDataAfterDate, NoDataBeforeDate, NoDataOnDate
from zipline.data.session_bars import CurrencyAwareSessionBarReader
from zipline.utils.calendar_utils import get_calendar
from zipline.utils.cli import maybe_show_progress
from zipline.utils.functional import apply
from zipline.utils.input_validation import expect_element
from zipline.utils.memoize import lazyval
from zipline.utils.numpy_utils import float64_dtype, iNaT, uint32_dtype

from ._equities import _compute_row_slices, _read_bcolz_data, _decode_categorical_features

logger = logging.getLogger("UsEquityPricing")

OHLC = frozenset(["open", "high", "low", "close"])
US_EQUITY_PRICING_BCOLZ_COLUMNS = (
    "open",
    "high",
    "low",
    "close",
    "volume",
    "day",
    "id",
)

UINT32_MAX = np.iinfo(np.uint32).max

def check_uint32_safe(value, colname):
    if value >= UINT32_MAX:
        raise ValueError("Value %s from column '%s' is too large" % (value, colname))


@expect_element(invalid_data_behavior={"warn", "raise", "ignore"})
def winsorise_uint32(data, invalid_data_behavior, column=None, *columns):
    """Drops any record where a value would not fit into a uint32.
    
    Parameters
    ----------
    data : np.ndarray
        2D array (rows x columns) containing numeric values to clamp to the uint32 range.
    invalid_data_behavior : {'warn', 'raise', 'ignore'}
        What to do when data is outside the bounds of a uint32.
    column : int, optional
        Column index (0-based) to check. If None, all columns are checked.
    *columns : iterable[int]
        Additional column indices to check.
    
    Returns
    -------
    truncated : np.ndarray
        ``data`` with values that do not fit into a uint32 zeroed out.
    """
    if not isinstance(data, np.ndarray):
        raise TypeError(f"Expected np.ndarray, got {type(data)}")
    
    if data.ndim != 2:
        raise ValueError(f"Expected 2D array, got {data.ndim}D array")
    
    if column is None:
        col_indices = list(range(data.shape[1]))
    else:
        col_indices = list((column,) + columns)
    
    # Create mask for out-of-range values
    mask = np.zeros_like(data, dtype=bool)
    for col_idx in col_indices:
        mask[:, col_idx] = data[:, col_idx] > UINT32_MAX
    
    if invalid_data_behavior != "ignore":
        # Also mask NaN values
        for col_idx in col_indices:
            mask[:, col_idx] |= np.isnan(data[:, col_idx])
    else:
        # Replace NaN with 0 when ignoring invalid data
        data[:] = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
    
    if mask.any():
        if invalid_data_behavior == "raise":
            raise ValueError(
                "%d values out of bounds for uint32"
                % (mask.sum(),)
            )
        if invalid_data_behavior == "warn":
            warnings.warn(
                "Ignoring %d values because they are out of bounds for"
                " uint32"
                % (mask.sum(),),
                stacklevel=3,  # one extra frame for `expect_element`
            )
    
    # Always zero out invalid values, regardless of behavior mode
    data[mask] = 0
    return data


class BcolzDailyBarWriter:
    """Class capable of writing daily OHLCV data to disk in a format that can
    be read efficiently by BcolzDailyOHLCVReader.

    Parameters
    ----------
    filename : str
        The location at which we should write our output.
    calendar : zipline.utils.calendar.trading_calendar
        Calendar to use to compute asset calendar offsets.
    start_session: pd.Timestamp
        Midnight UTC session label.
    end_session: pd.Timestamp
        Midnight UTC session label.

    See Also
    --------
    zipline.data.bcolz_daily_bars.BcolzDailyBarReader
    """

    _csv_dtypes = {
        "open": float64_dtype,
        "high": float64_dtype,
        "low": float64_dtype,
        "close": float64_dtype,
        "volume": float64_dtype,
    }

    def __init__(self, filename, calendar, start_session, end_session):
        self._filename = filename
        start_session = start_session.tz_localize(None)
        end_session = end_session.tz_localize(None)

        if start_session != end_session:
            if not calendar.is_session(start_session):
                raise ValueError("Start session %s is invalid!" % start_session)
            if not calendar.is_session(end_session):
                raise ValueError("End session %s is invalid!" % end_session)

        self._start_session = start_session
        self._end_session = end_session

        self._calendar = calendar

    @property
    def progress_bar_message(self):
        return "Merging daily equity files:"

    def progress_bar_item_show_func(self, value):
        return value if value is None else str(value[0])

    def _collect_batch_metadata(self, item):
        """Process a single DataFrame to collect feature statistics.
        
        This method extracts metadata from custom features (non-OHLCV columns) in a DataFrame.
        The collected metadata is used later to determine how features should be encoded,
        scaled, and stored in the bcolz ctable.
        
        Metadata Extracted:
        -------------------
        For each custom feature, the following metadata is collected:
        
        1. **semantic_dtype**: 'numeric' or 'categorical'
           - Determines whether the feature is treated as numeric or categorical data
           - Example: 'pe' (float) -> 'numeric', 'sector' (string) -> 'categorical'
        
        2. **unique_values**: set of unique string values (categorical only)
           - All distinct non-null values found in the feature
           - Used to create encoding/decoding maps for categorical features
           - Example: {'Technology', 'Finance', 'Healthcare'} for a 'sector' feature
        
        3. **is_integer**: bool (numeric only)
           - Whether the feature's dtype is integer-based
           - Determines if the feature should be scaled by *1000 or stored as raw uint32
           - Example: volume-like features (int64) -> True, price-like features (float64) -> False
        
        4. **min_value**: float (numeric only)
           - Minimum value found in the feature (excluding outliers)
           - Outliers are excluded if abs(value) * 1000 > UINT32_MAX (for floats)
           - Used to calculate negative_offset for preserving negative values
           - Example: PB ratio with values [-2.5, 1.0, 3.5] -> min_value: -2.5
        
        Example:
        --------
        Input DataFrame with features:
            date       | pe   | pb   | sector
            -----------|------|------|--------
            2020-01-01 | 15.5 | 2.3  | Tech
            2020-01-02 | 16.0 | 2.5  | Tech
            2020-01-03 | 14.5 | -1.2 | Finance
        
        Returns:
            {
                'column_names': {'date', 'pe', 'pb', 'sector'},
                'feature_stats': {
                    'pe': {
                        'semantic_dtype': 'numeric',
                        'is_integer': False,
                        'min_value': 14.5,
                        'unique_values': None
                    },
                    'pb': {
                        'semantic_dtype': 'numeric',
                        'is_integer': False,
                        'min_value': -1.2,
                        'unique_values': None
                    },
                    'sector': {
                        'semantic_dtype': 'categorical',
                        'unique_values': {'Tech', 'Finance'},
                        'is_integer': None,
                        'min_value': None
                    }
                }
            }
        
        Parameters
        ----------
        item : tuple
            Tuple of (sid, df) where df contains OHLCV + features.
        
        Returns
        -------
        partial_stats : dict
            Dictionary with keys:
            - 'column_names': set of column names (including OHLCV and features)
            - 'feature_stats': dict mapping feature names to their statistics
              Each feature's stats include: semantic_dtype, unique_values (categorical),
              is_integer (numeric), min_value (numeric)
        """
        if len(item) != 2:
            raise ValueError(f"Expected (sid, df) tuple, got: {item}")
        
        _, df = item
        column_names = set(df.columns)
        ohlcv_columns = {'open', 'high', 'low', 'close', 'volume'}
        feature_columns = [col for col in df.columns if col not in ohlcv_columns]
        
        partial_feature_stats = {}
        
        for feature_name in feature_columns:
            feature_series = df[feature_name]
            dtype = feature_series.dtype
            
            # Detect categorical data using efficient dtype checks
            is_object_dtype = (dtype == 'object' or dtype == object or 
                             pd.api.types.is_object_dtype(dtype))
            is_categorical_dtype = pd.api.types.is_categorical_dtype(dtype)
            semantic_dtype = 'categorical' if (is_object_dtype or is_categorical_dtype) else 'numeric'
            
            stats = {
                'semantic_dtype': semantic_dtype,
                'unique_values': set() if semantic_dtype == 'categorical' else None,
                'is_integer': None,
                'min_value': None,
            }
            
            if semantic_dtype == 'categorical':
                # Extract unique values: convert to native Python strings for encoding_map consistency.
                arr = feature_series.values
                valid_mask = pd.notna(arr)
                if valid_mask.any():
                    unique_arr = np.unique(arr[valid_mask].astype(str))
                    # Convert np.str_ to native str to match encoding_map keys in to_ctable.
                    stats['unique_values'] = {str(val) for val in unique_arr}
            else:
                # Numeric processing: use numpy for faster operations
                is_integer = pd.api.types.is_integer_dtype(dtype)
                stats['is_integer'] = is_integer
                
                # Convert to numpy array for faster filtering and min calculation
                arr = feature_series.values
                valid_mask = pd.notna(arr)
                
                if valid_mask.any():
                    valid_arr = arr[valid_mask]
                    
                    # Determine outlier threshold
                    if is_integer:
                        max_safe_abs_value = float(UINT32_MAX)
                    else:
                        max_safe_abs_value = UINT32_MAX / 1000.0
                    
                    # Use numpy for fast filtering and min calculation
                    abs_valid = np.abs(valid_arr)
                    in_range_mask = abs_valid <= max_safe_abs_value
                    
                    if in_range_mask.any():
                        # Use numpy nanmin for safety (though we already filtered NaN)
                        stats['min_value'] = float(np.nanmin(valid_arr[in_range_mask]))
            
            partial_feature_stats[feature_name] = stats
        
        return {
            'column_names': column_names,
            'feature_stats': partial_feature_stats
        }

    def _unify_feature_metadata(self, data_iter):
        """Process DataFrames sequentially, merge statistics, and generate unified metadata.
        
        This method performs the complete workflow:
        1. Processes all assets' DataFrames sequentially using _collect_batch_metadata
        2. Merges partial statistics from all DataFrames
        3. Generates final unified metadata dictionary for data conversion and storage
        
        Metadata Merging Strategy:
        --------------------------
        For each feature, partial statistics from all batches are merged using the following rules:
        
        **Categorical Features:**
        - **unique_values**: Union of all unique values across all batches
          Example: Batch1 has {'Tech', 'Finance'}, Batch2 has {'Tech', 'Healthcare'}
          -> Merged: {'Tech', 'Finance', 'Healthcare'}
        
        **Numeric Features:**
        - **is_integer**: AND operation - ALL batches must be integer, otherwise treated as float
          If ANY batch has float dtype, the feature is marked as float (requiring scaling)
          Example: Batch1 (int64), Batch2 (float64) -> is_integer: False (treated as float)
          Example: Batch1 (int64), Batch2 (int64) -> is_integer: True (no scaling needed)
        - **min_value**: Minimum value across ALL batches (excluding outliers)
          Example: Batch1 min=-2.5, Batch2 min=-1.0 -> Merged min: -2.5
        
        Final Metadata Generation:
        -------------------------
        After merging all batch statistics, final metadata is generated:
        
        **For Categorical Features:**
        - Creates encoding_map: string -> uint32 (1-indexed, 0 reserved for NaN)
          Example: {'Tech': 1, 'Finance': 2, 'Healthcare': 3}
        - Creates decoding_map: uint32 -> string (reverse mapping)
          Example: {1: 'Tech', 2: 'Finance', 3: 'Healthcare', 0: None}
        
        **For Numeric Features:**
        - **scale_with_thousand**: False if is_integer=True, True otherwise
          Determines if values should be scaled by *1000 before uint32 conversion
        - **negative_offset**: Calculated as -min_value if min_value < 0, else 0.0
          Used to shift negative values into uint32 range (offset added during write,
          subtracted during read to restore original negative values)
          Example: min_value=-2.5 -> negative_offset=2.5
        
        Example:
        --------
        Batch 1 (Asset A): pb feature with values [1.5, 2.0, -1.2]
        Batch 2 (Asset B): pb feature with values [3.0, 2.5, -0.5]
        
        Merged statistics:
            - semantic_dtype: 'numeric'
            - is_integer: False
            - min_value: -1.2 (minimum across all batches)
        
        Final metadata:
            - dtype: 'uint32'
            - semantic_dtype: 'numeric'
            - scale_with_thousand: True
            - negative_offset: 1.2
        
        Why Sequential Processing?
        --------------------------
        We use sequential processing instead of parallel processing for the following reasons:
        - Generator memory efficiency: Parallel processing requires converting the generator
          to a list (list(data_iter)), which loads all DataFrames into memory at once,
          defeating the memory-efficient nature of generators.
        - Serialization overhead: Parallel processing requires pickling DataFrames for
          inter-process communication, which is expensive for large DataFrames.
        - Process creation cost: For small datasets, the overhead of creating worker
          processes outweighs the benefits of parallelization.
        - Sequential processing maintains the generator's lazy evaluation benefits,
          processing one DataFrame at a time with minimal memory footprint.
        
        Parameters
        ----------
        data_iter : iterable
            Iterator yielding (sid, df) tuples where df contains OHLCV + features.
        
        Returns
        -------
        unified_metadata : dict
            Dictionary mapping feature names to their metadata.
            Each feature's metadata includes:
            - 'dtype': Storage dtype (always 'uint32')
            - 'semantic_dtype': 'numeric' or 'categorical'
            - For categorical: 'encoding_map' and 'decoding_map'
            - For numeric: 'scale_with_thousand' and 'negative_offset'
        all_column_names_set : set
            Set containing all column names collected across all assets.
        """
        # Step 1: Process DataFrames sequentially and collect statistics
        feature_stats = {}
        all_column_names_set = set()
        
        for item in data_iter:
            # Process single DataFrame to collect partial statistics
            result = self._collect_batch_metadata(item)
            
            # Merge column names
            all_column_names_set.update(result['column_names'])
            
            # Merge feature statistics
            for feature_name, partial_stats in result['feature_stats'].items():
                if feature_name not in feature_stats:
                    # First encounter: use partial stats as-is
                    feature_stats[feature_name] = partial_stats.copy()
                else:
                    # Merge with existing stats
                    existing = feature_stats[feature_name]
                    
                    # Merge unique_values (for categorical)
                    if existing['unique_values'] is not None and partial_stats['unique_values'] is not None:
                        existing['unique_values'].update(partial_stats['unique_values'])
                    
                    # Update is_integer flag: if ANY batch is float, mark as float (not integer)
                    # This ensures mixed int/float batches are treated as float (requiring scaling)
                    # Logic: ALL batches must be integer to be treated as integer
                    partial_is_integer = partial_stats.get('is_integer')
                    if partial_is_integer is False:
                        # If this batch is float, the feature must be treated as float
                        existing['is_integer'] = False
                    elif partial_is_integer is True:
                        # Only keep True if we haven't seen a float batch yet
                        # If existing is already False (from a previous float batch), keep it False
                        if existing.get('is_integer') is not False:
                            existing['is_integer'] = True
                    
                    # Update min_value (take minimum across all batches)
                    partial_min = partial_stats.get('min_value')
                    if partial_min is not None:
                        if existing['min_value'] is None:
                            existing['min_value'] = partial_min
                        else:
                            existing['min_value'] = min(existing['min_value'], partial_min)
        
        # Step 2: Generate unified metadata from merged statistics
        unified_metadata = {}
        for feature_name, stats in feature_stats.items():
            semantic_dtype = stats.get('semantic_dtype', 'numeric')
            is_categorical = (semantic_dtype == 'categorical')
            
            metadata = {
                'dtype': 'uint32',
                'semantic_dtype': semantic_dtype,
            }
            
            if is_categorical:
                # Create encoding/decoding maps: convert all values to strings for consistency.
                # This ensures encoding_map keys match astype(str) results in to_ctable.
                unique_values = stats.get('unique_values', set())
                if unique_values:
                    sorted_values = sorted(str(val) for val in unique_values)
                    encoding_map = {val: idx + 1 for idx, val in enumerate(sorted_values)}
                    decoding_map = {idx + 1: val for idx, val in enumerate(sorted_values)}
                    decoding_map[0] = None  # 0 reserved for NaN
                    
                    metadata['encoding_map'] = encoding_map
                    metadata['decoding_map'] = decoding_map
            else:
                # Numeric features: integers are stored without scaling
                metadata['scale_with_thousand'] = not bool(stats.get('is_integer', False))
                min_value = stats.get('min_value')
                if min_value is not None and pd.notna(min_value) and min_value < 0:
                    metadata['negative_offset'] = float(-min_value)
                else:
                    metadata['negative_offset'] = 0.0
            
            unified_metadata[feature_name] = metadata
        
        return unified_metadata, all_column_names_set

    def write(
        self, data, assets=None, show_progress=False, invalid_data_behavior="warn"
    ):
        """Write OHLCV data and optional custom features to bcolz ctable.

        Parameters
        ----------
        data : iterable[tuple[int, pandas.DataFrame]]
            The data chunks to write. Each chunk should be a tuple of:
            - (sid, df) where df contains OHLCV columns (open, high, low, close, volume)
              and optionally custom feature columns. All columns must have the same length
              and be aligned with the same date index.
        assets : set[int], optional
            The assets that should be in ``data``. If this is provided
            we will check ``data`` against the assets and provide better
            progress information.
        show_progress : bool, optional
            Whether or not to show a progress bar while writing.
        invalid_data_behavior : {'warn', 'raise', 'ignore'}, optional
            What to do when data is encountered that is outside the range of
            a uint32.

        Returns
        -------
        table : bcolz.ctable
            The newly-written table with OHLCV and feature columns. 
        """
        # Data is in (sid, df) format from factory functions (hdfdir, csvdir)
        # df contains OHLCV + Features in a single DataFrame
        # data is a generator (original Zipline pattern)
        
        # Use itertools.tee() to split iterator for two passes
        # This allows us to iterate twice while maintaining the original generator pattern
        data_iter1, data_iter2 = tee(data, 2)
        
        # Pass 1: Process DataFrames sequentially, merge statistics, and generate unified metadata
        unified_metadata, all_column_names_set = self._unify_feature_metadata(data_iter1)
        
        # Convert collected column names to sorted list (OHLCV first, then features)
        # Note: 'day' and 'id' are added by to_ctable(), so they're not in all_column_names_set yet
        # But we know they will be present in all ctables, so we include them
        ohlcv_list = ['open', 'high', 'low', 'close', 'volume', 'day', 'id']
        feature_list = sorted([col for col in all_column_names_set if col not in {'open', 'high', 'low', 'close', 'volume'}])
        all_column_names = ohlcv_list + feature_list
        
        # Pass 2: Create Ctable for each asset with unified metadata
        # unified_metadata is column-based (not asset-based), so it applies to all assets
        # to_ctable will automatically use the metadata for features present in the DataFrame
        ctx = maybe_show_progress(
            ((sid, self.to_ctable(df, invalid_data_behavior, feature_metadata=unified_metadata)) 
             for sid, df in data_iter2),
            show_progress=show_progress,
            item_show_func=self.progress_bar_item_show_func,
            label=self.progress_bar_message,
            length=len(assets) if assets is not None else None,
        )
        
        with ctx as it:
            return self._write_internal(
                it, 
                assets, 
                feature_metadata=unified_metadata,  # Pass unified metadata (from all assets)
                all_column_names=all_column_names  # Pass all column names
            )

    def write_csvs(self, asset_map, show_progress=False, invalid_data_behavior="warn"):
        """Read CSVs as DataFrames from our asset map.

        Parameters
        ----------
        asset_map : dict[int -> str]
            A mapping from asset id to file path with the CSV data for that
            asset
        show_progress : bool
            Whether or not to show a progress bar while writing.
        invalid_data_behavior : {'warn', 'raise', 'ignore'}
            What to do when data is encountered that is outside the range of
            a uint32.
        """
        read = partial(
            pd.read_csv,
            parse_dates=["day"],
            index_col="day",
            dtype=self._csv_dtypes,
        )
        return self.write(
            ((asset, read(path)) for asset, path in asset_map.items()),
            assets=asset_map.keys(),
            show_progress=show_progress,
            invalid_data_behavior=invalid_data_behavior,
        )

    def _write_internal(self, iterator, assets, feature_metadata=None, all_column_names=None):
        """Merge multiple asset ctables into a single unified ctable.
        
        This method takes individual asset ctables (created by `to_ctable()`) and
        merges them into one consolidated ctable containing all assets' data.
        
        Role:
        - Merges multiple asset ctables (already in uint32 format) into one ctable
        - Handles missing columns by filling with zeros (uint32)
        - Calculates first_row, last_row, and calendar_offset for each asset
        - Stores feature_metadata in the final table's attrs
        
        Parameters
        ----------
        iterator : iterable[tuple[int, bcolz.ctable]]
            Iterator yielding (asset_id, ctable) pairs. Each ctable contains a single
            asset's OHLCV + features data, already converted to uint32 by `to_ctable()`.
        assets : set[int], optional
            Set of expected asset IDs for validation.
        feature_metadata : dict[str, dict], optional
            Unified metadata for custom features (collected in Pass 1). Stored in
            the final table's attrs for later retrieval by the reader.
        all_column_names : list[str], optional
            List of all column names from all assets (collected in Pass 1). Ensures
            all assets have the same column structure, with missing columns filled
            with zeros.
        
        Returns
        -------
        ctable : bcolz.ctable
            Unified ctable containing all assets' data with consistent column structure.
        """
        total_rows = 0
        first_row = {}
        last_row = {}
        calendar_offset = {}

        # Initialize columns with all_column_names (collected in Pass 1)
        # all columns must been converted to uint32 at to_ctable 
        columns = {}
        for colname in all_column_names:
            columns[colname] = carray(np.array([], dtype=uint32_dtype))

        earliest_date = None
        sessions = self._calendar.sessions_in_range(
            self._start_session, self._end_session
        )

        if assets is not None:

            @apply
            def iterator(iterator=iterator, assets=set(assets)):
                for asset_id, table in iterator:
                    if asset_id not in assets:
                        raise ValueError("unknown asset id %r" % asset_id)
                    yield asset_id, table

        for asset_id, table in iterator:
            nrows = len(table)
            
            # Append all columns from this asset's table (OHLCV + features)
            for column_name in all_column_names:
                if column_name == "id":
                    # We know what the content of this column is, so don't
                    # bother reading it.
                    columns["id"].append(
                        np.full((nrows,), asset_id, dtype="uint32"),
                    )
                    continue

                # Check if this asset has this column
                if column_name in table.names:
                    # All other columns (OHLCV + features) are already in uint32 format
                    # from to_ctable(), so we can directly append them
                    columns[column_name].append(table[column_name])
                else:
                    # This asset doesn't have this column (e.g., feature missing for this asset)
                    # Fill with zeros (uint32) - this represents missing data
                    columns[column_name].append(np.zeros(nrows, dtype=uint32_dtype))

            if earliest_date is None:
                earliest_date = table["day"][0]
            else:
                earliest_date = min(earliest_date, table["day"][0])

            # Bcolz doesn't support ints as keys in `attrs`, so convert
            # assets to strings for use as attr keys.
            asset_key = str(asset_id)

            # Calculate the index into the array of the first and last row
            # for this asset. This allows us to efficiently load single
            # assets when querying the data back out of the table.
            first_row[asset_key] = total_rows
            last_row[asset_key] = total_rows + nrows - 1
            total_rows += nrows

            asset_first_day = pd.Timestamp(table["day"][0], unit="s").normalize()
            asset_last_day = pd.Timestamp(table["day"][-1], unit="s").normalize()

            asset_sessions = sessions[
                sessions.slice_indexer(asset_first_day, asset_last_day)
            ]
            if len(table) != len(asset_sessions):

                missing_sessions = asset_sessions.difference(
                    pd.to_datetime(np.array(table["day"]), unit="s")
                ).tolist()

                extra_sessions = (
                    pd.to_datetime(np.array(table["day"]), unit="s")
                    .difference(asset_sessions)
                    .tolist()
                )
                raise AssertionError(
                    f"Got {len(table)} rows for daily bars table with "
                    f"first day={asset_first_day.date()}, last "
                    f"day={asset_last_day.date()}, expected {len(asset_sessions)} rows.\n"
                    f"Missing sessions: {missing_sessions}\nExtra sessions: {extra_sessions}"
                )

            # assert len(table) == len(asset_sessions), (

            # Calculate the number of trading days between the first date
            # in the stored data and the first date of **this** asset. This
            # offset used for output alignment by the reader.
            calendar_offset[asset_key] = sessions.get_loc(asset_first_day)

        # This writes the table to disk.
        # all_column_names was determined from the first table (OHLCV + features)
        full_table = ctable(
            columns=[columns[colname] for colname in all_column_names],
            names=all_column_names,
            rootdir=self._filename,
            mode="w",
        )

        full_table.attrs["first_trading_day"] = (
            earliest_date if earliest_date is not None else iNaT
        )

        full_table.attrs["first_row"] = first_row
        full_table.attrs["last_row"] = last_row
        full_table.attrs["calendar_offset"] = calendar_offset
        full_table.attrs["calendar_name"] = self._calendar.name
        full_table.attrs["start_session_ns"] = self._start_session.value
        full_table.attrs["end_session_ns"] = self._end_session.value
        
        # Store feature metadata if provided (already processed in to_ctable())
        if feature_metadata:
            full_table.attrs["features"] = feature_metadata
        
        full_table.flush()
        return full_table

    @expect_element(invalid_data_behavior={"warn", "raise", "ignore"})
    def to_ctable(self, raw_data, invalid_data_behavior, feature_metadata=None):
        """Convert a single asset's DataFrame to bcolz ctable with uint32 conversion.
        
        This method processes a single asset's OHLCV and custom features data,
        converting all columns to uint32 format for efficient storage in bcolz.
        
        Role:
        - Converts DataFrame (single asset) to ctable (single asset)
        - Applies data type conversion: float → uint32 (with * 1000 scaling for OHLCV/numeric),
          string → uint32 (with encoding_map for categorical)
        - All numeric columns (OHLCV + numeric features) use * 1000 scaling
        - Categorical features use encoding_map from unified metadata
        
        Parameters
        ----------
        raw_data : pd.DataFrame
            DataFrame containing OHLCV data (open, high, low, close, volume) and optionally
            custom feature columns. Features are automatically detected from columns that
            are not OHLCV.
        invalid_data_behavior : {'warn', 'raise', 'ignore'}
            How to handle data outside uint32 range after scaling.
        feature_metadata : dict[str, dict], optional
            Unified metadata from Pass 1 (collected from all assets). If provided, this
            metadata will be used for feature conversion. If None, features will not be processed.
        
        Returns
        -------
        ctable : bcolz.ctable
            Single asset's ctable with all columns converted to uint32 format.
            Columns include: OHLCV (open, high, low, close, volume), day, id, and
            custom features (if any).
        """
        if isinstance(raw_data, ctable):
            # we already have a ctable so do nothing
            return raw_data

        # Auto-detect features from raw_data columns
        # raw_data contains OHLCV + features in a single DataFrame
        ohlcv_columns = {'open', 'high', 'low', 'close', 'volume'}
        feature_columns = [col for col in raw_data.columns if col not in ohlcv_columns]
        
        # Separate columns by type: numeric (OHLC + numeric features) vs categorical
        # Note: volume is treated like a no-scale column
        numeric_feature_columns_scaled = []
        numeric_feature_columns_no_scale = []
        categorical_columns = []
        
        feature_offsets = {}
        if feature_columns and feature_metadata:
            for col in feature_columns:
                if col in feature_metadata:
                    feature_meta = feature_metadata[col]
                    semantic_dtype = feature_meta.get('semantic_dtype', 'numeric')
                    if semantic_dtype == 'categorical':
                        categorical_columns.append(col)
                    else:
                        if feature_meta.get('scale_with_thousand', True):
                            numeric_feature_columns_scaled.append(col)
                        else:
                            numeric_feature_columns_no_scale.append(col)
                        offset = float(feature_meta.get('negative_offset', 0.0) or 0.0)
                        if offset:
                            feature_offsets[col] = offset
        
        # Process all numeric columns that need scaling (OHLC + numeric features) together
        all_scaled_columns = list(OHLC) + numeric_feature_columns_scaled
        
        # Check all numeric columns (OHLC + numeric features + volume) together
        # Same as original Zipline: winsorise_uint32(raw_data, invalid_data_behavior, "volume", *OHLC)
        all_numeric_columns = all_scaled_columns + numeric_feature_columns_no_scale + ["volume"]
        numeric_subset = raw_data[all_numeric_columns].to_numpy(copy=True)
        
        # Apply negative offsets vectorized (faster than loop)
        if feature_offsets:
            # Create offset array matching column order
            offset_array = np.array([feature_offsets.get(col_name, 0.0) for col_name in all_numeric_columns])
            if offset_array.any():
                # Vectorized addition: add offsets to all columns at once
                numeric_subset += offset_array
        
        winsorise_uint32(
            numeric_subset,
            invalid_data_behavior,
            *tuple(range(numeric_subset.shape[1])),
        )
        
        # Find column indices for scaled vs no-scale columns
        scaled_col_indices = [all_numeric_columns.index(col) for col in all_scaled_columns]
        no_scale_col_indices = [all_numeric_columns.index(col) for col in (numeric_feature_columns_no_scale + ["volume"])]
        
        # Initialize processed DataFrame
        processed = pd.DataFrame(index=raw_data.index)
        
        # Process scaled columns: scale by 1000 and convert to uint32
        if all_scaled_columns:
            # Use numeric_subset directly (avoid DataFrame roundtrip)
            scaled_subset = numeric_subset[:, scaled_col_indices]
            scaled_values = (np.nan_to_num(scaled_subset) * 1000).round()
            scaled_values = np.nan_to_num(scaled_values, nan=0.0, posinf=0.0, neginf=0.0)
            
            for idx, col_name in enumerate(all_scaled_columns):
                processed[col_name] = scaled_values[:, idx].astype(np.uint32)
        
        # Process no-scale columns: direct uint32 conversion
        if no_scale_col_indices:
            no_scale_subset = numeric_subset[:, no_scale_col_indices]
            no_scale_values = np.nan_to_num(no_scale_subset, nan=0.0, posinf=0.0, neginf=0.0)
            
            for idx, col_name in enumerate(numeric_feature_columns_no_scale + ["volume"]):
                processed[col_name] = no_scale_values[:, idx].astype(np.uint32)
        
        # Add day column
        dates = raw_data.index.values.astype("datetime64[s]")
        check_uint32_safe(dates.max().view(np.int64), "day")
        processed["day"] = dates.astype("uint32")
        
        # Process categorical features if present
        if categorical_columns and feature_metadata:
            for feature_name in categorical_columns:
                feature_meta = feature_metadata[feature_name]
                encoding_map = feature_meta.get('encoding_map', None)
                
                if encoding_map is None:
                    raise ValueError(
                        f"Feature '{feature_name}' is categorical but has no encoding_map. "
                        "This should have been generated in Pass 1."
                    )
                
                feature_series = raw_data[feature_name]
                feature_array = feature_series.values
                
                # Categorical: encode strings to integers using vectorized operations
                # Create mask for valid (non-NaN) values
                valid_mask = pd.notna(feature_array)
                
                # Initialize encoded array with zeros (NaN -> 0)
                encoded_array = np.zeros(len(feature_array), dtype=np.uint32)
                
                # Only process valid values
                if valid_mask.any():
                    valid_values = feature_array[valid_mask]
                    str_values = valid_values.astype(str)
                    
                    # Vectorized encoding: map string values to integers
                    # Use list comprehension for mapping (faster than loop for small maps)
                    encoded_values = np.array([encoding_map.get(str_val, 0) for str_val in str_values], dtype=np.uint32)
                    encoded_array[valid_mask] = encoded_values
                
                processed[feature_name] = encoded_array
        
        return ctable.fromdataframe(processed)


class BcolzDailyBarReader(CurrencyAwareSessionBarReader):
    """Reader for raw pricing data written by BcolzDailyOHLCVWriter.

    Parameters
    ----------
    table : bcolz.ctable
        The ctable contaning the pricing data, with attrs corresponding to the
        Attributes list below.
    read_all_threshold : int
        The number of equities at which; below, the data is read by reading a
        slice from the carray per asset.  above, the data is read by pulling
        all of the data for all assets into memory and then indexing into that
        array for each day and asset pair.  Used to tune performance of reads
        when using a small or large number of equities.

    Attributes
    ----------
    The table with which this loader interacts contains the following
    attributes:

    first_row : dict
        Map from asset_id -> index of first row in the dataset with that id.
    last_row : dict
        Map from asset_id -> index of last row in the dataset with that id.
    calendar_offset : dict
        Map from asset_id -> calendar index of first row.
    start_session_ns: int
        Epoch ns of the first session used in this dataset.
    end_session_ns: int
        Epoch ns of the last session used in this dataset.
    calendar_name: str
        String identifier of trading calendar used (ie, "NYSE").

    We use first_row and last_row together to quickly find ranges of rows to
    load when reading an asset's data into memory.

    We use calendar_offset and calendar to orient loaded blocks within a
    range of queried dates.

    Notes
    ------
    A Bcolz CTable is comprised of Columns and Attributes.
    The table with which this loader interacts contains the following columns:

    ['open', 'high', 'low', 'close', 'volume', 'day', 'id'].

    The data in these columns is interpreted as follows:

    - Price columns ('open', 'high', 'low', 'close') are interpreted as 1000 *
      as-traded dollar value.
    - Volume is interpreted as as-traded volume.
    - Day is interpreted as seconds since midnight UTC, Jan 1, 1970.
    - Id is the asset id of the row.

    The data in each column is grouped by asset and then sorted by day within
    each asset block.

    The table is built to represent a long time range of data, e.g. ten years
    of equity data, so the lengths of each asset block is not equal to each
    other. The blocks are clipped to the known start and end date of each asset
    to cut down on the number of empty values that would need to be included to
    make a regular/cubic dataset.

    When read across the open, high, low, close, and volume with the same
    index should represent the same asset and day.

    See Also
    --------
    zipline.data.bcolz_daily_bars.BcolzDailyBarReader
    """

    def __init__(self, table, read_all_threshold=3000):
        self._maybe_table_rootdir = table
        # Cache of fully read np.array for the carrays in the daily bar table.
        # raw_array does not use the same cache, but it could.
        # Need to test keeping the entire array in memory for the course of a
        # process first.
        self._spot_cols = {}
        self.PRICE_ADJUSTMENT_FACTOR = 0.001
        self._read_all_threshold = read_all_threshold

    @lazyval
    def _table(self):
        maybe_table_rootdir = self._maybe_table_rootdir
        if isinstance(maybe_table_rootdir, ctable):
            return maybe_table_rootdir
        return ctable(rootdir=maybe_table_rootdir, mode="r")

    @lazyval
    def sessions(self):
        if "calendar" in self._table.attrs.attrs:
            # backwards compatibility with old formats, will remove
            return pd.DatetimeIndex(self._table.attrs["calendar"])
        else:
            cal = get_calendar(self._table.attrs["calendar_name"])
            start_session_ns = self._table.attrs["start_session_ns"]

            start_session = pd.Timestamp(start_session_ns)

            end_session_ns = self._table.attrs["end_session_ns"]
            end_session = pd.Timestamp(end_session_ns)

            sessions = cal.sessions_in_range(start_session, end_session)

            return sessions

    @lazyval
    def _first_rows(self):
        return {
            int(asset_id): start_index
            for asset_id, start_index in self._table.attrs["first_row"].items()
        }

    @lazyval
    def _last_rows(self):
        return {
            int(asset_id): end_index
            for asset_id, end_index in self._table.attrs["last_row"].items()
        }

    @lazyval
    def _calendar_offsets(self):
        return {
            int(id_): offset
            for id_, offset in self._table.attrs["calendar_offset"].items()
        }
    
    @lazyval
    def _feature_metadata(self):
        """Get feature metadata from table attrs.
        
        Note: bcolz attrs stores data as JSON, which converts dictionary keys to strings.
        For decoding_map, we need to convert string keys back to integers.
        """
        try:
            raw_metadata = self._table.attrs["features"]
            # Convert decoding_map keys from string to int: JSON serialization converts int keys to strings.
            processed_metadata = {}
            for feat_name, feat_meta in raw_metadata.items():
                processed_meta = feat_meta.copy()
                if 'decoding_map' in processed_meta:
                    decoding_map = processed_meta['decoding_map']
                    if decoding_map:
                        sample_key = next(iter(decoding_map.keys()))
                        if isinstance(sample_key, str):
                            # Convert string keys back to integers for uint32 lookup.
                            processed_meta['decoding_map'] = {
                                int(k): v for k, v in decoding_map.items()
                            }
                processed_metadata[feat_name] = processed_meta
            return processed_metadata
        except KeyError:
            return {}
    
    def feature_names(self):
        """
        Get list of custom feature names available in this bundle.

        Returns
        -------
        feature_names : list[str]
            List of feature names, sorted alphabetically.
        """
        return sorted(self._feature_metadata.keys())
    
    def feature_metadata(self):
        """
        Get feature metadata dictionary.

        Returns
        -------
        feature_metadata : dict
            Dictionary mapping feature names to their metadata.
            Each feature's metadata includes:
            - 'dtype': Storage dtype (e.g., 'uint32', 'float64')
            - 'scaling_factor': Scaling factor applied (1.0 if no scaling)
            - 'semantic_dtype': 'numeric' or 'categorical'
            - 'encoding_map': (categorical only) Maps string -> int
            - 'decoding_map': (categorical only) Maps int -> string
        """
        return self._feature_metadata.copy()

    @lazyval
    def first_trading_day(self):
        try:
            return pd.Timestamp(self._table.attrs["first_trading_day"], unit="s")
        except KeyError:
            return None

    @lazyval
    def trading_calendar(self):
        if "calendar_name" in self._table.attrs.attrs:
            return get_calendar(self._table.attrs["calendar_name"])
        else:
            return None

    @property
    def last_available_dt(self):
        return self.sessions[-1]

    def _compute_slices(self, start_idx, end_idx, assets):
        """Compute the raw row indices to load for each asset on a query for the
        given dates after applying a shift.

        Parameters
        ----------
        start_idx : int
            Index of first date for which we want data.
        end_idx : int
            Index of last date for which we want data.
        assets : pandas.Int64Index
            Assets for which we want to compute row indices

        Returns
        -------
        A 3-tuple of (first_rows, last_rows, offsets):
        first_rows : np.array[intp]
            Array with length == len(assets) containing the index of the first
            row to load for each asset in `assets`.
        last_rows : np.array[intp]
            Array with length == len(assets) containing the index of the last
            row to load for each asset in `assets`.
        offset : np.array[intp]
            Array with length == (len(asset) containing the index in a buffer
            of length `dates` corresponding to the first row of each asset.

            The value of offset[i] will be 0 if asset[i] existed at the start
            of a query.  Otherwise, offset[i] will be equal to the number of
            entries in `dates` for which the asset did not yet exist.
        """
        # The core implementation of the logic here is implemented in Cython
        # for efficiency.
        return _compute_row_slices(
            self._first_rows,
            self._last_rows,
            self._calendar_offsets,
            start_idx,
            end_idx,
            assets,
        )

    def load_raw_arrays(self, columns, start_date, end_date, assets):
        start_idx = self._load_raw_arrays_date_to_index(start_date)
        end_idx = self._load_raw_arrays_date_to_index(end_date)

        first_rows, last_rows, offsets = self._compute_slices(
            start_idx,
            end_idx,
            assets,
        )
        
        # Separate columns by processing type
        # All numeric columns (OHLCV + numeric features) are processed together
        # Categorical features are processed separately
        ohlcv_columns = [col for col in columns if col in US_EQUITY_PRICING_BCOLZ_COLUMNS]
        feature_columns = [col for col in columns if col not in US_EQUITY_PRICING_BCOLZ_COLUMNS]
        
        # Separate features by type (numeric vs categorical) and scaling strategy
        numeric_features_scaled = []
        numeric_features_no_scale = []
        categorical_features = []
        missing_feature_columns = []
        feature_offsets = {}
        feature_meta_cache = {}
        
        feature_metadata = self._feature_metadata

        if feature_columns:
            for col_name in feature_columns:
                if col_name not in self._table.names:
                    missing_feature_columns.append(col_name)
                    continue
                
                feature_meta = feature_metadata.get(col_name, {})
                feature_meta_cache[col_name] = feature_meta
                semantic_dtype = feature_meta.get('semantic_dtype', 'numeric')
                decoding_map = feature_meta.get('decoding_map', None)
                
                if semantic_dtype == 'categorical' and decoding_map is not None:
                    categorical_features.append(col_name)
                else:
                    if feature_meta.get('scale_with_thousand', True):
                        numeric_features_scaled.append(col_name)
                    else:
                        numeric_features_no_scale.append(col_name)
                    
                    offset = float(feature_meta.get('negative_offset', 0.0) or 0.0)
                    if offset:
                        feature_offsets[col_name] = offset
        
        # Determine which OHLCV price columns were requested (exclude volume for scaling step)
        ohlc_price_columns = [col for col in ohlcv_columns if col in OHLC]
        include_volume = "volume" in ohlcv_columns
        
        # Combine OHLCV price columns and numeric features that use scaling
        all_numeric_columns_scaled = ohlc_price_columns + numeric_features_scaled
        
        # Columns that should bypass scaling (volume + integer-like features)
        numeric_no_scale_columns = []
        if include_volume:
            numeric_no_scale_columns.append("volume")
        numeric_no_scale_columns.extend(numeric_features_no_scale)
        
        shape = (end_idx - start_idx + 1, len(assets))
        result_map = {}  # Map column name to result array
        read_all_flag = len(assets) > self._read_all_threshold
        
        # Read numeric columns (scaled + no-scale) in a single pass
        numeric_columns_to_read = all_numeric_columns_scaled + numeric_no_scale_columns
        if numeric_columns_to_read:
            no_scale_set = set(numeric_no_scale_columns)
            numeric_results = _read_bcolz_data(
                self._table,
                shape,
                numeric_columns_to_read,
                first_rows,
                last_rows,
                offsets,
                read_all_flag,
                no_scale_columns=no_scale_set if no_scale_set else None,
            )
            for col_name, result in zip(numeric_columns_to_read, numeric_results):
                # Ensure float64 for downstream math
                result = result.astype(np.float64, copy=False)
                result[result == 0] = np.nan
                offset = feature_offsets.get(col_name, 0.0)
                if offset:
                    result = result - offset
                result_map[col_name] = result
        
        # Read categorical features separately
        if categorical_features:
            categorical_uint32_results = _read_bcolz_data(
                self._table, shape, categorical_features,
                first_rows, last_rows, offsets, read_all_flag,
                no_scale_columns=set(categorical_features),
            )
            for col_name, feature_uint32 in zip(categorical_features, categorical_uint32_results):
                decoding_map = feature_meta_cache.get(col_name, {}).get('decoding_map', {})
                # Use Cython function for fast decoding
                feature_array = _decode_categorical_features(feature_uint32, decoding_map)
                result_map[col_name] = feature_array
        
        # Add NaN arrays for missing features
        for col_name in missing_feature_columns:
            result_map[col_name] = np.full(shape, np.nan, dtype=np.float64)
        
        # Return results in original column order
        return [result_map[col] for col in columns]

    def _load_raw_arrays_date_to_index(self, date):
        try:
            # TODO get_loc is deprecated but get_indexer doesnt raise and error
            return self.sessions.get_loc(date)
        except KeyError as exc:
            raise NoDataOnDate(date) from exc

    def _spot_col(self, colname):
        """Get the colname from daily_bar_table and read all of it into memory,
        caching the result.

        Parameters
        ----------
        colname : string
            A name of a OHLCV carray in the daily_bar_table

        Returns
        -------
        array (uint32)
            Full read array of the carray in the daily_bar_table with the
            given colname.
        """
        try:
            col = self._spot_cols[colname]
        except KeyError:
            col = self._spot_cols[colname] = self._table[colname]
        return col

    def get_last_traded_dt(self, asset, day):
        volumes = self._spot_col("volume")

        search_day = day

        while True:
            try:
                ix = self.sid_day_index(asset, search_day)
            except NoDataBeforeDate:
                return pd.NaT
            except NoDataAfterDate:
                prev_day_ix = self.sessions.get_loc(search_day) - 1
                if prev_day_ix > -1:
                    search_day = self.sessions[prev_day_ix]
                continue
            except NoDataOnDate:
                return pd.NaT
            if volumes[ix] != 0:
                return search_day
            prev_day_ix = self.sessions.get_loc(search_day) - 1
            if prev_day_ix > -1:
                search_day = self.sessions[prev_day_ix]
            else:
                return pd.NaT

    def sid_day_index(self, sid, day):
        """

        Parameters
        ----------
        sid : int
            The asset identifier.
        day : datetime64-like
            Midnight of the day for which data is requested.

        Returns
        -------
        int
            Index into the data tape for the given sid and day.
            Raises a NoDataOnDate exception if the given day and sid is before
            or after the date range of the equity.
        """
        try:
            day_loc = self.sessions.get_loc(day)
        except Exception as exc:
            raise NoDataOnDate(
                "day={0} is outside of calendar={1}".format(day, self.sessions)
            ) from exc
        offset = day_loc - self._calendar_offsets[sid]
        if offset < 0:
            raise NoDataBeforeDate(
                "No data on or before day={0} for sid={1}".format(day, sid)
            )
        ix = self._first_rows[sid] + offset
        if ix > self._last_rows[sid]:
            raise NoDataAfterDate(
                "No data on or after day={0} for sid={1}".format(day, sid)
            )
        return ix

    def get_value(self, sid, dt, field):
        """Get a single value for a sid and date.

        Parameters
        ----------
        sid : int
            The asset identifier.
        dt : datetime64-like
            Midnight of the day for which data is requested.
        field : string
            The field name. e.g. ('open', 'high', 'low', 'close', 'volume') for OHLCV,
            or feature name for custom features.

        Returns
        -------
        float
            The value for the given field of the given sid on the given day.
            Raises a NoDataOnDate exception if the given day and sid is before
            or after the date range of the equity.
            Returns np.nan if the day is within the date range, but the value is
            missing or 0 (for OHLCV price fields).
        """
        ix = self.sid_day_index(sid, dt)
        
        # Check if it's a feature column
        if field not in US_EQUITY_PRICING_BCOLZ_COLUMNS:
            # Feature column
            if field not in self._table.names:
                return np.nan
            
            feature_meta = self._feature_metadata.get(field, {})
            semantic_dtype = feature_meta.get('semantic_dtype', 'numeric')
            decoding_map = feature_meta.get('decoding_map', None)
            
            feature_col = self._spot_col(field)
            value = feature_col[ix]
            
            # Apply inverse scaling or decoding based on metadata
            if semantic_dtype == 'categorical' and decoding_map is not None:
                # Categorical: decode integer back to string
                return None if value == 0 else decoding_map.get(int(value), None)
            else:
                # Numeric: Always use OHLCV-like inverse scaling (* 0.001)
                # No need to check scaling_factor - all numeric features use * 1000 during write
                value = float(value) * 0.001
                
                # Return NaN for missing or zero values
                return np.nan if (value == 0 or np.isnan(value)) else value
        
        # OHLCV column
        price = self._spot_col(field)[ix]
        if field != "volume":
            if price == 0:
                return np.nan
            else:
                return price * 0.001
        else:
            return price

    def currency_codes(self, sids):
        # XXX: This is pretty inefficient. This reader doesn't really support
        # country codes, so we always either return USD or None if we don't
        # know about the sid at all.
        first_rows = self._first_rows
        out = []
        for sid in sids:
            if sid in first_rows:
                out.append("USD")
            else:
                out.append(None)
        return np.array(out, dtype=object)
