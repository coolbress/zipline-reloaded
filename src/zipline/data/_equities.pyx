#
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
import bcolz
cimport cython
from cpython cimport bool

from numpy import (
    array,
    float64,
    full,
    intp,
    uint32,
    zeros,
)
from numpy cimport (
    float64_t,
    intp_t,
    ndarray,
    uint32_t,
    uint8_t,
)
from libc.math cimport NAN
ctypedef object carray_t
ctypedef object ctable_t
ctypedef object Timestamp_t
ctypedef object DatetimeIndex_t
ctypedef object Int64Index_t


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef _compute_row_slices(dict asset_starts_absolute,
                          dict asset_ends_absolute,
                          dict asset_starts_calendar,
                          intp_t query_start,
                          intp_t query_end,
                          Int64Index_t requested_assets):
    """
    Core indexing functionality for loading raw data from bcolz.

    For each asset in requested assets, computes three values:

    1.) The index in the raw bcolz data of first row to load.
    2.) The index in the raw bcolz data of the last row to load.
    3.) The index in the dates of our query corresponding to the first row for
        each asset. This is non-zero iff the asset's lifetime begins partway
        through the requested query dates.

    Values for unknown sids will be populated with a value of -1.

    Parameters
    ----------
    asset_starts_absolute : dict
        Dictionary containing the index of the first row of each asset in the
        bcolz file from which we will query.
    asset_ends_absolute : dict
        Dictionary containing the index of the last row of each asset in the
        bcolz file from which we will query.
    asset_starts_calendar : dict
        Dictionary containing the index of in our calendar corresponding to the
        start date of each asset
    query_start : intp
        Start index in our calendar of the dates for which we're querying.
    query_end : intp
        End index in our calendar of the dates for which we're querying.
    requested_assets : pandas.Int64Index
        The assets for which we want to load data.

    Returns
    -------
    first_rows, last_rows, offsets : 3-tuple of ndarrays
    """
    cdef:
        intp_t nassets = len(requested_assets)

        # For each sid, we need to compute the following:
        ndarray[dtype=intp_t, ndim=1] first_row_a = full(nassets, -1, dtype=intp)
        ndarray[dtype=intp_t, ndim=1] last_row_a = full(nassets, -1, dtype=intp)
        ndarray[dtype=intp_t, ndim=1] offset_a = full(nassets, -1, dtype=intp)

        # Loop variables.
        intp_t i
        intp_t asset
        intp_t asset_start_data
        intp_t asset_end_data
        intp_t asset_start_calendar
        intp_t asset_end_calendar

        # Flag to check whether we should raise an error because we don't know
        # about any of the requested sids.
        uint8_t any_hits = 0

    for i, asset in enumerate(requested_assets):
        if asset not in asset_starts_absolute:
            # This is an unknown asset, leave its slot empty.
            continue

        any_hits = 1

        asset_start_data = asset_starts_absolute[asset]
        asset_end_data = asset_ends_absolute[asset]
        asset_start_calendar = asset_starts_calendar[asset]
        asset_end_calendar = (
            asset_start_calendar + (asset_end_data - asset_start_data)
        )

        # If the asset started during the query, then start with the asset's
        # first row.
        # Otherwise start with the asset's first row + the number of rows
        # before the query on which the asset existed.
        first_row_a[i] = (
            asset_start_data + max(0, (query_start - asset_start_calendar))
        )
        # If the asset ended during the query, the end with the asset's last
        # row.
        # Otherwise, end with the asset's last row minus the number of rows
        # after the query for which the asset
        last_row_a[i] = (
            asset_end_data - max(0, asset_end_calendar - query_end)
        )
        # If the asset existed on or before the query, no offset.
        # Otherwise, offset by the number of rows in the query in which the
        # asset did not yet exist.
        offset_a[i] = max(0, asset_start_calendar - query_start)

    if not any_hits:
        raise ValueError('At least one valid asset id is required.')

    return first_row_a, last_row_a, offset_a


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef _read_bcolz_data(ctable_t table,
                       tuple shape,
                       list columns,
                       intp_t[:] first_rows,
                       intp_t[:] last_rows,
                       intp_t[:] offsets,
                       bool read_all,
                       set no_scale_columns=None):
    """Load raw bcolz data for the given columns and indices.

    Parameters
    ----------
    table : bcolz.ctable
        The table from which to read.
    shape : tuple (length 2)
        The shape of the expected output arrays.
    columns : list[str]
        List of column names to read.
    first_rows : ndarray[intp]
    last_rows : ndarray[intp]
    offsets : ndarray[intp
        Arrays in the format returned by _compute_row_slices.
    read_all : bool
        Whether to read_all sid data at once, or to read a silce from the
        carray for each sid.
    no_scale_columns : set[str], optional
        Set of column names that should NOT be scaled by 0.001.
        Defaults to {'volume', 'day', 'id'} (columns that don't need scaling).
        All other columns (OHLCV prices and numeric features) are automatically
        scaled by 0.001 to restore decimal precision.

    Returns
    -------
    results : list of ndarray
        A 2D array of shape `shape` for each column in `columns`.
        Numeric columns (OHLCV prices and numeric features) are returned as
        float64 arrays scaled by 0.001. Other columns (volume, day, id, categorical)
        are returned as uint32 arrays.
    """
    cdef:
        int nassets
        str column_name
        carray_t carray
        ndarray[dtype=uint32_t, ndim=1] raw_data
        ndarray[dtype=uint32_t, ndim=2] outbuf
        ndarray[dtype=uint8_t, ndim=2, cast=True] where_nan
        ndarray[dtype=float64_t, ndim=2] outbuf_as_float
        intp_t asset
        intp_t out_idx
        intp_t raw_idx
        intp_t first_row
        intp_t last_row
        intp_t offset
        list results = []
        set no_scale_cols

    # Default no_scale_columns to volume, day, id (columns that don't need scaling)
    if no_scale_columns is None:
        no_scale_cols = {'volume', 'day', 'id'}
    else:
        no_scale_cols = no_scale_columns | {'day', 'id'}

    ndays = shape[0]
    nassets = shape[1]
    if not nassets== len(first_rows) == len(last_rows) == len(offsets):
        raise ValueError("Incompatible index arrays.")

    for column_name in columns:
        outbuf = zeros(shape=shape, dtype=uint32)
        if read_all:
            raw_data = table[column_name][:]

            for asset in range(nassets):
                first_row = first_rows[asset]
                if first_row == -1:
                    # This is an unknown asset, leave its slot empty.
                    continue

                last_row = last_rows[asset]
                offset = offsets[asset]

                if first_row <= last_row:
                    outbuf[offset:offset + (last_row + 1 - first_row), asset] =\
                        raw_data[first_row:last_row + 1]
                else:
                    continue
        else:
            carray = table[column_name]

            for asset in range(nassets):
                first_row = first_rows[asset]
                if first_row == -1:
                    # This is an unknown asset, leave its slot empty.
                    continue

                last_row = last_rows[asset]
                offset = offsets[asset]
                out_start = offset
                out_end = (last_row - first_row) + offset + 1
                if first_row <= last_row:
                    outbuf[offset:offset + (last_row + 1 - first_row), asset] =\
                        carray[first_row:last_row + 1]
                else:
                    continue

        if column_name not in no_scale_cols:
            # Scale by 0.001 to restore decimal precision
            # Applies to all numeric columns: OHLCV prices and numeric features
            # (all use * 1000 scaling during write, so * 0.001 here)
            where_nan = (outbuf == 0)
            outbuf_as_float = outbuf.astype(float64) * .001
            outbuf_as_float[where_nan] = NAN
            results.append(outbuf_as_float)
        else:
            # Return as-is for volume, day, id, and categorical features
            # (volume/day/id don't need scaling, categorical will be decoded in Python)
            results.append(outbuf)
    return results


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef _decode_categorical_features(ndarray[dtype=uint32_t, ndim=2] uint32_data,
                                   dict decoding_map):
    """Decode categorical features from uint32 to string/object arrays.
    
    Parameters
    ----------
    uint32_data : ndarray[uint32, ndim=2]
        2D array of shape (n_dates, n_assets) containing encoded categorical values.
    decoding_map : dict[int, str or None]
        Dictionary mapping uint32 encoded values to their string representations.
        Key 0 is reserved for NaN/None.
    
    Returns
    -------
    decoded_array : ndarray[object, ndim=2]
        2D array of shape (n_dates, n_assets) containing decoded string values.
        Missing values (0) are set to None.
    """
    cdef:
        intp_t n_dates = uint32_data.shape[0]
        intp_t n_assets = uint32_data.shape[1]
        intp_t row_idx, asset_idx
        uint32_t uint32_val
        object decoded_val
        ndarray[dtype=object, ndim=2] decoded_array
    
    # Create object array initialized with None
    decoded_array = full((n_dates, n_assets), None, dtype=object)
    
    # Decode each value using the decoding map
    # Note: decoding_map[0] = None is intentional (missing value marker)
    # If decoding_map doesn't contain a value, it means the data is corrupted
    # or the encoding/decoding maps are out of sync. We treat such values as missing (None).
    for row_idx in range(n_dates):
        for asset_idx in range(n_assets):
            uint32_val = uint32_data[row_idx, asset_idx]
            if uint32_val != 0:
                # Look up in decoding map
                # If key doesn't exist, treat as missing (None)
                decoded_val = decoding_map.get(uint32_val)
                decoded_array[row_idx, asset_idx] = decoded_val
            # else: leave as None (already initialized) - missing value (0 -> None via decoding_map[0])
            # Note: decoding_map[0] = None, so 0 values will decode to None
    
    # Convert None to empty string: pandas Categorical doesn't allow None in categories.
    # Empty string is used as missing_value for LabelArray compatibility.
    for row_idx in range(n_dates):
        for asset_idx in range(n_assets):
            if decoded_array[row_idx, asset_idx] is None:
                decoded_array[row_idx, asset_idx] = ''
    
    return decoded_array
