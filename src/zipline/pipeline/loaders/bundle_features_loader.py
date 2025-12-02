"""
PipelineLoader for bundle custom features.

This loader reads custom features from a BcolzDailyBarReader
that contains feature data in the bundle.

Users should define their own DataSet class (similar to USEquityPricing)
with Column definitions for each feature, then use this loader to load
the data from the bundle.

Example:
    from zipline.pipeline.data import DataSet, Column
    from zipline.utils.numpy_utils import float64_dtype, categorical_dtype
    from zipline.pipeline.domain import US_EQUITIES

    class MyFeatures(DataSet):
        domain = US_EQUITIES
        pe = Column(float64_dtype)
        pbr = Column(float64_dtype)
        sector = Column(categorical_dtype, missing_value=None)
"""

from zipline.lib.adjusted_array import AdjustedArray
from zipline.utils.numpy_utils import repeat_first_axis, categorical_dtype

from .base import PipelineLoader
from .utils import shift_dates


class BundleFeaturesLoader(PipelineLoader):
    """A PipelineLoader for loading custom features from a bundle.

    Parameters
    ----------
    raw_price_reader : zipline.data.session_bars.SessionBarReader
        Reader providing raw prices and custom features.
        Must be a BcolzDailyBarReader with feature metadata.
    """

    def __init__(self, raw_price_reader):
        self.raw_price_reader = raw_price_reader

    def load_adjusted_array(self, domain, columns, dates, sids, mask):
        """
        Load custom feature data from the bundle.

        Parameters
        ----------
        domain : zipline.pipeline.domain.Domain
            The domain for which to load data.
        columns : list[zipline.pipeline.data.BoundColumn]
            Columns to load. These should be custom feature columns
            from a user-defined DataSet (e.g., MyFeatures).
        dates : pd.DatetimeIndex
            Dates for which to load data.
        sids : np.array[int64]
            Asset IDs for which to load data.
        mask : np.array[bool]
            Boolean mask indicating which (date, asset) pairs are valid.

        Returns
        -------
        out : dict[BoundColumn -> AdjustedArray]
            Dictionary mapping each requested column to its AdjustedArray.
        """
        # Similar to EquityPricingLoader, we need to shift dates back by one
        # session to get data that would be known at the start of each date.
        sessions = domain.sessions()
        shifted_dates = shift_dates(sessions, dates[0], dates[-1], shift=1)

        # Get feature column names
        feature_colnames = [c.name for c in columns]

        # Load feature data using the reader's load_raw_arrays method
        # This method already handles feature columns and applies inverse scaling
        raw_feature_arrays = self.raw_price_reader.load_raw_arrays(
            feature_colnames,
            shifted_dates[0],
            shifted_dates[-1],
            sids,
        )

        # Create AdjustedArray for each feature column
        # Custom features don't have adjustments (like splits/dividends)
        out = {}
        for c, c_raw in zip(columns, raw_feature_arrays):
            # Categorical features: use empty string as missing_value.
            # _decode_categorical_features converts None to '' for pandas Categorical compatibility.
            if c.dtype == categorical_dtype:
                missing_val = ''
            else:
                missing_val = c.missing_value
            
            out[c] = AdjustedArray(
                c_raw.astype(c.dtype),
                adjustments={},  # No adjustments for custom features
                missing_value=missing_val,
            )

        return out


