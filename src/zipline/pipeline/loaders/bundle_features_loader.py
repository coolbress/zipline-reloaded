"""PipelineLoader for custom feature columns colocated with bar data.

Reads non-OHLCV columns (sector, industry, pb, pe, derived factor scores,
etc.) that the bundle producer wrote alongside the OHLCV columns into the
same session bar reader. Backend-agnostic — works with any
``CurrencyAwareSessionBarReader`` whose ``load_raw_arrays`` exposes the
requested column names, including ``BcolzDailyBarReader`` (legacy) and
``ArcticDailyBarReader``.

Users define their own ``DataSet`` (parallel to ``USEquityPricing``) with
``Column`` entries for each custom feature, then register this loader for
that DataSet.

Example:

    from zipline.pipeline.data import DataSet, Column
    from zipline.utils.numpy_utils import float64_dtype, categorical_dtype
    from zipline.pipeline.domain import US_EQUITIES

    class MyFeatures(DataSet):
        domain = US_EQUITIES
        pe     = Column(float64_dtype)
        pbr    = Column(float64_dtype)
        sector = Column(categorical_dtype, missing_value="")  # explicit

    engine = SimplePipelineEngine(
        get_loader={
            USEquityPricing: EquityPricingLoader(bar_reader, adj_reader, fx_reader),
            MyFeatures:      BundleFeaturesLoader(bar_reader),
        },
        asset_finder=asset_finder,
    )
"""

from zipline.lib.adjusted_array import AdjustedArray

from .base import PipelineLoader
from .utils import shift_dates


class BundleFeaturesLoader(PipelineLoader):
    """PipelineLoader for custom feature columns colocated with OHLCV.

    Parameters
    ----------
    raw_price_reader : zipline.data.session_bars.SessionBarReader
        Any reader whose ``load_raw_arrays(columns, start, end, sids)``
        returns the requested custom columns. The reader is treated as a
        pure column store — column-name lookup is the only contract.
    """

    def __init__(self, raw_price_reader):
        self.raw_price_reader = raw_price_reader

    def load_adjusted_array(self, domain, columns, dates, sids, mask):
        # Like EquityPricingLoader, shift back one session so each row holds
        # the value that would have been known at the start of that date
        # (point-in-time correctness).
        sessions = domain.sessions()
        shifted_dates = shift_dates(sessions, dates[0], dates[-1], shift=1)

        feature_colnames = [c.name for c in columns]
        raw_feature_arrays = self.raw_price_reader.load_raw_arrays(
            feature_colnames,
            shifted_dates[0],
            shifted_dates[-1],
            sids,
        )

        # Custom features carry no split/dividend adjustments — they are
        # already point-in-time values written by the bundle producer.
        # The caller's declared `missing_value` on each Column is the source
        # of truth; backends that need a specific sentinel (e.g. bcolz
        # categorical decoding emits '') must reflect that in the Column
        # definition rather than have the loader override it.
        return {
            c: AdjustedArray(
                c_raw.astype(c.dtype),
                adjustments={},
                missing_value=c.missing_value,
            )
            for c, c_raw in zip(columns, raw_feature_arrays)
        }


