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
            writer.write(
                _hdf5_pricing_iter(hdf, symbols, tframe, metadata, divs_splits, show_progress),
                show_progress=show_progress,
            )

        metadata["exchange"] = "HDF5" # exchange 값 설정해도 asset db 에는 적용안되는데 원인은 모르겠음..

        asset_db_writer.write(equities=metadata)

        divs_splits["divs"]["sid"] = divs_splits["divs"]["sid"].astype(int)
        divs_splits["splits"]["sid"] = divs_splits["splits"]["sid"].astype(int)
        adjustment_writer.write(
            splits=divs_splits["splits"], dividends=divs_splits["divs"]
        )
        
def _hdf5_pricing_iter(hdf, symbols, tframe, metadata, divs_splits, show_progress):
    with maybe_show_progress(
        symbols, show_progress, label=f"Loading {tframe} pricing data: "
    ) as it:
        for sid, symbol in enumerate(it):
            logger.debug(f"{symbol}: sid {sid}")
            grp = hdf[symbol]

            # HDF5 파일의 루트 레벨에서 직접 심볼에 접근. 각 심볼 그룹 아래에 'open', 'high', 'low', 'close', 'volume', 'dates' 데이터셋이 있어야 합니다.
            df = pd.DataFrame({
                'open': grp['open'][:],
                'high': grp['high'][:],
                'low': grp['low'][:],
                'close': grp['close'][:],
                'volume': grp['volume'][:]
            }, index=pd.to_datetime(grp['dates'][:]).tz_localize('UTC')) # zipline 은 UTC timezone을 사용하므로, HDF5 파일의 날짜를 UTC로 변환합니다.

            start_date = df.index[0]
            end_date = df.index[-1]

            ac_date = end_date + pd.Timedelta(days=1)
            metadata.iloc[sid] = (
                start_date.tz_convert(None),
                end_date.tz_convert(None),
                ac_date.tz_convert(None),
                symbol
            )

            _process_corporate_actions(grp, sid, divs_splits)

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