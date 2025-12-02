import errno
import os

import click
import logging
import pandas as pd

import zipline
from zipline.data import bundles as bundles_module
from zipline.utils.calendar_utils import get_calendar
from zipline.utils.compat import wraps
from zipline.utils.cli import Date, Timestamp
from zipline.utils.run_algo import _run, BenchmarkSpec, load_extensions
from zipline.extensions import create_args

try:
    __IPYTHON__
except NameError:
    __IPYTHON__ = False


@click.group()
@click.option(
    "-e",
    "--extension",
    multiple=True,
    help="File or module path to a zipline extension to load.",
)
@click.option(
    "--strict-extensions/--non-strict-extensions",
    is_flag=True,
    help="If --strict-extensions is passed then zipline will not "
    "run if it cannot load all of the specified extensions. "
    "If this is not passed or --non-strict-extensions is passed "
    "then the failure will be logged but execution will continue.",
)
@click.option(
    "--default-extension/--no-default-extension",
    is_flag=True,
    default=True,
    help="Don't load the default zipline extension.py file in $ZIPLINE_HOME.",
)
@click.option(
    "-x",
    multiple=True,
    help="Any custom command line arguments to define, in key=value form.",
)
@click.pass_context
def main(ctx, extension, strict_extensions, default_extension, x):
    """Top level zipline entry point."""
    # install a logging handler before performing any other operations

    logging.basicConfig(
        format="[%(asctime)s-%(levelname)s][%(name)s]\n %(message)s",
        level=logging.INFO,
        datefmt="%Y-%m-%dT%H:%M:%S%z",
    )

    create_args(x, zipline.extension_args)
    load_extensions(
        default_extension,
        extension,
        strict_extensions,
        os.environ,
    )


def extract_option_object(option):
    """Convert a click.option call into a click.Option object.

    Parameters
    ----------
    option : decorator
        A click.option decorator.

    Returns
    -------
    option_object : click.Option
        The option object that this decorator will create.
    """

    @option
    def opt():
        pass

    return opt.__click_params__[0]


def ipython_only(option):
    """Mark that an option should only be exposed in IPython.

    Parameters
    ----------
    option : decorator
        A click.option decorator.

    Returns
    -------
    ipython_only_dec : decorator
        A decorator that correctly applies the argument even when not
        using IPython mode.
    """
    if __IPYTHON__:
        return option

    argname = extract_option_object(option).name

    def d(f):
        @wraps(f)
        def _(*args, **kwargs):
            kwargs[argname] = None
            return f(*args, **kwargs)

        return _

    return d


DEFAULT_BUNDLE = "quandl"


@main.command()
@click.option(
    "-f",
    "--algofile",
    default=None,
    type=click.File("r"),
    help="The file that contains the algorithm to run.",
)
@click.option(
    "-t",
    "--algotext",
    help="The algorithm script to run.",
)
@click.option(
    "-D",
    "--define",
    multiple=True,
    help="Define a name to be bound in the namespace before executing"
    " the algotext. For example '-Dname=value'. The value may be any "
    "python expression. These are evaluated in order so they may refer "
    "to previously defined names.",
)
@click.option(
    "--data-frequency",
    type=click.Choice({"daily", "minute"}),
    default="daily",
    show_default=True,
    help="The data frequency of the simulation.",
)
@click.option(
    "--capital-base",
    type=float,
    default=10e6,
    show_default=True,
    help="The starting capital for the simulation.",
)
@click.option(
    "-b",
    "--bundle",
    default=DEFAULT_BUNDLE,
    metavar="BUNDLE-NAME",
    show_default=True,
    help="The data bundle to use for the simulation.",
)
@click.option(
    "--bundle-timestamp",
    type=Timestamp(),
    default=pd.Timestamp.utcnow(),
    show_default=False,
    help="The date to lookup data on or before.\n" "[default: <current-time>]",
)
@click.option(
    "-bf",
    "--benchmark-file",
    default=None,
    type=click.Path(exists=True, dir_okay=False, readable=True, path_type=str),
    help="The csv file that contains the benchmark returns",
)
@click.option(
    "--benchmark-symbol",
    default=None,
    type=click.STRING,
    help="The symbol of the instrument to be used as a benchmark "
    "(should exist in the ingested bundle)",
)
@click.option(
    "--benchmark-sid",
    default=None,
    type=int,
    help="The sid of the instrument to be used as a benchmark "
    "(should exist in the ingested bundle)",
)
@click.option(
    "--no-benchmark",
    is_flag=True,
    default=False,
    help="If passed, use a benchmark of zero returns.",
)
@click.option(
    "-s",
    "--start",
    type=Date(as_timestamp=True),
    help="The start date of the simulation.",
)
@click.option(
    "-e",
    "--end",
    type=Date(as_timestamp=True),
    help="The end date of the simulation.",
)
@click.option(
    "-o",
    "--output",
    default="-",
    metavar="FILENAME",
    show_default=True,
    help="The location to write the perf data. If this is '-' the perf will"
    " be written to stdout.",
)
@click.option(
    "--trading-calendar",
    metavar="TRADING-CALENDAR",
    default="XNYS",
    help="The calendar you want to use e.g. XLON. XNYS is the default.",
)
@click.option(
    "--print-algo/--no-print-algo",
    is_flag=True,
    default=False,
    help="Print the algorithm to stdout.",
)
@click.option(
    "--metrics-set",
    default="default",
    help="The metrics set to use. New metrics sets may be registered in your"
    " extension.py.",
)
@click.option(
    "--blotter",
    default="default",
    help="The blotter to use.",
    show_default=True,
)
@ipython_only(
    click.option(
        "--local-namespace/--no-local-namespace",
        is_flag=True,
        default=None,
        help="Should the algorithm methods be " "resolved in the local namespace.",
    )
)
@click.pass_context
def run(
    ctx,
    algofile,
    algotext,
    define,
    data_frequency,
    capital_base,
    bundle,
    bundle_timestamp,
    benchmark_file,
    benchmark_symbol,
    benchmark_sid,
    no_benchmark,
    start,
    end,
    output,
    trading_calendar,
    print_algo,
    metrics_set,
    local_namespace,
    blotter,
):
    """Run a backtest for the given algorithm."""
    # check that the start and end dates are passed correctly
    if start is None and end is None:
        # check both at the same time to avoid the case where a user
        # does not pass either of these and then passes the first only
        # to be told they need to pass the second argument also
        ctx.fail(
            "must specify dates with '-s' / '--start' and '-e' / '--end'",
        )
    if start is None:
        ctx.fail("must specify a start date with '-s' / '--start'")
    if end is None:
        ctx.fail("must specify an end date with '-e' / '--end'")

    if (algotext is not None) == (algofile is not None):
        ctx.fail(
            "must specify exactly one of '-f' / "
            "'--algofile' or"
            " '-t' / '--algotext'",
        )

    trading_calendar = get_calendar(trading_calendar)

    benchmark_spec = BenchmarkSpec.from_cli_params(
        no_benchmark=no_benchmark,
        benchmark_sid=benchmark_sid,
        benchmark_symbol=benchmark_symbol,
        benchmark_file=benchmark_file,
    )

    return _run(
        initialize=None,
        handle_data=None,
        before_trading_start=None,
        analyze=None,
        algofile=algofile,
        algotext=algotext,
        defines=define,
        data_frequency=data_frequency,
        capital_base=capital_base,
        bundle=bundle,
        bundle_timestamp=bundle_timestamp,
        start=start,
        end=end,
        output=output,
        trading_calendar=trading_calendar,
        print_algo=print_algo,
        metrics_set=metrics_set,
        local_namespace=local_namespace,
        environ=os.environ,
        blotter=blotter,
        benchmark_spec=benchmark_spec,
        custom_loader=None,
    )


def zipline_magic(line, cell=None):
    """The zipline IPython cell magic."""
    load_extensions(
        default=True,
        extensions=[],
        strict=True,
        environ=os.environ,
    )
    try:
        return run.main(
            # put our overrides at the start of the parameter list so that
            # users may pass values with higher precedence
            [
                "--algotext",
                cell,
                "--output",
                os.devnull,  # don't write the results by default
            ]
            + (
                [
                    # these options are set when running in line magic mode
                    # set a non None algo text to use the ipython user_ns
                    "--algotext",
                    "",
                    "--local-namespace",
                ]
                if cell is None
                else []
            )
            + line.split(),
            "%s%%zipline" % ((cell or "") and "%"),
            # don't use system exit and propogate errors to the caller
            standalone_mode=False,
        )
    except SystemExit as exc:
        # https://github.com/mitsuhiko/click/pull/533
        # even in standalone_mode=False `--help` really wants to kill us ;_;
        if exc.code:
            raise ValueError(
                "main returned non-zero status code: %d" % exc.code
            ) from exc


@main.command()
@click.option(
    "-b",
    "--bundle",
    default=DEFAULT_BUNDLE,
    metavar="BUNDLE-NAME",
    show_default=True,
    help="The data bundle to ingest.",
)
@click.option(
    "--assets-version",
    type=int,
    multiple=True,
    help="Version of the assets db to which to downgrade.",
)
@click.option(
    "--show-progress/--no-show-progress",
    default=True,
    help="Print progress information to the terminal.",
)
def ingest(bundle, assets_version, show_progress):
    """Ingest the data for the given bundle."""
    bundles_module.ingest(
        bundle,
        os.environ,
        pd.Timestamp.utcnow(),
        assets_version,
        show_progress,
    )


@main.command()
@click.option(
    "-b",
    "--bundle",
    default=DEFAULT_BUNDLE,
    metavar="BUNDLE-NAME",
    show_default=True,
    help="The data bundle to clean.",
)
@click.option(
    "-e",
    "--before",
    type=Timestamp(),
    help="Clear all data before TIMESTAMP."
    " This may not be passed with -k / --keep-last",
)
@click.option(
    "-a",
    "--after",
    type=Timestamp(),
    help="Clear all data after TIMESTAMP"
    " This may not be passed with -k / --keep-last",
)
@click.option(
    "-k",
    "--keep-last",
    type=int,
    metavar="N",
    help="Clear all but the last N downloads."
    " This may not be passed with -e / --before or -a / --after",
)
def clean(bundle, before, after, keep_last):
    """Clean up data downloaded with the ingest command."""
    bundles_module.clean(
        bundle,
        before,
        after,
        keep_last,
    )


@main.command()
def bundles():
    """List all of the available data bundles."""
    for bundle in sorted(bundles_module.bundles.keys()):
        if bundle.startswith("."):
            # hide the test data
            continue
        try:
            ingestions = list(map(str, bundles_module.ingestions_for_bundle(bundle)))
        except OSError as e:
            if e.errno != errno.ENOENT:
                raise
            ingestions = []

        # If we got no ingestions, either because the directory didn't exist or
        # because there were no entries, print a single message indicating that
        # no ingestions have yet been made.
        for timestamp in ingestions or ["<no ingestions>"]:
            click.echo("%s %s" % (bundle, timestamp))


@main.command()
@click.option(
    "-b",
    "--bundle",
    default=DEFAULT_BUNDLE,
    metavar="BUNDLE-NAME",
    show_default=True,
    help="The data bundle to update.",
)
@click.option(
    "-t",
    "--timestamp",
    type=Timestamp(),
    default=None,
    help="Timestamp of the bundle ingestion to update. If not provided, uses most recent.",
)
@click.option(
    "-f",
    "--file",
    "data_file",
    type=click.Path(exists=True),
    required=True,
    help="Path to data file (CSV, Parquet, or HDF5) containing new data.",
)
@click.option(
    "--format",
    "data_format",
    type=click.Choice(["csv", "parquet", "hdf5"]),
    default="parquet",
    help="Format of the data file.",
)
@click.option(
    "--symbol-column",
    default="symbol",
    help="Column name for symbol identifiers (for CSV/Parquet).",
)
def update_bundle(bundle, timestamp, data_file, data_format, symbol_column):
    """Update existing V2 bundle with new data (incremental update).
    
    This command adds new data to an existing V2 bundle without re-ingesting
    the entire dataset. Only works with V2 bundles (created with use_v2_writer=True).
    
    Examples:
        # Update with Parquet file
        zipline update-bundle -b my_bundle_v2 -f new_data.parquet
        
        # Update with CSV file
        zipline update-bundle -b my_bundle_v2 -f new_data.csv --format csv
        
        # Update specific timestamp
        zipline update-bundle -b my_bundle_v2 -f new_data.parquet -t 2024-01-01
    """
    try:
        from zipline.data.bcolz_daily_bars import BcolzDailyBarWriter
        from zipline.data.bundles.parquetdir import _parquet_pricing_iter
        from zipline.data.bundles.hdfdir import _hdf5_pricing_iter
        import polars as pl
    except ImportError as e:
        click.echo(f"Error: V2 Writer not available: {e}", err=True)
        raise click.Abort()
    
    # Get bundle path
    if timestamp is None:
        timestamp = pd.Timestamp.utcnow()
    
    try:
        timestr = bundles_module.most_recent_data(bundle, timestamp, environ=os.environ)
    except ValueError as e:
        click.echo(f"Error: {e}", err=True)
        raise click.Abort()
    
    daily_bars_path = bundles_module.daily_equity_path(
        bundle, timestr, environ=os.environ
    )
    
    # V2 bundle: check if path is a directory (V2) or file (V1)
    if os.path.isdir(daily_bars_path):
        # V2 bundle: metadata.json is in the directory
        metadata_path = os.path.join(daily_bars_path, "metadata.json")
    else:
        # V1 bundle: not supported for updates
        click.echo(
            f"Error: Bundle '{bundle}' is not a V2 bundle. "
            f"V2 bundles require use_v2_writer=True during ingest. "
            f"Found path: {daily_bars_path}",
            err=True
        )
        raise click.Abort()
    
    if not os.path.exists(metadata_path):
        click.echo(
            f"Error: V2 bundle metadata not found at {metadata_path}",
            err=True
        )
        raise click.Abort()
    
    # Open bundle for writing
    try:
        from zipline.assets import AssetFinder
        writer = BcolzDailyBarWriter.open(daily_bars_path)
        asset_finder = AssetFinder(
            bundles_module.asset_db_path(bundle, timestr, environ=os.environ)
        )
    except Exception as e:
        click.echo(f"Error opening bundle: {e}", err=True)
        raise click.Abort()
    
    click.echo(f"Updating bundle '{bundle}' at {daily_bars_path}")
    
    # Read and process data based on format
    if data_format == "parquet":
        try:
            df_all = pl.read_parquet(data_file).to_pandas()
            symbols = sorted(df_all[symbol_column].unique())
            
            # Get symbol to sid mapping
            all_assets = asset_finder.retrieve_all(asset_finder.sids)
            symbol_to_sid = {asset.symbol: asset.sid for asset in all_assets}
            
            for symbol in symbols:
                if symbol not in symbol_to_sid:
                    click.echo(f"  Warning: Symbol '{symbol}' not found in bundle, skipping", err=True)
                    continue
                
                sid = symbol_to_sid[symbol]
                df_symbol = df_all[df_all[symbol_column] == symbol].copy()
                if df_symbol.empty:
                    continue
                
                # Prepare DataFrame with OHLCV
                datetime_col = df_symbol.index.name if isinstance(df_symbol.index, pd.DatetimeIndex) else df_symbol.columns[0]
                if datetime_col not in df_symbol.index.names and datetime_col not in df_symbol.columns:
                    datetime_col = df_symbol.columns[0]
                
                if datetime_col in df_symbol.columns:
                    df_symbol = df_symbol.set_index(datetime_col)
                
                df_symbol = df_symbol[['open', 'high', 'low', 'close', 'volume']].copy()
                df_symbol.index = pd.to_datetime(df_symbol.index).tz_localize(None)
                df_symbol = df_symbol.sort_index()
                
                try:
                    writer.append_sid(sid, df_symbol, allow_overlap=True)
                    click.echo(f"  ✓ Updated sid={sid} ({symbol}): {len(df_symbol)} days")
                except Exception as e:
                    click.echo(f"  ✗ Failed to update sid={sid} ({symbol}): {e}", err=True)
        
        except Exception as e:
            click.echo(f"Error processing Parquet file: {e}", err=True)
            raise click.Abort()
    
    elif data_format == "csv":
        click.echo("CSV format update not yet implemented. Please use Parquet format.")
        raise click.Abort()
    
    elif data_format == "hdf5":
        click.echo("HDF5 format update not yet implemented. Please use Parquet format.")
        raise click.Abort()
    
    click.echo(f"✓ Bundle update completed!")


@main.command()
@click.option(
    "-b",
    "--bundle",
    default=DEFAULT_BUNDLE,
    metavar="BUNDLE-NAME",
    show_default=True,
    help="The data bundle to add feature to.",
)
@click.option(
    "-t",
    "--timestamp",
    type=Timestamp(),
    default=None,
    help="Timestamp of the bundle ingestion. If not provided, uses most recent.",
)
@click.option(
    "-f",
    "--file",
    "feature_file",
    type=click.Path(exists=True),
    required=True,
    help="Path to feature data file (CSV, Parquet, or HDF5). Expected format: date x sid wide format (CSV/Parquet) or HDF5 with symbol groups.",
)
@click.option(
    "--feature-name",
    default=None,
    help="Name of the feature to add (e.g., 'pe', 'pbr', 'market_cap'). Required for CSV/Parquet, optional for HDF5 (auto-detect all features).",
)
@click.option(
    "--format",
    "data_format",
    type=click.Choice(["csv", "parquet", "hdf5"]),
    default=None,
    help="Format of the feature data file. If not specified, auto-detect from file extension.",
)
@click.option(
    "--dtype",
    default="float64",
    type=click.Choice(["float64", "float32", "int64", "int32", "uint32", "uint64"]),
    help="Data type for feature storage.",
)
@click.option(
    "--scaling-factor",
    default=1.0,
    type=float,
    help="Scaling factor for feature storage (e.g., 1000 for prices).",
)
@click.option(
    "--symbol-column",
    default="symbol",
    help="Column name for symbol identifiers (for mapping to sids).",
)
def add_feature(
    bundle, timestamp, feature_file, feature_name, data_format,
    dtype, scaling_factor, symbol_column
):
    """Add a custom feature to an existing V2 bundle.
    
    This command adds a feature column to all assets in a V2 bundle.
    
    For CSV/Parquet: The feature data file should be in wide format (date x sid/symbol).
    For HDF5: The file should have the same structure as OHLCV HDF5 file, with feature
    datasets in each symbol group. All features in the file will be automatically detected
    and added (OHLCV and corporate actions are excluded).
    
    Examples:
        # Add PE ratio feature from Parquet file
        zipline add-feature -b my_bundle_v2 -f pe_data.parquet --feature-name pe
        
        # Add feature with custom dtype and scaling
        zipline add-feature -b my_bundle_v2 -f market_cap.parquet \\
            --feature-name market_cap --dtype float32 --scaling-factor 1000
        
        # Add all features from HDF5 file (auto-detect)
        zipline add-feature -b my_bundle_v2 -f features.h5 --format hdf5
    """
    try:
        from zipline.data.bcolz_daily_bars import BcolzDailyBarWriter, BcolzDailyBarReader
        from zipline.assets import AssetFinder
        import polars as pl
    except ImportError as e:
        click.echo(f"Error: V2 Writer not available: {e}", err=True)
        raise click.Abort()
    
    # Get bundle path
    if timestamp is None:
        timestamp = pd.Timestamp.utcnow()
    
    try:
        timestr = bundles_module.most_recent_data(bundle, timestamp, environ=os.environ)
    except ValueError as e:
        click.echo(f"Error: {e}", err=True)
        raise click.Abort()
    
    daily_bars_path = bundles_module.daily_equity_path(
        bundle, timestr, environ=os.environ
    )
    
    # V2 bundle: check if path is a directory (V2) or file (V1)
    if os.path.isdir(daily_bars_path):
        # V2 bundle: metadata.json is in the directory
        metadata_path = os.path.join(daily_bars_path, "metadata.json")
    else:
        # V1 bundle: not supported for feature addition
        click.echo(
            f"Error: Bundle '{bundle}' is not a V2 bundle. "
            f"V2 bundles require use_v2_writer=True during ingest. "
            f"Found path: {daily_bars_path}",
            err=True
        )
        raise click.Abort()
    
    if not os.path.exists(metadata_path):
        click.echo(
            f"Error: V2 bundle metadata not found at {metadata_path}",
            err=True
        )
        raise click.Abort()
    
    # Open bundle
    try:
        writer = BcolzDailyBarWriter.open(daily_bars_path)
        reader = BcolzDailyBarReader(daily_bars_path)
        asset_finder = AssetFinder(
            bundles_module.asset_db_path(bundle, timestr, environ=os.environ)
        )
    except Exception as e:
        click.echo(f"Error opening bundle: {e}", err=True)
        raise click.Abort()
    
    # Auto-detect format if not specified
    if data_format is None:
        if feature_file.endswith('.h5') or feature_file.endswith('.hdf5'):
            data_format = 'hdf5'
        elif feature_file.endswith('.parquet'):
            data_format = 'parquet'
        elif feature_file.endswith('.csv'):
            data_format = 'csv'
        else:
            click.echo(f"Error: Cannot auto-detect format for {feature_file}. Please specify --format", err=True)
            raise click.Abort()
    
    # Validate feature_name for CSV/Parquet
    if data_format in ['csv', 'parquet'] and not feature_name:
        click.echo(f"Error: --feature-name is required for {data_format} format", err=True)
        raise click.Abort()
    
    # Get symbol to sid mapping
    all_assets = asset_finder.retrieve_all(asset_finder.sids)
    symbol_to_sid = {asset.symbol: asset.sid for asset in all_assets}
    
    click.echo(f"  Found {len(symbol_to_sid)} assets in bundle")
    
    # Process based on format
    if data_format == "hdf5":
        # HDF5 format: extract all features automatically
        try:
            import h5py
            from zipline.data.bundles.hdfdir import _extract_features_from_hdf5
            
            # Create metadata DataFrame from asset_finder for symbol mapping
            metadata = pd.DataFrame({
                'symbol': [asset.symbol for asset in all_assets],
            })
            metadata.index = [asset.sid for asset in all_assets]
            
            # Get symbols from HDF5 file
            with h5py.File(feature_file, 'r') as hdf:
                hdf_symbols = sorted(hdf.keys())
                
                click.echo(f"Adding features from HDF5 file: {feature_file}")
                click.echo(f"  Found {len(hdf_symbols)} symbols in HDF5 file")
                
                # Extract and add all features
                _extract_features_from_hdf5(
                    hdf,
                    writer,
                    hdf_symbols,
                    metadata,
                    show_progress=True
                )
        
        except Exception as e:
            click.echo(f"Error processing HDF5 file: {e}", err=True)
            raise click.Abort()
    
    elif data_format in ["parquet", "csv"]:
        # CSV/Parquet format: single feature
        if not feature_name:
            click.echo(f"Error: --feature-name is required for {data_format} format", err=True)
            raise click.Abort()
        
        click.echo(f"Adding feature '{feature_name}' to bundle '{bundle}'")
        
        try:
            if data_format == "parquet":
                df = pl.read_parquet(feature_file).to_pandas()
            elif data_format == "csv":
                df = pd.read_csv(feature_file, index_col=0, parse_dates=True)
            
            # Ensure index is DatetimeIndex
            if not isinstance(df.index, pd.DatetimeIndex):
                df.index = pd.to_datetime(df.index)
            df.index = df.index.tz_localize(None)
            
            click.echo(f"  Feature data shape: {df.shape}")
            
            # Add feature to each asset
            added_count = 0
            for symbol, sid in symbol_to_sid.items():
                if symbol not in df.columns:
                    continue
                
                try:
                    # Get feature values for this asset
                    feature_series = df[symbol].dropna()
                    if len(feature_series) == 0:
                        continue
                    
                    # Get asset's OHLCV data to match dates
                    asset_df = reader.read_sid_df(sid)
                    if asset_df.empty:
                        continue
                    
                    # Align feature data with asset dates
                    feature_series = feature_series.reindex(asset_df.index, method='ffill')
                    
                    # Add feature
                    writer.add_feature(
                        sid,
                        feature_name,
                        feature_series,
                        dtype=dtype,
                        scaling_factor=scaling_factor,
                    )
                    added_count += 1
                    click.echo(f"  ✓ Added '{feature_name}' to sid={sid} ({symbol})")
                
                except Exception as e:
                    click.echo(
                        f"  ✗ Failed to add '{feature_name}' to sid={sid} ({symbol}): {e}",
                        err=True
                    )
            
            click.echo(f"✓ Feature '{feature_name}' added to {added_count} assets!")
        
        except Exception as e:
            click.echo(f"Error processing feature data: {e}", err=True)
            raise click.Abort()
    
    else:
        click.echo(f"Error: Unsupported format: {data_format}", err=True)
        raise click.Abort()


if __name__ == "__main__":
    main()
