"""
cli.py

Entry point for CLI.
"""

import os
import sys
import math
import logging
import datetime
import tempfile

import click
import tqdm

from . import __version__


@click.group('fowd', invoke_without_command=True)
@click.version_option(version=__version__)
@click.pass_context
def cli(ctx):
    """The command line interface for the Free Ocean Wave Dataset (FOWD) processing toolkit."""

    if ctx.invoked_subcommand is None:
        click.echo(ctx.get_help())
        return


@cli.command('process-cdip')
@click.argument('CDIP_FOLDER', type=click.Path(file_okay=False, readable=True, exists=True))
@click.option(
    '-o', '--out-folder',
    type=click.Path(file_okay=False, writable=True),
    required=True,
)
@click.option(
    '-n', '--nproc', default=None, type=int,
    help='Maximum number of parallel processes [default: number of CPU cores]'
)
def process_cdip(cdip_folder, out_folder, nproc):
    """Process all deployments of a CDIP station into one FOWD output file."""
    from .cdip import process_cdip_station
    from .logs import setup_file_logger

    os.makedirs(out_folder, exist_ok=True)

    logfile = os.path.join(
        out_folder,
        f'fowd_cdip_{datetime.datetime.today():%Y%m%dT%H%M%S}.log'
    )
    setup_file_logger(logfile)

    try:
        process_cdip_station(cdip_folder, out_folder, nproc=nproc)
    except Exception:
        click.echo('Error during processing', err=True)
        raise
    else:
        click.echo('Processing finished successfully')
    finally:
        click.echo(f'Log file written to {logfile}')


@cli.command('process-generic')
@click.argument('INFILE', type=click.Path(dir_okay=False, readable=True, exists=True))
@click.option(
    '-o', '--out-folder',
    type=click.Path(file_okay=False, writable=True),
    required=True,
)
@click.option(
    '--station-id', default=None,
    help='Station ID to use in outputs [default: use input file name]'
)
def process_generic(infile, station_id, out_folder):
    """Process a generic netCDF input file into a FOWD output file."""
    from .generic_source import process_file
    from .logs import setup_file_logger

    os.makedirs(out_folder, exist_ok=True)

    logfile = os.path.join(
        out_folder,
        f'fowd_generic_{datetime.datetime.today():%Y%m%dT%H%M%S}.log'
    )
    setup_file_logger(logfile)

    try:
        process_file(infile, out_folder, station_id=station_id)
    except Exception:
        click.echo('Error during processing', err=True)
        raise
    else:
        click.echo('Processing finished successfully')
    finally:
        click.echo(f'Log file written to {logfile}')


@cli.command('run-tests')
@click.option('-o', '--out-folder', type=click.Path(file_okay=False, writable=True))
def run_tests(out_folder):
    """Run unit tests and sanity checks."""
    import pytest
    from .sanity.run_sanity_checks import run_all

    if out_folder is None:
        out_folder = tempfile.mkdtemp(prefix='fowd_sanity_')

    os.makedirs(out_folder, exist_ok=True)

    click.echo('Running unit tests ...')
    exit_code = pytest.main([
        '-x',
        os.path.join(os.path.dirname(__file__), 'tests')
    ])

    click.echo('')
    click.echo('Running sanity checks ...')
    run_all(out_folder)
    click.echo(f'Sanity check results written to {out_folder}')
    click.echo(click.style('Make sure to check whether outputs are as expected.', bold=True))

    if exit_code > 0:
        sys.exit(exit_code)


@cli.command('plot-qc')
@click.argument('QC_INFILE', type=click.Path(dir_okay=False, readable=True))
@click.option('-o', '--out-folder', type=click.Path(file_okay=False, writable=True))
def plot_qc(qc_infile, out_folder):
    """Generate plots from QC log files."""
    from .postprocessing import plot_qc

    if out_folder is None:
        out_folder = tempfile.mkdtemp(prefix='fowd_qc_')

    click.echo('Plotting QC records ...')
    plot_qc(qc_infile, out_folder)
    click.echo(f'Results written to {out_folder}')


@cli.command('postprocess')
@click.argument('INPUT_FILES', type=click.Path(dir_okay=False, readable=True), nargs=-1)
@click.option('-o', '--out-folder', type=click.Path(file_okay=False, writable=True), required=True)
def postprocess_cdip(input_files, out_folder):
    """Filter some invalid measurements from FOWD CDIP output."""
    import xarray as xr

    from .postprocessing import filter_cdip, CDIP_DEPLOYMENT_BLACKLIST
    from .logs import setup_file_logger
    from .output import write_records
    from .cdip import EXTRA_METADATA as CDIP_EXTRA_METADATA

    os.makedirs(out_folder, exist_ok=True)

    logfile = os.path.join(
        out_folder,
        f'fowd_postprocessing_{datetime.datetime.today():%Y%m%dT%H%M%S}.log'
    )
    setup_file_logger(logfile)
    logger = logging.getLogger(__name__)

    pbar = tqdm.tqdm(input_files, desc='Post-processing FOWD files')

    for infile in pbar:
        pbar.set_postfix(file=os.path.basename(infile))

        filename, ext = os.path.splitext(os.path.basename(infile))
        outfile = os.path.join(out_folder, f'{filename}_filtered{ext}')

        with xr.open_dataset(infile, cache=False) as ds:
            logger.info(f'Processing {infile}')

            station_name = str(ds.meta_station_name.values[0])
            num_records = len(ds['wave_id_local'])

            is_cdip = station_name.startswith('CDIP_')
            if is_cdip and CDIP_DEPLOYMENT_BLACKLIST.get(station_name[5:]) == '*':
                logger.info('All deployments blacklisted, skipping')
                continue

            include_direction = 'direction_sampling_time' in ds.variables

            out_metadata = {}
            if is_cdip:
                out_metadata.update(CDIP_EXTRA_METADATA)

            out_metadata['postprocessing'] = 'filtered'
            out_metadata['postprocessing_input_uuid'] = ds.attrs['uuid']

            num_filtered = {}
            # xarray comes to a crawl for smaller chunks for some reason
            chunk_size = 1_000_000

            record_generator = tqdm.tqdm(
                filter_cdip(ds, num_filtered, chunk_size=chunk_size),
                total=math.ceil(num_records / chunk_size),
                leave=False
            )

            write_records(
                record_generator,
                outfile, station_name,
                extra_metadata=out_metadata,
                include_direction=include_direction,
            )

            for filter_name, filter_num in num_filtered.items():
                logger.info(f'[{filter_name}]: Filtered {filter_num} seas')

    click.echo(f'Results written to {out_folder}')


def entrypoint():
    try:
        cli(obj={})
    except Exception:
        logger = logging.getLogger(__name__)
        logger.exception('Uncaught exception!', exc_info=True)
        raise
