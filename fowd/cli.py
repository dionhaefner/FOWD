"""
cli.py

Entry point for CLI.
"""

import sys
import os
import datetime
import tempfile

import click

from . import __version__


@click.group('fowd', invoke_without_command=True)
@click.version_option(version=__version__)
@click.pass_context
def cli(ctx):
    """The command line interface for the Free Ocean Wave Dataset (FOWD) creation toolkit."""

    if ctx.invoked_subcommand is None:
        click.echo(ctx.get_help())
        return


@cli.command('process-cdip')
@click.argument('CDIP_FOLDER')
@click.option(
    '-o', '--out-folder',
    type=click.Path(file_okay=False, writable=True, exists=False),
    required=True,
)
@click.option('-n', '--nproc', default=None, type=int)
def process_cdip(cdip_folder, out_folder, nproc):
    """Process all deployments of a CDIP station into one FOWD file."""
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
@click.argument('INFILE')
@click.option(
    '-o', '--out-folder',
    type=click.Path(file_okay=False, writable=True, exists=False),
    required=True,
)
@click.option(
    '--station-id', default=None,
    help='Station ID to use in outputs [default: use input file name]'
)
def process_generic(infile, station_id, out_folder):
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
@click.option('-o', '--out-folder', type=click.Path(file_okay=False, writable=True, exists=False))
def run_tests(out_folder):
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


@cli.command('postprocess')
@click.argument('QC_INFILE')
@click.option('-o', '--out-folder', type=click.Path(file_okay=False, writable=True, exists=False))
def postprocess(qc_infile, out_folder):
    from .postprocessing import plot_qc

    if out_folder is None:
        out_folder = tempfile.mkdtemp(prefix='fowd_qc_')

    click.echo('Plotting QC records ...')
    plot_qc(qc_infile, out_folder)
    click.echo(f'Results written to {out_folder}')


def entrypoint():
    try:
        cli(obj={})
    except Exception:
        import logging
        logger = logging.getLogger(__name__)
        logger.exception('Uncaught exception!', exc_info=True)
        raise
