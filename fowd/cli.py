"""
main.py

Entry point for CLI.
"""

import os
import datetime

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


@cli.command('from-cdip')
@click.argument('CDIP_FOLDER')
@click.option(
    '-o', '--out-folder',
    type=click.Path(file_okay=False, writable=True, exists=False),
    required=True,
)
@click.option(
    '--dump-qc', is_flag=True, default=False,
    help='Dump records that fail QC to JSON file',
)
@click.option('-n', '--nproc', default=None, type=int)
def from_cdip(cdip_folder, out_folder, dump_qc, nproc):
    from .cdip import process_cdip_station
    from .logs import setup_file_logger

    os.makedirs(out_folder, exist_ok=True)

    logfile = os.path.join(
        out_folder,
        f'fowd_cdip_{datetime.datetime.today():%Y%m%dT%H%M%S}.log'
    )
    setup_file_logger(logfile)

    if dump_qc:
        qc_outfile = os.path.join(
            out_folder,
            f'fowd_cdip_{datetime.datetime.today():%Y%m%dT%H%M%S}_qc.json'
        )
        with open(qc_outfile, 'w'):
            pass
    else:
        qc_outfile = None

    try:
        process_cdip_station(cdip_folder, out_folder, nproc=nproc, qc_outfile=qc_outfile)
    except Exception:
        click.echo('Error during processing', err=True)
        raise
    else:
        click.echo('Processing finished successfully')
    finally:
        if dump_qc:
            click.echo(f'QC information written to {qc_outfile}')
        click.echo(f'Log file written to {logfile}')


def entrypoint():
    try:
        cli(obj={})
    except Exception:
        import logging
        logger = logging.getLogger(__name__)
        logger.exception('Uncaught exception!', exc_info=True)
        raise
