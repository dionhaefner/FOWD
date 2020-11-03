"""
generic_source.py

Generic input file processing into FOWD datasets.

Input files are assumed to contain the following variables:
- time
- displacement

Attributes:
- sampling_rate
- water_depth
- longitude
- latitude

"""

import os
import logging

import numpy as np
import xarray as xr

from .constants import SSH_REFERENCE_INTERVAL
from .output import write_records
from .processing import compute_wave_records, read_pickle_outfile_chunks, read_pickle_statefile

logger = logging.getLogger(__name__)


# dataset-specific helpers

def add_surface_elevation(data):
    """Add surface elevation variable to xarray Dataset."""

    dt = float(1 / data.sampling_rate)
    window_size = int(60 * SSH_REFERENCE_INTERVAL / dt)

    data['mean_displacement'] = (
        data['displacement'].rolling(
            {'time': window_size},
            min_periods=60,
            center=False
        ).mean()
    )

    data['surface_elevation'] = data['displacement'] - data['mean_displacement']
    return data


class InvalidFile(Exception):
    pass


def get_input_data(filepath):
    """Read input file as xarray Dataset."""

    allowed_vars = [
        'time', 'displacement'
    ]

    def drop_unnecessary(ds):
        # check whether all variables are present
        if any(var not in ds.variables for var in allowed_vars):
            raise RuntimeError(
                f'Input file does not contain all required variables ({allowed_vars})'
            )

        for v in ds.variables:
            if v not in allowed_vars:
                ds = ds.drop(v)

        return ds

    data = drop_unnecessary(xr.open_dataset(filepath))
    data = add_surface_elevation(data)
    return data


#

def get_wave_records(filepath, out_folder, qc_outfile=None):
    """Process a single file and write results to pickle file."""
    filename = os.path.basename(filepath)
    outfile = os.path.join(out_folder, f'{filename}.waves.pkl')
    statefile = os.path.join(out_folder, f'{filename}.state.pkl')

    # parse file into xarray Dataset
    data = get_input_data(filepath)

    # extract relevant quantities from xarray dataset
    t = np.ascontiguousarray(data['time'].values)
    z = np.ascontiguousarray(data['displacement'].values)
    z_normalized = np.ascontiguousarray(data['surface_elevation'].values)

    meta_args = dict(
        filepath=filepath,
        uuid=data.attrs.get('uuid', '<not available>'),
        latitude=data.attrs['latitude'],
        longitude=data.attrs['longitude'],
        water_depth=np.float64(data.attrs['water_depth']),
        sampling_rate=data.attrs['sampling_rate']
    )

    del data  # reduce memory pressure

    compute_wave_records(
        t, z, z_normalized, outfile, statefile, meta_args,
        qc_outfile=qc_outfile
    )

    return outfile, statefile


def process_file(input_file, out_folder, station_id=None):
    """Process a single generic input file."""

    if station_id is None:
        station_id = os.path.splitext(os.path.basename(input_file))[0]

    qc_outfile = os.path.join(out_folder, f'fowd_{station_id}.qc.json')

    logger.info('Starting processing for file %s', station_id)

    result_file, state_file = get_wave_records(
        input_file, out_folder=out_folder, qc_outfile=qc_outfile
    )

    if result_file is None or state_file is None:
        logger.warning('Processing skipped for file %s', input_file)
        return

    num_waves = 0
    for record_chunk in read_pickle_outfile_chunks(result_file):
        if record_chunk:
            num_waves += len(record_chunk['wave_id_local'])

    if not num_waves:
        logger.warning('No data found in file %s', input_file)
        return

    # get QC information
    qc_flags_fired = read_pickle_statefile(state_file)['num_flags_fired']

    # log progress
    logger.info(
        'Processing finished for file %s', input_file
    )
    logger.info('  Found %s waves', num_waves)
    logger.info('  Number of QC flags fired:')
    for key, val in qc_flags_fired.items():
        logger.info(f'      {key} {val:>6d}')

    logger.info('Processing done')

    # write output
    result_generator = filter(None, read_pickle_outfile_chunks(result_file))
    out_file = os.path.join(out_folder, f'fowd_{station_id}.nc')
    logger.info('Writing output to %s', out_file)
    write_records(result_generator, out_file, station_id, num_records=num_waves)
