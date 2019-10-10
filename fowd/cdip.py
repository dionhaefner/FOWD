"""
cdip.py

CDIP input file processing into FOWD datasets
"""

import os
import sys
import glob
import pickle
import logging
import functools
import multiprocessing
import concurrent.futures

import tqdm
import numpy as np
import xarray as xr

from .constants import SSH_REFERENCE_INTERVAL
from .output import write_records
from .processing import compute_wave_records, read_pickled_records

logger = logging.getLogger(__name__)


# dataset-specific helpers

def mask_invalid(data):
    """Mask all data that has error flags set"""

    xyz_invalid = (data['xyzFlagPrimary'] > 2) | (data['xyzFlagSecondary'] > 0)
    data['xyzZDisplacement'][xyz_invalid] = np.nan

    time_invalid = (data['waveFlagPrimary'] > 2) | (data['waveFlagSecondary'] > 0)
    freq_invalid = (data['waveFrequencyFlagPrimary'] > 2) | (data['waveFrequencyFlagSecondary'] > 0)

    data['waveDp'][time_invalid] = np.nan

    for var in ('waveSpread', 'waveMeanDirection', 'waveEnergyDensity'):
        data[var][time_invalid, :] = np.nan
        data[var][:, freq_invalid] = np.nan

    return data


def add_surface_elevation(data):
    """Add surface elevation variable to CDIP xarray Dataset"""

    dt = float(1 / data.xyzSampleRate.values)
    window_size = int(60 * SSH_REFERENCE_INTERVAL / dt)

    data['meanDisplacement'] = (
        data['xyzZDisplacement'].rolling({'xyzTime': window_size}, min_periods=60).mean()
    )

    data['xyzSurfaceElevation'] = data['xyzZDisplacement'] - data['meanDisplacement']
    return data


class InvalidCDIPFile(Exception):
    pass


def get_cdip_data(filepath):
    """Read CDIP input file as xarray Dataset"""

    allowed_vars = [
        'xyzStartTime', 'xyzZDisplacement', 'xyzSampleRate',
        'xyzFlagPrimary', 'xyzFlagSecondary',
        'waveTime', 'waveTimeBounds', 'waveFlagPrimary', 'waveFlagSecondary',
        'waveFrequency', 'waveFrequencyFlagPrimary', 'waveFrequencyFlagSecondary',
        'waveDp', 'waveSpread', 'waveMeanDirection', 'waveEnergyDensity',
        'metaWaterDepth', 'metaDeployLongitude', 'metaDeployLatitude',
    ]

    def drop_unnecessary(ds):
        if 'xyzStartTime' not in ds.variables:
            # some older deployments don't have any xyz data
            raise InvalidCDIPFile()

        xyz_time = (
            ds['xyzStartTime'] + (np.timedelta64(1, 's') * ds['xyzCount'] / ds['xyzSampleRate'])
        )

        for v in ds.variables:
            if v not in allowed_vars:
                ds = ds.drop(v)

        ds['xyzTime'] = xyz_time
        ds = ds.swap_dims({'xyzCount': 'xyzTime'})
        return ds

    data = drop_unnecessary(xr.open_dataset(filepath))
    data = mask_invalid(data)
    data = add_surface_elevation(data)
    return data


#

def get_cdip_wave_records(filepath, out_folder, qc_outfile=None):
    """Process a single file and write results to pickle file"""
    filename = os.path.basename(filepath)
    outfile = os.path.join(out_folder, f'{filename}.waves.pkl')
    statefile = os.path.join(out_folder, f'{filename}.state.pkl')

    # parse file into xarray Dataset
    try:
        data = get_cdip_data(filepath)
    except InvalidCDIPFile:
        return None, None

    # extract relevant quantities from xarray dataset
    t = np.ascontiguousarray(data['xyzTime'].values)
    z = np.ascontiguousarray(data['xyzZDisplacement'].values)
    z_normalized = np.ascontiguousarray(data['xyzSurfaceElevation'].values)

    direction_args = dict(
        direction_time=np.ascontiguousarray(data.waveTime.values),
        direction_frequencies=np.ascontiguousarray(data.waveFrequency.values),
        direction_spread=np.ascontiguousarray(data.waveSpread.values),
        direction_mean_direction=np.ascontiguousarray(data.waveMeanDirection.values),
        direction_energy_density=np.ascontiguousarray(data.waveEnergyDensity.values),
        direction_peak_direction=np.ascontiguousarray(data.waveDp.values),
    )

    meta_args = dict(
        filepath=filepath,
        uuid=data.attrs['uuid'],
        latitude=data.metaDeployLatitude,
        longitude=data.metaDeployLongitude,
        water_depth=np.float64(data.metaWaterDepth.values),
        sampling_rate=data.xyzSampleRate
    )

    del data  # reduce memory pressure

    compute_wave_records(
        t, z, z_normalized, outfile, statefile, meta_args,
        direction_args=direction_args, qc_outfile=qc_outfile
    )

    return outfile, statefile


def process_cdip_station(station_folder, out_folder, qc_outfile=None, nproc=None):
    """Process all deployments of a single CDIP station

    Supports processing in parallel (one process per input file).
    """
    station_folder = os.path.normpath(station_folder)
    assert os.path.isdir(station_folder)

    station_id = os.path.basename(station_folder)
    glob_pattern = os.path.join(station_folder, f'{station_id}_d??.nc')
    station_files = sorted(glob.glob(glob_pattern))

    num_inputs = len(station_files)

    if num_inputs == 0:
        raise RuntimeError('Given input folder does not contain any valid station files')

    if nproc is None:
        nproc = multiprocessing.cpu_count()

    nproc = min(nproc, num_inputs)

    wave_records = [None for _ in range(num_inputs)]

    do_work = functools.partial(get_cdip_wave_records, out_folder=out_folder, qc_outfile=qc_outfile)

    def handle_result(i, result, pbar):
        pbar.update(1)

        result_file, state_file = result
        filename = station_files[i]

        if result_file is None or state_file is None:
            logger.warning('Processing skipped for file %s', filename)
            return

        result_records = list(read_pickled_records(result_file))

        # concatenate subrecords
        wave_records[i] = {
            key: np.concatenate([subrecord[key] for subrecord in result_records])
            for key in result_records[0].keys()
        }

        # get QC information
        with open(state_file, 'rb') as f:
            qc_flags_fired = pickle.load(f)['num_flags_fired']

        # log progress
        num_done = sum(record is not None for record in wave_records)
        logger.info(
            'Processing finished for file %s (%s/%s done)', filename, num_done, num_inputs
        )
        logger.info('  Found %s waves', len(wave_records[i]['wave_id_local']))
        logger.info('  Number of QC flags fired:')
        for key, val in qc_flags_fired.items():
            logger.info(f'      {key} {val:>6d}')

    pbar_kwargs = dict(
        total=num_inputs, position=nproc, unit='file',
        desc='Processing files', dynamic_ncols=True,
        smoothing=0
    )

    logger.info('Starting processing for station %s (%s input files)', station_id, num_inputs)

    try:
        with tqdm.tqdm(**pbar_kwargs) as pbar:
            if nproc > 1:
                # process deployments in parallel
                with concurrent.futures.ProcessPoolExecutor(nproc) as executor:
                    try:
                        future_to_idx = {
                            executor.submit(do_work, station_file): i
                            for i, station_file in enumerate(station_files)
                        }

                        for future in concurrent.futures.as_completed(future_to_idx):
                            handle_result(future_to_idx[future], future.result(), pbar)

                    except Exception:
                        # abort workers immediately if anything goes wrong
                        for process in executor._processes.values():
                            process.terminate()
                        raise
            else:
                # sequential shortcut
                for i, result in enumerate(map(do_work, station_files)):
                    handle_result(i, result, pbar)

    finally:
        # reset cursor position
        sys.stderr.write('\n' * (nproc + 2))

    logger.info('Processing done')

    # remove skipped files
    wave_records = [subrecord for subrecord in wave_records if subrecord is not None]

    if not wave_records:
        logger.warn('Processed no files - no output to write')
        return

    # concatenate subrecords
    wave_records = {
        key: np.concatenate([subrecord[key] for subrecord in wave_records])
        for key in wave_records[0].keys()
    }

    # fix local id to be unique for the whole station
    wave_records['wave_id_local'] = np.arange(len(wave_records['wave_id_local']))

    # write output
    out_file = os.path.join(out_folder, f'fowd_cdip_{station_id}.nc')
    logger.info('Writing output to %s', out_file)
    station_name = f'CDIP_{station_id}'
    write_records(wave_records, out_file, station_name)
