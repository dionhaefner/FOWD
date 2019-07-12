"""
cdip.py

Process CDIP input files into FOWD datasets.
"""

import os
import sys
import glob
import pickle
import logging
import tempfile
import functools
import contextlib
import collections
import multiprocessing
import concurrent.futures

import tqdm
import numpy as np
import xarray as xr

from .constants import (
    SSH_REFERENCE_INTERVAL, WAVE_HISTORY_LENGTH, SEA_STATE_INTERVALS
)

from .operators import (
    find_wave_indices, add_prefix, get_time_index, get_station_meta, get_wave_parameters,
    get_sea_parameters, get_directional_parameters, check_quality_flags,
)

from .output import write_records

logger = logging.getLogger(__name__)


# constants

WAVE_PARAMS_DTYPE = [
    ('start_time', '<M8[ns]'),
    ('end_time', '<M8[ns]'),
    ('crest_height', '<f8'),
    ('trough_depth', '<f8'),
    ('height', '<f8'),
    ('zero_crossing_period', '<f8'),
    ('maximum_elevation_slope', '<f8')
]

WAVE_PARAMS_KEYS = [key for key, _ in WAVE_PARAMS_DTYPE]


#

def mask_invalid(data):
    xyz_invalid = (data['xyzFlagPrimary'] > 2) | (data['xyzFlagSecondary'] > 0)
    data['xyzZDisplacement'][xyz_invalid] = np.nan

    time_invalid = (data['waveFlagPrimary'] > 2) | (data['waveFlagSecondary'] > 0)
    freq_invalid = (data['waveFrequencyFlagPrimary'] > 2) | (data['waveFrequencyFlagSecondary'] > 0)

    data['waveDp'][time_invalid] = np.nan

    for var in ('waveSpread', 'waveMeanDirection', 'waveEnergyDensity'):
        data[var][time_invalid, :] = np.nan
        data[var][:, freq_invalid] = np.nan

    return data


def get_cdip_data(filepath):
    allowed_vars = [
        'xyzStartTime', 'xyzZDisplacement', 'xyzSampleRate',
        'xyzFlagPrimary', 'xyzFlagSecondary',
        'waveTime', 'waveTimeBounds', 'waveFlagPrimary', 'waveFlagSecondary',
        'waveFrequency', 'waveFrequencyFlagPrimary', 'waveFrequencyFlagSecondary',
        'waveDp', 'waveSpread', 'waveMeanDirection', 'waveEnergyDensity',
        'metaWaterDepth', 'metaDeployLongitude', 'metaDeployLatitude',
    ]

    def drop_unnecessary(ds):
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


def add_surface_elevation(data):
    dt = float(1 / data.xyzSampleRate.values)
    window_size = int(60 * SSH_REFERENCE_INTERVAL / dt)

    data['meanDisplacement'] = (
        data['xyzZDisplacement'].rolling({'xyzTime': window_size}, min_periods=60).mean()
    )

    data['xyzSurfaceElevation'] = data['xyzZDisplacement'] - data['meanDisplacement']
    return data


def relative_pid():
    # get relative PID of a pool process
    try:
        return multiprocessing.current_process()._identity[0]
    except IndexError:
        # not multiprocessing
        return 1


def get_cdip_wave_records(filepath, out_folder):
    outfile = tempfile.NamedTemporaryFile(delete=False, dir=out_folder, suffix='.incomplete.npy')
    outfile.close()
    outfile = outfile.name

    # parse file into xarray Dataset
    data = get_cdip_data(filepath)

    # initialize counters and containers
    wave_records = collections.defaultdict(list)
    wave_params_history = np.array([], dtype=WAVE_PARAMS_DTYPE)
    history_length = np.timedelta64(WAVE_HISTORY_LENGTH, 'm')

    num_flags_fired = {k: 0 for k in 'abcdefg'}

    last_wave_stop = 0
    last_directional_time_idx = None

    local_wave_id = 0

    # extract relevant quantities from xarray dataset
    t = np.ascontiguousarray(data['xyzTime'].values)
    z = np.ascontiguousarray(data['xyzZDisplacement'].values)
    z_normalized = np.ascontiguousarray(data['xyzSurfaceElevation'].values)

    water_depth = np.float64(data.metaWaterDepth.values)

    direction_time = np.ascontiguousarray(data.waveTime.values)
    direction_frequencies = np.ascontiguousarray(data.waveFrequency.values)
    direction_spread = np.ascontiguousarray(data.waveSpread.values)
    direction_mean_direction = np.ascontiguousarray(data.waveMeanDirection.values)
    direction_energy_density = np.ascontiguousarray(data.waveEnergyDensity.values)
    direction_peak_direction = np.ascontiguousarray(data.waveDp.values)

    # pre-compute station metadata
    station_meta = get_station_meta(
        filepath,
        data.attrs['uuid'],
        data.metaDeployLatitude,
        data.metaDeployLongitude,
        data.metaWaterDepth,
        data.xyzSampleRate
    )

    del data

    # run processing
    pbar_kwargs = dict(
        total=len(z), unit_scale=True, position=(relative_pid() - 1),
        dynamic_ncols=True, desc=os.path.basename(filepath),
        mininterval=0.25, maxinterval=1, smoothing=0.1,
        postfix=dict(waves_processed=0)
    )

    with tqdm.tqdm(**pbar_kwargs) as pbar:
        for wave_start, wave_stop in find_wave_indices(z_normalized):
            this_wave_records = {}

            pbar.update(wave_stop - last_wave_stop)
            last_wave_stop = wave_stop

            # compute wave parameters
            xyz_idx = slice(wave_start, wave_stop + 1)
            wave_params = get_wave_parameters(
                local_wave_id, t[xyz_idx], z_normalized[xyz_idx], water_depth, filepath)
            this_wave_records.update(
                add_prefix(wave_params, 'wave')
            )

            # roll over wave parameter history and append new record
            rollover_mask = (
                (wave_params['start_time'] - wave_params_history['start_time']) < history_length
            )

            new_history_row = np.array(
                [tuple(wave_params[key] for key in WAVE_PARAMS_KEYS)],
                dtype=WAVE_PARAMS_DTYPE
            )

            wave_params_history = np.concatenate([
                wave_params_history[rollover_mask],
                new_history_row
            ])

            # run QC
            sea_history_idx = slice(
                get_time_index(wave_params['start_time'] - history_length, t),
                wave_start
            )

            flags_fired = check_quality_flags(
                t[sea_history_idx],
                z_normalized[sea_history_idx],
                wave_params_history['zero_crossing_period'],
                wave_params_history['crest_height'],
                wave_params_history['trough_depth']
            )

            if flags_fired:
                for flag in flags_fired:
                    num_flags_fired[flag] += 1
                continue

            # add metadata
            this_wave_records.update(
                add_prefix(station_meta, 'meta')
            )

            # compute sea state parameters
            for sea_state_period in SEA_STATE_INTERVALS:
                offset = np.timedelta64(sea_state_period, 'm')

                sea_state_idx = slice(
                    get_time_index(wave_params['start_time'] - offset, t),
                    wave_start
                )

                wave_param_mask = (
                    (wave_params['start_time'] - wave_params_history['start_time']) < offset
                )

                sea_state_params = get_sea_parameters(
                    t[sea_state_idx],
                    z[sea_state_idx],
                    wave_params_history['height'][wave_param_mask],
                    wave_params_history['zero_crossing_period'][wave_param_mask],
                    water_depth
                )

                this_wave_records.update(
                    add_prefix(sea_state_params, f'sea_state_{sea_state_period}m')
                )

            # compute directional quantities
            directional_time_idx = get_time_index(
                wave_params['start_time'], direction_time, nearest=True
            )

            if directional_time_idx != last_directional_time_idx:
                # only re-compute if index has changed
                directional_params = get_directional_parameters(
                    direction_time[directional_time_idx],
                    direction_frequencies,
                    direction_spread[directional_time_idx],
                    direction_mean_direction[directional_time_idx],
                    direction_energy_density[directional_time_idx],
                    direction_peak_direction[directional_time_idx]
                )
                last_directional_time_idx = directional_time_idx

            this_wave_records.update(
                add_prefix(directional_params, 'direction')
            )

            # append to global record
            for var in this_wave_records.keys():
                wave_records[var].append(this_wave_records[var])

            local_wave_id += 1

            if local_wave_id % 1000 == 0:
                # prevent too frequent progress updates
                pbar.set_postfix(dict(waves_processed=str(local_wave_id)))

        pbar.update(len(z) - wave_stop)

    # convert records to NumPy array
    for key, val in wave_records.items():
        wave_records[key] = np.asarray(val)

    with open(outfile, 'wb') as f:
        pickle.dump(wave_records, f)

    return outfile, num_flags_fired


def process_cdip_station(station_folder, out_folder, nproc=None):
    station_folder = os.path.normpath(station_folder)
    assert os.path.isdir(station_folder)

    station_id = os.path.basename(station_folder)
    glob_pattern = os.path.join(station_folder, f'{station_id}_d??.nc')
    station_files = sorted(glob.glob(glob_pattern))

    num_inputs = len(station_files)
    num_done = 0

    if num_inputs == 0:
        raise RuntimeError('Given input folder does not contain any valid station files')

    if nproc is None:
        nproc = multiprocessing.cpu_count()

    nproc = min(nproc, num_inputs)

    wave_records = [[] for _ in range(num_inputs)]

    worker = functools.partial(get_cdip_wave_records, out_folder=out_folder)

    try:
        with contextlib.ExitStack() as es:
            pbar = es.enter_context(
                tqdm.tqdm(
                    total=num_inputs, position=nproc, unit='file',
                    desc='Processing files', dynamic_ncols=True,
                    smoothing=0
                )
            )

            def handle_result(i, result):
                nonlocal num_done
                num_done += 1

                result_file, qc_flags_fired = result

                with open(result_file, 'rb') as f:
                    wave_records[i] = pickle.load(f)

                try:
                    os.remove(result_file)
                except OSError:
                    pass

                filename = station_files[i]
                logger.info(
                    'Processing finished for file %s (%s/%s done)', filename, num_done, num_inputs
                )
                logger.info('  Found %s waves', len(wave_records[i]["wave_id_local"]))
                logger.info('  Number of QC flags fired:')
                for key, val in qc_flags_fired.items():
                    logger.info(f'      {key} {val:>6d}')

                pbar.update(1)

            if nproc > 1:
                # process deployments in parallel
                executor = es.enter_context(
                    concurrent.futures.ProcessPoolExecutor(nproc)
                )

                try:
                    future_to_idx = {
                        executor.submit(worker, station_file): i
                        for i, station_file in enumerate(station_files)
                    }

                    for future in concurrent.futures.as_completed(future_to_idx):
                        handle_result(future_to_idx[future], future.result())

                except Exception:
                    # kill workers immediately if anything goes wrong
                    for p in executor._processes.values():
                        p.terminate()
                    raise
            else:
                # sequential shortcut
                for i, result in enumerate(map(worker, station_files)):
                    handle_result(i, result)

    finally:
        # reset cursor position
        sys.stderr.write('\n' * (nproc + 2))

    # concatenate subrecords
    wave_records = {
        key: np.concatenate([subrecord[key] for subrecord in wave_records])
        for key in wave_records[0].keys()
    }

    # fix local id to be unique for the whole station
    wave_records['wave_id_local'] = np.arange(len(wave_records['wave_id_local']))

    # write output
    out_file = os.path.join(out_folder, f'fowd_cdip_{station_id}.nc')
    write_records(wave_records, out_file, station_id)
