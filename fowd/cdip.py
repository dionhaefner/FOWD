"""
cdip.py

Process CDIP input files into FOWD datasets.
"""

import os
import sys
import glob
import logging
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
            ds['xyzStartTime']
            + (1e9 * ds['xyzCount'] / ds['xyzSampleRate']).astype('timedelta64[ns]')
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


def get_cdip_wave_records(filepath):
    data = get_cdip_data(filepath)

    wave_records = collections.defaultdict(list)

    t = np.ascontiguousarray(data['xyzTime'].values)
    z = np.ascontiguousarray(data['xyzZDisplacement'].values)
    z_normalized = np.ascontiguousarray(data['xyzSurfaceElevation'].values)

    water_depth = np.float64(data.metaWaterDepth.values)

    wave_params_history = np.array([], dtype=WAVE_PARAMS_DTYPE)
    history_length = np.timedelta64(WAVE_HISTORY_LENGTH, 'm')

    num_flags_fired = {k: 0 for k in 'abcdefg'}

    last_wave_stop = 0
    last_directional_time_idx = None

    direction_time = np.ascontiguousarray(data.waveTime.values)
    direction_frequencies = np.ascontiguousarray(data.waveFrequency.values)
    direction_spread = np.ascontiguousarray(data.waveSpread.values)
    direction_mean_direction = np.ascontiguousarray(data.waveMeanDirection.values)
    direction_energy_density = np.ascontiguousarray(data.waveEnergyDensity.values)
    direction_peak_direction = np.ascontiguousarray(data.waveDp.values)

    station_meta = get_station_meta(
        filepath,
        data.attrs['uuid'],
        data.metaDeployLatitude,
        data.metaDeployLongitude,
        data.metaWaterDepth,
        data.xyzSampleRate
    )

    local_wave_id = 0

    pbar_kwargs = dict(
        total=len(z), unit_scale=True, position=(relative_pid() - 1),
        dynamic_ncols=True, desc=os.path.basename(filepath),
        mininterval=0.25, maxinterval=1
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
                wave_params['start_time'], data.waveTime.values, nearest=True
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

            if local_wave_id % 1000 == 0:
                pbar.set_postfix(dict(wave_id=str(local_wave_id)))

            local_wave_id += 1

            if local_wave_id > 5000:
                break

        pbar.update(len(z) - wave_stop)

    # convert record lists to NumPy arrays
    wave_records = {k: np.array(v) for k, v in wave_records.items()}

    return wave_records, num_flags_fired


def process_cdip_station(station_folder, out_folder, nproc=None):
    assert os.path.isdir(station_folder)
    station_id = os.path.basename(station_folder)
    glob_pattern = os.path.join(station_folder, f'{station_id}_d??.nc')
    station_files = sorted(glob.glob(glob_pattern))

    num_inputs = len(station_files)

    if nproc is None:
        nproc = multiprocessing.cpu_count()

    nproc = min(nproc, num_inputs)

    wave_records = [[] for _ in range(num_inputs)]

    def handle_qc_flags(qc_flags_fired, filename):
        qc_flag_str = '\n'.join(f'\t- {key}: {val}' for key, val in qc_flags_fired.items())
        logger.warn(f'QC flags fired for file {filename}:\n{qc_flag_str}')

    # process deployments in parallel
    try:
        with contextlib.ExitStack() as es:
            pbar = es.enter_context(
                tqdm.tqdm(
                    total=num_inputs, position=nproc, unit='file',
                    desc='Processing files', dynamic_ncols=True,
                    smoothing=0
                )
            )

            if nproc > 1:
                executor = es.enter_context(
                    concurrent.futures.ProcessPoolExecutor(nproc)
                )
                future_to_idx = {
                    executor.submit(get_cdip_wave_records, station_file): i
                    for i, station_file in enumerate(station_files)
                }

                for future in concurrent.futures.as_completed(future_to_idx):
                    i = future_to_idx[future]
                    wave_records[i], qc_flags_fired = future.result()

                    handle_qc_flags(qc_flags_fired, station_files[i])
                    pbar.update(1)
            else:
                for i, result in enumerate(map(get_cdip_wave_records, station_files)):
                    wave_records[i], qc_flags_fired = result

                    handle_qc_flags(qc_flags_fired, station_files[i])
                    pbar.update(1)

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
