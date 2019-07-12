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
    find_wave_indices, add_prefix, get_time_index, get_md5_hash, get_proc_version,
    get_station_meta, get_wave_parameters, get_sea_parameters, get_directional_parameters,
    check_quality_flags,
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


# utilities

def relative_pid():
    """Get relative PID of a pool process"""
    try:
        return multiprocessing.current_process()._identity[0]
    except IndexError:
        # not multiprocessing
        return 1


def read_pickled_records(input_file):
    """Read a sequence of pickled objects in the same file"""
    with open(input_file, 'rb') as f:
        unpickler = pickle.Unpickler(f)
        while True:
            try:
                yield unpickler.load()
            except EOFError:
                break


def initialize_processing(record_file, state_file, input_hash):
    """Parse temporary files to re-start processing if possible"""
    default_params = dict(
        last_wave_id=None,
        last_wave_end_time=None,
        wave_params_history=np.array([], dtype=WAVE_PARAMS_DTYPE),
        num_flags_fired={k: 0 for k in 'abcdefg'},
    )

    if not os.path.isfile(record_file) and not os.path.isfile(state_file):
        return default_params

    try:
        with open(state_file, 'rb') as f:
            saved_state = pickle.load(f)

        if saved_state['input_hash'] != input_hash:
            raise RuntimeError('Input hash has changed')

        if saved_state['processing_version'] != get_proc_version():
            raise RuntimeError('Processing version has changed')

        for row in read_pickled_records(record_file):
            last_wave_record = row

        wave_params_history = saved_state['wave_params_history']
        num_flags_fired = saved_state['num_flags_fired']
        last_wave_end_time = last_wave_record['wave_end_time'].max()
        last_wave_id = last_wave_record['wave_id_local'].max()

    except Exception as exc:
        logger.warn(
            f'Error while restarting processing from pickle files ({record_file}, {state_file}): '
            f'{exc!r} - starting from scratch'
        )

        if os.path.isfile(record_file):
            os.remove(record_file)

        if os.path.isfile(state_file):
            os.remove(state_file)

        return default_params

    return dict(
        last_wave_id=last_wave_id,
        last_wave_end_time=last_wave_end_time,
        wave_params_history=wave_params_history,
        num_flags_fired=num_flags_fired
    )


#

def get_cdip_wave_records(filepath, out_folder):
    """Process a single file and write results to pickle file"""
    filename = os.path.basename(filepath)

    # parse file into xarray Dataset
    data = get_cdip_data(filepath)

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
    input_hash = get_md5_hash(filepath)

    del data  # reduce memory pressure

    # initialize processing state
    outfile = os.path.join(out_folder, f'{filename}.waves.pkl')
    statefile = os.path.join(out_folder, f'{filename}.state.pkl')

    initial_state = initialize_processing(outfile, statefile, input_hash)

    if initial_state['last_wave_id'] is not None:
        local_wave_id = initial_state['last_wave_id'] + 1
    else:
        local_wave_id = 0

    if initial_state['last_wave_end_time'] is not None:
        start_idx = max(get_time_index(initial_state['last_wave_end_time'], t) - 1, 0)
    else:
        start_idx = 0

    wave_params_history = initial_state['wave_params_history']
    num_flags_fired = initial_state['num_flags_fired']

    history_length = np.timedelta64(WAVE_HISTORY_LENGTH, 'm')
    last_wave_stop = start_idx
    last_directional_time_idx = None
    wave_records = collections.defaultdict(list)

    # run processing
    pbar_kwargs = dict(
        total=len(z), unit_scale=True, position=(relative_pid() - 1),
        dynamic_ncols=True, desc=filename,
        mininterval=0.25, maxinterval=1, smoothing=0.1,
        postfix=dict(waves_processed=str(local_wave_id)), initial=start_idx
    )

    with tqdm.tqdm(**pbar_kwargs) as pbar:

        def handle_output(wave_records, wave_params_history, num_flags_fired):
            # convert records to NumPy array
            wave_records_np = {}
            for key, val in wave_records.items():
                wave_records_np[key] = np.asarray(val)

            # dump results to files
            with open(outfile, 'ab') as f:
                pickle.dump(wave_records_np, f)

            with open(statefile, 'wb') as f:
                pickle.dump({
                    'wave_params_history': wave_params_history,
                    'num_flags_fired': num_flags_fired,
                    'input_hash': input_hash,
                    'processing_version': get_proc_version(),
                }, f)

        for wave_start, wave_stop in find_wave_indices(z_normalized, start_idx=start_idx):
            this_wave_records = {}

            pbar.update(wave_stop - last_wave_stop)
            last_wave_stop = wave_stop

            # compute wave parameters
            xyz_idx = slice(wave_start, wave_stop + 1)
            wave_params = get_wave_parameters(
                local_wave_id, t[xyz_idx], z_normalized[xyz_idx], water_depth, input_hash
            )
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

                # skip further processing for this wave
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

            for var in this_wave_records.keys():
                wave_records[var].append(this_wave_records[var])

            local_wave_id += 1

            if local_wave_id % 1000 == 0:
                handle_output(wave_records, wave_params_history, num_flags_fired)
                wave_records.clear()
                pbar.set_postfix(dict(waves_processed=str(local_wave_id)))

        else:
            # all waves processed
            pbar.update(len(z) - wave_stop)

        if wave_records:
            handle_output(wave_records, wave_params_history, num_flags_fired)

        pbar.set_postfix(dict(waves_processed=str(local_wave_id)))

    return outfile, statefile


def process_cdip_station(station_folder, out_folder, nproc=None):
    """Process all deployments of a single CDIP station

    Supports processing in parallel (one process per input file).
    """
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

    pbar_kwargs = dict(
        total=num_inputs, position=nproc, unit='file',
        desc='Processing files', dynamic_ncols=True,
        smoothing=0
    )

    try:
        with tqdm.tqdm(**pbar_kwargs) as pbar:

            def handle_result(i, result):
                nonlocal num_done
                num_done += 1

                result_file, state_file = result
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
                filename = station_files[i]
                logger.info(
                    'Processing finished for file %s (%s/%s done)', filename, num_done, num_inputs
                )
                logger.info('  Found %s waves', len(wave_records[i]['wave_id_local']))
                logger.info('  Number of QC flags fired:')
                for key, val in qc_flags_fired.items():
                    logger.info(f'      {key} {val:>6d}')

                pbar.update(1)

            if nproc > 1:
                # process deployments in parallel
                with concurrent.futures.ProcessPoolExecutor(nproc) as executor:
                    try:
                        future_to_idx = {
                            executor.submit(worker, station_file): i
                            for i, station_file in enumerate(station_files)
                        }

                        for future in concurrent.futures.as_completed(future_to_idx):
                            handle_result(future_to_idx[future], future.result())

                    except Exception:
                        # abort workers immediately if anything goes wrong
                        for process in executor._processes.values():
                            process.terminate()
                        raise
            else:
                # sequential shortcut
                for i, result in enumerate(map(worker, station_files)):
                    handle_result(i, result)

    finally:
        # reset cursor position
        sys.stderr.write('\n' * (nproc + 2))

    logger.info('Processing done')

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
    write_records(wave_records, out_file, station_id)
