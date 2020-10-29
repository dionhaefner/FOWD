"""
processing.py

Main processing chain.
"""

import os
import json
import pickle
import logging
import warnings
import contextlib
import collections
import multiprocessing

import tqdm
import filelock
import numpy as np

from .constants import (
    WAVE_HISTORY_LENGTH, SEA_STATE_INTERVALS,
    QC_FAIL_LOG_THRESHOLD, QC_EXTREME_WAVE_LOG_THRESHOLD
)

from .operators import (
    find_wave_indices, add_prefix, get_time_index, get_md5_hash, get_proc_version,
    get_station_meta, get_wave_parameters, get_sea_parameters, get_directional_parameters,
    check_quality_flags, compute_significant_wave_height
)

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


# utilities

def relative_pid():
    """Get relative PID of a pool process."""
    try:
        return multiprocessing.current_process()._identity[0]
    except IndexError:
        # not multiprocessing
        return 1


def read_pickle_outfile_chunks(pickle_file):
    """Read a sequence of pickled objects in the same file."""
    if os.path.isfile(pickle_file):
        with open(pickle_file, 'rb') as f:
            unpickler = pickle.Unpickler(f)
            while True:
                try:
                    yield unpickler.load()
                except EOFError:
                    break


def read_pickle_statefile(state_file):
    """Read pickled state file."""
    with open(state_file, 'rb') as f:
        return pickle.load(f)


def qc_format(filename, flags_fired, rel_wave_height, t, z, z_raw, wave_period, crest_height,
              trough_depth):
    """Format QC records for JSON output."""
    time_offset = (t - t[0]) / np.timedelta64(1, 's')

    def format_float_arr(floatarr, precision):
        return [round(float(f), precision) for f in floatarr]

    return dict(
        filename=filename,
        flags_fired=flags_fired,
        relative_wave_height=float(rel_wave_height),
        start_date=str(t[0]),
        time=format_float_arr(time_offset, 4),
        elevation=format_float_arr(z, 4),
        wave_periods=format_float_arr(wave_period, 2),
        crest_heights=format_float_arr(crest_height, 2),
        trough_depths=format_float_arr(trough_depth, 2),
    )


def is_same_version(version1, version2):
    """Check whether two versions are equal.

    This is the case if minor and major version are the same (e.g. 2.4.1 and 2.4.3).
    """
    split_v1 = version1.split('.')
    split_v2 = version2.split('.')

    if len(split_v1) < 2 or len(split_v2) < 2:
        # unexpected format
        return False

    # only compare major and minor version
    return split_v1[:2] == split_v2[:2]


def initialize_processing(record_file, state_file, input_hash):
    """Parse temporary files to re-start processing if possible."""
    default_params = dict(
        last_wave_id=None,
        last_wave_end_time=None,
        wave_params_history=np.array([], dtype=WAVE_PARAMS_DTYPE),
        num_flags_fired={k: 0 for k in 'abcdefg'},
    )

    if not os.path.isfile(record_file) and not os.path.isfile(state_file):
        return default_params

    try:
        saved_state = read_pickle_statefile(state_file)

        if saved_state['input_hash'] != input_hash:
            raise RuntimeError('Input hash has changed')

        this_version = get_proc_version()
        if not is_same_version(saved_state['processing_version'], this_version):
            raise RuntimeError('Processing version has changed')

        for row in read_pickle_outfile_chunks(record_file):
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


@contextlib.contextmanager
def strict_filelock(target_file):
    """File-based lock that throws an exception if lock is already in place."""
    lockfile = f'{target_file}.lock'
    if os.path.isfile(lockfile):
        raise RuntimeError(
            f'File {target_file} appears to be locked. '
            'Make sure that no other process is accessing it, then remove manually.'
        )

    with open(lockfile, 'w'):
        pass

    try:
        yield
    finally:
        try:
            os.remove(lockfile)
        except OSError:
            warnings.warn(f'Could not remove lock file {lockfile}')


def compute_wave_records(time, elevation, elevation_normalized, outfile, statefile,
                         meta_args, direction_args=None, qc_outfile=None):
    """Compute all wave records in given input data, and dump results to pickle files."""
    # pre-compute station metadata
    filename = os.path.basename(meta_args['filepath'])
    water_depth = meta_args['water_depth']

    station_meta = get_station_meta(**meta_args)
    input_hash = get_md5_hash(meta_args['filepath'])

    if qc_outfile is not None:
        if not os.path.isfile(qc_outfile):
            with open(qc_outfile, 'w'):
                pass
        qc_lock = filelock.FileLock(f'{qc_outfile}.lock', timeout=10)

    def handle_output(wave_records, wave_params_history, num_flags_fired):
        # convert records to NumPy arrays
        wave_records_np = {}
        for key, val in wave_records.items():
            if key == 'wave_raw_elevation':
                wave_records_np[key] = np.empty(len(val), dtype=object)
                wave_records_np[key][...] = val
            else:
                wave_records_np[key] = np.asarray(val)

        # dump results to files
        if not os.path.exists(outfile):
            with open(outfile, 'wb'):
                pass

        if wave_records_np:
            with open(outfile, 'ab') as f:
                pickle.dump(wave_records_np, f)

        with open(statefile, 'wb') as f:
            pickle.dump({
                'wave_params_history': wave_params_history,
                'num_flags_fired': num_flags_fired,
                'input_hash': input_hash,
                'processing_version': get_proc_version(),
            }, f)

    # initialize processing state

    initial_state = initialize_processing(outfile, statefile, input_hash)

    if initial_state['last_wave_id'] is not None:
        local_wave_id = initial_state['last_wave_id'] + 1
    else:
        local_wave_id = 0

    if initial_state['last_wave_end_time'] is not None:
        start_idx = max(get_time_index(initial_state['last_wave_end_time'], time) - 1, 0)
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
        total=len(elevation), unit_scale=True, position=(relative_pid() - 1),
        dynamic_ncols=True, desc=filename,
        mininterval=0.25, maxinterval=1, smoothing=0.1,
        postfix=dict(waves_processed=str(local_wave_id)), initial=start_idx
    )

    with strict_filelock(outfile), tqdm.tqdm(**pbar_kwargs) as pbar:
        # initialize output files
        handle_output(wave_records, wave_params_history, num_flags_fired)

        for wave_start, wave_stop in find_wave_indices(elevation_normalized, start_idx=start_idx):
            pbar.update(wave_stop - last_wave_stop)
            last_wave_stop = wave_stop

            this_wave_records = {}

            # compute wave parameters
            xyz_idx = slice(wave_start, wave_stop + 1)
            wave_params = get_wave_parameters(
                local_wave_id, time[xyz_idx], elevation_normalized[xyz_idx], water_depth, input_hash
            )
            this_wave_records.update(
                add_prefix(wave_params, 'wave')
            )

            # roll over wave parameter history and append record for current wave
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
                get_time_index(wave_params['start_time'] - history_length, time),
                wave_stop + 1
            )

            qc_args = (
                time[sea_history_idx],
                elevation_normalized[sea_history_idx],
                elevation[sea_history_idx],
                wave_params_history['zero_crossing_period'],
                wave_params_history['crest_height'],
                wave_params_history['trough_depth']
            )

            flags_fired = check_quality_flags(*qc_args)

            if qc_outfile is not None:
                # write significant waves to QC dataset
                significant_waveheight = compute_significant_wave_height(
                    wave_params_history['height']
                )
                rel_waveheight = wave_params['height'] / significant_waveheight

                above_fail_threshold = rel_waveheight > QC_FAIL_LOG_THRESHOLD
                above_extreme_threshold = rel_waveheight > QC_EXTREME_WAVE_LOG_THRESHOLD
                write_qc = above_extreme_threshold or (flags_fired and above_fail_threshold)

                if write_qc:
                    with qc_lock, open(qc_outfile, 'a') as qcf:
                        qc_info = qc_format(filename, flags_fired, rel_waveheight, *qc_args)
                        qcf.write(json.dumps(qc_info) + '\n')

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
                    get_time_index(wave_params['start_time'] - offset, time),
                    wave_start
                )

                wave_param_timediff = wave_params['start_time'] - wave_params_history['start_time']
                wave_param_mask = np.logical_and(
                    # do not look into the future
                    wave_param_timediff > np.timedelta64(1, 'ms'),
                    # look at most sea_state_period minutes into the past
                    wave_param_timediff < offset
                )

                sea_state_params = get_sea_parameters(
                    time[sea_state_idx],
                    elevation[sea_state_idx],
                    wave_params_history['height'][wave_param_mask],
                    wave_params_history['zero_crossing_period'][wave_param_mask],
                    water_depth
                )

                this_wave_records.update(
                    add_prefix(sea_state_params, f'sea_state_{sea_state_period}m')
                )

            # compute directional quantities
            if direction_args is not None:
                directional_time_idx = get_time_index(
                    wave_params['start_time'], direction_args['direction_time'], nearest=True
                )

                if directional_time_idx != last_directional_time_idx:
                    # only re-compute if index has changed
                    directional_params = get_directional_parameters(
                        direction_args['direction_time'][directional_time_idx],
                        direction_args['direction_frequencies'],
                        direction_args['direction_spread'][directional_time_idx],
                        direction_args['direction_mean_direction'][directional_time_idx],
                        direction_args['direction_energy_density'][directional_time_idx],
                        direction_args['direction_peak_direction'][directional_time_idx]
                    )
                    last_directional_time_idx = directional_time_idx

                this_wave_records.update(
                    add_prefix(directional_params, 'direction')
                )

            for var in this_wave_records.keys():
                wave_records[var].append(this_wave_records[var])

            local_wave_id += 1

            # output and empty records in regular intervals
            if local_wave_id % 1000 == 0:
                handle_output(wave_records, wave_params_history, num_flags_fired)
                wave_records.clear()
                pbar.set_postfix(dict(waves_processed=str(local_wave_id)))

        else:
            # all waves processed
            pbar.update(len(elevation) - last_wave_stop)

        if wave_records:
            handle_output(wave_records, wave_params_history, num_flags_fired)

        pbar.set_postfix(dict(waves_processed=str(local_wave_id)))

    return outfile, statefile
