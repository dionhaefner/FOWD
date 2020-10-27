"""
postprocessing.py

Postprocessing of CDIP files and QC logs.
"""

import os
import json
import logging

import numpy as np
import tqdm

from .constants import QC_EXTREME_WAVE_LOG_THRESHOLD

logger = logging.getLogger(__name__)


def plot_qc(qcfile, outdir, exclude_flags=tuple('cefg'), plot_extreme=True):
    import matplotlib as mpl
    mpl.use('Agg')
    import matplotlib.pyplot as plt

    if not set(exclude_flags) <= set('abcdefg'):
        raise ValueError('exclude_flags can only contain {a, b, c, d, e, f, g}')

    with open(qcfile, 'r') as f:
        qc_records = [json.loads(line) for line in f]

    os.makedirs(outdir, exist_ok=True)

    basename = os.path.basename(qcfile).split('.')[0]

    i = 1

    for record in tqdm.tqdm(qc_records):
        qc_passed = len(record['flags_fired']) == 0

        process_record = (
            not qc_passed
            and all(flag not in exclude_flags for flag in record['flags_fired'])
        )

        if plot_extreme:
            process_record |= (
                qc_passed
                and record['relative_wave_height'] > QC_EXTREME_WAVE_LOG_THRESHOLD
            )

        if not process_record:
            continue

        mintime, maxtime = np.min(record['time']), np.max(record['time'])

        # don't plot records with extreme time jumps
        if maxtime - mintime > 10000 and 'e' in record['flags_fired']:
            continue

        elev_range = np.nanmax(np.abs(record['elevation']))

        info_left = [
            f'Wave height: {record["relative_wave_height"]:.2f} SWH',
            f'Record start time: {record["start_date"]}',
            f'Source file: {record["filename"]}',
        ]
        info_right = [
            f'QC flags fired: {record["flags_fired"]}' if not qc_passed else 'QC passed',
        ]

        fig, ax = plt.subplots(1, 1, figsize=(15, 4))
        plt.plot(record['time'], record['elevation'], linewidth=0.5)
        plt.xlim(mintime, maxtime)
        plt.ylim(-elev_range, elev_range)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Elevation (m)')
        ax.set_xticks(np.arange(mintime, maxtime, 10), minor=True)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.text(0.01, 1, '\n'.join(info_left), va='top', ha='left', transform=ax.transAxes)
        ax.text(0.99, 1, '\n'.join(info_right), va='top', ha='right', transform=ax.transAxes)
        fig.tight_layout()
        fig.savefig(os.path.join(outdir, f'{basename}_qc_{i:0>4}.pdf'))
        plt.close(fig)

        i += 1


CDIP_DEPLOYMENT_BLACKLIST = {
    '045p1': ['d01', 'd02', 'd03', 'd13', 'd15', 'd17', 'd19', 'd21'],
    '094p1': ['d01', 'd02', 'd03', 'd04', 'd05'],
    '096p1': ['d04'],
    '100p1': ['d11'],
    '106p1': ['d02'],
    '109p1': ['d05', 'd06'],
    '111p1': ['d06'],
    '132p1': ['d01'],
    '141p1': ['d03'],
    '142p1': ['d02', 'd15', 'd18'],
    '144p1': ['d01'],
    '146p1': ['d01', 'd02'],
    '158p1': ['d02', 'd04'],
    '162p1': ['d07'],
    '163p1': ['d01', 'd05'],
    '167p1': ['d01'],
    '172p1': ['d01'],
    '177p1': ['*'],
    '196p1': ['d04'],
    '201p1': ['d03'],
    '205p1': ['*'],
    '206p1': ['*'],
    '261p1': ['*'],
    '430p1': ['d06'],
    '431p1': ['d02'],
}


def apply_mask(ds, dim, mask):
    mask = mask.isel(meta_station_name=0)

    if mask.values.all():
        return ds

    idx = np.where(mask.values)[0]
    return ds.isel(wave_id_local=idx)


def remove_blacklisted(ds):
    deployment_files = np.unique(ds['meta_source_file_name'])
    whitelist = list(deployment_files)
    for f in deployment_files:
        for station, deployments in CDIP_DEPLOYMENT_BLACKLIST.items():
            if station in f:
                if '*' in deployments or any(d in f for d in deployments):
                    logger.info(f'Removing blacklisted deployment {f}')
                    whitelist.remove(f)

    mask = ds['meta_source_file_name'].isin(whitelist)
    num_filtered = mask.size - mask.sum().values
    return apply_mask(ds, 'wave_id_local', mask), num_filtered


def filter_low_swh(ds):
    mask = ds['sea_state_30m_significant_wave_height_spectral'] > 0.5
    num_filtered = mask.size - mask.sum().values
    return apply_mask(ds, 'wave_id_local', mask), num_filtered


def filter_undersampled(ds):
    nyquist_frequency = 0.5 * ds['meta_sampling_rate'].astype('float32')
    mean_frequency = 1. / ds['sea_state_30m_mean_period_spectral'].astype('float32')
    mask = 2.2 * mean_frequency < nyquist_frequency
    num_filtered = mask.size - mask.sum().values
    return apply_mask(ds, 'wave_id_local', mask), num_filtered


def filter_cdip(ds, num_filtered_dict=None, chunk_size=10_000):
    if num_filtered_dict is None:
        num_filtered_dict = {}
    else:
        num_filtered_dict.clear()

    num_records = len(ds['wave_id_local'])

    filters = {
        'blacklist': remove_blacklisted,
        'low_swh': filter_low_swh,
        'undersampled': filter_undersampled,
    }

    num_filtered_dict.update({f: 0 for f in filters})

    chunks = [
        slice(i, min(i + chunk_size, num_records))
        for i in range(0, num_records, chunk_size)
        if i < num_records
    ]

    for chunk_slice in chunks:
        dsi = ds.isel(wave_id_local=chunk_slice).load()

        for name, filter_fun in filters.items():
            dsi, n = filter_fun(dsi)
            num_filtered_dict[name] += n

            if len(dsi['wave_id_local']) == 0:
                dsi = None
                break

        yield dsi
