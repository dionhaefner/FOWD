import os
import json

import numpy as np
import tqdm

from .constants import QC_EXTREME_WAVE_LOG_THRESHOLD


def plot_qc(qcfile, outdir, plot_flags=tuple('abdg'), plot_extreme=True):
    import matplotlib as mpl
    mpl.use('Agg')
    import matplotlib.pyplot as plt

    if not set(plot_flags) <= set('abcdefg'):
        raise ValueError('plot_flags can only contain {a, b, c, d, e, f, g}')

    with open(qcfile, 'r') as f:
        qc_records = [json.loads(line) for line in f]

    os.makedirs(outdir, exist_ok=True)

    i = 1

    for record in tqdm.tqdm(qc_records):
        process_record = any(flag in plot_flags for flag in record['flags_fired'])

        if plot_extreme:
            process_record |= record['relative_wave_height'] > QC_EXTREME_WAVE_LOG_THRESHOLD

        if not process_record:
            continue

        mintime, maxtime = np.min(record['time']), np.max(record['time'])

        # don't plot records with extreme time jumps
        if maxtime - mintime > 10000 and 'e' in record['flags_fired']:
            continue

        elev_range = np.nanmax(np.abs(record['elevation']))

        info = [
            f'QC flags fired: {record["flags_fired"]}' if record['flags_fired'] else 'QC passed',
            f'Wave height: {record["relative_wave_height"]:.2f} SWH',
            f'Record start time: {record["start_date"]}',
            f'Source file: {record["filename"]}',
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
        ax.text(0.01, 1, '\n'.join(info), va='top', ha='left', transform=ax.transAxes)
        fig.tight_layout()
        fig.savefig(os.path.join(outdir, f'fowd_qc_{i:0>4}.pdf'), bbox_inches='tight')
        plt.close(fig)

        i += 1
