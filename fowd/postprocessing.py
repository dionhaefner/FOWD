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

    for i, record in enumerate(tqdm.tqdm(qc_records), 1):
        process_record = any(flag in plot_flags for flag in record['flags_fired'])

        if plot_extreme:
            process_record |= record['relative_wave_height'] > QC_EXTREME_WAVE_LOG_THRESHOLD

        if not process_record:
            continue

        mintime, maxtime = np.min(record['time']), np.max(record['time'])
        elev_range = np.nanmax(np.abs(record['elevation']))

        fig = plt.figure(figsize=(15, 4))
        ax = plt.gca()
        # ax = plt.axes([0, 0, 1, 1])
        plt.plot(record['time'], record['elevation'], linewidth=0.5)
        plt.xlim(mintime, maxtime)
        plt.ylim(-elev_range, elev_range)
        ax.set_xticks(np.arange(mintime, maxtime, 10), minor=True)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.set_title(f'QC flags fired: {record["flags_fired"]}')
        fig.tight_layout()
        fig.savefig(os.path.join(outdir, f'fowd_qc_{i}.pdf'))
        plt.close(fig)
