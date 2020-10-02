import os
import json

import numpy as np

import matplotlib as mpl
mpl.use('Agg')

import matplotlib.pyplot as plt  # noqa: E402


def run_sea_state(outdir):
    from fowd.sanity.testcases import TEST_CASES
    from fowd.operators import get_sea_parameters

    for description, case in TEST_CASES.items():
        sea_state = get_sea_parameters(
            time=case['time'],
            z_displacement=case['elevation'],
            wave_heights=np.array([]),
            wave_periods=np.array([]),
            water_depth=case['depth']
        )

        for v in ('start_time', 'end_time'):
            sea_state.pop(v)

        results = {}
        results['water_depth'] = case['depth']
        results['spectral_parameters'] = case['spectral_params']
        results['estimated_sea_state'] = sea_state

        with open(os.path.join(outdir, f'fowd_test_{description}_output.json'), 'w') as f:
            f.write(json.dumps(results, sort_keys=True, indent=4))

        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(4, 8))

        ax1.plot(case['frequencies'], case['wave_spectral_density'])
        ax1.set_ylabel('Wave spectral density (m$^2$/Hz)')
        ax1.set_xlabel('Frequency (Hz)')
        ax1.set_xlim(0, 1)
        ax1.spines['top'].set_visible(False)
        ax1.spines['right'].set_visible(False)

        ax2.plot(case['time'] / np.timedelta64(1, 's'), case['elevation'])
        ax2.set_xlabel('Time (s)')
        ax2.set_ylabel('Elevation (m)')
        ax2.set_xlim(0, 180)
        ax2.set_ylim(-2, 2)
        ax2.spines['top'].set_visible(False)
        ax2.spines['right'].set_visible(False)

        finite_mask = np.isfinite(case['elevation'])
        ax3.hist(case['elevation'][finite_mask], density=True)
        ax3.set_xlabel('Elevation (m)')
        ax3.set_ylabel('Density')
        ax3.spines['top'].set_visible(False)
        ax3.spines['right'].set_visible(False)

        ax1.set_title(description.replace('_', ' ').title())
        fig.tight_layout()

        fig.savefig(os.path.join(outdir, f'fowd_test_{description}_input.png'))
        plt.close(fig)


def run_directional(outdir):
    from fowd.sanity.testcases import DIRECTIONAL_TEST_CASES
    from fowd.operators import get_directional_parameters
    from fowd.constants import FREQUENCY_INTERVALS

    for description, case in DIRECTIONAL_TEST_CASES.items():
        directional_params = get_directional_parameters(
            time=case['time'],
            frequencies=case['frequencies'],
            directional_spread=case['directional_spread'],
            mean_direction=case['mean_direction'],
            spectral_energy_density=case['spectral_energy_density'],
            peak_wave_direction=case['peak_wave_direction']
        )

        for v in ('sampling_time',):
            directional_params.pop(v)

        results = {}
        results['spectral_parameters'] = case['spectral_params']
        results['estimated_directional_parameters'] = directional_params

        with open(os.path.join(outdir, f'fowd_test_{description}_output.json'), 'w') as f:
            f.write(json.dumps(results, sort_keys=True, indent=4))

        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(4, 8))

        ax1.plot(case['frequencies'], case['spectral_energy_density'])
        ax1.set_ylabel('Wave spectral density (m$^2$/Hz)')
        ax1.set_xlabel('Frequency (Hz)')
        ax1.set_xlim(0, 1)
        ax1.spines['top'].set_visible(False)
        ax1.spines['right'].set_visible(False)

        for i, (i_start, i_end) in enumerate(FREQUENCY_INTERVALS, 1):
            ax1.fill_between([i_start, i_end], [i, i], [i-1, i-1], alpha=0.4)
            ax1.text(0.5 * (i_start + i_end), i - 0.5, str(i), ha='center', va='center')

        ax2.plot(case['frequencies'], case['directional_spread'])
        ax2.set_ylabel('Directional spread (deg)')
        ax2.set_xlabel('Frequency (Hz)')
        ax2.set_xlim(0, 1)
        ax2.spines['top'].set_visible(False)
        ax2.spines['right'].set_visible(False)

        ax3.plot(case['frequencies'], case['mean_direction'])
        ax3.set_ylabel('Mean direction (deg)')
        ax3.set_xlabel('Frequency (Hz)')
        ax3.set_xlim(0, 1)
        ax3.spines['top'].set_visible(False)
        ax3.spines['right'].set_visible(False)

        ax1.set_title(description.replace('_', ' ').title())
        fig.tight_layout()

        fig.savefig(os.path.join(outdir, f'fowd_test_{description}_input.png'))
        plt.close(fig)


def run_all(outdir):
    run_sea_state(outdir)
    run_directional(outdir)
