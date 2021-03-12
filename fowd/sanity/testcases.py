"""
sanity/testcases.py

Definition of sanity check test cases.
"""

import numpy as np
from scipy.special import gamma

SEED = 17
WATER_DEPTH = 500


def ochi_test_case(t, spectral_params, random_phases=True):
    random_state = np.random.RandomState(SEED)
    f = np.linspace(1e-5, 3 * np.pi, 100_000)
    spectrum = ochi_hubble_spectrum(f, **spectral_params)
    elevation = timeseries_from_spectrum(
        t, f, spectrum, random_phases=random_phases, random_state=random_state
    )

    # throw in 1% NaNs for good measure
    num_times = len(times)
    random_idx = np.random.choice(np.arange(num_times), size=num_times // 100, replace=False)
    elevation[random_idx] = np.nan

    return dict(
        time=(1e9 * t).astype('timedelta64[ns]'),
        elevation=elevation,
        frequencies=f,
        wave_spectral_density=spectrum,
        spectral_params=spectral_params,
        depth=WATER_DEPTH
    )


def single_group_test_case(t):
    f = np.arange(0.01, 1, 0.001)
    spectrum = 0.5 / (f[1] - f[0]) * (np.isclose(f, 0.1) | np.isclose(f, 0.11))
    elevation = np.sin(2 * np.pi * 0.1 * t) + np.sin(2 * np.pi * 0.11 * t)

    return dict(
        time=(1e9 * t).astype('timedelta64[ns]'),
        elevation=elevation,
        frequencies=f,
        wave_spectral_density=spectrum,
        spectral_params={},
        depth=WATER_DEPTH
    )


def directional_test_case(spectral_params):
    f = np.linspace(1e-3, 1, 1000)
    spectrum = ochi_hubble_spectrum(f, **spectral_params)
    directional_spread = np.linspace(0, 180, len(f))
    mean_direction = np.linspace(360, 0, len(f))

    return dict(
        time=None,
        frequencies=f,
        directional_spread=directional_spread,
        mean_direction=mean_direction,
        wave_spectral_density=spectrum,
        peak_wave_direction=None,
        spectral_params=spectral_params
    )


def timeseries_from_spectrum(t, f, spectrum, random_phases=True, random_state=np.random):
    out = 0.
    for fi, si in zip(f, spectrum):
        if random_phases:
            phase = random_state.uniform(0, 2 * np.pi)
        else:
            phase = np.pi
        out += np.sqrt(2 * si * (f[1] - f[0])) * np.cos(2 * np.pi * fi * t + phase)

    out -= np.mean(out)
    return out


def ochi_hubble_spectrum(f, swh_wind, peak_period_wind, shape_wind,
                         swh_swell, peak_period_swell, shape_swell):
    def ochi_component(f, swh, peak_period, shape):
        return (
            swh ** 2 * peak_period * (shape + 0.25) ** shape
            / (4 * gamma(shape) * (peak_period * f) ** (4 * shape + 1))
            * np.exp(-(shape + 0.25) / (peak_period * f) ** 4)
        )

    psd = (
        ochi_component(f, swh_wind, peak_period_wind, shape_wind)
        + ochi_component(f, swh_swell, peak_period_swell, shape_swell)
    )

    return psd


# 30m time series at 1.28Hz sampling resolution
times = np.arange(0, 1800, 1. / 1.28)

ochi_params = {
    'swell_dominated': dict(
        swh_wind=1,
        peak_period_wind=5,
        shape_wind=1,
        swh_swell=2,
        peak_period_swell=16,
        shape_swell=1
    ),
    'narrow_swell': dict(
        swh_wind=0,
        peak_period_wind=5,
        shape_wind=.33,
        swh_swell=2,
        peak_period_swell=16,
        shape_swell=3
    ),
    'wide_swell': dict(
        swh_wind=0,
        peak_period_wind=5,
        shape_wind=.33,
        swh_swell=2,
        peak_period_swell=16,
        shape_swell=0.33
    ),
    'wind_dominated': dict(
        swh_wind=2,
        peak_period_wind=5,
        shape_wind=1,
        swh_swell=1,
        peak_period_swell=16,
        shape_swell=1
    ),
}

TEST_CASES = {
    typ: ochi_test_case(
        times,
        ochi_params[typ],
    )
    for typ in ochi_params
}

TEST_CASES['single_group'] = single_group_test_case(times)
TEST_CASES['phase_alignment'] = ochi_test_case(
    times, ochi_params['swell_dominated'], random_phases=False
)


DIRECTIONAL_TEST_CASES = {
    'swell_dominated_directional': directional_test_case(
        ochi_params['swell_dominated'],
    ),
    'wind_dominated_directional': directional_test_case(
        ochi_params['wind_dominated'],
    )
}
