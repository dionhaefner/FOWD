import math
import functools
import hashlib
import os

import numpy as np
import scipy.stats
import scipy.signal
import scipy.integrate

from .constants import (
    GRAVITY, DENSITY, FREQUENCY_INTERVALS, RAW_ELEVATION_SIZE,
    QC_FLAG_A_THRESHOLD, QC_FLAG_B_THRESHOLD, QC_FLAG_C_THRESHOLD,
    QC_FLAG_D_THRESHOLD, QC_FLAG_F_THRESHOLD, QC_FLAG_G_THRESHOLD
)


# helper functions

def memoize(func):
    cache = {}

    @functools.wraps(func)
    def memoized(*args):
        if args not in cache:
            cache[args] = func(*args)
        return cache[args]

    return memoized


def get_time_index(target_time, time_records, nearest=False):
    record_length = len(time_records)
    pos = np.searchsorted(time_records, target_time, side='right')

    if nearest and 0 < pos < record_length:
        diff = abs(time_records[pos] - target_time)
        otherdiff = abs(time_records[pos - 1] - target_time)
        if diff > otherdiff:
            pos -= 1

    return min(pos, record_length - 1)


def find_wave_indices(z):
    active = False
    wave_start = 0

    for i in range(len(z) - 1):
        if math.isnan(z[i]):
            # discard wave if it contains invalid values
            active = False

        if z[i] < 0 and z[i + 1] >= 0:
            if active:
                yield (wave_start, i + 1)

            wave_start = i
            active = True


def add_prefix(dic, prefix):
    return {f'{prefix}_{key}': value for key, value in dic.items()}


# metadata

@memoize
def get_md5_hash(filepath, blocksize=1024 * 1024):
    m5 = hashlib.md5()
    with open(filepath, 'rb') as f:
        chunk = f.read(blocksize)
        while chunk:
            m5.update(chunk)
            chunk = f.read(blocksize)
    return m5.hexdigest()


def create_wave_id(start_time, end_time, filename, proc_version):
    input_hash = get_md5_hash(filename)

    m5 = hashlib.md5()
    for v in (start_time, end_time, input_hash, proc_version):
        if isinstance(v, str):
            v = v.encode('utf-8')
        m5.update(bytes(v))
    return m5.hexdigest()


def get_proc_version():
    from . import __version__
    return __version__


# wave-based operators

def compute_period(t, z):
    assert z[0] < 0 and z[1] >= 0, z
    assert z[-2] < 0 and z[-1] >= 0, z

    # interpolate linearly to find zero-crossings
    t_start = t[0] - z[0] * (t[1] - t[0]) / (z[1] - z[0])
    t_end = t[-2] - z[-2] * (t[-1] - t[-2]) / (z[-1] - z[-2])

    period = np.float64(t_end - t_start) / 1e9
    return period


def compute_zero_crossing_wavelength(period, water_depth, gravity=GRAVITY):
    return 1. / frequency_to_wavenumber(1. / period, water_depth, gravity)


def compute_maximum_slope(t, elevation):
    t_seconds = 1e-9 * t.astype(np.float64)
    return np.nanmax(np.abs(np.gradient(elevation, t_seconds)))


def compute_crest_height(elevation):
    return np.nanmax(elevation)


def compute_trough_depth(elevation):
    return np.nanmin(elevation)


def compute_wave_height(crest_height, trough_depth):
    assert crest_height > trough_depth
    return crest_height - trough_depth


# aggregates

def integrate(y, x):
    return np.trapz(y, x)


def compute_ssh(displacement):
    return np.nanmean(displacement)


def compute_elevation(displacement, ssh):
    return displacement - ssh


def compute_significant_wave_height(waveheights):
    waveheights = waveheights[np.isfinite(waveheights)]

    if len(waveheights) < 20:
        return np.nan

    largest_third = np.quantile(waveheights, 2./3.)
    return np.mean(waveheights[waveheights >= largest_third])


def compute_mean_wave_period(wave_periods):
    if len(wave_periods) < 5:
        return np.nan

    return np.mean(wave_periods)


def compute_skewness(elevation):
    return np.nansum(elevation ** 3) / np.nansum(elevation ** 2) ** (3 / 2)


def compute_excess_kurtosis(elevation):
    return np.nansum(elevation ** 4) / np.nansum(elevation ** 2) ** 2 - 3


def compute_valid_data_ratio(elevation):
    return np.count_nonzero(np.isfinite(elevation)) / elevation.size


def compute_spectral_density(elevation, sample_rate):
    elevation[np.isnan(elevation)] = 0.

    def noop(x): return x  # data is already detrended
    return scipy.signal.periodogram(elevation, 1 / sample_rate, detrend=noop)


def compute_spectral_density_smooth(elevation, sample_rate):
    elevation[np.isnan(elevation)] = 0.

    def noop(x): return x  # data is already detrended
    return scipy.signal.welch(elevation, 1 / sample_rate, nperseg=128, detrend=noop)


def get_interval_mask(domain, lower_limit=None, upper_limit=None):
    if lower_limit is None:
        lower_limit = -float('inf')
    if upper_limit is None:
        upper_limit = float('inf')

    return (domain >= lower_limit) & (domain <= upper_limit)


def compute_definite_integral(domain, quantity, lower_limit=None, upper_limit=None):
    mask = get_interval_mask(domain, lower_limit, upper_limit)
    return integrate(quantity[mask], domain[mask])


def compute_energy_spectrum(wave_spectral_density, gravity, density):
    return wave_spectral_density * gravity * density


def compute_peak_frequency(frequencies, wave_spectral_density):
    return (
        integrate(frequencies * wave_spectral_density ** 4, frequencies)
        / integrate(wave_spectral_density ** 4, frequencies)
    )


def frequency_to_wavenumber(frequency, water_depth, gravity):
    """Approximate inverse dispersion relation for linear waves"""
    alpha_k = (2 * np.pi * frequency) ** 2 * water_depth / gravity
    beta_k = alpha_k / np.sqrt(np.tanh(alpha_k))

    # side-step numerical issues in extreme cases
    if beta_k > 100 or beta_k < 0.01:
        return beta_k / water_depth

    return (
        (alpha_k + beta_k ** 2 * np.cosh(beta_k) ** (-2))
        / (water_depth * (np.tanh(beta_k) + beta_k * np.cosh(beta_k) ** (-2)))
    )


def compute_nth_moment(frequencies, wave_spectral_density, n):
    if n == 0:
        return integrate(wave_spectral_density, frequencies)

    return integrate(frequencies ** n * wave_spectral_density, frequencies)


def compute_mean_wave_period_spectral(zeroth_moment, frequencies, wave_spectral_density):
    second_moment = compute_nth_moment(frequencies, wave_spectral_density, 2)
    return np.sqrt(zeroth_moment / second_moment)


def compute_significant_wave_height_spectral(zeroth_moment):
    return 4 * np.sqrt(zeroth_moment)


def compute_steepness(zeroth_moment, peak_wavenumber):
    return np.sqrt(2 * zeroth_moment) * peak_wavenumber


def compute_bandwidth_narrowness(zeroth_moment, frequencies, wave_spectral_density):
    first_moment = compute_nth_moment(frequencies, wave_spectral_density, 1)
    second_moment = compute_nth_moment(frequencies, wave_spectral_density, 2)
    narrowness = np.sqrt(zeroth_moment * second_moment / first_moment ** 2 - 1)
    return narrowness


def compute_bandwidth_broadness(zeroth_moment, frequencies, wave_spectral_density):
    fourth_moment = compute_nth_moment(frequencies, wave_spectral_density, 4)
    second_moment = compute_nth_moment(frequencies, wave_spectral_density, 2)
    narrowness = np.sqrt(1 - second_moment ** 2 /
                         (zeroth_moment * fourth_moment))
    return narrowness


def compute_bandwidth_quality(zeroth_moment, frequencies, wave_spectral_density):
    q_p = 2 / zeroth_moment ** 2 * \
        integrate(frequencies * wave_spectral_density ** 2, frequencies)
    return 1. / (np.sqrt(np.pi) * q_p)


def compute_benjamin_feir_index(bandwidth, steepness, water_depth, peak_wavenumber):
    kd = peak_wavenumber * water_depth

    # side-step numerical issues
    if kd > 100:
        nu = alpha = beta = 1
    else:
        nu = 1 + 2 * kd / np.sinh(2 * kd)
        alpha = -nu ** 2 + 2 + 8 * kd ** 2 * \
            np.cosh(2 * kd) / np.sinh(2 * kd) ** 2
        beta = (
            (np.cosh(4 * kd) + 8 - 2 * np.tanh(kd) ** 2) / (8 * np.sinh(kd) ** 4)
            - (2 * np.cosh(kd) ** 2 + 0.5 * nu) ** 2 /
            (np.sinh(2 * kd) ** 2 * (kd / np.tanh(kd) - (nu / 2) ** 2))
        )

    return steepness / bandwidth * nu * np.sqrt(np.maximum(beta / alpha, 0))


# spectral information

def circular_convolution(coords, angle, operand):
    """Computes the weighted convolution of an angle with another function"""
    finite_mask = np.isfinite(angle) & np.isfinite(operand)
    coords = coords[finite_mask]
    angle = angle[finite_mask]
    operand = operand[finite_mask]

    if coords.size == 0:
        return np.nan

    x, y = np.sin(np.radians(angle)), np.cos(np.radians(angle))
    norm = integrate(operand, x=coords)
    xconv = integrate(x * operand, x=coords) / norm
    yconv = integrate(y * operand, x=coords) / norm
    res = np.arctan2(xconv, yconv)
    res_deg = np.degrees(res)
    if res_deg < 0:
        res_deg += 360
    return res_deg


def compute_dominant_spread(frequencies, spread, energy_density):
    return circular_convolution(frequencies, spread, energy_density)


def compute_dominant_direction(frequencies, direction, energy_density):
    return circular_convolution(frequencies, direction, energy_density)


# QC

def check_flag_a(zero_crossing_periods, threshold=QC_FLAG_A_THRESHOLD):
    """Check for excessively long waves"""
    return np.all(zero_crossing_periods > threshold)


def check_flag_b(time, elevation, zero_crossing_periods, threshold=QC_FLAG_B_THRESHOLD):
    """Check for unphysical gradients"""
    if not np.any(np.isfinite(elevation)) or not np.any(np.isfinite(zero_crossing_periods)):
        return True

    time_in_seconds = 1e-9 * time.astype(np.float64)
    limit_rate_of_change = (
        2 * np.pi * np.nanstd(elevation) / np.nanmean(zero_crossing_periods)
        * np.sqrt(2 * np.log(len(zero_crossing_periods)))
    )
    measured_rate_of_change = np.gradient(elevation, time_in_seconds)
    measured_rate_of_change[np.isnan(measured_rate_of_change)] = 0.
    return np.any(np.abs(measured_rate_of_change) > threshold * limit_rate_of_change)


def check_flag_c(elevation, threshold=QC_FLAG_C_THRESHOLD):
    """Check for locked-in measurement runs"""

    def rle(inarray):
        """Get all run lengths contained in a NumPy array"""
        n = len(inarray)
        y = inarray[1:] != inarray[:-1]
        i = np.append(np.where(y), n - 1)
        return np.diff(np.append(-1, i))

    return np.any(rle(elevation) > threshold)


def check_flag_d(elevation, wave_crests, wave_troughs, threshold=QC_FLAG_D_THRESHOLD):
    """Check for unrealistically extreme elevations"""
    elevation_stdev = np.nanstd(elevation)

    def exceeds(arr):
        return np.abs(arr) > threshold * elevation_stdev

    return np.any(exceeds(wave_crests)) or np.any(exceeds(wave_troughs))


def check_flag_e(times):
    """Check for records that are not equally spaced in time"""
    sampling_rates = np.around(np.diff(times.astype(np.float64)), 6)
    return len(np.unique(sampling_rates)) == 1


def check_flag_f(elevation, threshold=QC_FLAG_F_THRESHOLD):
    """Check for a high ratio of missing values"""
    return np.count_nonzero(~np.isfinite(elevation)) / elevation.size > threshold


def check_flag_g(zero_crossing_periods, threshold=QC_FLAG_G_THRESHOLD):
    return len(zero_crossing_periods) < 100


def check_quality_flags(time, elevation, zero_crossing_periods, wave_crests, wave_troughs):
    triggered_flags = []

    if check_flag_a(zero_crossing_periods):
        triggered_flags.append('a')

    if check_flag_b(time, elevation, zero_crossing_periods):
        triggered_flags.append('b')

    if check_flag_c(elevation):
        triggered_flags.append('c')

    if check_flag_d(elevation, wave_crests, wave_troughs):
        triggered_flags.append('d')

    if check_flag_e(time):
        triggered_flags.append('e')

    if check_flag_f(elevation):
        triggered_flags.append('f')

    if check_flag_g(zero_crossing_periods):
        triggered_flags.append('g')

    return triggered_flags


# top-level functions

def get_station_meta(filepath, uuid, lat, lon, depth, rate):
    filename = os.path.basename(filepath)

    return {
        'source_file_name': filename,
        'source_file_uuid': uuid,
        'deploy_latitude': float(lat),
        'deploy_longitude': float(lon),
        'water_depth': float(depth),
        'sampling_rate': float(rate),
    }


def get_wave_parameters(local_id, t, z, water_depth, filepath):
    if not len(t) == len(z):
        raise ValueError('t and z must have equal lengths')

    proc_version = get_proc_version()
    global_id = create_wave_id(t[0], t[-1], filepath, proc_version)

    wave_period = compute_period(t, z)
    wavelength = compute_zero_crossing_wavelength(wave_period, water_depth)
    maximum_slope = compute_maximum_slope(t, z)
    crest_height = compute_crest_height(z)
    trough_depth = compute_trough_depth(z)
    wave_height = compute_wave_height(crest_height, trough_depth)

    raw_elevation = np.full(RAW_ELEVATION_SIZE, np.nan)
    raw_elevation[:len(z) - 2] = z[1:-1]

    return {
        'start_time': t[0],
        'end_time': t[-1],
        'id_local': local_id,
        'id_global': global_id,
        'zero_crossing_period': wave_period,
        'zero_crossing_wavelength': wavelength,
        'maximum_elevation_slope': maximum_slope,
        'crest_height': crest_height,
        'trough_depth': trough_depth,
        'height': wave_height,
        'raw_elevation': raw_elevation,
    }


def get_sea_parameters(time, z_displacement, wave_heights, wave_periods, water_depth,
                       gravity=GRAVITY, density=DENSITY):
    sample_rate = np.around(1e-9 * np.diff(time).astype('float64'), 6)
    assert len(np.unique(sample_rate)) == 1
    sample_rate = sample_rate[0]

    ssh = compute_ssh(z_displacement)
    elevation = compute_elevation(z_displacement, ssh)
    significant_wave_height_direct = compute_significant_wave_height(wave_heights)
    mean_period_direct = compute_mean_wave_period(wave_periods)
    skewness = compute_skewness(elevation)
    excess_kurtosis = compute_excess_kurtosis(elevation)
    valid_data_ratio = compute_valid_data_ratio(z_displacement)

    frequencies, wave_spectral_density = compute_spectral_density_smooth(elevation, sample_rate)
    zeroth_moment = compute_nth_moment(frequencies, wave_spectral_density, 0)
    significant_wave_height_spectral = compute_significant_wave_height_spectral(zeroth_moment)

    mean_period_spectral = compute_mean_wave_period_spectral(
        zeroth_moment,
        frequencies,
        wave_spectral_density
    )

    peak_frequency = compute_peak_frequency(frequencies, wave_spectral_density)
    peak_wavenumber = frequency_to_wavenumber(peak_frequency, water_depth, gravity)
    peak_wavelength = 1. / peak_wavenumber
    peak_period = 1. / peak_frequency

    steepness = compute_steepness(zeroth_moment, peak_wavenumber)

    bandwidth_quality = compute_bandwidth_quality(zeroth_moment, frequencies, wave_spectral_density)
    bandwidth_narrowness = compute_bandwidth_narrowness(
        zeroth_moment, frequencies, wave_spectral_density
    )

    bfi = compute_benjamin_feir_index(bandwidth_quality, steepness, water_depth, peak_wavenumber)

    spectral_energy_density = compute_energy_spectrum(wave_spectral_density, gravity, density)

    energy_in_frequency_interval = [
        compute_definite_integral(frequencies, spectral_energy_density, *frequency_interval)
        for frequency_interval in FREQUENCY_INTERVALS
    ]

    return {
        'start_time': time[0],
        'end_time': time[-1],

        'sea_surface_height': ssh,
        'significant_wave_height_direct': significant_wave_height_direct,
        'significant_wave_height_spectral': significant_wave_height_spectral,
        'mean_period_direct': mean_period_direct,
        'mean_period_spectral': mean_period_spectral,
        'skewness': skewness,
        'kurtosis': excess_kurtosis,
        'valid_data_ratio': valid_data_ratio,

        'peak_wave_period': peak_period,
        'peak_wavelength': peak_wavelength,
        'steepness': steepness,
        'bandwidth_quality_factor': bandwidth_quality,
        'bandwidth_narrowness': bandwidth_narrowness,
        'benjamin_feir_index': bfi,

        'energy_in_frequency_interval': energy_in_frequency_interval,
    }


def get_directional_parameters(time, frequencies, directional_spread, mean_direction,
                               spectral_energy_density, peak_wave_direction):
    dominant_directional_spreads = []
    dominant_directions = []

    for frequency_interval in FREQUENCY_INTERVALS:
        interval_mask = get_interval_mask(frequencies, *frequency_interval)

        dominant_directional_spreads.append(
            compute_dominant_spread(
                frequencies[interval_mask],
                directional_spread[interval_mask],
                spectral_energy_density[interval_mask]
            )
        )
        dominant_directions.append(
            compute_dominant_direction(
                frequencies[interval_mask],
                mean_direction[interval_mask],
                spectral_energy_density[interval_mask]
            )
        )

    assert len(dominant_directional_spreads) == len(FREQUENCY_INTERVALS)
    assert len(dominant_directions) == len(FREQUENCY_INTERVALS)

    return {
        'sampling_time': time,
        'dominant_spread_in_frequency_interval': dominant_directional_spreads,
        'dominant_direction_in_frequency_interval': dominant_directions,
        'peak_wave_direction': peak_wave_direction,
    }
