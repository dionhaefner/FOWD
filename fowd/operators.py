"""
operators.py

Operators for all physical quantities and metadata.
"""

import os
import math
import hashlib

import numpy as np
import scipy.stats
import scipy.signal

from .constants import (
    GRAVITY, DENSITY, FREQUENCY_INTERVALS,
    QC_FLAG_A_THRESHOLD, QC_FLAG_B_THRESHOLD, QC_FLAG_C_THRESHOLD,
    QC_FLAG_D_THRESHOLD, QC_FLAG_E_THRESHOLD, QC_FLAG_F_THRESHOLD,
    QC_FLAG_G_THRESHOLD,
    SPECTRUM_WINDOW_SIZE,
)


# helper functions

def get_time_index(target_time, time_records, nearest=False):
    """Find index of given target time in (sorted) records."""
    record_length = len(time_records)
    pos = np.searchsorted(time_records, target_time, side='right')

    if nearest and 0 < pos < record_length:
        diff = abs(time_records[pos] - target_time)
        otherdiff = abs(time_records[pos - 1] - target_time)
        if diff > otherdiff:
            pos -= 1

    return min(pos, record_length - 1)


def find_wave_indices(z, start_idx=0):
    """Generator that yields start and end index of every wave in surface elevation record z."""
    assert start_idx < len(z) - 1
    active = False
    wave_start = start_idx

    for i in range(wave_start, len(z) - 1):
        if math.isnan(z[i]):
            # discard wave if it contains invalid values
            active = False

        if z[i] < 0 and z[i + 1] >= 0:
            if active:
                yield (wave_start, i + 1)

            wave_start = i
            active = True


def compute_dynamic_window_size(t, elevation, min_length, max_length, num_windows, num_samples):
    """Compute best window size from data by minimizing fluctuations around mean energy drift.

    Reference:

        Fedele, Francesco, et al. “Large Nearshore Storm Waves off the Irish Coast.”
        Scientific Reports, vol. 9, no. 1, 1, Nature Publishing Group, Oct. 2019, p. 15406.
        www.nature.com, doi:10.1038/s41598-019-51706-8.

    """
    best_window_length = None
    best_window_stationarity = float('inf')

    if isinstance(min_length, np.timedelta64):
        min_length = int(min_length / np.timedelta64(1, 's'))

    if isinstance(max_length, np.timedelta64):
        max_length = int(max_length / np.timedelta64(1, 's'))

    window_lengths = np.linspace(min_length, max_length, num_windows, dtype='int')

    for window_length in window_lengths:
        window_stationarity = []
        window_offsets = np.linspace(0, window_length, num_samples, endpoint=False, dtype='int')

        for window_offset in window_offsets:
            if window_offset >= len(t):
                continue

            current_time_idx = window_offset
            window_stds = []
            while True:
                next_time = t[current_time_idx] + np.timedelta64(window_length, 's')
                if next_time > t[-1]:
                    break

                next_time_idx = get_time_index(next_time, t)
                window_stds.append(np.nanstd(elevation[current_time_idx:next_time_idx], ddof=1.5))
                current_time_idx = next_time_idx

            window_stds = np.asarray(window_stds)
            window_stationarity.append(np.nanstd(window_stds[1:] / window_stds[:-1] - 1, ddof=1))

        mean_window_stationarity = np.nanmean(window_stationarity)

        if mean_window_stationarity < best_window_stationarity:
            best_window_length = window_length
            best_window_stationarity = mean_window_stationarity

    if best_window_length is None:
        return min_length

    return best_window_length


def add_prefix(dic, prefix):
    """Adds a prefix to every key in given dictionary."""
    return {f'{prefix}_{key}': value for key, value in dic.items()}


# metadata

def get_md5_hash(filepath, blocksize=1024 * 1024):
    """Get MD5 hash of input file."""
    m5 = hashlib.md5()
    with open(filepath, 'rb') as f:
        chunk = f.read(blocksize)
        while chunk:
            m5.update(chunk)
            chunk = f.read(blocksize)
    return m5.hexdigest()


def get_proc_version():
    """Get current processing version."""
    from . import __version__
    return __version__


# wave-based operators

def compute_period(t, z):
    """Computes zero-crossing period from given surface elevation time series.

    Record must include a single wave. Finds zero-crossings through linear interpolation.
    """
    assert z[0] < 0 and z[1] >= 0, z
    assert z[-2] < 0 and z[-1] >= 0, z

    # interpolate linearly to find zero-crossings
    t_start = t[0] - z[0] * (t[1] - t[0]) / (z[1] - z[0])
    t_end = t[-2] - z[-2] * (t[-1] - t[-2]) / (z[-1] - z[-2])

    period = (t_end - t_start) / np.timedelta64(1, 's')  # convert to seconds
    return period


def compute_zero_crossing_wavelength(period, water_depth, gravity=GRAVITY):
    """Computes zero-crossing wavelength from given period.

    This uses the dispersion relation for linear waves.
    """
    return wavenumber_to_wavelength(
        frequency_to_wavenumber(1. / period, water_depth, gravity)
    )


def compute_maximum_slope(t, elevation):
    """Computes maximum slope (dy/dt) of given surface elevation time series."""
    t_seconds = (t - t[0]) / np.timedelta64(1, 's')
    return np.nanmax(np.abs(np.gradient(elevation, t_seconds)))


def compute_crest_height(elevation):
    """Computes crest height from given surface elevation time series.

    May contain only a single wave.
    """
    return np.nanmax(elevation)


def compute_trough_depth(elevation):
    """Computes trough depth from given surface elevation time series.

    May contain only a single wave.
    """
    return np.nanmin(elevation)


def compute_wave_height(crest_height, trough_depth):
    """Computes wave height from given crest height and trough depth."""
    assert crest_height > trough_depth
    return crest_height - trough_depth


def compute_ursell_number(wave_height, wavelength, water_depth):
    """Computes Ursell number of a single wave."""
    return wave_height * wavelength ** 2 / water_depth ** 3


# aggregates

def integrate(y, x):
    """Computes numerical integral ∫ y(x) dx."""
    mask = np.isfinite(y) & np.isfinite(x)
    return np.trapz(y[mask], x[mask])


def compute_elevation(displacement):
    """Computes surface elevation from raw displacement."""
    return displacement - np.nanmean(displacement)


def compute_significant_wave_height(waveheights):
    """Compute significant wave height H_1/3 from given list of wave heights."""
    waveheights = waveheights[np.isfinite(waveheights)]

    if len(waveheights) < 20:
        return np.nan

    largest_third = np.quantile(waveheights, 2./3.)
    return np.mean(waveheights[waveheights >= largest_third])


def compute_maximum_wave_height(waveheights):
    """Compute maximum wave hehight from given list of wave heights."""
    waveheights = waveheights[np.isfinite(waveheights)]

    if len(waveheights) < 20:
        return np.nan

    return np.max(waveheights)


def compute_mean_wave_period(wave_periods):
    """Compute mean wave period from given list of wave periods."""
    if len(wave_periods) < 5:
        return np.nan

    return np.mean(wave_periods)


def compute_skewness(elevation):
    """Compute surface elevation skewness."""
    return np.nanmean(elevation ** 3) / np.nanmean(elevation ** 2) ** (3 / 2)


def compute_excess_kurtosis(elevation):
    """Compute surface elevation excess kurtosis."""
    return np.nanmean(elevation ** 4) / np.nanmean(elevation ** 2) ** 2 - 3


def compute_valid_data_ratio(elevation):
    """Compute ratio of valid to invalid data."""
    return np.count_nonzero(np.isfinite(elevation)) / elevation.size


def compute_spectral_density(elevation, sample_dt):
    """Compute wave spectral density from surface elevation time series.

    This uses Welch's method with a Hann window and zero-padded FFTs.

    Returns (frequencies, spectral_density).
    """
    elevation[np.isnan(elevation)] = 0.
    sample_dt = float(sample_dt)
    nperseg = round(SPECTRUM_WINDOW_SIZE / sample_dt)
    nfft = 2 ** (math.ceil(math.log(nperseg, 2)))  # round to next higher power of 2
    return scipy.signal.welch(
        elevation, 1 / sample_dt, window='hann',
        nperseg=nperseg, nfft=nfft, noverlap=nperseg // 2,
    )


def get_interval_mask(domain, lower_limit=None, upper_limit=None):
    """Given a domain and limits, returns a boolean mask with True values within the limits."""
    if lower_limit is None:
        lower_limit = -float('inf')
    if upper_limit is None:
        upper_limit = float('inf')

    return (domain >= lower_limit) & (domain <= upper_limit)


def compute_definite_integral(domain, quantity, lower_limit=None, upper_limit=None):
    """Compute a definite integral within given upper and lower limits."""
    mask = get_interval_mask(domain, lower_limit, upper_limit)
    return integrate(quantity[mask], domain[mask])


def compute_energy_spectrum(wave_spectral_density, gravity, density):
    """Compute energy density from wave spectral density."""
    return wave_spectral_density * gravity * density


def compute_relative_energy(frequencies, wave_spectral_density, zeroth_moment,
                            lower_limit, upper_limit):
    """Compute relative energy within lower and upper limits from wave spectral density."""
    return (
        compute_definite_integral(frequencies, wave_spectral_density, lower_limit, upper_limit)
        / zeroth_moment
    )


def compute_peak_frequency(frequencies, wave_spectral_density):
    """Compute peak frequency from wave spectral density.

    Reference:

        Young, I. R. “The Determination of Confidence Limits Associated with Estimates
        of the Spectral Peak Frequency.” Ocean Engineering, vol. 22, no. 7, Oct. 1995,
        pp. 669–86. ScienceDirect, doi:10.1016/0029-8018(95)00002-3.

    """
    return (
        integrate(frequencies * wave_spectral_density ** 4, frequencies)
        / integrate(wave_spectral_density ** 4, frequencies)
    )


def frequency_to_wavenumber(frequency, water_depth, gravity):
    """Approximate inverse dispersion relation for linear waves.

    Reference:

        Holthuijsen, Leo H. Waves in Oceanic and Coastal Waters.
        Cambridge University Press, 2010.

    """
    alpha_k = (2 * np.pi * frequency) ** 2 * water_depth / gravity
    beta_k = alpha_k / np.sqrt(np.tanh(alpha_k))

    # side-step numerical issues in extreme cases
    if beta_k > 100 or beta_k < 0.01:
        return beta_k / water_depth

    return (
        (alpha_k + beta_k ** 2 * np.cosh(beta_k) ** (-2))
        / (water_depth * (np.tanh(beta_k) + beta_k * np.cosh(beta_k) ** (-2)))
    )


def wavenumber_to_wavelength(wavenumber):
    """Convert wave number to wavelength."""
    return 2 * np.pi / wavenumber


def compute_nth_moment(frequencies, wave_spectral_density, n):
    """Compute nth spectral moment of given wave spectral density."""
    if n == 0:
        return integrate(wave_spectral_density, frequencies)

    return integrate(frequencies ** n * wave_spectral_density, frequencies)


def compute_mean_wave_period_spectral(zeroth_moment, frequencies, wave_spectral_density):
    """Compute mean zero-crossing period from wave spectral density."""
    second_moment = compute_nth_moment(frequencies, wave_spectral_density, 2)
    return np.sqrt(zeroth_moment / second_moment)


def compute_significant_wave_height_spectral(zeroth_moment):
    """Compute significant wave height from wave spectral density."""
    return 4 * np.sqrt(zeroth_moment)


def compute_steepness(zeroth_moment, peak_wavenumber):
    """Compute characteristic steepness from given peak wave number."""
    return np.sqrt(2 * zeroth_moment) * peak_wavenumber


def compute_bandwidth_narrowness(zeroth_moment, first_moment, frequencies, wave_spectral_density):
    """Compute spectral bandwidth (narrowness)."""
    second_moment = compute_nth_moment(frequencies, wave_spectral_density, 2)
    narrowness = np.sqrt(zeroth_moment * second_moment / first_moment ** 2 - 1)
    return narrowness


def compute_bandwidth_peakedness(zeroth_moment, frequencies, wave_spectral_density):
    """Compute spectral bandwidth (peakedness).

    Reference:

        Serio, Marina, et al. “On the Computation of the Benjamin-Feir Index.”
        Nuovo Cimento Della Societa Italiana Di Fisica C, vol. 28, Nov. 2005, pp. 893–903.
        ResearchGate, doi:10.1393/ncc/i2005-10134-1.

    """
    q_p = 2 / zeroth_moment ** 2 * integrate(frequencies * wave_spectral_density ** 2, frequencies)
    return 1. / (np.sqrt(np.pi) * q_p)


def compute_benjamin_feir_index(bandwidth, steepness, water_depth, peak_wavenumber):
    """Compute Benjamin-Feir index (BFI) from bandwidth and steepness estimates.

    Reference:

        Serio, Marina, et al. “On the Computation of the Benjamin-Feir Index.”
        Nuovo Cimento Della Societa Italiana Di Fisica C, vol. 28, Nov. 2005, pp. 893–903.
        ResearchGate, doi:10.1393/ncc/i2005-10134-1.

    """
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


def compute_crest_trough_correlation_spectral(zeroth_moment, first_moment, frequencies,
                                              wave_spectral_density):
    """Compute crest-trough correlation of the wave record from spectral density.

    Reference:

        Tayfun, M. Aziz, and Francesco Fedele. "Wave-Height Distributions and Nonlinear Effects."
        Ocean Engineering, vol. 34, no. 11, Aug. 2007, pp. 1631–49. ScienceDirect,
        doi:10.1016/j.oceaneng.2006.11.006.


    """
    # convert f -> omega and S_f -> S_omega
    angular_frequencies = 2 * np.pi * frequencies
    angular_density = wave_spectral_density / (2 * np.pi)

    # first moment in frequency domain -> no factor pi/2
    t_bar = zeroth_moment / first_moment

    arg = angular_frequencies * t_bar / 2.

    c_rho = integrate(angular_density * np.cos(arg), angular_frequencies)
    c_lambda = integrate(angular_density * np.sin(arg), angular_frequencies)

    return 1. / zeroth_moment * np.sqrt(c_rho ** 2 + c_lambda ** 2)


# directional information

def circular_weighted_average(coords, angle, weights=None):
    """Compute the average of an angle, optionally weighted with another function."""
    if weights is None:
        weights = np.ones_like(angle)

    finite_mask = np.isfinite(angle) & np.isfinite(weights)
    coords = coords[finite_mask]
    angle = angle[finite_mask]
    weights = weights[finite_mask]

    if coords.size == 0:
        return np.nan

    x, y = np.sin(np.radians(angle)), np.cos(np.radians(angle))
    xconv = integrate(x * weights, x=coords)
    yconv = integrate(y * weights, x=coords)
    res = np.arctan2(xconv, yconv)
    res_deg = np.degrees(res) % 360
    return res_deg


def compute_dominant_spread(frequencies, spread, energy_density):
    """Compute dominant directional spread."""
    return circular_weighted_average(frequencies, spread, weights=energy_density)


def compute_dominant_direction(frequencies, direction, energy_density):
    """Compute dominant wave direction."""
    return circular_weighted_average(frequencies, direction, weights=energy_density)


def compute_directionality_index(frequencies, spread, spectral_bandwidth, energy_density):
    """Compute directionality index R.

    Reference:

        Fedele, Francesco, et al. “Large Nearshore Storm Waves off the Irish Coast.”
        Scientific Reports, vol. 9, no. 1, 1, Nature Publishing Group, Oct. 2019, p. 15406.
        www.nature.com, doi:10.1038/s41598-019-51706-8.

    """
    total_squared_spread = (
        integrate(np.radians(spread) ** 2 * energy_density, frequencies)
        / integrate(energy_density, frequencies)
    )
    return total_squared_spread / (2 * spectral_bandwidth ** 2)


# QC

def check_flag_a(zero_crossing_periods, threshold=QC_FLAG_A_THRESHOLD):
    """Check for excessively long waves."""
    return np.any(zero_crossing_periods > threshold)


def check_flag_b(time, elevation, zero_crossing_periods, threshold=QC_FLAG_B_THRESHOLD):
    """Check for extreme gradients."""
    if not np.any(np.isfinite(elevation)) or not np.any(np.isfinite(zero_crossing_periods)):
        return True

    time_in_seconds = (time - time[0]) / np.timedelta64(1, 's')
    limit_rate_of_change = (
        2 * np.pi * np.nanstd(elevation) / np.nanmean(zero_crossing_periods)
        * np.sqrt(2 * np.log(len(zero_crossing_periods)))
    )
    measured_rate_of_change = np.gradient(elevation, time_in_seconds)
    measured_rate_of_change[np.isnan(measured_rate_of_change)] = 0.
    return np.any(np.abs(measured_rate_of_change) > threshold * limit_rate_of_change)


def check_flag_c(elevation, threshold=QC_FLAG_C_THRESHOLD):
    """Check for locked-in measurement runs."""

    def rle(inarray):
        """Get all run lengths contained in a NumPy array."""
        n = len(inarray)
        y = inarray[1:] != inarray[:-1]
        i = np.append(np.where(y), n - 1)
        return np.diff(np.append(-1, i))

    return np.any(rle(elevation) > threshold)


def check_flag_d(elevation, wave_crests, wave_troughs, threshold=QC_FLAG_D_THRESHOLD):
    """Check for unrealistically extreme elevations."""
    elevation_median = np.nanmedian(elevation)
    elevation_mad = np.nanmedian(np.abs(elevation - elevation_median))

    def is_outlier(val):
        # scale MAD so it converges to standard deviation
        # see Huber, Peter J. Robust statistics. Springer Berlin Heidelberg, 2011
        magic_number = 1.483
        return np.abs(val) > threshold * elevation_mad * magic_number

    return np.any(is_outlier(wave_crests)) or np.any(is_outlier(wave_troughs))


def check_flag_e(time, threshold=QC_FLAG_E_THRESHOLD):
    """Check for records that are not equally spaced in time."""
    sampling_dt = np.around(np.diff(time) / np.timedelta64(1, 's'), QC_FLAG_E_THRESHOLD)
    return len(np.unique(sampling_dt)) > 1


def check_flag_f(elevation, threshold=QC_FLAG_F_THRESHOLD):
    """Check for a high ratio of missing values."""
    return np.count_nonzero(~np.isfinite(elevation)) / elevation.size > threshold


def check_flag_g(zero_crossing_periods, threshold=QC_FLAG_G_THRESHOLD):
    """Check for low number of waves."""
    return len(zero_crossing_periods) < threshold


def check_quality_flags(time, elevation, elevation_raw, zero_crossing_periods, wave_crests,
                        wave_troughs):
    """Check all quality flags.

    Returns a list containing all failed QC flags.
    """
    triggered_flags = []

    if check_flag_a(zero_crossing_periods):
        triggered_flags.append('a')

    if check_flag_b(time, elevation, zero_crossing_periods):
        triggered_flags.append('b')

    if check_flag_c(elevation_raw):
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

def get_station_meta(filepath, uuid, latitude, longitude, water_depth, sampling_rate):
    """Get all station metadata."""
    filename = os.path.basename(filepath)

    return {
        'source_file_name': filename,
        'source_file_uuid': uuid,
        'deploy_latitude': float(latitude),
        'deploy_longitude': float(longitude),
        'water_depth': float(water_depth),
        'sampling_rate': float(sampling_rate),
    }


def get_wave_parameters(local_id, t, z, water_depth, input_hash):
    """Get all wave parameters."""
    if not len(t) == len(z):
        raise ValueError('t and z must have equal lengths')

    wave_period = compute_period(t, z)
    wavelength = compute_zero_crossing_wavelength(wave_period, water_depth)
    maximum_slope = compute_maximum_slope(t, z)
    crest_height = compute_crest_height(z)
    trough_depth = compute_trough_depth(z)
    wave_height = compute_wave_height(crest_height, trough_depth)
    ursell_number = compute_ursell_number(wave_height, wavelength, water_depth)

    raw_elevation = z[1:-1].astype('float32')

    return {
        'start_time': t[0],
        'end_time': t[-1],
        'id_local': local_id,
        'zero_crossing_period': wave_period,
        'zero_crossing_wavelength': wavelength,
        'maximum_elevation_slope': maximum_slope,
        'crest_height': crest_height,
        'trough_depth': trough_depth,
        'height': wave_height,
        'ursell_number': ursell_number,
        'raw_elevation': raw_elevation,
    }


def get_sea_parameters(time, z_displacement, wave_heights, wave_periods, water_depth,
                       gravity=GRAVITY, density=DENSITY):
    """Get all sea state parameters."""
    sample_dt = np.around(np.diff(time) / np.timedelta64(1, 's'), 6)
    sample_dt = sample_dt[0]

    elevation = compute_elevation(z_displacement)
    significant_wave_height_direct = compute_significant_wave_height(wave_heights)
    maximum_wave_height = compute_maximum_wave_height(wave_heights)
    mean_period_direct = compute_mean_wave_period(wave_periods)
    skewness = compute_skewness(elevation)
    excess_kurtosis = compute_excess_kurtosis(elevation)
    valid_data_ratio = compute_valid_data_ratio(z_displacement)

    frequencies, wave_spectral_density = compute_spectral_density(elevation, sample_dt)
    zeroth_moment = compute_nth_moment(frequencies, wave_spectral_density, 0)
    first_moment = compute_nth_moment(frequencies, wave_spectral_density, 1)
    significant_wave_height_spectral = compute_significant_wave_height_spectral(zeroth_moment)

    rel_maximum_wave_height = maximum_wave_height / significant_wave_height_spectral

    mean_period_spectral = compute_mean_wave_period_spectral(
        zeroth_moment,
        frequencies,
        wave_spectral_density
    )

    peak_frequency = compute_peak_frequency(frequencies, wave_spectral_density)
    peak_wavenumber = frequency_to_wavenumber(peak_frequency, water_depth, gravity)
    peak_wavelength = wavenumber_to_wavelength(peak_wavenumber)
    peak_period = 1. / peak_frequency

    steepness = compute_steepness(zeroth_moment, peak_wavenumber)

    bandwidth_peakedness = compute_bandwidth_peakedness(
        zeroth_moment, frequencies, wave_spectral_density
    )
    bandwidth_narrowness = compute_bandwidth_narrowness(
        zeroth_moment, first_moment, frequencies, wave_spectral_density
    )

    bfi_peakedness = compute_benjamin_feir_index(
        bandwidth_peakedness, steepness, water_depth, peak_wavenumber
    )
    bfi_narrowness = compute_benjamin_feir_index(
        bandwidth_narrowness, steepness, water_depth, peak_wavenumber
    )

    crest_trough_correlation = compute_crest_trough_correlation_spectral(
        zeroth_moment, first_moment, frequencies, wave_spectral_density
    )

    spectral_energy_density = compute_energy_spectrum(wave_spectral_density, gravity, density)

    energy_in_frequency_interval = [
        compute_definite_integral(
            frequencies, spectral_energy_density, *frequency_interval
        ) for frequency_interval in FREQUENCY_INTERVALS
    ]

    rel_energy_in_frequency_interval = [
        compute_relative_energy(
            frequencies, wave_spectral_density, zeroth_moment, *frequency_interval
        ) for frequency_interval in FREQUENCY_INTERVALS
    ]

    return {
        'start_time': time[0],
        'end_time': time[-1],

        'significant_wave_height_direct': significant_wave_height_direct,
        'significant_wave_height_spectral': significant_wave_height_spectral,
        'mean_period_direct': mean_period_direct,
        'mean_period_spectral': mean_period_spectral,
        'maximum_wave_height': maximum_wave_height,
        'rel_maximum_wave_height': rel_maximum_wave_height,
        'skewness': skewness,
        'kurtosis': excess_kurtosis,
        'valid_data_ratio': valid_data_ratio,

        'peak_wave_period': peak_period,
        'peak_wavelength': peak_wavelength,
        'steepness': steepness,
        'bandwidth_peakedness': bandwidth_peakedness,
        'bandwidth_narrowness': bandwidth_narrowness,
        'benjamin_feir_index_peakedness': bfi_peakedness,
        'benjamin_feir_index_narrowness': bfi_narrowness,

        'crest_trough_correlation': crest_trough_correlation,

        'energy_in_frequency_interval': energy_in_frequency_interval,
        'rel_energy_in_frequency_interval': rel_energy_in_frequency_interval,
    }


def get_directional_parameters(time, frequencies, directional_spread, mean_direction,
                               wave_spectral_density, peak_wave_direction):
    """Get all directional parameters."""
    dominant_directional_spreads = []
    dominant_directions = []

    for frequency_interval in FREQUENCY_INTERVALS:
        interval_mask = get_interval_mask(frequencies, *frequency_interval)

        dominant_directional_spreads.append(
            compute_dominant_spread(
                frequencies[interval_mask],
                directional_spread[interval_mask],
                wave_spectral_density[interval_mask]
            )
        )
        dominant_directions.append(
            compute_dominant_direction(
                frequencies[interval_mask],
                mean_direction[interval_mask],
                wave_spectral_density[interval_mask]
            )
        )

    assert len(dominant_directional_spreads) == len(FREQUENCY_INTERVALS)
    assert len(dominant_directions) == len(FREQUENCY_INTERVALS)

    zeroth_moment = compute_nth_moment(frequencies, wave_spectral_density, 0)
    first_moment = compute_nth_moment(frequencies, wave_spectral_density, 1)
    spectral_bandwidth = compute_bandwidth_narrowness(
        zeroth_moment, first_moment, frequencies, wave_spectral_density
    )
    directionality_index = compute_directionality_index(
        frequencies, directional_spread, spectral_bandwidth, wave_spectral_density
    )

    return {
        'sampling_time': time,
        'dominant_spread_in_frequency_interval': dominant_directional_spreads,
        'dominant_direction_in_frequency_interval': dominant_directions,
        'peak_wave_direction': peak_wave_direction,
        'directionality_index': directionality_index,
    }
