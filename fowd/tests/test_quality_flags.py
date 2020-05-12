from fowd import operators

import numpy as np


def create_datetime(offset):
    return np.datetime64('now') + (1e9 * offset).astype('timedelta64[ns]')


def test_flag_a():
    res = operators.check_flag_a(
        np.array([0, 0, 0, 1 + 1e-4, 0, 0]),
        threshold=1
    )
    assert res

    res = operators.check_flag_a(
        np.array([0, 0, 0, 1 - 1e-4, 0, 0]),
        threshold=1
    )
    assert not res


def test_flag_b():
    time = np.arange(0, 180, 1)
    time_s = create_datetime(time)

    elevation = np.cos(2 * np.pi * 0.1 * time)
    elevation[10:20] = np.nan

    zero_crossing_periods = 10 * np.ones(18)
    zero_crossing_periods[5:10] = np.nan

    res = operators.check_flag_b(time_s, elevation, zero_crossing_periods, threshold=1)
    assert not res

    # make first wave 4 times higher
    elevation[:10] *= 4
    res = operators.check_flag_b(time_s, elevation, zero_crossing_periods, threshold=1)
    assert res


def test_flag_c():
    elevation = np.arange(100)
    res = operators.check_flag_c(elevation, threshold=10)
    assert not res

    elevation[5:15] = 1
    res = operators.check_flag_c(elevation, threshold=10)
    assert not res

    elevation[5:16] = 1
    res = operators.check_flag_c(elevation, threshold=10)
    assert res


def test_flag_d():
    time = np.arange(0, 180, 1)
    elevation = np.cos(2 * np.pi * 0.1 * time)
    elevation[10:20] = np.nan

    wave_crests = np.ones(18)
    wave_troughs = -np.ones(18)

    res = operators.check_flag_d(
        elevation, wave_crests, wave_troughs, threshold=8
    )
    assert not res

    # make first wave 4 times higher
    elevation[:10] *= 4
    wave_crests[0] = 4
    wave_troughs[0] = -4

    res = operators.check_flag_d(
        elevation, wave_crests, wave_troughs, threshold=8
    )
    assert not res

    # make first wave 10 times higher
    elevation[:10] *= 2.5
    wave_crests[0] = 10
    wave_troughs[0] = -10

    res = operators.check_flag_d(
        elevation, wave_crests, wave_troughs, threshold=8
    )
    assert res


def test_flag_e():
    time = np.arange(0, 180, 1.28)
    time_s = create_datetime(time)

    res = operators.check_flag_e(time_s)
    assert not res

    time[-1] += 1
    time_s = create_datetime(time)

    res = operators.check_flag_e(time_s)
    assert res


def test_flag_f():
    elevation = np.ones(100)

    elevation[:50] = np.nan
    res = operators.check_flag_f(elevation, threshold=0.5)
    assert not res

    elevation[:51] = np.nan
    res = operators.check_flag_f(elevation, threshold=0.5)
    assert res


def test_flag_g():
    zero_crossing_periods = np.ones(10)
    res = operators.check_flag_g(zero_crossing_periods, threshold=10)
    assert not res

    zero_crossing_periods = np.ones(9)
    res = operators.check_flag_g(zero_crossing_periods, threshold=10)
    assert res
