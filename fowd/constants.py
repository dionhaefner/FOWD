"""
constants.py

Constants relevant for processing.
"""

# length of SSH window to subtract before finding waves (in  minutes)
SSH_REFERENCE_INTERVAL = 30

# length of wave history to keep in memory for QC and statistics (in minutes)
WAVE_HISTORY_LENGTH = 30

# time series window size for spectrum calculation (in seconds)
SPECTRUM_WINDOW_SIZE = 180

# frequency bands, lower and upper limit (in Hz)
FREQUENCY_INTERVALS = (
    (0, 0.05),
    (0.05, 0.1),
    (0.1, 0.25),
    (0.25, 1.5),
    (0.08, 0.5)
)

# sea state aggregation periods (in minutes)
SEA_STATE_INTERVALS = ('dynamic',)

# settings for dynamic sea state aggregation period
# the shortest and longest permitted dynamic window sizes (in minutes)
DYNAMIC_WINDOW_LENGTH_BOUNDS = (10, 60)
# time in minutes after which the dynamic window size is re-computed
DYNAMIC_WINDOW_UPDATE_FREQUENCY = 30
# dynamic window calculation is based on this many minutes before and after current wave
DYNAMIC_WINDOW_REFERENCE_PERIOD = (12 * 60, 0)
# total number of different window lengths to try within length bounds
NUM_DYNAMIC_WINDOWS = 11
# number of times each window is re-evaluated for better statistics
NUM_DYNAMIC_WINDOW_SAMPLES = 10

# gravitational acceleration in m/s^2
GRAVITY = 9.81

# reference density in kg/m^3
DENSITY = 1024

# QC thresholds
# maximum allowed zero-crossing period (in seconds)
QC_FLAG_A_THRESHOLD = 25
# maximum allowed multiple of limit rate of change
QC_FLAG_B_THRESHOLD = 2
# maximum allowed number of consecutive identical values
QC_FLAG_C_THRESHOLD = 10
# maximum allowed exceedance of surface elevation MADN
QC_FLAG_D_THRESHOLD = 8
# maximum float precision to which sampling rate has to be uniform
QC_FLAG_E_THRESHOLD = 2
# maximum allowed ratio of invalid data
QC_FLAG_F_THRESHOLD = 0.05
# minimum number of zero-crossing periods in wave history
QC_FLAG_G_THRESHOLD = 100

# QC logging options
# wave height (in units of sig. wave height) for which failed QC is logged
QC_FAIL_LOG_THRESHOLD = 2
# wave height (in units of sig. wave height) above which waves are always logged
QC_EXTREME_WAVE_LOG_THRESHOLD = 2.5
