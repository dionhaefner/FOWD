"""
constants.py

Constants relevant for processing
"""

# length of SSH window to subtract before finding waves (in  minutes)
SSH_REFERENCE_INTERVAL = 30

# length of wave history to keep in memory for QC and statistics (in minutes)
WAVE_HISTORY_LENGTH = 30

# time series window size for spectrum calculation (in seconds)
SPECTRUM_WINDOW_SIZE = 90

# frequency bands, lower and upper limit (in Hz)
FREQUENCY_INTERVALS = (
    (0, 0.05),
    (0.05, 0.1),
    (0.1, 0.25),
    (0.25, 1.5),
    (0.08, 0.5)
)

# sea state aggregation periods (in minutes)
SEA_STATE_INTERVALS = (10, 30)

# gravitational acceleration in m/s^2
GRAVITY = 9.81

# water density in kg/m^3
DENSITY = 1024

# QC thresholds
QC_FLAG_A_THRESHOLD = 25  # maximum allowed zero-crossing period (in seconds)
QC_FLAG_B_THRESHOLD = 2  # maximum allowed multiple of limit rate of change
QC_FLAG_C_THRESHOLD = 10  # maximum allowed number of consecutive identical values
QC_FLAG_D_THRESHOLD = 8  # maximum allowed exceedance of surface elevation MADN
QC_FLAG_E_THRESHOLD = None  # not used
QC_FLAG_F_THRESHOLD = 0.05  # maximum allowed ratio of invalid data
QC_FLAG_G_THRESHOLD = 100  # minimum number of zero-crossing periods in wave history
