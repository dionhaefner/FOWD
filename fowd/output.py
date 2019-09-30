"""
output.py

Output metadata and I/O handling
"""

import datetime

from .constants import FREQUENCY_INTERVALS, SEA_STATE_INTERVALS
from .operators import get_proc_version

import numpy as np
import netCDF4


# times are measured in milliseconds since this date
TIME_ORIGIN = '1980-01-01'

# fill values to use
FILL_VALUE_NUMBER = -9999
FILL_VALUE_STR = 'MISSING'

# chunk sizes to use for each dimension
CHUNKSIZES = {
    'meta_station_name': 1,
    'wave_id_local': 1000,
    'meta_frequency_band': len(FREQUENCY_INTERVALS),
}

DATASET_VARIABLES = dict(
    # metadata
    meta_source_file_name=dict(
        dims=('wave_id_local',),
        dtype=str,
        attrs=dict(
            long_name='File name of raw input data file',
        )
    ),
    meta_source_file_uuid=dict(
        dims=('wave_id_local',),
        dtype=str,
        attrs=dict(
            long_name='UUID of raw input data file',
        )
    ),

    # wave data
    wave_id_global=dict(
        dims=('wave_id_local',),
        dtype=str,
        attrs=dict(
            long_name='Unique identifier for any given wave',
        )
    ),

    wave_start_time=dict(
        dims=('wave_id_local',),
        dtype='int64',
        attrs=dict(
            long_name='Wave start time',
            units=f'milliseconds since {TIME_ORIGIN}',
        ),

    ),

    wave_end_time=dict(
        dims=('wave_id_local',),
        dtype='int64',
        attrs=dict(
            long_name='Wave end time',
            units=f'milliseconds since {TIME_ORIGIN}',
        ),
    ),

    wave_zero_crossing_period=dict(
        dims=('wave_id_local',),
        dtype='float32',
        attrs=dict(
            long_name='Wave zero-crossing period relative to 30m SSH',
            units='seconds',
            comment='Zero-crossings determined through linear interpolation',
        )
    ),

    wave_zero_crossing_wavelength=dict(
        dims=('wave_id_local',),
        dtype='float32',
        attrs=dict(
            long_name='Wave zero-crossing wavelength relative to 30m SSH',
            units='meters',
        )
    ),

    wave_raw_elevation=dict(
        dims=('wave_id_local',),
        dtype='vlen',
        attrs=dict(
            long_name='Raw surface elevation relative to 30m SSH',
            units='meters',
            comment='Spacing in time as given by meta_sampling_rate',
        )
    ),

    wave_crest_height=dict(
        dims=('wave_id_local',),
        dtype='float32',
        attrs=dict(
            long_name='Wave crest height relative to 30m SSH',
            units='meters',
        )
    ),

    wave_trough_depth=dict(
        dims=('wave_id_local',),
        dtype='float32',
        attrs=dict(
            long_name='Wave trough depth relative to 30m SSH',
            units='meters',
        )
    ),

    wave_height=dict(
        dims=('wave_id_local',),
        dtype='float32',
        attrs=dict(
            long_name='Absolute wave height relative to 30m SSH',
            units='meters',
        )
    ),

    wave_ursell_number=dict(
        dims=('wave_id_local',),
        dtype='float32',
        attrs=dict(
            long_name='Ursell number',
            units='1',
            valid_min='0',
        )
    ),

    wave_maximum_elevation_slope=dict(
        dims=('wave_id_local',),
        dtype='float32',
        attrs=dict(
            long_name='Maximum slope of surface elevation in time',
            units='m s-1',
        )
    ),

    # station metadata
    meta_deploy_latitude=dict(
        dims=('wave_id_local',),
        dtype='float32',
        attrs=dict(
            long_name='Deploy latitude of instrument',
            units='degrees_north',
        )
    ),

    meta_deploy_longitude=dict(
        dims=('wave_id_local',),
        dtype='float32',
        attrs=dict(
            long_name='Deploy longitude of instrument',
            units='degrees_east',
        )
    ),

    meta_water_depth=dict(
        dims=('wave_id_local',),
        dtype='float32',
        attrs=dict(
            long_name='Water depth at deployment location',
            units='meters',
            positive='down',
        )
    ),

    meta_sampling_rate=dict(
        dims=('wave_id_local',),
        dtype='float32',
        attrs=dict(
            long_name='Measurement sampling frequency in time',
            units='hertz',
        )
    ),
)

# sea state parameter metadata
for interval in SEA_STATE_INTERVALS:
    DATASET_VARIABLES.update({
        f'sea_state_{interval}m_start_time': dict(
            dims=('wave_id_local',),
            dtype='int64',
            attrs=dict(
                long_name='Sea state aggregation start time',
                units=f'milliseconds since {TIME_ORIGIN}',
            )
        ),
        f'sea_state_{interval}m_end_time': dict(
            dims=('wave_id_local',),
            dtype='int64',
            attrs=dict(
                long_name='Sea state aggregation end time',
                units=f'milliseconds since {TIME_ORIGIN}',
            )
        ),
        f'sea_state_{interval}m_significant_wave_height_spectral': dict(
            dims=('wave_id_local',),
            dtype='float32',
            attrs=dict(
                long_name='Significant wave height estimated from spectral density spectrum (Hm0)',
                units='meters',
            )
        ),
        f'sea_state_{interval}m_significant_wave_height_direct': dict(
            dims=('wave_id_local',),
            dtype='float32',
            attrs=dict(
                long_name='Significant wave height estimated from wave history (H1/3)',
                units='meters',
            )
        ),
        f'sea_state_{interval}m_maximum_wave_height': dict(
            dims=('wave_id_local',),
            dtype='float32',
            attrs=dict(
                long_name='Maximum wave height estimated from wave history',
                units='meters',
            )
        ),
        f'sea_state_{interval}m_mean_period_direct': dict(
            dims=('wave_id_local',),
            dtype='float32',
            attrs=dict(
                long_name='Mean zero-crossing period estimated from wave history',
                units='seconds',
            )
        ),
        f'sea_state_{interval}m_mean_period_spectral': dict(
            dims=('wave_id_local',),
            dtype='float32',
            attrs=dict(
                long_name='Mean wave period estimated from spectral density spectrum',
                units='seconds',
            )
        ),
        f'sea_state_{interval}m_skewness': dict(
            dims=('wave_id_local',),
            dtype='float32',
            attrs=dict(
                long_name='Skewness of sea surface elevation',
                units='1',
            )
        ),
        f'sea_state_{interval}m_kurtosis': dict(
            dims=('wave_id_local',),
            dtype='float32',
            attrs=dict(
                long_name='Excess kurtosis of sea surface elevation',
                units='1',
            )
        ),
        f'sea_state_{interval}m_valid_data_ratio': dict(
            dims=('wave_id_local',),
            dtype='float32',
            attrs=dict(
                long_name='Ratio of valid measurements to all measurements',
                valid_min=0,
                valid_max=1,
                units='1',
            )
        ),
        f'sea_state_{interval}m_peak_wave_period': dict(
            dims=('wave_id_local',),
            dtype='float32',
            attrs=dict(
                long_name='Dominant wave period',
                units='seconds',
            )
        ),
        f'sea_state_{interval}m_peak_wavelength': dict(
            dims=('wave_id_local',),
            dtype='float32',
            attrs=dict(
                long_name='Dominant wavelength',
                units='meters',
            )
        ),
        f'sea_state_{interval}m_steepness': dict(
            dims=('wave_id_local',),
            dtype='float32',
            attrs=dict(
                long_name='Dominant wave steepness',
                units='1',
            )
        ),
        f'sea_state_{interval}m_bandwidth_peakedness': dict(
            dims=('wave_id_local',),
            dtype='float32',
            attrs=dict(
                long_name=(
                    'Spectral bandwidth estimated through spectral peakedness '
                    '(quality factor)',
                ),
                units='1',
            )
        ),
        f'sea_state_{interval}m_bandwidth_narrowness': dict(
            dims=('wave_id_local',),
            dtype='float32',
            attrs=dict(
                long_name='Spectral bandwidth estimated through spectral narrowness',
                units='1',
            )
        ),
        f'sea_state_{interval}m_benjamin_feir_index_peakedness': dict(
            dims=('wave_id_local',),
            dtype='float32',
            attrs=dict(
                long_name='Benjamin-Feir index estimated through steepness and peakedness',
                units='1',
            )
        ),
        f'sea_state_{interval}m_benjamin_feir_index_narrowness': dict(
            dims=('wave_id_local',),
            dtype='float32',
            attrs=dict(
                long_name='Benjamin-Feir index estimated through steepness and narrowness',
                units='1',
            )
        ),
        f'sea_state_{interval}m_groupiness': dict(
            dims=('wave_id_local',),
            dtype='float32',
            attrs=dict(
                long_name='Groupiness parameter (r) estimated from spectral density',
                units='1',
                valid_min='0',
                valid_max='1',
            )
        ),
        f'sea_state_{interval}m_energy_in_frequency_interval': dict(
            dims=('wave_id_local', 'meta_frequency_band'),
            dtype='float32',
            attrs=dict(
                long_name='Total power contained in frequency band',
                units='watts',
            )
        ),
    })

# directional parameter metadata
DATASET_VARIABLES.update(dict(
    direction_sampling_time=dict(
        dims=('wave_id_local',),
        dtype='int64',
        attrs=dict(
            long_name='Time at which directional quantities are sampled',
        ),
        units=f'milliseconds since {TIME_ORIGIN}'
    ),
    direction_dominant_spread_in_frequency_interval=dict(
        dims=('wave_id_local', 'meta_frequency_band'),
        dtype='float32',
        attrs=dict(
            long_name='Dominant directional spread in frequency band',
            units='degrees',
            valid_min=0,
            valid_max=90,
        )
    ),
    direction_dominant_direction_in_frequency_interval=dict(
        dims=('wave_id_local', 'meta_frequency_band'),
        dtype='float32',
        attrs=dict(
            long_name='Dominant wave direction in frequency band',
            units='degrees',
            valid_min=0,
            valid_max=360,
        )
    ),
    direction_peak_wave_direction=dict(
        dims=('wave_id_local',),
        dtype='float32',
        attrs=dict(
            long_name='Peak wave direction relative to normal-north',
            units='degrees',
            valid_min=0,
            valid_max=360
        )
    ),
))


freq_lower, freq_upper = list(zip(*FREQUENCY_INTERVALS))

# additional output variables that are constant across stations
EXTRA_VARIABLES = dict(
    meta_frequency_band_lower=dict(
        data=np.array(freq_lower, dtype='float32'),
        dims=('meta_frequency_band',),
        attrs=dict(
            long_name='Lower limit of frequency band',
            units='hertz',
        ),
    ),
    meta_frequency_band_upper=dict(
        data=np.array(freq_upper, dtype='float32'),
        dims=('meta_frequency_band',),
        attrs=dict(
            long_name='Upper limit of frequency band',
            units='hertz',
        ),
    ),
)


# attributes added to coordinate variables
COORD_ATTRS = dict(
    meta_station_name=dict(
        long_name='Name of original measurement station',
    ),
    wave_id_local=dict(
        long_name='Incrementing wave ID for given station',
        comment=(
            'This ID is not guaranteed to denote the same wave between data versions. '
            'Use wave_id_global instead.'
        ),
    ),
    meta_frequency_band=dict(
        long_name='Index of frequency band',
        comment=(
            'Frequency ranges are given by '
            '(meta_frequency_band_lower, meta_frequency_band_upper)'
        ),
    ),
)


def write_records(wave_records, filename, station_name):
    """Write given records to netCDF4"""

    dataset_metadata = dict(
        id=f'FOWD_{station_name}',
        title=f'Free Ocean Wave Dataset (FOWD), station {station_name}',
        summary=(
            'A catalogue of ocean waves and associated sea states, derived from in-situ '
            'measurement data.'
        ),
        project='Free Ocean Wave Dataset (FOWD)',
        keywords=(
            'EARTH SCIENCE, OCEANS, OCEAN WAVES, GRAVITY WAVES, WIND WAVES, '
            'SIGNIFICANT WAVE HEIGHT, WAVE FREQUENCY, WAVE PERIOD, WAVE SPECTRA'
        ),
        processing_version=get_proc_version(),
        processing_url='https://github.com/dionhaefner/FOWD',
        date_created=f'{datetime.datetime.utcnow():%Y-%m-%dT%H:%M:%S.%f}',
        creator_name='NBI Copenhagen',
        creator_url='https://climate-geophysics.nbi.ku.dk/research/oceanography/',
        creator_email='dion.haefner@nbi.ku.dk',
        institution='Niels Bohr Institute, University of Copenhagen',
        contributor_name='CDIP, CDBW/USACE',
        contributor_role='station operation, station funding',
        geospatial_lat_units='degrees_north',
        geospatial_lat_resolution=1e-5,
        geospatial_lon_units='degrees_east',
        geospatial_lon_resolution=1e-5,
        geospatial_vertical_units='meters',
        geospatial_vertical_origin='sea surface height',
        geospatial_vertical_positive='up',
        time_coverage_start=str(wave_records['wave_start_time'].min()),
        time_coverage_end=str(wave_records['wave_end_time'].max()),
        source='insitu observations',
        license='These data may be redistributed and used without restriction.',
        acknowledgment=(
            'CDIP is supported by the U.S. Army Corps of Engineers (USACE) and the California '
            'Department of Boating and Waterways (CDBW). The instrument that collected this '
            'dataset was funded by CDBW/USACE and operated by CDIP.'
        ),
        comment=(
            ''
        ),
    )

    dimension_data = (
        # (name, dtype, data)
        ('meta_station_name', str, np.array([np.string_(station_name)])),
        ('wave_id_local', 'int64', wave_records['wave_id_local']),
        ('meta_frequency_band', 'uint8', np.arange(len(FREQUENCY_INTERVALS))),
    )

    with netCDF4.Dataset(filename, 'w') as f:
        # set global metadata
        for attr, val in dataset_metadata.items():
            setattr(f, attr, val)

        # some variables have variable length
        vlen_type = f.createVLType('float32', 'float_array')

        # create dimensions
        for dim, dtype, val in dimension_data:
            f.createDimension(dim, len(val))
            v = f.createVariable(dim, dtype, (dim,))
            v[:] = val

        for name, meta in DATASET_VARIABLES.items():
            # add meta_station_name as additional scalar dimension
            data = wave_records[name][None, ...]
            dims = ('meta_station_name',) + meta['dims']

            # compression args
            extra_args = dict(
                zlib=True, fletcher32=True,
                chunksizes=[CHUNKSIZES[dim] for dim in dims]
            )

            # determine dtype
            if meta['dtype'] == 'vlen':
                dtype = vlen_type
            else:
                dtype = meta['dtype']

            # add correct fill value
            is_number = np.issubdtype(dtype, np.floating) or np.issubdtype(dtype, np.integer)
            if is_number and dtype is not vlen_type:
                extra_args.update(fill_value=FILL_VALUE_NUMBER)
            elif np.issubdtype(dtype, np.character):
                extra_args.update(fill_value=FILL_VALUE_STR)

            # convert datetimes to ms since time origin
            if np.issubdtype(data.dtype, np.datetime64):
                data = (data - np.datetime64(TIME_ORIGIN)) / np.timedelta64(1, 'ms')

            v = f.createVariable(name, dtype, dims, **extra_args)
            v[...] = data

            # attach attributes
            for attr, val in meta['attrs'].items():
                setattr(v, attr, val)

        # add extra variables
        for name, meta in EXTRA_VARIABLES.items():
            v = f.createVariable(name, meta['data'].dtype, meta['dims'])
            v[:] = meta['data']
            for attr, val in meta['attrs'].items():
                setattr(v, attr, val)

        # add coordinate attributes
        for coord, attrs in COORD_ATTRS.items():
            for attr, val in attrs.items():
                setattr(f[coord], attr, val)
