"""
output.py

Defines output metadata and handles I/O
"""

import datetime

from .constants import FREQUENCY_INTERVALS, SEA_STATE_INTERVALS, RAW_ELEVATION_SIZE
from .operators import get_proc_version

import numpy as np
import xarray as xr


DATASET_VARIABLES = dict(
    # metadata
    meta_source_file_name=dict(
        dims=('meta_station_name', 'wave_id_local',),
        attrs=dict(
            long_name='File name of raw input data file',
        )
    ),
    meta_source_file_uuid=dict(
        dims=('meta_station_name', 'wave_id_local',),
        attrs=dict(
            long_name='UUID of raw input data file',
        )
    ),

    wave_id_global=dict(
        dims=('meta_station_name', 'wave_id_local',),
        attrs=dict(
            long_name='Unique identifier for any given wave',
        )
    ),

    wave_start_time=dict(
        dims=('meta_station_name', 'wave_id_local',),
        attrs=dict(
            long_name='UTC wave start time',
        ),
    ),

    wave_end_time=dict(
        dims=('meta_station_name', 'wave_id_local',),
        attrs=dict(
            long_name='UTC wave end time',
        ),
    ),

    wave_zero_crossing_period=dict(
        dims=('meta_station_name', 'wave_id_local',),
        attrs=dict(
            long_name='Wave zero-crossing period relative to 30m SSH',
            units='seconds',
            comment='Zero-crossings determined through linear interpolation',
        )
    ),

    wave_zero_crossing_wavelength=dict(
        dims=('meta_station_name', 'wave_id_local',),
        attrs=dict(
            long_name='Wave zero-crossing wavelength relative to 30m SSH',
            units='meters',
        )
    ),

    wave_raw_elevation=dict(
        dims=('meta_station_name', 'wave_id_local', 'wave_raw_elevation_time_step'),
        attrs=dict(
            long_name='Raw surface elevation relative to 30m SSH',
            units='meters',
            comment='Spacing in time as given by sampling_rate',
        )
    ),

    wave_crest_height=dict(
        dims=('meta_station_name', 'wave_id_local',),
        attrs=dict(
            long_name='Wave crest height relative to 30m SSH',
            units='meters',
        )
    ),

    wave_trough_depth=dict(
        dims=('meta_station_name', 'wave_id_local',),
        attrs=dict(
            long_name='Wave trough depth relative to 30m SSH',
            units='meters',
        )
    ),

    wave_height=dict(
        dims=('meta_station_name', 'wave_id_local',),
        attrs=dict(
            long_name='Absolute wave height relative to 30m SSH',
            units='meters',
        )
    ),

    wave_maximum_elevation_slope=dict(
        dims=('meta_station_name', 'wave_id_local',),
        attrs=dict(
            long_name='Maximum slope of surface elevation in time',
            units='meters per second',
        )
    ),

    # metadata
    meta_deploy_latitude=dict(
        dims=('meta_station_name', 'wave_id_local',),
        attrs=dict(
            long_name='Deploy latitude of instrument',
            units='degrees_north',
        )
    ),

    meta_deploy_longitude=dict(
        dims=('meta_station_name', 'wave_id_local',),
        attrs=dict(
            long_name='Deploy longitude of instrument',
            units='degrees_east',
        )
    ),

    meta_water_depth=dict(
        dims=('meta_station_name', 'wave_id_local',),
        attrs=dict(
            long_name='Water depth at deployment location',
            units='meters',
            positive='down',
        )
    ),

    meta_sampling_rate=dict(
        dims=('meta_station_name', 'wave_id_local',),
        attrs=dict(
            long_name='Measurement sampling frequency in time',
            units='Hz',
        )
    ),
)

# sea state parameters
for interval in SEA_STATE_INTERVALS:
    DATASET_VARIABLES.update({
        f'sea_state_{interval}m_start_time': dict(
            dims=('meta_station_name', 'wave_id_local',),
            attrs=dict(
                long_name='UTC sea state aggregation start time',
            )
        ),
        f'sea_state_{interval}m_end_time': dict(
            dims=('meta_station_name', 'wave_id_local',),
            attrs=dict(
                long_name='UTC sea state aggregation end time',
            )
        ),
        f'sea_state_{interval}m_significant_wave_height_spectral': dict(
            dims=('meta_station_name', 'wave_id_local',),
            attrs=dict(
                long_name='Significant wave height estimated from spectral density spectrum (Hm0)',
                units='meters',
            )
        ),
        f'sea_state_{interval}m_significant_wave_height_direct': dict(
            dims=('meta_station_name', 'wave_id_local',),
            attrs=dict(
                long_name='Significant wave height estimated from wave history (H1/3)',
                units='meters',
            )
        ),
        f'sea_state_{interval}m_mean_period_direct': dict(
            dims=('meta_station_name', 'wave_id_local',),
            attrs=dict(
                long_name='Mean zero-crossing period estimated from wave history',
                units='seconds',
            )
        ),
        f'sea_state_{interval}m_mean_period_spectral': dict(
            dims=('meta_station_name', 'wave_id_local',),
            attrs=dict(
                long_name='Mean wave period estimated from spectral density spectrum',
                units='seconds',
            )
        ),
        f'sea_state_{interval}m_sea_surface_height': dict(
            dims=('meta_station_name', 'wave_id_local',),
            attrs=dict(
                long_name='Sea surface height (mean of vertical displacement)',
                units='meters',
            )
        ),
        f'sea_state_{interval}m_skewness': dict(
            dims=('meta_station_name', 'wave_id_local',),
            attrs=dict(
                long_name='Skewness of sea surface elevation',
            )
        ),
        f'sea_state_{interval}m_kurtosis': dict(
            dims=('meta_station_name', 'wave_id_local',),
            attrs=dict(
                long_name='Excess kurtosis of sea surface elevation',
            )
        ),
        f'sea_state_{interval}m_valid_data_ratio': dict(
            dims=('meta_station_name', 'wave_id_local',),
            attrs=dict(
                long_name='Ratio of valid measurements to all measurements',
                valid_min=0,
                valid_max=1,
            )
        ),
        f'sea_state_{interval}m_peak_wave_period': dict(
            dims=('meta_station_name', 'wave_id_local',),
            attrs=dict(
                long_name='Dominant wave period',
                units='seconds',
            )
        ),
        f'sea_state_{interval}m_peak_wavelength': dict(
            dims=('meta_station_name', 'wave_id_local',),
            attrs=dict(
                long_name='Dominant wavelength',
                units='meters',
            )
        ),
        f'sea_state_{interval}m_steepness': dict(
            dims=('meta_station_name', 'wave_id_local',),
            attrs=dict(
                long_name='Dominant wave steepness',
            )
        ),
        f'sea_state_{interval}m_bandwidth_quality_factor': dict(
            dims=('meta_station_name', 'wave_id_local',),
            attrs=dict(
                long_name=(
                    'Spectral bandwidth estimated through spectral peakedness '
                    '(quality factor)',
                )
            )
        ),
        f'sea_state_{interval}m_bandwidth_narrowness': dict(
            dims=('meta_station_name', 'wave_id_local',),
            attrs=dict(
                long_name='Spectral bandwidth estimated through spectral narrowness',
            )
        ),
        f'sea_state_{interval}m_energy_in_frequency_interval': dict(
            dims=('meta_station_name', 'wave_id_local', 'meta_frequency_band'),
            attrs=dict(
                long_name='Total power contained in frequency band',
                units='W',
            )
        ),
    })

# directional parameters
DATASET_VARIABLES.update(dict(
    direction_sampling_time=dict(
        dims=('meta_station_name', 'wave_id_local',),
        attrs=dict(
            long_name='Time at which directional quantities are sampled',
        )
    ),
    direction_dominant_spread_in_frequency_interval=dict(
        dims=('meta_station_name', 'wave_id_local', 'meta_frequency_band'),
        attrs=dict(
            long_name='Dominant directional spread in frequency band',
            units='degrees',
            valid_min=0,
            valid_max=90,
        )
    ),
    direction_dominant_direction_in_frequency_interval=dict(
        dims=('meta_station_name', 'wave_id_local', 'meta_frequency_band'),
        attrs=dict(
            long_name='Dominant wave direction in frequency band',
            units='degrees',
            valid_min=0,
            valid_max=360,
        )
    ),
    direction_peak_wave_direction=dict(
        dims=('meta_station_name', 'wave_id_local',),
        attrs=dict(
            long_name='Peak wave direction relative to normal-north',
            units='degrees',
            valid_min=0,
            valid_max=360
        )
    ),
))


freq_lower, freq_upper = list(zip(*FREQUENCY_INTERVALS))

EXTRA_VARIABLES = dict(
    meta_frequency_band_lower=dict(
        data=np.array(freq_lower, dtype='float32'),
        dims=('meta_frequency_band',),
        attrs=dict(
            long_name='Lower limit of frequency band',
            units='Hz',
        ),
    ),
    meta_frequency_band_upper=dict(
        data=np.array(freq_upper, dtype='float32'),
        dims=('meta_frequency_band',),
        attrs=dict(
            long_name='Upper limit of frequency band',
            units='Hz',
        ),
    ),
)


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
)


def create_output_dataset(wave_records, station_name):
    dataset_metadata = dict(
        id='FOWD_%s' % station_name,
        title='',
        summary='',
        project='Free Ocean Wave Dataset (FOWD)',
        keywords=(
            'EARTH SCIENCE, OCEANS, OCEAN WAVES, GRAVITY WAVES, WIND WAVES, '
            'SIGNIFICANT WAVE HEIGHT, WAVE FREQUENCY, WAVE PERIOD, WAVE SPECTRA'
        ),
        processing_version=get_proc_version(),
        processing_url='',
        date_created=f'{datetime.datetime.utcnow():%Y%m%dT%H%M%S}',
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

    def prepare_data(data):
        if np.issubdtype(data.dtype, np.float64):
            data = data.astype('float32')

        return data[None, ...]

    data_vars = {
        name: (meta['dims'], prepare_data(
            wave_records[name]), meta['attrs'])
        for name, meta in DATASET_VARIABLES.items()
    }

    data_vars.update({
        name: (meta['dims'], meta['data'], meta['attrs'])
        for name, meta in EXTRA_VARIABLES.items()
    })

    out_dataset = xr.Dataset(
        data_vars=data_vars,
        coords={
            'meta_station_name': [station_name],
            'wave_id_local': wave_records['wave_id_local'],
            'wave_raw_elevation_time_step': np.arange(RAW_ELEVATION_SIZE),
            'meta_frequency_band': np.arange(len(FREQUENCY_INTERVALS)),
        },
        attrs=dataset_metadata
    )

    for coord, attrs in COORD_ATTRS.items():
        out_dataset[coord].attrs.update(attrs)

    return out_dataset


def write_records(wave_records, filename, station_name):
    out_dataset = create_output_dataset(wave_records, station_name)
    out_dataset.to_netcdf(filename, engine='h5netcdf', format='NETCDF4')
